from __future__ import annotations

from statistics import mean, stdev
import numpy as np

from numpy.typing import NDArray
from typing import List, Dict

from tqdm import tqdm

import torch
import torch.utils.data
from torch.utils.data import DataLoader

from models import ProtoNet
from data_loader import PointCloudDataset, NPYDataset
from config import Config, DatasetParams
from utils.plot import plot_support, plot_query, plot_few_shot_test_acc_trend

def test(model: ProtoNet, dataset: PointCloudDataset, k_shot: int, batch_size: int, num_worker: int):
    num_classes = (dataset.labels.unique().shape[0])
    

    if dataset.labels.shape[0]%num_classes != 0:
        if dataset.labels.shape[0]%num_classes > 2:
            num_classes = num_classes - 2
        elif dataset.labels.shape[0]%num_classes <= 2:
            num_classes = num_classes - 1

    exmps_per_class = dataset.labels.shape[0] // num_classes

    with torch.no_grad():
        model.eval()
        test_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_worker)

        pcd_features = []
        pcd_targets = []

        pcd_samples = []
        pcd_labels = []

        for pcds, targets in tqdm(test_loader, "Extracting point cloud features"):
            pcd_samples.append(pcds.cpu().numpy())
            pcd_labels.append(targets.cpu().numpy())

            pcds = pcds.to(model.device).float().cuda()
            features = model(pcds)

            pcd_features.append(features.detach().cpu())
            pcd_targets.append(targets)

        pcd_targets = torch.cat(pcd_targets, dim=0)
        pcd_features = torch.cat(pcd_features, dim=0)

        pcd_samples = np.concatenate(pcd_samples, axis=0)
        pcd_labels = np.concatenate(pcd_labels, axis=0)

        pcd_targets, sort_idx = pcd_targets.sort()

        sort_label_idx = np.argsort(pcd_labels)
        pcd_labels = np.sort(pcd_labels)

        pcd_targets = pcd_targets.view((num_classes, exmps_per_class)).transpose(0, 1)
        pcd_labels = (pcd_labels.reshape((num_classes, exmps_per_class))).T

        pcd_features = pcd_features[sort_idx].view(num_classes, exmps_per_class, -1).transpose(0, 1)

        pcd_samples = pcd_samples[sort_label_idx].reshape(num_classes, exmps_per_class, pcd_samples.shape[1], pcd_samples.shape[2]).transpose(1,0,2,3).copy()

    accuracies = []
    predicted_targets = []
    support_samples: List[Dict[str, NDArray]] = []
    query_samples: List[Dict[str, NDArray]] = []

    for k_idx in tqdm(range(0, pcd_features.shape[0], k_shot), "Evaluating prototype classification", leave=False):
        # Select support set and calculate prototypes
        support_sample = pcd_samples[k_idx: k_idx + k_shot]
        support_sample: NDArray = support_sample.reshape(support_sample.shape[0]*support_sample.shape[1], support_sample.shape[2], support_sample.shape[3])
        support_label: NDArray = pcd_labels[k_idx : k_idx + k_shot].flatten()

        support_samples.append({'pcd': support_sample, 'label': support_label}) # type: ignore

        k_pcd_feats = pcd_features[k_idx : k_idx + k_shot].flatten(0, 1)
        k_targets = pcd_targets[k_idx : k_idx + k_shot].flatten(0, 1)

        prototypes, proto_classes = model.compute_prototypes(k_pcd_feats, k_targets)
        # Evaluate accuracy on the rest of the dataset
        batch_acc = 0
        for e_idx in range(0, pcd_features.shape[0], k_shot):
            if k_idx == e_idx:  # Do not evaluate on the support set examples
                continue
            e_pcd_feats = pcd_features[e_idx : e_idx + k_shot].flatten(0, 1)
            e_targets = pcd_targets[e_idx : e_idx + k_shot].flatten(0, 1)

            query_sample = pcd_samples[e_idx : e_idx + k_shot]
            query_sample = query_sample.reshape(query_sample.shape[0]*query_sample.shape[1], query_sample.shape[2], query_sample.shape[3])
            query_label = pcd_labels[e_idx : e_idx + k_shot].flatten()
            query_samples.append({'pcd': query_sample, 'label': query_label}) # type: ignore

            logits, _, acc = model.classify_features(prototypes, proto_classes, e_pcd_feats, e_targets)

            batch_acc += acc.item()

            predicted_targets.append(logits.cpu().numpy())
        batch_acc /= pcd_features.shape[0] // k_shot - 1
        accuracies.append(batch_acc)

    return (mean(accuracies), stdev(accuracies)), (support_samples, query_samples, predicted_targets)


def prepare_dataset(config: Config):
    dataset_params = config.dataset_params
    test_params = config.testing_params

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = NPYDataset(dataset_params.dataset, dataset_params.label)

    if test_params.dataset_path is None:
        test_set = PointCloudDataset(dataset.pcds_to_tensor(), dataset.labels_to_tensor())
    else:
        test_set = PointCloudDataset.from_pickle(test_params.dataset_path)

    model = ProtoNet(dataset_params.num_classes, device=device)
    model.load_state_dict(torch.load(test_params.model_state))

    test_proto(model, test_set, test_params.k_shots, dataset_params, dataset.unique_classes_map)


def plot_support_and_query_results(support_sammples:List[Dict[str, NDArray]], query_samples: List[Dict[str, NDArray]], predicted_targets: List[NDArray], dataset_uniq_label_map: Dict[int, str], root_directory: str, ds_name:str, k_shot: int):

    plot_support(support_sammples, dataset_uniq_label_map, root_directory, ds_name, k_shot)
    plot_query(query_samples, predicted_targets, dataset_uniq_label_map, root_directory, ds_name, k_shot)

def test_proto(model: ProtoNet, test_set: PointCloudDataset, k_shots: List[int], dataset_params: DatasetParams, dataset_uniq_label_map: Dict[int, str]):
    print(f'===========Begin testing on {dataset_params.name}=============')
    protonet_accuracies = dict()

    for k in k_shots:
        protonet_accuracies[k], data_feats = test(model, test_set, k, dataset_params.batch_size, dataset_params.data_loader_num_workers)
        support_samples, query_samples, predicted_targets = data_feats
        print(
            "Accuracy for k=%i: %4.2f%% (+-%4.2f%%)"
            % (k, 100.0 * protonet_accuracies[k][0], 100 * protonet_accuracies[k][1]))

        plot_support_and_query_results(support_samples, query_samples, predicted_targets, dataset_uniq_label_map, dataset_params.experiment_result_uri, dataset_params.name, k)


    plot_few_shot_test_acc_trend(protonet_accuracies, 'ProtoNet', root_director=dataset_params.experiment_result_uri, ds_name=dataset_params.name)


