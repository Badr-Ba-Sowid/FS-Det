from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns
from statistics import mean, stdev
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

import torch
import torch.utils.data
from torch.utils.data import DataLoader

from models import ProtoNet
from data_loader import PointCloudDataset, NPYDataset
from config import Config

def test(model: ProtoNet, dataset: PointCloudDataset, k_shot: int, batch_size: int, num_worker: int):
    num_classes = (dataset.labels.unique().shape[0])
    

    if dataset.labels.shape[0]%num_classes == 0:
        num_classes = (dataset.labels.unique().shape[0])
    else:
        num_classes = (dataset.labels.unique().shape[0]) -1

    exmps_per_class = dataset.labels.shape[0] // num_classes

    with torch.no_grad():
        model.eval()
        
        test_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_worker)

        pcd_features = []
        pcd_targets = []

        for pcds, targets in tqdm(test_loader, "Extracting point cloud features"):
            pcds = pcds.to(model.device).float().cuda()
            features = model(pcds)

            pcd_features.append(features.detach().cpu())
            pcd_targets.append(targets)
        
        pcd_targets = torch.cat(pcd_targets, dim=0)
        pcd_features = torch.cat(pcd_features, dim=0)

        pcd_targets, sort_idx = pcd_targets.sort()
        pcd_targets = pcd_targets.view((num_classes, exmps_per_class)).transpose(0, 1)
        pcd_features = pcd_features[sort_idx].view(num_classes, exmps_per_class, -1).transpose(0, 1)

    accuracies = []
    predicted_targets = []
    for k_idx in tqdm(range(0, pcd_features.shape[0], k_shot), "Evaluating prototype classification", leave=False):
        # Select support set and calculate prototypes
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

            predicted, _, acc = model.classify_features(prototypes, proto_classes, e_pcd_feats, e_targets)
            batch_acc += acc.item()
            print(k_idx)
            print(predicted.shape)
            predicted_targets.append(predicted.cpu().numpy())
        batch_acc /= pcd_features.shape[0] // k_shot - 1
        accuracies.append(batch_acc)

    return (mean(accuracies), stdev(accuracies)), (pcd_features, pcd_targets, predicted_targets)

def plot_few_shot(acc_dict, name, color=None, ax=None):
    sns.set()
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    ks = sorted(list(acc_dict.keys()))
    mean_accs = [acc_dict[k][0] for k in ks]
    std_accs = [acc_dict[k][1] for k in ks]
    ax.plot(ks, mean_accs, marker="o", markeredgecolor="k", markersize=6, label=name, color=color)
    ax.fill_between(
        ks,
        [m - s for m, s in zip(mean_accs, std_accs)], # type: ignore
        [m + s for m, s in zip(mean_accs, std_accs)],  # type: ignore
        alpha=0.2,
        color=color,
    )
    ax.set_xticks(ks)
    ax.set_xlim([ks[0] - 1, ks[-1] + 1])  # type: ignore
    ax.set_xlabel("Number of shots per class", weight="bold")
    ax.set_ylabel("Accuracy", weight="bold")
    if len(ax.get_title()) == 0:
        ax.set_title("Few-Shot Performance " + name, weight="bold")
    else:
        ax.set_title(ax.get_title() + " and " + name, weight="bold")
    ax.legend()
    
    plt.savefig('few_shot_performance')

def prepare_dataset(config: Config):
    dataset_params = config.dataset_params
    test_params = config.testing_params

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = NPYDataset(dataset_params.dataset, dataset_params.label)

    test_set = PointCloudDataset(dataset.pcds_to_tensor(), dataset.labels_to_tensor())

    model = ProtoNet(dataset_params.num_classes, device=device)
    model.load_state_dict(torch.load(test_params.model_state))

    test_proto(model, test_set, test_params.k_shots, dataset_params.batch_size, dataset_params.data_loader_num_workers)

def plot_test_results(test_set: PointCloudDataset, predicted_targets: list[NDArray]):
    fig = plt.figure(figsize=(12,4))
    # labels = test_set.labels.cpu().numpy()
    # print("target " ,len(predicted_targets))


    # for i, (actual, predicted) in enumerate(zip(labels[:5], predicted_targets[:5])):
    #     print(actual, "-", np.argmax(predicted, axis=1).shape)
    #     point_cloud = test_set.pcds[i]
    #     x = point_cloud[0, :]
    #     y = point_cloud[1, :]
    #     z = point_cloud[2, :]

    #     ax = fig.add_subplot(1, 5, i+1, projection='3d')

    #     img = ax.scatter(x, y, z, cmap=plt.hot())
    #     # fig.colorbar(img)
    #     ax.set_xlabel('X')
    #     ax.set_ylabel('Y')
    #     ax.set_zlabel('Z')
    # plt.show()

def test_proto(model: ProtoNet, test_set: PointCloudDataset, k_shots: list[int], batch_size: int, num_worker: int):
    print('===========Begin testing=============')
    protonet_accuracies = dict()

    for k in k_shots:
        protonet_accuracies[k], data_feats = test(model, test_set, k, batch_size, num_worker)
        _, actual_target, predicted_targets = data_feats
        print(
            "Accuracy for k=%i: %4.2f%% (+-%4.2f%%)"
            % (k, 100.0 * protonet_accuracies[k][0], 100 * protonet_accuracies[k][1]))
        # if(k ==5):
            # plot_test_results(actual_target, predicted_targets)


    plot_few_shot(protonet_accuracies, 'ProtoNet')


