from __future__ import annotations

from tqdm import tqdm
from typing import Tuple, List, Dict, Union
from numpy.typing import NDArray
import numpy as np

import torch

import torch.utils.data
from torch import  optim
from torch.utils.data import DataLoader

from models import ProtoNet, ProtoNetParallerWrapper
from data_loader import FewShotBatchSampler, PointCloudDataset, NPYDataset
from config import Config
from test.test_protonet import test_proto
from utils.plot import plot_training_val_fewshot, plot_query, plot_support
from .utils import save_support_query_prediction_samples

def evaluate(model: Union[ProtoNet, ProtoNetParallerWrapper], val_loader: DataLoader) -> Tuple[float, ...]:
    
    with torch.no_grad():
        model.eval()

        total_loss = 0
        total_acc = 0
        acc = 0

        for batch in tqdm(val_loader, desc='Evaluating batch'):
            point_clouds, labels = batch
            
            point_clouds, labels = point_clouds.to(model.device), labels.to(model.device)
            pcd_embeddings = model(point_clouds)

            support_features, query_features, support_labels, query_labels = split_batch(pcd_embeddings, labels) 

            prototypes, prototype_labels = model.compute_prototypes(support_features, support_labels)

            _, loss, acc = model.classify_features(prototypes, prototype_labels, query_features, query_labels)

            total_loss += loss.item()
            total_acc += acc.item()

        return total_loss/(len(val_loader)), total_acc/(len(val_loader)) # type: ignore

def train(config: Config):
    training_params = config.trainig_params
    few_shot_params = config.few_shot_params
    dataset_params = config.dataset_params
    testing_params = config.testing_params

    torch.manual_seed(training_params.seed)


    dataset_params.name = f'{dataset_params.name}_{few_shot_params.n_ways}_{few_shot_params.k_shots}'
    dataset_params.batch_size = few_shot_params.n_ways*few_shot_params.k_shots

    device_ids = training_params.device_ids
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProtoNet(num_classes=dataset_params.num_classes, device=device, pretrained_ckpts=None, use_attention=training_params.attention)

    if device_ids is not None:
        model = ProtoNetParallerWrapper(model, device_ids)
    model.to(device)

    print(f'====================Using {device} to train {dataset_params.name} Dataset======================')

    dataset = NPYDataset(dataset_params.dataset, dataset_params.label)

    optimizer = optim.AdamW(model.parameters(), lr=training_params.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=training_params.epochs, eta_min=(training_params.learning_rate)*0.0001, last_epoch=-1)

    train_cls_idx, validation_cls_idx, test_cls_idx = dataset.train_test_val_class_indices_split(train_ratio=training_params.training_split, seed=training_params.seed)

    train_set = PointCloudDataset.from_dataset(dataset.pcds_to_tensor(), dataset.labels_to_tensor(), train_cls_idx)
    validation_set = PointCloudDataset.from_dataset(dataset.pcds_to_tensor(), dataset.labels_to_tensor(), validation_cls_idx)
    test_set = PointCloudDataset.from_dataset(dataset.pcds_to_tensor(), dataset.labels_to_tensor(), test_cls_idx)

    if testing_params.dataset_path is not None:
        test_set.save_dataset(testing_params.dataset_path)

    train_batch_sampler = FewShotBatchSampler(train_set.labels, n_ways=few_shot_params.n_ways, k_shots=few_shot_params.k_shots, include_query=True)
    val_batch_sampler = FewShotBatchSampler(validation_set.labels, n_ways=few_shot_params.n_ways, k_shots=few_shot_params.k_shots, include_query=True)

    train_loader = DataLoader(
        train_set,
        batch_sampler=train_batch_sampler,
        num_workers=int(dataset_params.data_loader_num_workers))


    val_loader = DataLoader(
        validation_set,
        batch_sampler=val_batch_sampler,
        num_workers=int(dataset_params.data_loader_num_workers))

    training_loss_per_epoch = []
    training_acc_per_epoch = []
    val_loss_per_epoch = []
    val_acc_per_epoch = []
    best_val_acc = 0
    best_loss_val = float('inf')

    support_samples: List[Dict[str, NDArray]] = []
    query_samples: List[Dict[str, NDArray]] = []
    predicted_logits: List[Dict[str, NDArray]] = []

    for epoch in range(training_params.epochs):
        model.train()
        total_loss = 0
        total_acc = 0

        for batch in tqdm(train_loader, desc=f'Training batch'):
            point_clouds, labels = batch

            pcd_support , pcd_query = np.split(point_clouds, 2, axis=0)

            point_clouds, labels = point_clouds.to(device), labels.to(device)

            pcd_embeddings = model(point_clouds)

            support_features, query_features, support_labels, query_labels = split_batch(pcd_embeddings, labels)


            prototypes, classes = model.compute_prototypes(support_features, support_labels)
            logits, loss, acc = model.classify_features(prototypes, classes, query_features, query_labels)

            if(epoch == 0 or epoch==9):
                support_samples.append({'pcd': pcd_support, 'label': support_labels.cpu().numpy()})
                query_samples.append({'pcd': pcd_query, 'label': query_labels.cpu().numpy()})
                predicted_logits.append({'logtis': logits.cpu().detach().numpy()}) 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc +=acc.item()


        scheduler.step()

        val_loss, val_acc = evaluate(model, val_loader)

        print(f"Epoch {epoch+1}/{training_params.epochs}, Training_loss: {total_loss/(len(train_loader))}, Training_acc:{total_acc/len(train_loader)}, Val_loss: {val_loss}, Val_acc: {val_acc}") # type: ignore

        if val_acc > best_val_acc and val_loss < best_loss_val:
            if epoch > 4:
                best_val_acc = val_acc
                best_loss_val = val_loss
                print('best accuracy')
                print(best_val_acc)
                print('Saving a checkpoint')
                torch.save(model.state_dict(), f'{training_params.ckpts}/{dataset_params.name}_2')

        training_loss_per_epoch.append(total_loss/(len(train_loader)))
        training_acc_per_epoch.append(total_acc/(len(train_loader)))
        val_acc_per_epoch.append(val_acc)
        val_loss_per_epoch.append(val_loss)


    # plot_training_val_fewshot(training_loss_per_epoch, val_loss_per_epoch, dataset_params.experiment_result_uri, dataset_params.name, 'Loss')
    # plot_training_val_fewshot(training_acc_per_epoch, val_acc_per_epoch, dataset_params.experiment_result_uri,dataset_params.name, 'Accuracy')
    # plot_support(support_samples, dataset.unique_classes_map, dataset_params.experiment_result_uri, dataset_params.name, few_shot_params.k_shots, 'train', few_shot_params.n_ways)
    # plot_query(query_samples, predicted_logits, dataset.unique_classes_map, dataset_params.experiment_result_uri, dataset_params.name, few_shot_params.k_shots, 'train', few_shot_params.n_ways)

    # save_support_query_prediction_samples(support_samples, query_samples, predicted_logits, f'{dataset_params.name}_{few_shot_params.n_ways}_{few_shot_params.k_shots}')
    test_proto(model, test_set, testing_params.k_shots, dataset_params, dataset.unique_classes_map)


def split_batch(pcd_features: torch.Tensor, labels: torch.Tensor):
    """
        To be called after iterating on train loader that returns a batch
        to retrive support, query sets with their respective labels for few shot learning.
    """

    support_imgs, query_imgs = pcd_features.chunk(2, dim=0)
    support_targets, query_targets = labels.chunk(2, dim=0)

    return support_imgs, query_imgs, support_targets, query_targets