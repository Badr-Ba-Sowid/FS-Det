from __future__ import annotations

from tqdm import tqdm
from typing import Tuple

import torch
import torch.utils.data
from torch import  optim
from torch.utils.data import DataLoader

from models import ProtoNet
from data_loader import ModelNet40C, FewShotBatchSampler, PointCloudDataset
from config import TrainingConfig



def evaluate(model: ProtoNet, val_loader: DataLoader) -> Tuple[float, ...]:
    model.eval()

    with torch.no_grad():
        total_loss = 0
        total_acc = 0
        acc = 0

        for batch in tqdm(val_loader, desc='Evaluating batch'):
            point_clouds, labels = batch

            pcd_embeddings = model(point_clouds)

            support_features, query_features, support_labels, query_labels = split_batch(pcd_embeddings, labels) 
            prototypes, prototype_labels = model.compute_prototypes(support_features, support_labels)
            predicted_labels = model.predict(prototypes, query_features)

            loss = model.compute_loss(prototypes, prototype_labels, query_features, query_labels)
            total_loss += loss.item()

            actual_labels = (prototype_labels[None:] == query_labels[:, None]).long().argmax(dim=-1)
            acc =(predicted_labels.argmax(dim=1) == actual_labels).float().mean().item()
            total_acc += acc

        return total_loss/len(val_loader), acc/len(val_loader)

def train(config_uri: str):
    config = TrainingConfig.from_file(config_uri)
    training_params = config.trainig_params
    training_uris = config.uris
    few_shot_params = config.few_shot_params
    dataset_params = config.dataset_params

    torch.manual_seed(training_params.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = ProtoNet(num_classes=dataset_params.num_classes, device=device, pretrained_ckpts='ckpts/mode_net_40/model_net_40')

    optimizer = optim.AdamW(model.parameters(), lr=training_params.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=training_params.scheduler_steps, gamma=training_params.scheduler_gamma)

    model_net_dataset = ModelNet40C(training_uris.dataset, training_uris.label)
    train_cls_idx, validation_cls_idx, _ = model_net_dataset.train_test_val_class_indices_split(train_ratio=training_params.training_split, seed=training_params.seed)

    train_set = PointCloudDataset.from_dataset(model_net_dataset.pcds_to_tensor(), model_net_dataset.labels_to_tensor(), train_cls_idx)
    validation_set = PointCloudDataset.from_dataset(model_net_dataset.pcds_to_tensor(), model_net_dataset.labels_to_tensor(), validation_cls_idx)
    
    train_loader = DataLoader(
        train_set,
        batch_sampler=FewShotBatchSampler(train_set.labels, n_ways=few_shot_params.n_ways, k_shots=few_shot_params.k_shots),
        num_workers=int(training_params.data_loader_num_workers))

    val_loader = DataLoader(
        validation_set,
        batch_sampler=FewShotBatchSampler(validation_set.labels, n_ways=few_shot_params.n_ways, k_shots=few_shot_params.k_shots),
        num_workers=int(training_params.data_loader_num_workers))
    
    
    for epoch in range(training_params.epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f'Training batch'):
            point_clouds, labels = batch

            pcd_embeddings = model(point_clouds)

            support_features, query_features, support_labels, query_labels = split_batch(pcd_embeddings, labels)

            prototypes, classes = model.compute_prototypes(support_features, support_labels)
            loss = model.compute_loss(prototypes, classes, query_features, query_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        val_loss, val_acc = evaluate(model, val_loader)

        print(f"Epoch {epoch+1}/{training_params.epochs}, Loss: {total_loss/len(train_loader)}, Val_loss: {val_loss}, Val_acc: {val_acc}")

    torch.save(model.state_dict(), training_uris.ckpts)


def split_batch(pcd_features: torch.Tensor, labels: torch.Tensor):
    """
        To be called after iterating on train loader that returns a batch
        to retrive support, query sets with their respective labels for few shot learning.
    """
    support_imgs, query_imgs = pcd_features.chunk(2, dim=0)
    support_targets, query_targets = labels.chunk(2, dim=0)
    return support_imgs, query_imgs, support_targets, query_targets