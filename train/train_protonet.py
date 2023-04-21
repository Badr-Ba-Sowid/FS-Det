from __future__ import annotations

from tqdm import tqdm
from typing import Tuple
import matplotlib.pyplot as plt

import torch
import torch.utils.data
from torch import  optim
from torch.utils.data import DataLoader

from models import ProtoNet
from data_loader import ModelNet40C, FewShotBatchSampler, PointCloudDataset
from config import Config
from test.test_protonet import test_proto


def evaluate(model: ProtoNet, val_loader: DataLoader) -> Tuple[float, ...]:

    with torch.no_grad():
        model.eval()

        total_loss = 0
        total_acc = 0
        acc = 0

        for batch in tqdm(val_loader, desc='Evaluating batch'):
            point_clouds, labels = batch

            pcd_embeddings = model(point_clouds)

            support_features, query_features, support_labels, query_labels = split_batch(pcd_embeddings, labels) 
            prototypes, prototype_labels = model.compute_prototypes(support_features, support_labels)

            _, loss, acc = model.classify_features(prototypes, prototype_labels, query_features, query_labels)
            
            total_loss += loss.item()
            total_acc += acc.item()

        return total_loss/len(val_loader), total_acc/len(val_loader)

def train(config: Config):
    training_params = config.trainig_params
    few_shot_params = config.few_shot_params
    dataset_params = config.dataset_params

    torch.manual_seed(training_params.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'====================Using {device} to train======================')

    model = ProtoNet(num_classes=dataset_params.num_classes, device=device)

    optimizer = optim.AdamW(model.parameters(), lr=training_params.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=training_params.epochs, eta_min=(training_params.learning_rate)*0.001, last_epoch=-1)

    model_net_dataset = ModelNet40C(dataset_params.dataset, dataset_params.label)
    train_cls_idx, validation_cls_idx, test_cls_idx = model_net_dataset.train_test_val_class_indices_split(train_ratio=training_params.training_split, seed=training_params.seed)

    train_set = PointCloudDataset.from_dataset(model_net_dataset.pcds_to_tensor(), model_net_dataset.labels_to_tensor(), train_cls_idx)
    validation_set = PointCloudDataset.from_dataset(model_net_dataset.pcds_to_tensor(), model_net_dataset.labels_to_tensor(), validation_cls_idx)
    test_set = PointCloudDataset.from_dataset(model_net_dataset.pcds_to_tensor(), model_net_dataset.labels_to_tensor(), test_cls_idx)

    train_loader = DataLoader(
        train_set,
        batch_sampler=FewShotBatchSampler(train_set.labels, n_ways=few_shot_params.n_ways, k_shots=few_shot_params.k_shots),
        num_workers=int(dataset_params.data_loader_num_workers))

    val_loader = DataLoader(
        validation_set,
        batch_sampler=FewShotBatchSampler(validation_set.labels, n_ways=few_shot_params.n_ways, k_shots=few_shot_params.k_shots),
        num_workers=int(dataset_params.data_loader_num_workers))
    
    training_loss_per_epoch = []
    val_loss_per_epoch = []
    best_val_acc = 0

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

        print(f"Epoch {epoch+1}/{training_params.epochs}, Training_loss: {total_loss/len(train_loader)}, Val_loss: {val_loss}, Val_acc: {val_acc}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print('best accuracy')
            print(best_val_acc)
            print('Saving a checkpoint')
            torch.save(model.state_dict(), training_params.ckpts)

        training_loss_per_epoch.append(total_loss/len(train_loader))
        val_loss_per_epoch.append(val_loss)

    plot_training_val_loss(training_loss_per_epoch, val_loss_per_epoch)

    test_proto(model, test_set, config.testing_params.k_shots, dataset_params.batch_size, dataset_params.data_loader_num_workers)

def plot_training_val_loss(training_loss: list[float], val_loss: list[float]):
    print('Saving training/validation losses plot')
    epochs = range(1, len(training_loss) + 1)
    plt.plot(epochs, training_loss, 'b-', label='Training loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_plot')


def split_batch(pcd_features: torch.Tensor, labels: torch.Tensor):
    """
        To be called after iterating on train loader that returns a batch
        to retrive support, query sets with their respective labels for few shot learning.
    """
    support_imgs, query_imgs = pcd_features.chunk(2, dim=0)
    support_targets, query_targets = labels.chunk(2, dim=0)
    return support_imgs, query_imgs, support_targets, query_targets