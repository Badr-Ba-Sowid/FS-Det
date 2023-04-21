from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns
from statistics import mean, stdev

from tqdm import tqdm

import torch
import torch.utils.data
from torch.utils.data import DataLoader

from models import ProtoNet
from data_loader import PointCloudDataset

def test(model: ProtoNet, dataset: PointCloudDataset, k_shot: int):
    num_classes = (dataset.labels.unique().shape[0]) -1
    exmps_per_class = dataset.labels.shape[0] // num_classes

    with torch.no_grad():
        model.eval()
        
        test_loader = DataLoader(dataset, batch_size=128, num_workers=4)

        pcd_features = []
        pcd_targets = []

        for pcds, targets in tqdm(test_loader, "Extracting point cloud features"):
            pcds = pcds.to(model.device)
            features = model(pcds)

            pcd_features.append(features.detach().cpu())
            pcd_targets.append(targets)
        
        pcd_targets = torch.cat(pcd_targets, dim=0)
        pcd_features = torch.cat(pcd_features, dim=0)

        pcd_targets, sort_idx = pcd_targets.sort()
        pcd_targets = pcd_targets.view((num_classes, exmps_per_class)).transpose(0, 1)
        pcd_features = pcd_features[sort_idx].view(num_classes, exmps_per_class, -1).transpose(0, 1)

    accuracies = []
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

            _, _, acc = model.classify_features(prototypes, proto_classes, e_pcd_feats, e_targets)
            batch_acc += acc.item()

        batch_acc /= pcd_features.shape[0] // k_shot - 1
        accuracies.append(batch_acc)

    return (mean(accuracies), stdev(accuracies)), (pcd_features, pcd_targets)

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

def begin_test(config: str):
    pass
