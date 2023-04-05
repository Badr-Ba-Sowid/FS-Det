from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pointnet import PointNetEncoder

__all__ = ['ProtoNet']


class ProtoNet(nn.Module):
    def __init__(self, num_classes: int, device: torch.device) -> None:
        super(ProtoNet, self).__init__()
        self.pointnet = PointNetEncoder(num_classes)
        self.device = device

    def forward(self, x):
        return self.pointnet(x)

    @staticmethod
    def compute_prototypes(support_feature: torch.Tensor, support_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            returns : prototypes and their corresponding label classes in Tensor.
        """

        classes, _ = torch.unique(support_labels).sort()
        prototypes = []
        for c in classes:
            p = support_feature[torch.where(support_labels == c)[0]].mean(dim=0)
            prototypes.append(p)

        prototypes = torch.stack(prototypes, dim =0)

        return prototypes, classes

    def euclidean_distance(self, prototypes: torch.Tensor, query_features: torch.Tensor) -> torch.Tensor:
        return torch.pow(prototypes[None, :] - query_features[:, None], 2).sum(dim=2)
    
    def predict(self, prototypes: torch.Tensor, query_features: torch.Tensor) -> torch.Tensor:
        distances =  self.euclidean_distance(prototypes, query_features)

        return F.log_softmax(-distances, dim=-1)

    def compute_loss(self, prototypes: torch.Tensor, prototype_labels: torch.Tensor, query_features: torch.Tensor, query_labels: torch.Tensor) -> torch.Tensor:
        log_p_y = self.predict(prototypes, query_features)

        labels = (prototype_labels[None:] == query_labels[:, None]).long().argmax(dim=-1)

        return F.cross_entropy(log_p_y, labels)