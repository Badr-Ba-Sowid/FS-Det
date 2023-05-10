from __future__ import annotations

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pointnet import PointNetEncoder, PointNetfeat

__all__ = ['ProtoNet', 'ProtoNetParallerWrapper']


class ProtoNet(nn.Module):
    def __init__(self, num_classes: int, device: torch.device, pretrained_ckpts: Optional[str]=None, use_attention: bool=False) -> None:
        super(ProtoNet, self).__init__()
        self.encoder = PointNetEncoder(device, num_classes, use_attention)
        self.encoder.to(device)
        self.device = device
        
        if pretrained_ckpts is not None:
            pretrained_dict = torch.load(pretrained_ckpts)
            self.encoder.load_state_dict(pretrained_dict)

    def forward(self, x):
        x = self.encoder(x)
        return x

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

        return F.log_softmax(-distances, dim=1).to(self.device)

    def classify_features(self, prototypes: torch.Tensor, prototype_cls: torch.Tensor, query_features: torch.Tensor, query_cls: torch.Tensor) -> Tuple[torch.Tensor,...]:
        logits = self.predict(prototypes, query_features)

        loss = self.compute_loss(prototypes, prototype_cls, query_features, query_cls)
        actual_labels = ((prototype_cls[None:] == query_cls[:, None]).long().argmax(dim=-1)).to(self.device)

        acc =((logits.argmax(dim=1) == actual_labels).float().mean())
        return logits, loss, acc


    def compute_loss(self, prototypes: torch.Tensor, prototype_labels: torch.Tensor, query_features: torch.Tensor, query_labels: torch.Tensor) -> torch.Tensor:
        logits = self.predict(prototypes, query_features)
        labels = ((prototype_labels[None:] == query_labels[:, None]).long().argmax(dim=-1)).to(self.device)

        return F.cross_entropy(logits, labels).to(self.device)


class ProtoNetParallerWrapper(nn.DataParallel):
    def __init__(self, module: ProtoNet, device_ids) -> None:
        super(ProtoNetParallerWrapper, self).__init__(module, device_ids=device_ids)
        self.device = 'cuda'

    def compute_prototypes(self, support_feature: torch.Tensor, support_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
       return self.module.compute_prototypes(support_feature, support_labels)

    def euclidean_distance(self, prototypes: torch.Tensor, query_features: torch.Tensor) -> torch.Tensor:
        return self.module.euclidean_distance(prototypes, query_features)

    def predict(self, prototypes: torch.Tensor, query_features: torch.Tensor) -> torch.Tensor:
        return self.module.predict(prototypes, query_features )

    def classify_features(self, prototypes: torch.Tensor, prototype_cls: torch.Tensor, query_features: torch.Tensor, query_cls: torch.Tensor) -> Tuple[torch.Tensor,...]:
        return self.module.classify_features(prototypes, prototype_cls, query_features, query_cls)

    def compute_loss(self, prototypes: torch.Tensor, prototype_labels: torch.Tensor, query_features: torch.Tensor, query_labels: torch.Tensor) -> torch.Tensor:
        return self.module.compute_loss(prototypes, prototype_labels, query_features, query_labels)
