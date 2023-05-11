import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['DGCNN']

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base
    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature


class DGCNNEncoder(nn.Module):
    def __init__(self, k, maxpool=False):
        super(DGCNNEncoder, self).__init__()
        self.k = k
        emb_dims = 1024
        self.maxpool = maxpool
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(emb_dims)

        self.conv1 =  nn.Conv2d(6, 64, kernel_size=1, bias=False)
        self.lr1 =  nn.LeakyReLU(negative_slope=0.2)

        self.conv2 = nn.Conv2d(64*2, 64, kernel_size=1, bias=False)
        self.lr2 = nn.LeakyReLU(negative_slope=0.2)
        self.conv3 = nn.Conv2d(64*2, 128, kernel_size=1, bias=False)
        self.lr3 = nn.LeakyReLU(negative_slope=0.2)
        self.conv4 = nn.Conv2d(128*2, 256, kernel_size=1, bias=False)
        self.lr4 = nn.LeakyReLU(negative_slope=0.2)
        self.conv5 = nn.Conv1d(512, emb_dims, kernel_size=1, bias=False)
        self.lr5 = nn.LeakyReLU(negative_slope=0.2)


    def forward(self, x):
        graphs = []
        x = get_graph_feature(x, k=self.k)
        x = self.lr1(self.bn1(self.conv1(x)))
        graphs.append(x.max(dim=-1, keepdim=False)[0])

        x = get_graph_feature(graphs[0], k=self.k)
        x = self.lr2(self.bn2(self.conv2(x)))

        graphs.append(x.max(dim=-1, keepdim=False)[0])

        x = get_graph_feature(graphs[1], k=self.k)
        x = self.lr3(self.bn3(self.conv3(x)))
        graphs.append(x.max(dim=-1, keepdim=False)[0])

        x = get_graph_feature(graphs[2], k=self.k)
        x = self.lr4(self.bn4(self.conv4(x)))
        graphs.append(x.max(dim=-1, keepdim=False)[0])

        x = torch.cat(graphs, dim=1)

        x = self.lr5(self.bn5(self.conv5(x)))
        if self.maxpool:
            return x.max(dim=-1, keepdim=False)[0]
        return x[0]

class DGCNN(nn.Module):
    def __init__(self, k):
        super(DGCNN, self).__init__()
        self.k = k
        emb_dims = 1024
        dropout = 0.5
        self.encoder = DGCNNEncoder(k)
        self.linear1 = nn.Linear(emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=dropout)
        self.linear3 = nn.Linear(256, self.k)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.encoder(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x
