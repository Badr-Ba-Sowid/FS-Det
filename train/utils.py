
import numpy as np
from numpy.typing import NDArray
from typing import List, Dict
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def save_support_query_prediction_samples(support: List[Dict[str, NDArray]], query:List[Dict[str, NDArray]], logits: List[Dict[str, NDArray]], filename_prefix: str):
    support_arr = np.stack(support, axis=0) # type: ignore
    query_arr = np.stack(query, axis=0) # type: ignore
    logits_arr = np.stack(logits, axis=0) # type: ignore

    np.save(f'data/test_set/support_{filename_prefix}.npy', support_arr)
    np.save(f'data/test_set/query_{filename_prefix}.npy', query_arr)
    np.save(f'data/test_set/predictions_{filename_prefix}.npy', logits_arr)

def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss
