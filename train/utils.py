
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from typing import List

def plot_training_val_fewshot(training_arr: List[float], val_arr: List[float], root_directory: str, ds_name:str, mode: str):
    plt.clf()
    print(f'Saving training/validation {mode} plot')
    epochs = range(1, len(training_arr) + 1)
    plt.plot(epochs, training_arr, 'b-', label='Training')
    plt.plot(epochs, val_arr, 'r-', label='Validation')
    plt.title(f'Training and validation {mode}')
    plt.xlabel('Epochs')
    plt.ylabel(mode)
    plt.legend()
    plt.savefig(f'{root_directory}/training_{mode}_plot_{ds_name}')

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