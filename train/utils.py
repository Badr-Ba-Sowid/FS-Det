
import matplotlib.pyplot as plt

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