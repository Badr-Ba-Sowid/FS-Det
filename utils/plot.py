from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import seaborn as sns
from typing import List, Dict


def plot_train_test_data(train_data: list[float], test_data: list[float], label):
    plt.clf()
    print('Saving training/validation losses plot')
    epochs = range(1, len(train_data) + 1)
    plt.plot(epochs, train_data, 'b-', label='Training ' + label)
    plt.plot(epochs, test_data, 'r-', label='Validation ' + label)
    plt.title('Training and validation ' + label)
    plt.xlabel('Epochs')
    plt.ylabel(label)
    plt.legend()
    plt.savefig(label + '_plot')

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

def plot_few_shot_test_acc_trend(acc_dict:Dict, name: str, color=None, ax=None, root_director: str='', ds_name: str='unknown_ds'):
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

    plt.savefig(f'{root_director}/fewshot_performance_{ds_name}')

def plot_support(support_sammples:List[Dict[str, NDArray]], dataset_uniq_label_map: Dict[int, str], root_directory: str, ds_name:str, k_shot: int):
    plt.clf()

    num_samples = len(np.unique(support_sammples[0]['label']))
    fig = plt.figure(figsize=(15, 10*num_samples))

    random_index = np.random.randint(len(support_sammples))
    batch_sample_dict = support_sammples[random_index]

    unique_labels = np.unique(batch_sample_dict['label'])

    # create a new dictionary with one example for each unique label
    new_dict = {'label': np.empty((len(unique_labels), 1)), 
                'pcd': np.empty((len(unique_labels), batch_sample_dict['pcd'].shape[1], batch_sample_dict['pcd'].shape[2]))}

    for i, label in enumerate(unique_labels):
        # find the index of the first occurrence of the label in the original label array
        index = np.where(batch_sample_dict['label'] == label)[0][0]
        
        # copy the label and pcd data to the new dictionary
        new_dict['label'][i] = batch_sample_dict['label'][index]
        new_dict['pcd'][i] = batch_sample_dict['pcd'][index]

    for i in range(len(new_dict['label'])):
        label = new_dict['label'][i]
        pcd = new_dict['pcd'][i]

        ax = fig.add_subplot(num_samples, 1, i+1, projection='3d')    
        img = ax.scatter(pcd[0, :], pcd[1,:], pcd[2, :], cmap=plt.hot())

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Support Sample {i+1}: Label = {dataset_uniq_label_map[int(label)]}')

        fig.colorbar(img, ax=ax)

    plt.savefig(f'{root_directory}/support_samples_{ds_name}_{k_shot}_test')

def plot_query(query_samples: List[Dict[str, NDArray]], predicted_targets: List[NDArray], dataset_uniq_label_map: Dict[int, str], root_directory: str, ds_name:str, k_shot: int):
    plt.clf()
    num_samples = 5
    fig = plt.figure(figsize=(15, 10*num_samples))
    random_index = np.random.randint(len(query_samples))

    for i, (query, predicted) in enumerate(zip(query_samples[random_index:(random_index+num_samples)], predicted_targets[random_index:(random_index+num_samples)])):
        labels = query['label']

        random_index = np.random.randint(len(query))

        label = query['label'][random_index]
        pcd = query['pcd'][random_index]
        predicted_label = labels[np.argmax(predicted, axis=1)][random_index]

        ax = fig.add_subplot(num_samples, 1, i+1, projection='3d')
        img = ax.scatter(pcd[0, :], pcd[1,:], pcd[2, :], cmap=plt.hot())

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Query Sample {i+1}: Predicted Label = {dataset_uniq_label_map[int(predicted_label)]}, Actual Label = {dataset_uniq_label_map[int(label)]}')

        fig.colorbar(img, ax=ax)

    plt.savefig(f'{root_directory}/query_samples_{ds_name}_{k_shot}_test')