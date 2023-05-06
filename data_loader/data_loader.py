import numpy as np
import matplotlib.pyplot as plt

import pickle
import random
import numpy as np

from collections import defaultdict
from typing import List, Tuple
from numpy.typing import NDArray
from typing_extensions import  Self

import torch
from torch.utils.data import Dataset, Subset, random_split, Sampler



__all__ = ['ModelNet40CFewShot', 'FewShotBatchSampler', 'PointCloudDataset', 'NPYDataset']


# class ModelNet40C(Dataset):
#     def __init__(self, data_path:str, label_path:str):
#         self.point_clouds = np.load(data_path, allow_pickle=True)
#         self.point_clouds =  np.swapaxes(self.point_clouds, 1, 2)
#         self.labels_dict = np.load(label_path, allow_pickle=True)
#         labels_ids: List[int] = [label.get('class_id') for label in self.labels_dict]
#         self.labels_txt: List[str] = [label.get('class_name') for label in self.labels_dict]
#         self.labels_ids = np.array(labels_ids).reshape(len(labels_ids), 1)


#     def __getitem__(self, point_cloud_idx:int):
#         sample =  self.point_clouds[point_cloud_idx]
#         label = self.labels_ids[point_cloud_idx]

#         return torch.tensor(sample), torch.tensor(label, dtype=torch.long)
        
#     def __len__(self) -> int:
#         return len(self.point_clouds)

#     def plot(self, point_cloud_idx:int):
#         point_cloud = self.point_clouds[point_cloud_idx]
#         x = point_cloud[:, 0]
#         y = point_cloud[:, 1]
#         z = point_cloud[:, 2]
        
#         fig = plt.figure(figsize=(9,7))
#         ax = fig.add_subplot(projection='3d')
#         img = ax.scatter(x, y, z, cmap=plt.hot())
#         fig.colorbar(img)

#         ax.set_xlabel('X')
#         ax.set_ylabel('Y')
#         ax.set_zlabel('Z')
#         plt.savefig('vis')

#     def remove_class(self, cls_index):
#         indeces = np.where(self.labels_ids == cls_index)
#         point_clouds = []
#         for i in indeces:
#             point_clouds.append(self.point_clouds[i])
#             self.point_clouds = np.delete(self.point_clouds, i)
#             self.labels_ids = np.delete(self.labels_ids, i)

#         return point_clouds

#     def train_val_test_split(self, train_ratio: float = 0.7, validation_ratio: float = 0.1) -> List[Subset[Self]]:
#         train_size = int(train_ratio * self.__len__())
#         remaining_size = self.__len__() - train_size
#         validation_size = int(validation_ratio* remaining_size)
#         test_size = remaining_size - validation_size

#         return random_split(self , [train_size, validation_size, test_size])
    
#     def train_test_val_class_indices_split(self, train_ratio: float = 0.8, seed: int=42) -> Tuple[torch.Tensor, ...]:
#         labels = self.labels_ids

#         labels_size = len(np.unique(labels))
#         train_size = int(train_ratio * labels_size)
#         validation_size = int((labels_size - train_size)/ 2)

#         torch.manual_seed(int(seed))
#         classes = torch.randperm(labels_size)

#         return classes[:train_size], classes[train_size:train_size+(validation_size)], classes[train_size+(validation_size):]
    
#     def labels_to_tensor(self) -> torch.Tensor:
#         return torch.from_numpy(self.labels_ids)
    
#     def pcds_to_tensor(self) -> torch.Tensor:
#         return torch.from_numpy(self.point_clouds)


# class ShapeNet55C(Dataset):
#     def __init__(self, data_path:str, label_path:str):
#         self.point_clouds = np.load(data_path, allow_pickle=True)
#         self.point_clouds =  np.swapaxes(self.point_clouds, 1, 2)
#         self.labels_dict = np.load(label_path, allow_pickle=True)

#         labels_ids: List[int] = [label.get('class_id') for label in self.labels_dict]
#         labels_txt: List[str] = [label.get('class_name') for label in self.labels_dict]
#         self.labels_ids: NDArray = np.array(labels_ids).reshape(len(labels_ids), 1)
#         self.labels_txt: NDArray = np.array(labels_txt).reshape(len(labels_txt), 1)

#     def __getitem__(self, point_cloud_idx:int):
#         sample =  self.point_clouds[point_cloud_idx]
#         label = self.labels_ids[point_cloud_idx]

#         return torch.tensor(sample), torch.tensor(label, dtype=torch.long)
        
#     def __len__(self) -> int:
#         return len(self.point_clouds)

#     def plot(self, point_cloud_idx:int):
#         point_cloud = self.point_clouds[point_cloud_idx]
#         x = point_cloud[:, 0]
#         y = point_cloud[:, 1]
#         z = point_cloud[:, 2]
        
#         fig = plt.figure(figsize=(9,7))
#         ax = fig.add_subplot(projection='3d')
#         img = ax.scatter(x, y, z, cmap=plt.hot())
#         fig.colorbar(img)

#         ax.set_xlabel('X')
#         ax.set_ylabel('Y')
#         ax.set_zlabel('Z')

#     def remove_class(self, cls_index):
#         indeces = np.where(self.labels_ids == cls_index)
#         point_clouds = []
#         for i in indeces:
#             point_clouds.append(self.point_clouds[i])
#             self.point_clouds = np.delete(self.point_clouds, i)
#             self.labels_ids = np.delete(self.labels_ids, i)
#             self.labels_txt = np.delete(self.labels_txt, i)

#         return point_clouds

#     def labels_to_tensor(self) -> torch.Tensor:
#         return torch.from_numpy(self.labels_ids)
    
#     def pcds_to_tensor(self) -> torch.Tensor:
#         return torch.from_numpy(self.point_clouds)

#     def train_test_val_class_indices_split(self, train_ratio: float = 0.8, seed: int=42) -> Tuple[torch.Tensor, ...]:
#         labels = self.labels_ids

#         labels_size = len(np.unique(labels))
#         train_size = int(train_ratio * labels_size)
#         validation_size = int((labels_size - train_size)/ 2)
#         torch.manual_seed(int(seed))
#         classes = torch.randperm(labels_size)

#         return classes[:train_size], classes[train_size:train_size+(validation_size)], classes[train_size+(validation_size):]

class NPYDataset(Dataset):
    """"Responsible for reading the point cloud datasets that NPY format."""

    def __init__(self, data_path:str, label_path:str):
        """
        data_path: the point cloud dataset path where the file is in form of .npy file and the size of the dataset is (n, num_points, pcd_channles) 
        label_path: the corresponding label indices to its point cloud where the label is encoded into indices in range of [0,num_classes].
        """
        self.point_clouds = np.load(data_path, allow_pickle=True)
        self.point_clouds =  np.swapaxes(self.point_clouds, 1, 2)
        self.labels_dict = np.load(label_path, allow_pickle=True)

        labels_ids: List[int] = [label.get('class_id') for label in self.labels_dict]
        labels_txt: List[str] = [label.get('class_name') for label in self.labels_dict]
        self.labels_ids: NDArray = np.array(labels_ids).reshape(len(labels_ids), 1)
        self.labels_txt: NDArray = np.array(labels_txt).reshape(len(labels_txt), 1)

    def __getitem__(self, point_cloud_idx:int):
        sample =  self.point_clouds[point_cloud_idx]
        label = self.labels_ids[point_cloud_idx]

        return torch.tensor(sample), torch.tensor(label, dtype=torch.long)
        
    def __len__(self) -> int:
        return len(self.point_clouds)

    def plot(self, point_cloud_idx:int):
        point_cloud = self.point_clouds[point_cloud_idx]
        x = point_cloud[:, 0]
        y = point_cloud[:, 1]
        z = point_cloud[:, 2]
        
        fig = plt.figure(figsize=(9,7))
        ax = fig.add_subplot(projection='3d')
        img = ax.scatter(x, y, z, cmap=plt.hot())
        fig.colorbar(img)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    def remove_class(self, cls_index):
        indeces = np.where(self.labels_ids == cls_index)
        point_clouds = []
        for i in indeces:
            point_clouds.append(self.point_clouds[i])
            self.point_clouds = np.delete(self.point_clouds, i)
            self.labels_ids = np.delete(self.labels_ids, i)
            self.labels_txt = np.delete(self.labels_txt, i)

        return point_clouds

    def labels_to_tensor(self) -> torch.Tensor:
        return torch.from_numpy(self.labels_ids).float()
    
    def pcds_to_tensor(self) -> torch.Tensor:
        return torch.from_numpy(self.point_clouds)

    def train_val_test_split(self, train_ratio: float = 0.7, validation_ratio: float = 0.1) -> List[Subset[Self]]:
        train_size = int(train_ratio * self.__len__())
        remaining_size = self.__len__() - train_size
        validation_size = int(validation_ratio* remaining_size)
        test_size = remaining_size - validation_size

        return random_split(self , [train_size, validation_size, test_size])

    def train_test_val_class_indices_split(self, train_ratio: float = 0.8, seed: int=42) -> Tuple[torch.Tensor, ...]:
        labels = self.labels_ids

        labels_size = len(np.unique(labels))
        train_size = int(train_ratio * labels_size)
        validation_size = int((labels_size - train_size)/ 2)
        torch.manual_seed(int(seed))
        classes = torch.randperm(labels_size)

        return classes[:train_size], classes[train_size:train_size+(validation_size)], classes[train_size+(validation_size):]

class PointCloudDataset(Dataset):
    def __init__(self, point_clouds: torch.Tensor, labels: torch.Tensor) -> None:
        super().__init__()

        self.pcds = point_clouds
        self.labels = labels

    @staticmethod
    def from_dataset(all_point_clouds: torch.Tensor, all_labels: torch.Tensor, label_indices: torch.Tensor):
        class_mask = (all_labels[:, None] == label_indices[None, :]).any(dim=-1)

        return PointCloudDataset(point_clouds=all_point_clouds[class_mask.squeeze(), :], labels=all_labels[class_mask])
    
    @staticmethod
    def from_pickle(file_name: str):
        with open(file_name, 'rb') as dataset_file:
            dataset  = pickle.load(dataset_file)
        return dataset
    
    def save_dataset(self, file_name: str):
        with open(file_name, 'wb') as dataset_file:
            pickle.dump(self, dataset_file)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.pcds[idx], self.labels[idx]

    def __len__(self) -> int:
        return self.pcds.shape[0]

class FewShotBatchSampler(Sampler):
    def __init__(self, dataset_labels: torch.Tensor, n_ways: int, k_shots: int, include_query: bool=False, shuffle: bool=True, shuffle_once: bool=True) -> None:
        """
        Inputs:
            dataset_labels - PyTorch tensor of the labels of the data elements.
            N_way - Number of classes to sample per batch.
            K_shot - Number of examples to sample per class in the batch.
            include_query - If True, returns batch of size N_way*K_shot*2, which
                            can be split into support and query set. Simplifies
                            the implementation of sampling the same classes but
                            distinct examples for support and query set.
            shuffle - If True, examples and classes are newly shuffled in each
                    iteration (for training)
            shuffle_once - If True, examples and classes are shuffled once in
                        the beginning, but kept constant across iterations
                        (for validation)        
        """
        self.dataset_labels = dataset_labels
        self.n_way = n_ways
        self.k_shot = k_shots
        self.include_query = include_query
        self.shuffle = shuffle
        self.shuffle_once = shuffle_once
        if self.include_query:
            self.k_shot *= 2
        self.batch_size = self.n_way * self.k_shot

        self._arrange_classes_per_batch()

    def _arrange_classes_per_batch(self) -> None:
        self.classes: list[int] = (torch.unique(self.dataset_labels).tolist())
        self.num_classes = len(self.classes)
        self.indices_per_class: dict[int, torch.Tensor] = {}
        self.batches_per_class: dict[int, int] = {} # Number of K-shot batches that each class can provide

        for c in self.classes:
            self.indices_per_class[c] = torch.where(self.dataset_labels==c)[0]
            self.batches_per_class[c] = self.indices_per_class[c].shape[0] // self.k_shot

        self.iterations: int = sum(self.batches_per_class.values()) // self.n_way
        self.class_list: list[int] = [c for c in self.classes for _ in range(self.batches_per_class[c])]

        if self.shuffle_once or self.shuffle:
            self.shuffle_data()

        else:
            # For testing, we iterate over classes instead of shuffling them
            sort_idxs: list[int] = [ i+ p* self.num_classes for i , c in enumerate(self.classes) for p in range(self.batches_per_class[c])]
            self.class_list: list[int] = np.array(self.class_list)[np.argsort(sort_idxs)].tolist()

    def shuffle_data(self) -> None:
        for c in self.classes:
            perm = torch.randperm(self.indices_per_class[c].shape[0])
            self.indices_per_class[c] = self.indices_per_class[c][perm]

        random.shuffle(self.class_list)

    def __iter__(self):
        if self.shuffle:
            self.shuffle_data()

        start_index = defaultdict(int)

        for i in range(self.iterations):
            class_batch: list[int] = self.class_list[i * self.n_way: (i+1) * self.n_way]

            index_batch = []
            for c in class_batch:
                index_batch.extend(self.indices_per_class[c][start_index[c] : start_index[c] + self.k_shot])

                start_index[c] += self.k_shot

            if self.include_query:
                # If we return support+query set, sort them so that they are easy to split
                index_batch = index_batch[::2] + index_batch[1::2]
            yield index_batch

    def __len__(self):
        return self.iterations

class ModelNet40CFewShot:
    
    def __init__(self, dataset_path):
        with open(dataset_path, 'rb') as f:
            self.dataset = pickle.load(f)
            print((self.dataset['train'][3]))
            self.plot()
    
    def plot(self):
        point_cloud = self.dataset["train"][5][0][:,:3 ]
        x = point_cloud[:, 0]
        y = point_cloud[:, 1]
        z = point_cloud[:, 2]
        
        fig = plt.figure(figsize=(9,7))
        ax = fig.add_subplot(projection='3d')
        img = ax.scatter(x, y, z, cmap=plt.hot())
        fig.colorbar(img)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')