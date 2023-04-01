
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch
import pickle

class ModelNet40C:

    def __init__(self, data_path:str, label_path:str):
        self.data = np.load(data_path)
        self.data =  np.swapaxes(self.data, 1, 2)
        self.label = np.load(label_path)

    def __getitem__(self, point_cloud_idx:int):
        sample =  self.data[point_cloud_idx]
        label = self.label[point_cloud_idx]
        return torch.tensor(sample), torch.tensor(label, dtype=torch.long)
        
    def __len__(self):
        return len(self.data)

    def plot(self, point_cloud_idx:int):
        point_cloud = self.data[point_cloud_idx]
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
        plt.show()

    def remove_class(self, cls_index):
        indeces = np.where(self.label == cls_index)
        point_clouds = []
        for i in indeces:
            point_clouds.append(self.data[i])
            self.data = np.delete(self.data, i)
            self.label = np.delete(self.label, i)
        return point_clouds
    
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
        plt.show()

if __name__ == "__main__":    
    data = ModelNet40C("../data/model_net_40c/data_original.npy", "../data/model_net_40c/label.npy")
    point_clouds = data.remove_class(1)
    print(len(np.unique(data.label)))
    # # data.plot(1)
    # data = ModelNet40CFewShot("../data/model_net_40c_few_shot/4.pkl")
    