
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
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
    


if __name__ == "__main__":    
    data = ModelNet40C("../data/model_net_40c/data_original.npy", "../data/model_net_40c/label.npy")
    point_clouds = data.remove_class(1)
    print(len(np.unique(data.label)))
    # # data.plot(1)
    # data = ModelNet40CFewShot("../data/model_net_40c_few_shot/4.pkl")
    