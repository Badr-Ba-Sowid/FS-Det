
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List 
from torch.utils.data import Dataset, DataLoader

#Base class whic contains useful helpers
class FSDETDataset(Dataset):
    def __init__(self, data_path:str, label_path:str):
        self.data = np.load(data_path)
        self.label = np.load(label_path)
    
    def __getitem__(self, point_cloud_idx:int):
        sample =  self.data[point_cloud_idx, :, :]
        label = self.label[point_cloud_idx]

        return {
            "sample" : torch.tensor(sample, dtype = torch.float),
            "label" : torch.tensor(label, dtype=torch.int)
        }
    def __len__(self):
        return len(self.data)

    def plot(self, point_cloud_idx:int):
        point_cloud = self.data[point_cloud_idx]
        x = point_cloud[:, 0]
        y = point_cloud[:, 1]
        z = point_cloud[:, 2]

        fig = plt.figure(figsize=(12,7))
        ax = fig.add_subplot(projection='3d')
        img = ax.scatter(x, y, z, cmap=plt.hot())
        fig.colorbar(img)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

class ModelNet40C(FSDETDataset):
    pass

if __name__ == "__main__":
    # td, tl, tst_d, tst_l= parse_split_data("../data/model_net_40c/data_original.npy", "../data/model_net_40c/label.npy", 0.7, MODEL_NET_40C_NUM_CLS)
    
    data = ModelNet40C("../data/model_net_40c/data_original.npy", "../data/model_net_40c/label.npy")
    