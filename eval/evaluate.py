from __future__ import annotations
import torch
import sys
import matplotlib.pyplot as plt
sys.path.insert(0,"..")

from models.pointnet import PointNetCls
from data_loader.data_loader import ModelNet40C
from utils.point_cloud_utils import plot


classes = [
"airplane",
"bathtub"   ,
"bed"     ,
"bench"	   ,
"bookshelf",
"bottle"	   ,
"bowl"	   ,
"car"	       ,
"chair"	   ,
"cone"	   ,
"cup"	       ,
"curtain"	   ,
"desk"	   ,
"door"	   ,
"dresser"	   ,
"flower_pot",
"glass_box",
"guitar"	   ,
"keyboard",
"lamp"	   ,
"laptop"	   ,
"mantel"	   ,
"monitor"	   ,
"night_stand",
"person"	   ,
"piano"	   ,
"plant"	   ,
"radio"	   ,
"range_hood",
"sink"	   ,
"sofa"	   ,
"stairs"	   ,
"stool"	   ,
"table"	   ,
"tent"	   ,
"toilet"	   ,
"tv_stand",
"vase"	   ,
"wardrobe",
"xbox"	   ]

if __name__ == '__main__':
    dataset = ModelNet40C("../data/model_net_40c/data_original.npy", "../data/model_net_40c/label.npy")

    classifier = PointNetCls(k = 40)
    classifier.load_state_dict(torch.load("../ckpt/point_net_general_batch_size_32_classes_40"))
    classifier = classifier.eval()
    point_cloud, label = dataset.__getitem__(1)
    pred, _, _ = classifier(torch.unsqueeze(point_cloud, 0))
    plot(point_cloud, classes[pred.max(1)[1]])


