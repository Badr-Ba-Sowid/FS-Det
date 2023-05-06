import numpy as np
from pathlib import Path
import json

shapenet_path = 'path to unziped shapenet model'
shapenet_labels_path = 'path to labels json file'

def preprocess_shapenet_labels():
    directory = Path(shapenet_path)
    labels = None
    shape_net_55_labels = []
    
    with open(shapenet_path) as f:
        labels = json.load(f)

    labels_keys: list[str] = list(labels.keys())
    pcd_data = []
    for i, (file_path) in enumerate(directory.iterdir()):
        print(f'saving label num {i+1}')
        pcd  = np.load(file_path)
        pcd_data.append(pcd)
        file_name = file_path.stem
        class_id = file_name.split('-')[0]
        class_txt = labels[class_id]
        print(class_id)
        print(class_txt)
        print(labels_keys.index(class_id))
        shape_net_55_labels.append({'class_id': int(labels_keys.index(class_id)), 'class_name': class_txt})
    
    stack_labels = np.stack(shape_net_55_labels, axis=0)
    stack_data  = np.stack(pcd_data, axis=0)
    print('stack labels')
    print(stack_labels.shape)
    print('stack pcd')
    print(stack_data.shape)

    np.save('./preprocessed_shapenet/shape_net_labels.npy', stack_labels)
    print('saved labels')
    np.save('./preprocessed_shapenet/shape_net_pcd.npy', stack_data)
    print('saved point cloud samples')


preprocess_shapenet_labels()


