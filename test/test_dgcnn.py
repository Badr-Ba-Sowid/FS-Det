from __future__ import annotations

import torch.nn.functional as F
import torch.utils.data

from config import Config
from data_loader import NPYDataset
from models import DGCNN
from tqdm import tqdm
import pickle
from train.utils import cal_loss

def test_ckpt(config: Config):
    dataset_params = config.dataset_params
    test_params = config.testing_params
    model = DGCNN(dataset_params.num_classes)
    model = model.cuda()
    model.load_state_dict(torch.load(test_params.model_state))
    data_loader = prepare_dataloader(dataset_params, test_params)
    accuracy, total_loss = test(model, data_loader, loss_fn=cal_loss)
    print("=============Results 🙅‍♂️=============")
    print("Accuracy : ", accuracy)

def prepare_dataloader(dataset_params, test_params):
    with open(test_params.dataset_path, "rb") as f:
        dataset = pickle.load(f)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=dataset_params.batch_size,
        shuffle=False,
        num_workers=int(dataset_params.data_loader_num_workers))


def test(model, data_loader, loss_fn):
    model.eval()
    total_correct = 0
    total_loss = 0

    for _, data in tqdm(enumerate(data_loader, 0)):
        point_cloud, label = data
        point_cloud, label = point_cloud.cuda(), label.cuda()

        pred = model(point_cloud)
        total_loss += loss_fn(pred, torch.flatten(label)).item()

        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(torch.flatten(label)).sum().item()
        total_correct += correct

    accuracy = total_correct / len(data_loader.dataset)
    
    return accuracy, total_loss / len(data_loader)
