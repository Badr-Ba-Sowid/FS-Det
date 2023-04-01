from __future__ import annotations

import torch.utils.data

from tqdm import tqdm

from config import TrainingConfig
from data_loader import ModelNet40C
from models import PointNetCls

def test(config_uri: str):
    training_config = TrainingConfig.from_file(config_uri)
    training_params = training_config.trainig_params
    training_uris = training_config.uris

    dataset = ModelNet40C(training_uris.dataset, training_uris.label)
    
    _, test_set = dataset.train_and_test_split(training_params.training_split)
    test_data_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=training_params.batch_size,
        shuffle=True,
        num_workers=int(training_params.data_loader_num_workers))

    classifier = PointNetCls(k = 40)
    classifier.eval()
    classifier.cuda()


    total_correct = 0
    total_testset = 0
    for _, data in tqdm(enumerate(test_data_loader, 0)):
        point_cloud, label = data
        point_cloud, label = point_cloud.cuda(), label.cuda()

        pred, _, _ = classifier(point_cloud)
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(torch.flatten(label)).sum()
        total_correct+= correct.item()
        total_testset += point_cloud.size()[0]

    return total_correct/float(total_testset)