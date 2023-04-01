from __future__ import annotations

import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim

from config import TrainingConfig
from data_loader import ModelNet40C
from models import PointNetCls

def train(config_uri: str):
    training_config = TrainingConfig.from_file(config_uri)
    training_params = training_config.trainig_params
    training_uris = training_config.uris

    torch.manual_seed(training_params.seed)

    dataset = ModelNet40C(training_uris.dataset, training_uris.label)

    train_set, _ = dataset.train_and_test_split(training_params.training_split)

    train_data_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=training_params.batch_size,
        shuffle=True,
        num_workers=int(training_params.data_loader_num_workers))

    classifier = PointNetCls(k = 40)
    classifier.cuda()
    classifier.train()

    optimizer = optim.Adam(classifier.parameters(), lr = training_params.learning_rate, betas= (0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                        step_size=training_params.scheduler_steps, 
                                        gamma=training_params.scheduler_gamma)
    loss_fn = F.nll_loss

    for epoch in range(training_params.epoch):
        for i, data in enumerate(train_data_loader, 0):
            point_clouds, labels = data
            point_clouds, labels = point_clouds.cuda(), labels.cuda()
            optimizer.zero_grad()
            output, _, _ = classifier(point_clouds)
            loss = loss_fn(output, torch.flatten(labels))
            loss.backward()
            optimizer.step()
            if i%10 == 0:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {loss:.4f}')
        scheduler.step()

    torch.save(classifier.state_dict(), training_uris.ckpts)

