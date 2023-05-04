from __future__ import annotations

import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim

from config import Config
from data_loader import NPYDataset
from models import PointNetCls

def train(config: Config):
    training_params = config.trainig_params
    dataset_params = config.dataset_params

    torch.manual_seed(training_params.seed)

    dataset = NPYDataset(dataset_params.dataset, dataset_params.label)

    train_set, _, _= dataset.train_val_test_split(train_ratio=training_params.training_split)

    train_data_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=dataset_params.batch_size,
        shuffle=True,
        num_workers=int(dataset_params.data_loader_num_workers))

    classifier = PointNetCls(k = 40)
    classifier.cuda()
    classifier.train()

    optimizer = optim.Adam(classifier.parameters(), lr = training_params.learning_rate, betas= (0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                        step_size=training_params.scheduler_steps, 
                                        gamma=training_params.scheduler_gamma)
    loss_fn = F.nll_loss

    for epoch in range(training_params.epochs):
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

    torch.save(classifier.state_dict(), training_params.ckpts)

