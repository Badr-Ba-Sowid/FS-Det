from __future__ import annotations

import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim

from config import Config
from data_loader import NPYDataset
from models import PointNetCls
from utils.plot import plot_train_test_data
from test.test_pointnet import test


def point_net_train(config: Config):

    training_params = config.trainig_params
    dataset_params = config.dataset_params

    torch.manual_seed(training_params.seed)

    dataset = NPYDataset(dataset_params.dataset, dataset_params.label)

    classifier = PointNetCls(k = dataset_params.num_classes)
    classifier.cuda()
    classifier.train()

    optimizer = optim.Adam(classifier.parameters(), lr = training_params.learning_rate, betas= (0.9, 0.999))

    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                        step_size=training_params.scheduler_steps,
                                        gamma=training_params.scheduler_gamma)

    loss_fn = F.nll_loss

    train_set, val_set, test_set = dataset.train_val_test_split(train_ratio=training_params.training_split)

    train_data_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=dataset_params.batch_size,
        shuffle=True,
        num_workers=int(dataset_params.data_loader_num_workers))

    val_data_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=dataset_params.batch_size,
        shuffle=True,
        num_workers=int(dataset_params.data_loader_num_workers))

    test_data_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=dataset_params.batch_size,
        shuffle=True,
        num_workers=int(dataset_params.data_loader_num_workers))

    t_a, t_l, v_a, v_l = train(classifier, train_data_loader, val_data_loader,
                                    optimizer, scheduler, loss_fn,
                                    training_params.epochs,
                                    training_params.ckpts)

    plot_train_test_data(t_a, v_a, "Accuracy")
    plot_train_test_data(t_l, v_l, "Loss")
    # accuracy = test(classifier, test_data_loader, loss_fn)
    # print("Final accuracy ", accuracy)



def train(model, train_data_loader, val_data_loader,
           optimizer, scheduler, loss_fn, epochs, ckpts):

    total_train_loss, total_val_loss = [], []
    total_train_accuracy, total_val_accuracy = [], []

    best_val = 0
    for epoch in range(epochs):
        running_loss = 0
        total_correct = 0
        for i, data in enumerate(train_data_loader, 0):
            point_clouds, labels = data
            point_clouds, labels = point_clouds.cuda(), labels.cuda()
            optimizer.zero_grad()
            output, _, _ = model(point_clouds)
            loss = loss_fn(output, torch.flatten(labels))
            running_loss += loss.item()
            y_predict = output.max(1, keepdim=True)[1]
            total_correct += y_predict.eq(labels.view_as(y_predict)).sum().item()
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {loss:.4f}')
        scheduler.step()
        
        val_accuracy, val_loss = test(model, val_data_loader, loss_fn)
        total_train_loss.append(running_loss/len(train_data_loader))
        total_train_accuracy.append(total_correct / len(train_data_loader.dataset))
        total_val_loss.append(val_loss)
        total_val_accuracy.append(val_accuracy)
        if(val_accuracy > best_val):
            best_val = val_accuracy
            torch.save(model.state_dict(), ckpts)

    return total_train_accuracy, total_train_loss, total_val_accuracy, total_val_loss
