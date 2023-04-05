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

    train_set, _, _= dataset.train_val_test_split(train_ratio=training_params.training_split)

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
    torch.save(classifier.state_dict(), "../ckpt/point_net_batch_size_32_classes_40")

def test(test_data_loader):
    total_correct = 0
    total_test_set = 0
    for _, data in tqdm(enumerate(test_data_loader, 0)):
        point_cloud, label = data
        point_cloud, label = point_cloud.cuda(), label.cuda()
        pred, _, _ = classifier(point_cloud)
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(torch.flatten(label)).sum()
        total_correct+= correct.item()
        total_test_set += point_cloud.size()[0]
    return total_correct/float(total_test_set)


def train_step(model : torch.nn.Module, loss_fn : Callable, x :torch.Tensor, y :torch.tensor):
    x, y = x.cuda(), y.cuda()
    output, _, _ = model(x)
    return loss_fn(output, y)

#TODO add type of gradient
def update_weights(model : torch.nn.Module, gradient, learning_rate : float ):
    for i, param in enumerate(model.parameters()):
        param = learning_rate * gradient[i]
    
def inner_loop_update(model : torch.nn.Module, loss_fn : Callable, support_x : torch.Tensor, support_y : torch.tensor, learning_rate : float):
    for x, y in zip(support_x, support_y):
        loss = train_step(model, loss_fn, x, y)
        model.zero_grad()
        gradient = torch.autograd.grad(loss, model.parameters(), create_graph = True)
        update_weights(model , gradient, learning_rate)

def evaluate_query_set(model : torch.nn.Module, loss_fn : Callable, query_x : torch.Tensor, query_y : torch.tensor):
    loss = []
    for x, y in zip(query_x, query_y):
        output = model(x)
        loss.append(loss_fn(output, y))
    return loss

def meta_train_step(model : torch.nn.Module, loss_fn : Callable, support_x : torch.Tensor, support_y : torch.tensor, query_x : torch.Tensor, query_y : torch.tensor, inner_learning_rate : float, inner_loop_steps : int):
    #Train and update using the support set
    for _ in range(inner_loop_steps):
        inner_loop_update(model, loss_fn, x_support, y_support, inner_learning_rate)
    #Test using the query set
    return evaluate_query_set(query_x, query_y)
    

def meta_train(model : torch.nn.Module, loss_fn : Callable, meta_optimizer : Optimizer, data_loader : torch.utils.data.DataLoader, inner_learning_rate : float, inner_loop_steps : int, ckpt_path: str):
    #TODO split the data
    
    for ***:
        loss = meta_train_step(model, support_x, support_y, query_x, query_y, inner_learning_rate, inner_loop_steps)
        mean_loss = outer_loss.mean()
        mean_loss.backward()
        meta_optimizer.step()
        print(f'Task: {task} | loss: {loss:.4f}')

    torch.save(model.state_dict(), ckpt_path)

def meta_test(model : torch.nn.Module, loss_fn : Callable, meta_optimizer : Optimizer, inner_learning_rate : float, inner_loop_steps : int, ckpt_path: str):
    pass    

    torch.save(classifier.state_dict(), training_uris.ckpts)

if __name__ == '__main__':
    # classifier = classifier.train()
    # train(train_data_loader)
    # classifier = classifier.eval()
    # accuracy = test(test_data_loader)
    # print("************ Test Complete ************\n", "Final accuracy = " , accuracy)

    seed = random.randint(1, 10000)  # fix seed
    print("Random Seed: ", seed)
    random.seed(seed)
    torch.manual_seed(seed)

    try:
        os.makedirs("../ckpt/point_net")
    except OSError:
        pass
    
    dataset = ModelNet40C("../data/model_net_40c/data_original.npy", "../data/model_net_40c/label.npy")
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset , [train_size, test_size])

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=int(2))

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=int(2))

    classifier = PointNetCls(k = 40)
    optimizer = Optimizer.Adam(classifier.parameters(), lr = 0.001, betas= (0.9, 0.999))
    scheduler = Optimizer.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    loss_fn = F.nll_loss
    classifier.cuda()
    meta_train(classifier, loss_fn, optimizer, train_data_loader, 0.001, 10, "../ckpt/point_net_few_shot")


