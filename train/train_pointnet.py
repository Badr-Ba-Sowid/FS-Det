import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
import sys
sys.path.insert(0,"..")
from data_loader.data_loader import ModelNet40C
from models.pointnet import PointNetCls, feature_transform_regularizer
import random
import os
from tqdm import tqdm

BATCH_SIZE = 34
seed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", seed)
random.seed(seed)
torch.manual_seed(seed)


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
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=int(2))

try:
    os.makedirs("../ckpt/point_net")
except OSError:
    pass

classifier = PointNetCls(k = 40)
optimizer = optim.Adam(classifier.parameters(), lr = 0.001, betas= (0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
loss_fn = F.nll_loss
classifier.cuda()

def train(train_data_loader):
    for epoch in range(32):
        for i, data in enumerate(train_data_loader, 0):
            point_clouds, labels = data
            point_clouds, labels = point_clouds.cuda(), labels.cuda()
            optimizer.zero_grad()
            output, _, _ = classifier(point_clouds)
            loss = F.nll_loss(output, torch.flatten(labels))
            loss.backward()
            optimizer.step()
            if i%10 == 0:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {loss:.4f}')
        scheduler.step()
    torch.save(classifier.state_dict(), "../ckpt/point_net_batch_size_32_classes_40")

def test(test_data_loader):
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



if __name__ == '__main__':
    classifier = classifier.train()
    train(train_data_loader)
    classifier = classifier.eval()
    accuracy = test(test_data_loader)
    print("************ Test Complete ************\n", "Final accuracy = " , accuracy)
