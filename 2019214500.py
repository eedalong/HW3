import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import argparse
import numpy as np
from utils import LoadData
from PIL import Image
class HWDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        data = torch.FloatTensor(self.data[index].reshape([1,28,28]))

        return data, int(self.label[index])

    def __len__(self):
        return len(self.data)

class ComplicateModule(torch.nn.Module):
    def __init__(self):
        super(ComplicateModule, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 20, 5, 1)
        self.bn = torch.nn.BatchNorm2d(20)
        self.conv2 = torch.nn.Conv2d(20, 50, 5, 1)
        self.fc1 = torch.nn.Linear(4 * 4 * 50, 500)
        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()
        self.relu3 = torch.nn.ReLU()

    def forward(self,x ):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu1(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.relu2(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = self.relu3(self.fc1(x))
        return x

class HWModel(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.compact_model = ComplicateModule()
        self.fc2 = torch.nn.Linear(500, dim)

    def forward(self, x):
        x = self.compact_model(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1), x

class HWModel2(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 256)
        self.bn = torch.nn.BatchNorm1d(256)
        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, dim)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), x

def train(model, device, train_loader, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, _ = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            pass
            #print('{:2.0f}%  Loss {}'.format(10 * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, _ = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    print('Loss: {}  Accuracy: {}%)\n'.format(
        test_loss, 100 * correct / len(test_loader.dataset)))


if __name__ == "__main__":


    # define settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=50,
                        help='number of classes used')
    parser.add_argument('--num_samples_train', type=int, default=15,
                        help='number of samples per class used for training')
    parser.add_argument('--num_samples_test', type=int, default=5,
                        help='number of samples per class used for testing')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed')
    args = parser.parse_args()

    train_image, train_label, test_image, test_label = LoadData(args.num_classes, args.num_samples_train,
                                                                args.num_samples_test, args.seed)
    train_loader = torch.utils.data.DataLoader(HWDataset(train_image, train_label), shuffle=True, batch_size=8)
    test_loader = torch.utils.data.DataLoader(HWDataset(test_image, test_label), shuffle=False, batch_size=1)

    model = HWModel2(args.num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(),lr=0.001, momentum=0.5)
    epoch_num = 500
    for epoch in range(epoch_num):
        train(model, device, train_loader, optimizer)
        if epoch % 10 == 0:
            test(model, device, test_loader)
    test(model, device, test_loader)



