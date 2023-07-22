import cv2 as cv
import numpy as np
import torchvision
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
from torch import nn,optim
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader



batch_size = 64#一个批次的大小,64张照片

#创建模型
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        # Comment:卷积层
        self.features = nn.Sequential( #输入是：28*28*1    卷积核3*3  步长1   填充  1
            nn.Conv2d(in_channels=1,out_channels=6,kernel_size=(3,3),stride=1,padding=1),
            nn.ReLU(),
            #卷积核2*2  步长2
            nn.MaxPool2d(kernel_size=2,stride=2), # 输出是:14*14*6
            #卷积核3*3  步长1   填充  1
            nn.Conv2d(in_channels=6,out_channels=16,kernel_size=(3,3),stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2), # 输出是:6*6*16  卷积核3*3  步长1   填充  1
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出是:3*3*32
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=288, out_features=1024),
            nn.ReLU(),# 输出是:512*1024
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),# 输出是:1024*512
            nn.Linear(in_features=512, out_features=10)# 输出是:512*10
        )
    def forward(self,x):
        # 定义了：向前传播
        x = self.features(x)
        x = torch.flatten(x,1)
        result = self.classifier(x)
        return result


if __name__ == '__main__':

    #下载数据
    train_data = datasets.MNIST(
        root="./data/",
        train=True,
        transform=transforms.ToTensor(),
        download=True)
    test_data = datasets.MNIST(
        root="./data/",
        train=False,
        transform=transforms.ToTensor(),
        download=True)
    #加载数据

    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        )
    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False,
        )

    lr = 0.28

    model = LeNet()
    optimizer = optim.SGD(params=model.parameters(), lr=lr)
    loss_train = []
    loss_func = nn.CrossEntropyLoss()


    for i in range(14):
        loss_temp = 0
        for j, (batch_data, batch_label) in enumerate(train_data_loader):
            batch_data, batch_label = batch_data.cuda(), batch_label.cuda()
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model.to(device)
            optimizer.zero_grad()
            prediction = model(batch_data)
            loss = loss_func(prediction, batch_label)
            loss_temp += loss.item()
            loss.backward()
            optimizer.step()
        loss_train.append(loss_temp / len(train_data_loader))

        print('[%d] loss: %.4f' % (i + 1, loss_temp / len(train_data_loader)))

    torch.save(model, 'number_ocr1.pth')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(loss_train)
    ax.set_xlabel('epoch', fontsize=14)
    ax.set_ylabel('loss', rotation=0, fontsize=14)
    plt.show()

    correct = 0
    for batch_data, batch_label in test_data_loader:
        # 放入GPU中
        batch_data, batch_label = batch_data.cuda(), batch_label.cuda()
        # batch_data, batch_label = batch_data.cpu(), batch_label.cpu()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        # 预测
        prediction = model(batch_data)
        prediction = prediction.cpu()
        batch_label = batch_label.cpu()
        #print(prediction)
        # 将预测值中最大的索引取出，其对应了不同类别值
        predicted = torch.max(prediction.data, 1)[1]
        #print(predicted)
        #print(batch_label)
        # 获取准确个数
        correct += (predicted == batch_label).sum()
        #print(correct)
    print(correct)
    #print(len(batch_data))
    print('准确率: %.2f %%' % (100 * correct / 10000))  #











