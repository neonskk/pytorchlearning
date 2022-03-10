import time

import torch
import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Linear
from torch.nn.modules.flatten import Flatten
from torch.utils.data import DataLoader

#定义训练的设备
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda')
#准备训练和测试的数据集
train_data = torchvision.datasets.CIFAR10(root="./CIFARdataset",train=True,transform=torchvision.transforms.ToTensor(),download=True)
test_data = torchvision.datasets.CIFAR10(root="./CIFARdataset",train=False,transform=torchvision.transforms.ToTensor(),download=True)

#计算数据集长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

#加载数据集
train_dataloader = DataLoader(train_data,batch_size=64)
test_dataloader = DataLoader(test_data,batch_size=64)

#创建网络
class putong(nn.Module):
    def __init__(self):
        super(putong, self).__init__()
        self.model1 = Sequential(
            Conv2d(3,32,5,padding=2),  #这里注意要使用逗号做分割！
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)   #最后一个不用打逗号
        )
    def forward(self,x):
        x = self.model1(x)
        return x

yang = putong()
yang.to(device)

#损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)

#优化器
#learning_rating = 0.01学习率设置
#1e-2 = 1x（10）^(-2) = 1/100 =0.01 用这个可以避免写0.01的0太多了写错了
learning_rata = 1e-2
optimizer = torch.optim.SGD(yang.parameters(),lr=learning_rata)

#设置训练网络的一些参数
#记录训练的次数
total_train_step = 0
#记录测试的次数
total_test_step = 0
#训练的轮数
epoch = 30

#添加tensorboard
writer = SummaryWriter('../logs_train')
start_time = time.time()
for i in range(epoch):
    print('-------------第{}轮训练开始-----------------'.format(i+1))

    #训练步骤开始
    yang.train()
    for data in train_dataloader:
        imgs,targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = yang(imgs)
        loss = loss_fn(outputs,targets)

        #优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0 :
            end_time = time.time()
            print("花费时间为{}".format(end_time - start_time))
            print("训练次数：{}，损失函数loss：{}".format(total_train_step,loss.item()))
            writer.add_scalar("train_loss",loss.item(),total_train_step)

    #测试步骤开始
    yang.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        #【with torch.no_grad()】强制之后的内容不进行计算图构建,也就是不生成grad_fn=<AddmmBackward>，减少现存开销
        for data in test_dataloader:
            imgs,targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = yang(imgs)
            loss = loss_fn(outputs,targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    print("整体测试集上的loss：{}".format(total_test_loss))
    print('total_accuracy的值是{}'.format(total_accuracy))
    print('test_data_size的值是{}'.format(test_data_size))
    print("整体测试集上的正确率：{}".format(float(total_accuracy)/float(test_data_size)))
    writer.add_scalar("test_loss",total_test_loss,total_test_step)
    writer.add_scalar("test_accuracy",total_accuracy/test_data_size,total_test_step)
    total_test_step = total_test_step + 1
    torch.save(yang,"yang_{}_gpu.pth".format(i))
    print("模型已保存")

writer.close()






