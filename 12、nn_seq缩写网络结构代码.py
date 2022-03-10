import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Linear, Sequential
from torch.nn.modules.flatten import Flatten
#Sequential用来将网络复杂的结构简化，并且可以为网络做上编号顺序
from torch.utils.tensorboard import SummaryWriter


class putong(nn.Module):
    def __init__(self):
        super(putong, self).__init__()
        # self.conv1 = Conv2d(3,32,5,padding=2)
        # self.maxpool1 = MaxPool2d(2)
        # self.conv2 = Conv2d(32,32,5,padding=2)
        # self.maxpool2 = MaxPool2d(2)
        # self.conv3 = Conv2d(32,64,5,padding=2)
        # self.maxpool3 = MaxPool2d(2)
        # self.flatten = Flatten()
        # #把数据展平
        # self.linear1 = Linear(1024,65)
        # self.linear2 = Linear(64, 10)
        # #线性层

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
        # x = self.conv1(x)
        # x = self.maxpool1(x)
        # x = self.conv2(x)
        # x = self.maxpool2(x)
        # x = self.conv3(x)
        # x = self.maxpool3(x)
        # x = self.flatten(x)
        # x = self.linear1(x)
        # x = self.linear2(x)
        x = self.model1(x)
        return x
noseq = putong()
print(noseq)
input = torch.ones((64,3,32,32))#创建一个全是1的64个一批次，3通道 32*32数据组。
output = noseq(input)

writer = SummaryWriter('./logs_seq')
writer.add_graph(noseq,input)
#计算图，可以在tensorboard显示网络的流程
writer.close()