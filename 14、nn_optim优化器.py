import torch.optim
import torchvision.datasets
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Linear, Sequential
from torch.nn.modules.flatten import Flatten
#Sequential用来将网络复杂的结构简化，并且可以为网络做上编号顺序
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
#-----
from tqdm import trange, tqdm
import time, random
#引入进度条
#---------


dataset = torchvision.datasets.CIFAR10('./CIFARdataset',train=False,transform=torchvision.transforms.ToTensor(),download=True)

dataloader = DataLoader(dataset,batch_size=1)
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
loss = nn.CrossEntropyLoss()
loss = loss.cuda() #给loss接入gpu训练
yang = putong()
yang = yang.cuda() #将模型接入gpu训练
optim = torch.optim.SGD(yang.parameters(),lr=0.01)
for epoch in tqdm(range(20)):
    running_loss = 0.0
    time.sleep(random.random())
    for data in dataloader:
        imges,targets = data
        imges = imges.cuda()
        targets = targets.cuda()#把数据放进gpu训练
        output = yang(imges)
        result_loss = loss(output,targets)
        optim.zero_grad()
        result_loss.backward()
        optim.step()
        running_loss = running_loss+result_loss
    print(running_loss)

