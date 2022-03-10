import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


# input = torch.tensor([[1,2,0,3,1],[0,1,2,3,1],[1,2,1,0,0],[5,2,3,1,1],[2,1,0,1,1]],dtype=torch.float32)
# input = torch.reshape(input,(-1,1,5,5))
# print(input.shape)

dataset = torchvision.datasets.CIFAR10("./CIFARdataset",train=False,download=True,transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset,batch_size=64)

class MAXPOOL(nn.Module):
#定义池化层类
    def __init__(self):
        super(MAXPOOL, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3,ceil_mode=False)
         #ceil_mode=True可以把未能被卷积核全部覆盖的区域也取最大值
    def forward(self,input):
        output = self.maxpool1(input)
        return output

yang = MAXPOOL()
# output = yang(input)
# print(output)

writer = SummaryWriter("logs_maxpool")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input",imgs,step)
    output = yang(imgs)
    writer.add_images("output",output,step)
    step = step + 1
writer.close()

