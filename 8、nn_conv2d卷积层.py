import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root="./CIFARdataset",train=False,transform=torchvision.transforms.ToTensor(),download=False)
dataloader = DataLoader(dataset,batch_size=64)

class JUANJI(nn.Module):
     def __init__(self):
         super(JUANJI, self).__init__()
         self.conv1 = Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0)
     def forward(self,x):
        x = self.conv1(x)
        return x
yang = JUANJI()
writer = SummaryWriter("./logs.nnModule")

step = 0
for data in dataloader:
    imgs, targets = data
    output = yang(imgs)
    # print(imgs.shape)
    # print(output.shape)
    writer.add_images("input",imgs,step)
    # writer.add_images("output",output,step)

    output = torch.reshape(output,(-1,3,30,30))
    writer.add_images("output",output,step)
    step = step + 1
writer.close()