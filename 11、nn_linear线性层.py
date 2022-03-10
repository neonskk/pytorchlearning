import torch
import torchvision.datasets
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("./CIFARdataset",train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader = DataLoader(dataset,batch_size=64,drop_last=True)

class linear(nn.Module):
    def __init__(self):
        super(linear, self).__init__()
        self.linear1 = Linear(196608,10)
    def forward(self, input):
        output = self.linear1(input)
        return output


yang = linear()
for data in dataloader:
    imgs,targets = data
    print(imgs.shape)
    # output = torch.reshape(imgs,(1,1,1,-1))
    output = torch.flatten(imgs)
    print(output.shape)
    output = yang(output)
    print(output.shape)
