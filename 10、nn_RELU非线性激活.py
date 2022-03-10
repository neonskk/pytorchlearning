import torch
import torchvision.datasets
from torch import nn
from torch.nn import ReLU,Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1,-0.5],[-1,3]])

input = torch.reshape(input,(-1,1,2,2))
print(input)
print(input.shape)

dataset = torchvision.datasets.CIFAR10("./CIFARdataset",train=False,download=True,transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset,batch_size=64)
class RELU1(nn.Module):
    def __init__(self):
        super(RELU1, self).__init__()
        self.relu = ReLU()
        self.sigmoid = Sigmoid()

    def forward(self,input):
        output = self.sigmoid(input)
        output2 = self.relu(input)
        return output,output2

yang = RELU1()
# output1,output2 = yang(input)
# print(output1)
# print(output2)

writer = SummaryWriter("logs_relu")
step = 0
for data in dataloader:
    imgs,targets = data
    writer.add_images("input",imgs,global_step=step)
    output,output2 = yang(imgs)
    writer.add_images("Sigmoid",output,global_step=step)
    writer.add_images("ReLU", output2, global_step=step)
    step = step + 1
writer.close()


