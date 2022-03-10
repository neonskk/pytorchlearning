import torchvision.models.vgg
from torch import nn

vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)

print(vgg16_true)

train_data = torchvision.datasets.CIFAR10("./CIFARdataset",train=True,transform=torchvision.transforms.ToTensor(),download=True)

vgg16_true.classifier.add_module("add_linear",nn.Linear(1000,10))
#使用add_module加在classifier层最后一个线性层
print(vgg16_true)
vgg16_false.classifier[6] = nn.Linear(4096,100)
#将第六行替换为Linear(4096,100)
print(vgg16_false)