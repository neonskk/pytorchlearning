import torch
from PIL import Image
import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Linear
from torch.nn.modules.flatten import Flatten

imge_path = "./images/feiji.jpg"
image = Image.open(imge_path)
print(image)

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),
                                            torchvision.transforms.ToTensor()])

image = transform(image)
print(image.shape)

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

model = torch.load("yang_29_gpu.pth",map_location=torch.device('cpu'))
print(model)
image = torch.reshape(image,(1,3,32,32))
model.eval()
with torch.no_grad():
    output = model(image)
print(output)
print(output.argmax(1))
