import torch

#方式1—》加载保存方式1的模型
import torchvision

model1 = torch.load('vgg16_mothod1.pth')
print(model1)

#方式2—》加载保存方式2的模型
# model2 = torch.load('vgg16_method2.pth')

vgg16 = torchvision.models.vgg16(pretrained=False)
#新建网络模型结构
vgg16.load_state_dict(torch.load('vgg16_method2.pth'))
#加载之前保存的字典参数
print(vgg16)