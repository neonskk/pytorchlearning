import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

#准备数据集
test_data = torchvision.datasets.CIFAR10(root="./CIFARdataset",train=False,transform=torchvision.transforms.ToTensor(),download=False)
test_loader = DataLoader(dataset=test_data,batch_size=64,shuffle=True,num_workers=0,drop_last=False)
# batch_size每次取64个打包
#shuffle=true 每一轮Epoch提取的照片都不一样，shuffle=false会让两轮提取的相同
#drop_last=true 如果最后剩下的不够64张就会舍去,这里的false就会让最后剩下16张单独一批

#测试数据集中第一张图片及其target
img, target = test_data[0]
print(img.shape)
print(target)

writer = SummaryWriter("dataloader")
step = 0
# for data in test_loader:
#     img, target = data
#     # print(img.shape)
#     # print(target)
#     writer.add_images("test_data", img, step)
#     step = step + 1

for epoch in range(2):
    for data in test_loader:
        img, target = data
        # print(img.shape)
        # print(target)
        writer.add_images("uuEpoch:{}".format(epoch), img,step)
        step = step + 1
#这里的epoch只是一个参数，和abcd没差别，改成其他的也可以

#epoch：1个epoch等于使用训练集中的全部样本训练一次
writer.close()