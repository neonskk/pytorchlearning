import torchvision
from torch.utils.tensorboard import SummaryWriter
from urllib3.filepost import writer

#数据集转化为tensor格式，然后录入到P5文件夹，在tensorboard面板显示
dataset_transfom = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
train_set = torchvision.datasets.CIFAR10(root="./CIFARdataset",train=True,transform=dataset_transfom,download=True)
test_set = torchvision.datasets.CIFAR10(root="./CIFARdataset",train=False,transform=dataset_transfom,download=True)

# print(test_set[0])
# print(test_set.classes)
#
# img, target = test_set[6]
# print(img)
# print(target)
# print(test_set.classes[target])
# img.show()

# print(test_set[0])

writer = SummaryWriter("P5")
for i in range(10):
    img, target = test_set[i]
    writer.add_image("test_set",img,i)
writer.close()