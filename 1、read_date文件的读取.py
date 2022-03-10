from torch.utils.data import Dataset
from PIL import Image
import os

# 读取文件夹里的数据，注意地址的拼接过程，以及两个数据集可以直接拼接
class MyDate(Dataset):

    def __init__(self,root_dir,label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir,self.label_dir) #os.path.join可以把地址依次拼接
        self.img_path = os.listdir(self.path)
        #os.listdir(path) 返回指定路径下所有文件和文件夹的名字，并存放于一个列表中。（方便下一步按照列表进行读取）

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir,self.label_dir,img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img,label
    #如果在类中定义了__getitem__()方法，那么他的实例对象（假设为P）就可以这样P[key]取值。
    # 当实例对象做P[key]运算时，就会调用类中的__getitem__()方法。
    #而且返回值就是__getitem__()方法中规定的return值。
    #注意 这里返回了img和label两个值，所以只写一个img = trains_dataset[124]就会报错

    def __len__(self):
        return len(self.img_path)
    #测量这个数据集长短的，如倒数第二行

root_dir = "dataset/train"
ants_label_dir = "ants"
bees_label_dir = "bees"
ant_dataset = MyDate(root_dir,ants_label_dir)
bee_dataset = MyDate(root_dir,bees_label_dir)

trains_dataset = ant_dataset+bee_dataset #数据集拼接

img, label = trains_dataset[122]
print(len(trains_dataset))
img.show()
