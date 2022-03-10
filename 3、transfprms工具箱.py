from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
#transforms是个工具箱，tensor|resize是里面有具体用法的工具
# tool=transforms.ToTensor()创建一个具体的工具

#python的用法 tensor数据类型
# 通过transform.ToTensor去解决两个问题。
# 1、how to use transforms （python）
# 2、为啥用tensor数据类型（它包含了多个参数在里面，比如大小，反向传播参数，梯度等等）

img_path = "dataset/train/bees/16838648_415acd9e3f.jpg"
img = Image.open(img_path) #python自有的图片读取代码 在PIL里面

writer = SummaryWriter("formlogs")

tensor_trans = transforms.ToTensor() #实例化对象
tensor_img = tensor_trans(img)

writer.add_image("tensor_img",tensor_img,)

writer.close()
