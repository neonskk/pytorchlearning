from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

# tesorboard最大的作用是可视化，让输入的图表可以打开观察过程

writer = SummaryWriter('logs')
#创建一个logs的文件夹放数据（图片和列表等）

image_path = "dataset/train/ants/5650366_e22b7e1065.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)
#把图片从JpegImageFile格式转化为numpy格式

print(img_array.shape)#显示一下结构

writer.add_image("test",img_array,2,dataformats="HWC")
#标题，图片输入，第几个，高宽通道数的shape要加dataformats="HWC"，否则报错

for i in range(100):
    writer.add_scalar('y=x', i, i)
#表格的标题，y的数据，x轴的数据，xy轴名称似乎不能改

writer.close()#图片数据录入完成，关闭录入
