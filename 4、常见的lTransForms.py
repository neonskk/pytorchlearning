from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("logs")
#writer是一个文件夹
img = Image.open("images/ceshi.jpg")
print(img)

#totensor来转换格式(tensor格式)
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
print(img_tensor.shape)
writer.add_image("ToTensor", img_tensor)
writer.close()

# Normalize归一化
print(img_tensor[0][0][0])
#归一化的计算公式input[channel] = (input[channel] - mean[channel]) / std[channel]
trans_norm = transforms.Normalize([0.6,0.6,0.6],[0.8,0.7,0.3])
#参数1【mean】：每个信道的平均值序列；参数2【std】：每个信道标准差序列 这里的数字是随便写的，可以自己改
#rgb是三信道
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
print(img_norm[1][1][1])
writer.add_image("Normalize",img_norm,2)

# Resize 改变图片大小
print(img)
img.show()
trans_resize = transforms.Resize(200)
img_resize = trans_resize(img)
print(img_resize)
img_resize.show()

#Compose()用法，需要一个列表。
#所以得到 Compose（[transforms参数1，transforms参数2,。。。]）
trans_resize_2 = transforms.Resize((512,512))
#PIL > PIL >tensor
trans_compose = transforms.Compose([trans_resize_2,trans_totensor])
img_resize_2 = trans_compose(img)
print(img_resize_2)
writer.add_image("Resize",img_resize_2,1)

#RandomCrop
#随意裁剪512大小的图片
trans_random = transforms.RandomCrop((200,200))
trans_compose_2 = transforms.Compose([trans_random,trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrop",img_crop,i)
print(img_crop.shape)

writer.close()
