from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as Data
import numpy as np

train_data=FashionMNIST(root=r'E:\learn\深度学习\训练数据\LeNet',
                        train=True,
                        transform=transforms.Compose([transforms.Resize(size=224),transforms.ToTensor()]),
                        download=True)

train_loader=Data.DataLoader(dataset=train_data,
                             batch_size=64,
                             shuffle=True,
                             num_workers=0)#这里用多进程会有问题

for step,(b_x,b_y) in enumerate(train_loader):
    if step >0:
        break
batch_x=b_x.squeeze().numpy()#将四维张量移除第一维，并转换成numpy
batch_y=b_y.squeeze().numpy()
class_label=train_data.classes
print(class_label)

