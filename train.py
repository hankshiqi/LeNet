import copy
import time

import torch
from torch import nn
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import LeNet


# 训练加载函数
def train_val_data_load_process():
    train_data = FashionMNIST(root=r'E:\learn\深度学习\训练数据\LeNet',
                              train=True,
                              transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]),
                              download=True)
    train_data, val_data = Data.random_split(train_data, [round(0.8 * len(train_data)), round(0.2 * len(train_data))])
    train_dataloader = Data.DataLoader(dataset=train_data,
                                       batch_size=32,
                                       shuffle=True,
                                       num_workers=2)
    val_dataloader = Data.DataLoader(dataset=val_data,
                                     batch_size=32,
                                     shuffle=True,
                                     num_workers=2)
    return train_dataloader, val_dataloader


# train_val_data_load_process()
def train_model_process(model, train_dataloader, val_dataloader, num_epochs):
    # 设定使用的设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 使用Adam优化器
    optimism = torch.optim.Adam(model.parameters(), lr=0.001)
    # 损失函数为交叉熵函数
    criterion = nn.CrossEntropyLoss()
    # 将模型放入训练设备中
    model = model.to(device)
    # 复制当前模型参数
    best_model_wts = copy.deepcopy(model.state_dict())

    # 初始化参数
    best_acc = 0.0
    # 训练集损失列表
    train_loss_all = []
    # 验证集损失列表
    val_loss_all = []
    # 训练集精确度列表
    train_acc_all = []
    # 验证集精确度列表
    val_acc_all = []
    since = time.time()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print("-" * 10)

        # 初始化loss值
        train_loss = 0.0
        # 初始化准确度
        train_acc = 0.0
        # 验证集loss值
        val_loss = 0.0
        # 验证集准确度
        val_acc = 0.0
        # 训练样本数量
        train_num = 0
        # 验证集样本数量
        val_num = 0
        for step, (b_x, b_y) in enumerate(train_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            # 设置模型为训练模型
            model.train()

            # 前向传播过程
            outputs = model(b_x)
            # 查找每一行中最大值对应的行标
            pre_lab = torch.argmax(outputs, dim=1)
            # 计算每个batch的损失函数
            loss = criterion(outputs, b_y)

            # 将梯度初始化为0
            optimism.zero_grad()
            # 反向传播计算
            loss.backward()
            # 根据网络反向传播梯度信息来更新网络的参数，以起到降低loss计算值的作用
            optimism.step()

            # 对损失函数进行累加
            train_loss += loss.item() * b_x.size(0)
            # 对准确度进行累加   ？？？
            train_acc += torch.sum(pre_lab == b_y.data)
            # 当前用于训练的样本数量
            train_num += b_x.size(0)
        for step, (b_x, b_y) in enumerate(val_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            model.eval()

            #得到评估结果
            output=model(b_x)
            # 查找每一行中最大值对应的行标
            pre_lab = torch.argmax(output, dim=1)
            # 计算每个batch的损失函数
            loss = criterion(output, b_y)
            #注意验证不需要反向传播没有更新参数的过程

            # 对损失函数结果进行累加
            val_loss += loss.item() * b_x.size(0)
            # 对准确度进行累加   ？？？
            val_acc += torch.sum(pre_lab == b_y.data)
            # 当前用于训练的样本数量
            val_num += b_x.size(0)

        #计算并保存每一次模型函数的准确率和loss
        train_loss_all.append(train_loss/train_num)
        val_loss_all.append(val_loss/val_num)
        train_acc_all.append(train_acc.double().item()/train_num)
        val_acc_all.append(val_acc.double().item()/val_num)
        print('{} train Loss: {:.4f} train Acc: {:.4f}'.format(epoch, train_loss_all[-1], train_acc_all[-1]))
        print('{} Val   Loss: {:.4f}  val  Acc: {:.4f}'.format(epoch, val_loss_all[-1], val_acc_all[-1]))
        #寻找最高准确度的权重参数
        if val_acc_all[-1] >best_acc:
            best_acc = val_acc_all[-1]
            #保存当前最高准确度，深拷贝保证后续的参数改变不会影响到最好状态的参数
            best_model_wts=copy.deepcopy(model.state_dict())
        #花费时间
        time_use=time.time()-since
        print('训练所耗费的时间{:.0f}m {:.0f}s'.format(time_use//60, time_use%60))

    #选择最优模型进行加载
    #保存最优参数
    torch.save(best_model_wts,'best_model_wts.pth')

    train_process=pd.DataFrame(data={"epoch":range(num_epochs),
                                    "train_loss": train_loss_all,
                                    "val_loss": val_loss_all,
                                    "train_acc": train_acc_all,
                                    "val_acc": val_acc_all})
    return train_process
#画图
def matplot_acc_loss(train_process):
    plt.figure(figsize=(12,4))
    #一行两列的第一张图
    plt.subplot(1,2,1)
    plt.plot(train_process["epoch"], train_process.train_loss,'ro-',label="train loss")
    plt.plot(train_process["epoch"], train_process.val_loss,'bs-',label="val loss")
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')

    plt.subplot(1, 2, 2)
    plt.plot(train_process["epoch"], train_process.train_acc, 'ro-', label="train acc")
    plt.plot(train_process["epoch"], train_process.val_acc, 'bs-', label="val acc")
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('acc')
if __name__ == '__main__':
    LeNet=LeNet()
    train_dataloader,val_dataloader=train_val_data_load_process()
    train_process=train_model_process(LeNet,train_dataloader,val_dataloader,10)
    matplot_acc_loss(train_process)
    plt.show()