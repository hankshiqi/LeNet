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
def test_data_load_process():
    test_data = FashionMNIST(root=r'E:\learn\深度学习\训练数据\LeNet',
                              train=True,
                              transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]),
                              download=True)
    test_dataloader = Data.DataLoader(dataset=test_data,
                                       batch_size=32,
                                       shuffle=True,
                                       num_workers=2)
    return test_dataloader
def test_model_process(model, test_dataloader):
    # 设定使用的设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 将模型放入训练设备中
    model = model.to(device)
    correct = 0.0
    test_num = 0
    # 关闭梯度计算,节省内存,加快测试速度,提高效率
    with torch.no_grad():
        for data, target in test_dataloader:
            # 将数据和标签放入训练设备中
            data, target = data.to(device), target.to(device)
            model.eval()
            # 前向传播
            output = model(data)
            # 计算预测标签
            pre_lab = torch.argmax(output, dim=1)
            # 计算正确标签
            correct += torch.sum(pre_lab == target.data).item()
            # 统计测试样本数
            test_num +=data.size(0)
    acc = correct / test_num
    print("测试集准确率为：",acc)
    return acc
if __name__ == '__main__':
    # 加载模型
    model = LeNet()
    # 加载测试数据
    model.load_state_dict(torch.load(r'best_model.pth'))
    # 测试模型
    test_loader = test_data_load_process()
    # test_acc = test_model_process(model, test_loader)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            model.eval()
            output = model(data)
            pre_lab = torch.argmax(output, dim=1)
            result = pre_lab[0].item()
            ans=target[0].item()
            print("Predicted: ", result, "  Actual: ", ans)
