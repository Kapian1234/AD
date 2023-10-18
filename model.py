import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd


# CBMA通道注意力
class ChannelAttention(nn.Module):
    def __init__(self, in_channel, ratio=8):
        super().__init__()
        self.avgPool = nn.AdaptiveAvgPool3d(1)
        self.maxPool = nn.AdaptiveMaxPool3d(1)

        self.fc1 = nn.Conv3d(in_channel, in_channel // ratio, kernel_size=1, bias=False)
        self.fc2 = nn.Conv3d(in_channel // ratio, in_channel, kernel_size=1, bias=False)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()



    def forward(self, x):
        # 分别对输入进行平均池化和最大池化
        x_avgPool = self.avgPool(x)
        x_maxPool = self.maxPool(x)

        # 使用卷积层和激活函数，对池化后的结果进行通道压缩和恢复
        avgPool = self.fc2(self.relu(self.fc1(x_avgPool)))
        maxPool = self.fc2(self.relu(self.fc1(x_maxPool)))

        # 将上面得到的两个结果相加并归一化，得到综合的通道注意力权重，并将权重与输入相乘
        output = x * self.sigmoid(avgPool + maxPool)

        return output


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv3d(1,32,kernel_size=1),
                                   nn.InstanceNorm3d(32),
                                   nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(nn.Conv3d(32,64,kernel_size=(3,3,3),stride=1,padding=1),
                                   nn.InstanceNorm3d(64),
                                   nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(nn.Conv3d(64,128,kernel_size=3,padding=1),
                                   nn.InstanceNorm3d(128),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(nn.Conv3d(128,256,kernel_size=3,padding=1),
                                   nn.InstanceNorm3d(256),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(nn.Conv3d(256,512,kernel_size=3,padding=1),
                                   nn.InstanceNorm3d(512),
                                   nn.ReLU(inplace=True))

        self.conv6 = nn.Sequential(nn.Conv3d(512,512,kernel_size=3),
                                   nn.InstanceNorm3d(512),
                                   nn.ReLU(inplace=True))
        self.MaxPool = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.ca1 = ChannelAttention(64)
        self.ca2 = ChannelAttention(128)
        self.ca3 = ChannelAttention(256)
        self.ca4 = ChannelAttention(512)
        self.average = nn.AdaptiveAvgPool3d(1) # C*1*1*1
        self.fc = nn.Linear(512,3)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.conv1(x)

        x = self.conv2(x)
        x = self.MaxPool(x)

        x = self.conv3(x)
        x = self.ca2(x)
        x = self.MaxPool(x)

        x = self.conv4(x)
        x = self.ca3(x)
        x = self.MaxPool(x)

        x = self.conv5(x)
        x = self.ca4(x)
        x = self.MaxPool(x)

        x = self.conv6(x)

        x = self.average(x)

        x = x_linear = x.view(x.shape[0], -1)
        x = self.dropout(x)
        output = self.fc(x)

        return x_linear, output
