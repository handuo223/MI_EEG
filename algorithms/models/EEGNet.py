import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import nn
 
 
 
 
 
class DepthwiseConv(nn.Module):
    def __init__(self, inp, oup):
        super(DepthwiseConv, self).__init__()
        self.depth_conv = nn.Sequential(
            # dw
            nn.Conv2d(inp, inp, kernel_size=3, stride=1, padding=1, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            # pw
            nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(oup)
        )
    
    def forward(self, x):
        
        return self.depth_conv(x)
 
 
 
 
class depthwise_separable_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(depthwise_separable_conv, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.depth_conv = nn.Conv2d(ch_in, ch_in, kernel_size=3, padding=1, groups=ch_in)
        self.point_conv = nn.Conv2d(ch_in, ch_out, kernel_size=1)
 
    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        
        return x
 
 
 
 
 
class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()
        self.T = 500 
        #Layer 1
        self.conv1 = nn.Conv2d(22,48,(3,3),padding=0)
        self.batchnorm1 = nn.BatchNorm2d(48,False)

        #Layer 2
        self.Depth_conv = DepthwiseConv(inp=48,oup=22)
        self.pooling1 = nn.AvgPool2d(4,4)

        #Layer 3
        self.Separable_conv = depthwise_separable_conv(ch_in=22, ch_out=48)
        self.batchnorm2 = nn.BatchNorm2d(48,False)
        self.pooling2 = nn.AvgPool2d(2,2)

        #全连接层      
        self.fc1 = nn.Linear(576, 256)
        self.fc2 = nn.Linear(256,64)
        self.fc3 = nn.Linear(64,4)
    
    def forward(self, item):
        
        x = F.relu(self.conv1(item))
        x = self.batchnorm1(x)
        x = F.relu(self.Depth_conv(x))
        x = self.pooling1(x)
        x = F.relu(self.Separable_conv(x))
        x = self.batchnorm2(x)
        x = self.pooling2(x)
        #flatten
        x = x.contiguous().view(x.size()[0],-1) 
        #view函数：-1为计算后的自动填充值=batch_size，或x = x.contiguous().view(batch_size,x.size()[0])
        x = F.relu(self.fc1(x))
        x = F.dropout(x,0.25)
        x = F.relu(self.fc2(x))
        x = F.dropout(x,0.5)
        x = F.softmax(self.fc3(x),dim=1)
        
        return x