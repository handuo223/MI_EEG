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
 
 
 
# ## 帮助函数
def show_plot(iteration,accuracy,loss):
    plt.plot(iteration,accuracy,loss)
    plt.show()
 
 
def test_show_plot(iteration,accuracy):
    plt.plot(iteration,accuracy)
    plt.show()
 
 
# ## 用于配置的帮助类
class Config():
    training_dir = "./data/faces/training/"
    testing_dir = "./data/faces/testing/"
    train_batch_size = 48 # 64
    test_batch_size = 20
    train_number_epochs = 100 # 100
    test_number_epochs = 20
 
 
class CNNNetDataset(Dataset):
    def __init__(self,file_path,target_path,transform=None,target_transform=None):
        self.file_path = file_path
        self.target_path = target_path
        self.data = self.parse_data_file(file_path)
        self.target = self.parse_target_file(target_path)
        
        self.transform = transform
        self.target_transform = target_transform
        
    def parse_data_file(self,file_path):
        
        data = torch.load(file_path)
        return np.array(data,dtype=np.float32)
    
    def parse_target_file(self,target_path):
        
        target = torch.load(target_path)
        return np.array(target,dtype=np.float32)
    
    def __len__(self):
        
        return len(self.data)
    
    def __getitem__(self,index):
        item = self.data[index,:]
        target = self.target[index]
        
        if self.transform:
            item = self.transform(item)
        if self.target_transform:
            target = self.target_transform(target)
        
        return item,target
 
 
 
 
class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        
        self.conv1 = nn.Conv2d(22,44,(1,3),stride=2)
        self.conv2 = nn.Conv2d(44,88,(1,3),stride=2)
        self.batchnorm1 = nn.BatchNorm2d(88,False)
        self.pooling1 = nn.MaxPool2d(2,2)
        
        self.conv3 = nn.Conv2d(88,44,(1,3),stride=2)
        #flatten
        self.fc1 = nn.Linear(88,64) 
        self.fc2 = nn.Linear(64,32)
        self.fc3 = nn.Linear(32,4)
 
    def forward(self,item):
    
        x = F.elu(self.conv1(item))
        x = F.elu(self.conv2(x))
        x = self.batchnorm1(x)
        x = self.pooling1(x)
        x = F.relu(self.conv3(x))
        #flatten
        x = x.contiguous().view(x.size()[0],-1) 
        #view函数：-1为计算后的自动填充值，这个值就是batch_size，或者x = x.contiguous().view(batch_size,x.size()[0])
        x = F.relu(self.fc1(x))
        x = F.dropout(x,0.25)
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x),dim=1) #self.sf =nn.Softmax(dim=1)
        return x