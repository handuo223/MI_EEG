import torchvision.transforms as transforms
from torch import device
from torch.utils.data import DataLoader
from BCIC.algorithms.models.CNNNet import *
import pandas as pd
from matplotlib import pyplot as plt
import torch


def accuracy(output, target):
    pred = torch.argmax(output, dim=1)
    pred = pred.float()
    correct = torch.sum(pred == target)
    return 100 * correct / len(target)
 
def plot_loss(epoch_number, loss):
    plt.plot(epoch_number, loss, color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss during test')
    plt.savefig("loss.jpg")
    plt.show()
    
def plot_accuracy(epoch_number, accuracy):
    plt.plot(epoch_number, accuracy, color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy during test')
    plt.savefig("accuracy.jpg")
    plt.show()
    
def plot_recall(epoch_number, recall):
    plt.plot(epoch_number, recall, color='purple', label='Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Rate')
    plt.title('Recall during test')
    plt.savefig("recall.jpg")
    plt.show()
 
def plot_precision(epoch_number,  precision):
    plt.plot(epoch_number, precision, color='black', label='Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Rate')
    plt.title('Precision during test')
    plt.savefig("precision.jpg")
    plt.show()
 
def plot_f1(epoch_number,  f1):
    plt.plot(epoch_number, f1, color='yellow', label='f1')
    plt.xlabel('Epoch')
    plt.ylabel('Rate')
    plt.title('f1 during test')
    plt.savefig("f1.jpg")
    plt.show()
    
def calc_recall_precision(output, target):
    pred = torch.argmax(output, dim=1)
    pred = pred.float()
    tp = ((pred == target) & (target == 1)).sum().item()  # 正确预测为“相同”的样本数
    tn = ((pred == target) & (target == 0)).sum().item()  # 正确预测为“不相同”的样本数
    fp = ((pred != target) & (target == 0)).sum().item()  # 错误预测为“相同”的样本数
    fn = ((pred != target) & (target == 1)).sum().item()  # 错误预测为“不相同”的样本数
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0  # 计算召回率
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0  # 计算精确度
    return recall, precision


EEGnetdata = CNNNetDataset('/home/handuo/test03/BCIC/preprocessed/A01_test_data.pt','/home/handuo/test03/BCIC/preprocessed/A01_test_label.pt' ,
                            transform=False,target_transform=False)

test_dataloader= DataLoader(EEGnetdata,shuffle=False,num_workers=0,batch_size=Config.test_batch_size,drop_last=True)

for epoch in range(0, Config.test_number_epochs):
    for i,data in enumerate(test_dataloader,0): #enumerate防止重复抽取到相同数据，数据取完就可以结束一个epoch
        item,target = data
        item,target= item.to(device),target.to(device)
        
        # optimizer.zero_grad() #grad归零
        output = net(item)  #输出
        # loss = criterion(output,target.long()) #算loss,target原先为Tensor类型，指定target为long类型即可。
        # loss.backward()   #反向传播算当前grad
        # optimizer.step()  #optimizer更新参数
        #求ACC标准流程
        Acc=accuracy(output, target)
        recall,precision=calc_recall_precision(output, target)
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        plot_accuracy(i, Acc)
        plot_f1(i,f1_score)
        if i % 10 == 0: #每10个epoch输出一次结果
                print("Epoch number {}\n Current Accuracy {}\n Current loss {}\n".format
                      (epoch, Acc.item(),loss.item()))
        iteration_number += 1
        counter.append(iteration_number)
        accuracy_history.append(train_accuracy.item())
        loss_history.append(loss.item())
        


