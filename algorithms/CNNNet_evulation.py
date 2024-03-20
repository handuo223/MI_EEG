
import torchvision.transforms as transforms
from torch import device
from torch.utils.data import DataLoader
from BCIC.algorithms.models.CNNNet import *
import pandas as pd
from matplotlib import pyplot as plt
import torch
import torchvision.models as model



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
print(len(EEGnetdata))

test_dataloader= DataLoader(EEGnetdata,shuffle=False,num_workers=0,batch_size=Config.test_batch_size,drop_last=True)
print(len(test_dataloader))

criterion = torch.nn.CrossEntropyLoss()
counter = []
loss_history = []
iteration_number = 0
test_correct = 0
total = 0
train_accuracy = []
correct = 0
total = 0
accuracy_history = []

# 定义一个与你训练模型相同架构的模型
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNNNet().to(device)  # 这里以ResNet-18为例，你需要根据你的实际模型进行调整

# 指定已经训练好的模型的文件路径
model_path = '/home/handuo/test03/The train.CNNNet.ph'

# 如果你的模型是在 GPU 上训练的，需要添加下面这行以在 CPU 上加载
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

# 将加载的权重加载到模型中
model.load_state_dict(checkpoint)
model.eval()

for epoch in range(0, Config.test_number_epochs):
    for i,data in enumerate(test_dataloader,0): #enumerate防止重复抽取到相同数据，数据取完就可以结束一个epoch
        item,target = data
        item,target= item.to(device),target.to(device)

        with torch.no_grad():
            output = model(item)
            loss = criterion(output, target.long())
            Acc=accuracy(output,target)
            recall,precision=calc_recall_precision(output,target)


        if i % 10 == 0: #每10个epoch输出一次结果
            print("Epoch number {}\n Current Accuracy {}\n Current loss {}\n".format
                      (epoch, Acc,recall,precision))
            
        plot_accuracy(i,Acc)
        plot_precision(i,precision)

        

