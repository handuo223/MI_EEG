import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader
from BCIC.algorithms.models.CNNNet import *
import pandas as pd
from sklearn.model_selection import train_test_split


train_EEGnetdata = CNNNetDataset('/home/handuo/test03/BCIC/preprocessed/A01_train_data.pt','/home/handuo/test03/BCIC/preprocessed/A01_train_label.pt' ,
                            transform=False,target_transform=False) 
train_dataloader  = DataLoader(train_EEGnetdata,shuffle=True,num_workers=0,batch_size=Config.train_batch_size,drop_last=True)
 
test_EEGnetdata = CNNNetDataset('/home/handuo/test03/BCIC/preprocessed/A01_test_data.pt','/home/handuo/test03/BCIC/preprocessed/A01_test_label.pt' ,
                            transform=False,target_transform=False)
test_dataloader= DataLoader(test_EEGnetdata,shuffle=False,num_workers=0,batch_size=Config.test_batch_size,drop_last=True)
 
val_EEGnetdata = CNNNetDataset('/home/handuo/test03/BCIC/preprocessed/A01_test_data.pt','/home/handuo/test03/BCIC/preprocessed/A01_test_label.pt' ,
                            transform=False,target_transform=False)
val_dataloader= DataLoader(val_EEGnetdata,shuffle=True,num_workers=0,batch_size=Config.test_batch_size,drop_last=True)


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = CNNNet().to(device)
criterion = torch.nn.CrossEntropyLoss()
#criterion = nn.MultiMarginLoss()
#optimizer = optim.SGD(net.parameters(),lr=0.8)
optimizer = optim.Adam(net.parameters(), lr=0.001)
counter = []
loss_history = []
iteration_number = 0
train_correct = 0
total = 0
train_accuracy = []
correct = 0
total = 0
classnum = 2
accuracy_history = []
 
net.train()
 
for epoch in range(0, Config.train_number_epochs):
    
    for i,data in enumerate(train_dataloader,0): #enumerate防止重复抽取到相同数据，数据取完就可以结束一个epoch
        item,target = data
        item,target= item.to(device),target.to(device)
        
        optimizer.zero_grad() #grad归零
        output = net(item)  #输出
        loss = criterion(output,target.long()) #算loss,target原先为Tensor类型，指定target为long类型即可。
        loss.backward()   #反向传播算当前grad
        optimizer.step()  #optimizer更新参数
        #求ACC标准流程
        predicted=torch.argmax(output, 1)
        train_correct += (predicted == target).sum().item()
        total+=target.size(0) # total += target.size
        train_accuracy = train_correct / total
        train_accuracy = np.array(train_accuracy)
        
        if i % 10 == 0: #每10个epoch输出一次结果
                print("Epoch number {}\n Current Accuracy {}\n Current loss {}\n".format
                      (epoch, train_accuracy.item(),loss.item()))
        iteration_number += 1
        counter.append(iteration_number)
        accuracy_history.append(train_accuracy.item())
        loss_history.append(loss.item())
        plt.plot(i, train_accuracy, color='orange')
        plt.savefig("accuracy.jpg")

    net.eval()  # 设置模型为评估模式
    val_correct = 0
    val_total = 0

    #with torch.no_grad():
    for data in val_dataloader:
            # 获取测试数据
            item, target = data
            item, target = item.to(device), target.to(device)

            # 进行推理
            output = net(item)
            predicted = torch.argmax(output, 1)
            # 计算准确率
            val_correct += (predicted == target).sum().item()
            val_total += target.size(0)
            # 计算测试集上的准确率
            val_accuracy = val_correct / val_total
            print(f"Epoch {epoch}: val_Accuracy: {val_accuracy}")
            plt.plot(i, val_accuracy, color='blue')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Accuracy during test')
            #plt.savefig("accuracy.jpg")
            plt.show()

# 保存模型
torch.save(net.state_dict(),"The train.CNNNet.ph")
