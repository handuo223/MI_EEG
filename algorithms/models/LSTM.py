from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score,accuracy_score,classification_report


dataset = np.load('/home/handuo/test02/test20231109/dataset/BCIC-2a/A01_train.npy')
label=np.load('/home/handuo/test02/test20231109/dataset/BCIC-2a/A01train_label.npy')
label = label.flatten()
train_data, test_data, train_label, test_label = train_test_split(dataset, label, test_size=0.3,random_state=5)
print(test_label)


def one_hot(y_, num_classes):
    y_ = y_.reshape(len(y_))
    y_ = [int(xx) for xx in y_]
    n_values = num_classes
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]

# check if a GPU is available
with_gpu = torch.cuda.is_available()
if with_gpu:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print('We are using %s now.' %device)


BATCH_size = test_data.shape[0] # use test_data as batch size
print(BATCH_size)

# feed data into dataloader
train_data= torch.tensor(train_data).to(device)
train_label = torch.tensor(train_label.flatten()).to(device)

train_data = Data.TensorDataset(train_data, train_label)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_size, shuffle=False)
test_data = torch.tensor(test_data).to(device)
test_label = torch.tensor(test_label.flatten()).to(device)

# 定义神经网络
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()

        self.lstm_layer = nn.LSTM(
            input_size=1000,
            hidden_size=2000,         # LSTM hidden unit
            num_layers=2,           # number of LSTM layer
            bias=True,
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, segment_length, no_feature)
        )

        self.out = nn.Linear(2000, 4)

    def forward(self, x):
        r_out, (h_n, h_c) = self.lstm_layer(x.float(), None)
        r_out = F.dropout(r_out, 0.3)

        test_output = self.out(r_out[:, -1, :]) # choose r_out at the last time step
        return test_output

lstm = LSTM()
lstm.to(device)
print(lstm)

#定义优化器、代价函数
optimizer = torch.optim.Adam(lstm.parameters(), lr=0.005, weight_decay=0.01)   # optimize all parameters
loss_func = nn.CrossEntropyLoss()

best_acc = []
best_auc = []
best_train_acc=[]

# 训练模型
start_time = time.perf_counter()
for epoch in range(11):
    for step, (train_x, train_y) in enumerate(train_loader):

        output = lstm(train_x)  # LSTM output of training data
        loss = loss_func(output, train_y.long())  # cross entropy loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

    if epoch % 10 == 0:
        test_output = lstm(test_data)  # LSTM output of test data
        test_loss = loss_func(test_output, test_label.long())
        #将测试数据集的真实值进行独热编码
        test_y_score = one_hot(test_label.data.cpu().numpy(),num_classes=4)  # .cpu() can be removed if your device is cpu.
        #将测试数据集训练出的结果进行softmax回归计算
        pred_score = F.softmax(test_output, dim=1).data.cpu().numpy()  
        #使用 ROC-AUC 来评估模型在测试集上的性能，其中 test_y_score 是真实标签的one-hot编码，而 pred_score 是模型的预测概率
        auc_score = average_precision_score(test_y_score, pred_score, average='macro')
        #auc_score = roc_auc_score(test_y_score, pred_score,multi_class='ovr')

        pred_y = torch.argmax(test_output, 1).data.cpu().numpy()
        pred_train = torch.argmax(output, 1).data.cpu().numpy()

        test_acc = accuracy_score(test_label.data.cpu().numpy(), pred_y)
        train_acc = accuracy_score(train_y.data.cpu().numpy(), pred_train)


        print('Epoch: ', epoch, '|train loss: %.4f' % loss.item(),
              ' train ACC: %.4f' % train_acc, '| test loss: %.4f' % test_loss.item(),
              'test ACC: %.4f' % test_acc, '| AUC-PR: %.4f' % auc_score)
        best_acc.append(test_acc)
        best_auc.append(auc_score)
        best_train_acc.append(train_acc)

        plt.plot(epoch, best_acc, label='Test Acc')
        plt.plot(epoch, best_train_acc, label='Train Acc')
        plt.plot(epoch, best_train_acc, label='AUC')
        plt.title('Training and Test Loss over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Acc')
        plt.legend()


current_time = time.perf_counter()
running_time = current_time - start_time
print(classification_report(test_label.data.cpu().numpy(), pred_y))
print('BEST TEST ACC: {}, AUC: {}'.format(max(best_acc), max(best_auc)))
print("Total Running Time: {} seconds".format(round(running_time, 2)))

