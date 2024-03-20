
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

