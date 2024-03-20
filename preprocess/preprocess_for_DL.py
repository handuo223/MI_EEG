import matplotlib.pyplot as plt
import mne
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy import signal
import torch
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

#要根据存放位置自己修改
filename = "BCIC/dataset/A01T.gdf"

#1、读取gdf原始数据集
raw_gdf = mne.io.read_raw_gdf(filename, stim_channel="auto",exclude=(["EOG-left", "EOG-central", "EOG-right"]),verbose='ERROR')

# 2、将原通道重命名为10-20系统中
raw_gdf.rename_channels(
            {'EEG-Fz': 'Fz', 'EEG-0': 'FC3', 'EEG-1': 'FC1', 'EEG-2': 'FCz', 'EEG-3': 'FC2', 'EEG-4': 'FC4',
             'EEG-5': 'C5', 'EEG-C3': 'C3', 'EEG-6': 'C1', 'EEG-Cz': 'Cz', 'EEG-7': 'C2', 'EEG-C4': 'C4', 'EEG-8': 'C6',
             'EEG-9': 'CP3', 'EEG-10': 'CP1', 'EEG-11': 'CPz', 'EEG-12': 'CP2', 'EEG-13': 'CP4',
             'EEG-14': 'P1', 'EEG-15': 'Pz', 'EEG-16': 'P2', 'EEG-Pz': 'POz'})

#3、处理异常值和空值
raw_gdf.load_data()
data = raw_gdf.get_data()

for i_chan in range(data.shape[0]):  # 遍历 22 channel
    # 将数组中的所有值设置为nan，然后将这些NaN值替换为该数组的均值。
    this_chan = data[i_chan]
    data[i_chan] = np.where(
        this_chan == np.min(this_chan), np.nan, this_chan
    )
    mask = np.isnan(data[i_chan])
    chan_mean = np.nanmean(data[i_chan])
    data[i_chan, mask] = chan_mean


# 4、对原始数据进行FIR带通滤波
raw_gdf.filter(7., 30., fir_design='firwin', skip_by_annotation='edge')

# 4、获取事件时间位置，返回事件和事件下标
events, events_id = mne.events_from_annotations(raw_gdf)
print('Number of events:', len(events))
print(events_id)
print(events)


# 5、分段：使用mne.Epochs获取需求的MI数据
# 选择范围为Cue后 1s - 4s 的数据
tmin, tmax = 1., 4.
# 四类 MI 对应的 events_id
#event_id = dict({'769': 7, '770': 8, '771': 9, '772': 10})
#event_id = dict({'770': 8, '771': 9, '772': 10})
event_id = dict({ '771': 9, '772': 10})
epochs = mne.Epochs(raw_gdf, events, event_id, tmin, tmax, proj=True, baseline=None, preload=True)


#6、切片：获取 events 的最后一列，作为label值
labels = epochs.events[:, -1].astype(float)
print(labels.shape)
# Get all epochs as a 3D array.
data = epochs.get_data()
print(data.shape)


#7、data的处理：
    #1)基线漂移
    #2）标准化
    #3）从3维变为4维

#8、label的处理
    #1）独热编码
    #2）转为tensor
    #3）降维处理



#7、滤波：带通滤波
# def filter(data):
#     wn1 = 2 * 8 / 250
#     wn2 = 2 * 30 / 250
#     b, a = signal.butter(2, [wn1, wn2], 'bandpass')
#     data = signal.filtfilt(b, a, data)

# # data=filter(data)
# print(data.shape)


#9、label数据处理：独热编码
def one_hot(y_):
    y_=y_-7
    y_ = y_.reshape(len(y_))
    y_ = [int(xx) for xx in y_]
    n_values = 4
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]

labels=one_hot(labels)
print(labels)

#把标签转换为tensor数据
transf = transforms.ToTensor()
d=transf(labels)  
label = torch.argmax(d, dim=2).long()
#torch.Size([1, 144])
print(label.shape)
 
#把标签降维[1,144]->[144,1]
label= label.reshape(144,1)
#torch.Size([144, 1])
print(label.shape)
h = torch.squeeze(label)#表示把x中维度大小为1的所有维都已删除
#torch.Size([144])
print(h.shape)



#7、data处理：去除基线漂移
def baselineFix(data_matrix, time_window=20):
        """
        进行基线校正的函数，根据时间窗口对数据进行滑动平均。

        参数：
        data_matrix (numpy.ndarray): 输入的二维数据矩阵。
        time_window (int): 滑动平均的时间窗口大小。

        返回：
        numpy.ndarray: 基线校正后的数据矩阵。
        """
        begin_point = 0
        end_point = time_window
        while end_point < data_matrix.shape[1]:
            baseline = np.mean(data_matrix[:, begin_point:end_point], axis=1, keepdims=True)
            data_matrix[:, begin_point:end_point] -= baseline
            begin_point += time_window
            end_point += time_window
        if end_point >= data_matrix.shape[1]:
            end_point = data_matrix.shape[1]
        return data_matrix
    
data = baselineFix(data)
print(data.shape)
#data数据处理：标准化
def Standard_process(data):
    # Assuming data is a 3D array with shape (samples, features, depth)
    # Reshape the data to (samples * features, depth) to perform standardization on each feature
    samples, features, depth = data.shape
    data_reshaped = data.reshape((samples * features, depth))
    
    # Standardize the data
    scaler = StandardScaler().fit(data_reshaped)
    x_train_reshaped = scaler.transform(data_reshaped)
    
    # Reshape the standardized data back to the original shape
    x_train = x_train_reshaped.reshape((samples, features, depth))
    
    return x_train

data=Standard_process(data)
print(data.shape)

#使用切片操作，舍弃最后一维，变为[144,22,750]
data = data[:, :, :-1]

#继续升维：从3维->4维
data = data.reshape(144,22,15,50)
print(data.shape)

# 记录原始 data 的形状是 [144, 22, 20, 30]
original_shape = data.shape

# 将多维数组展平为二维数组
data_flattened = data.reshape(data.shape[0], -1)

# 使用train_test_split划分数据
train_data, test_data, train_label, test_label = train_test_split(data_flattened, labels, test_size=0.4, random_state=5)


# 将 train_data 还原为四维数组
train_data_original_shape = train_data.reshape(-1, *original_shape[1:])


# 将 test_data 还原为四维数组
test_data_original_shape = test_data.reshape(-1, *original_shape[1:])



torch.save(train_data_original_shape,'/home/handuo/test03/BCIC/preprocessed/A01_train_data.pt')
torch.save(test_data_original_shape,'/home/handuo/test03/BCIC/preprocessed/A01_test_data.pt')
 
torch.save(h,'/home/handuo/test03/BCIC/preprocessed/A01_train_label.pt')
torch.save(h,'/home/handuo/test03/BCIC/preprocessed/A01_test_label.pt')

