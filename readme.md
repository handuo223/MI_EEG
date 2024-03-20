——————————————————————————————————————————
基本信息：
BCIC-2a数据集
二分类问题

预处理preprocess.py
——————————————————————————————————————————————
2023/11/14
mne库读取报错
raw_gdf = mne.io.read_raw_gdf(filename, stim_channel="auto",exclude=(["EOG-left", "EOG-central", "EOG-right"]),verbose='ERROR')
报错：numpy.core._exceptions._UFuncOutputCastingError: Cannot cast ufunc 'clip' output from dtype('float64') to dtype('uint32') with casting rule 'same_kind'
解决方案：https://blog.csdn.net/qq_36758270/article/details/131418753

已解决，可以用了！


———————————————————————————————————————————————
2023/11/17
调参日志tuning log

1、预处理中基线漂移的时间窗大小
33
n_components: 2  score: 0.4772727272727273
n_components: 4  score: 0.5454545454545454
n_components: 6  score: 0.6590909090909091
n_components: 8  score: 0.5909090909090909
n_components: 10  score: 0.7045454545454546
n_components: 12  score: 0.7045454545454546
n_components: 14  score: 0.6363636363636364
n_components: 16  score: 0.7272727272727273
n_components: 18  score: 0.7272727272727273

33->20
n_components: 2  score: 0.5227272727272727
n_components: 4  score: 0.5227272727272727
n_components: 6  score: 0.5909090909090909
n_components: 8  score: 0.5454545454545454
n_components: 10  score: 0.6363636363636364
n_components: 12  score: 0.6363636363636364
n_components: 14  score: 0.7272727272727273
n_components: 16  score: 0.7954545454545454
n_components: 18  score: 0.7954545454545454

有效果！

2、在预处理时增加标准化处理是否有效果，影响不大？
时间窗固定在33，加了标准化


不加标准化



——————————————————————————————————————————————————————
preprocess_for_DL.py
用深度学习模型之前，需要对数据集做更多的处理。
1、data：标准化、升到四维
2、label：独热编码、降到一维


CNNNet可以使用
输入data[288,22,15,50]

preprocess_for_DL.py第51行，修改时间从1-4s
第165行，data = data.reshape(288,22,15,50)

但CNNNet过拟合严重



EEGNet可以使用
输入data[288,22,20,50]

需要修改preprocess_for_DL.py第51行，修改时间从0-4s
第165行，data = data.reshape(288,22,20,50)




————————————————————————————————————————————————————
todo List
1、FBCSP
2、可视化
3、EEG-Net


二分类：
preprocess_for_DL:288要改成144
EEGNet_train:38
classnum=2

准确率在0.7左右
