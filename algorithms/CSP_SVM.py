
import warnings
import joblib
import mne
import numpy as np
from mne.decoding import CSP
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
warnings.filterwarnings('ignore')
np.set_printoptions(suppress=True)
mne.set_log_level('error')


#标准二分类CSP
def standardCSP():
    data = np.load('/home3/handuo/test01/BCIC/preprocessed/A01T_preprocessed.npy', allow_pickle=True)
    labels = np.load('/home3/handuo/test01/BCIC/preprocessed/A01T_preprocessed_label.npy', allow_pickle=True)

    train_data, test_data, train_label, test_label = train_test_split(data, labels, test_size=0.3,random_state=5)
    train_data = train_data.astype(np.float64).copy()
    train_label = train_label.astype(np.float64).copy()
    test_data = test_data.astype(np.float64).copy()
    test_label = test_label.astype(np.float64).copy()


    for i in range(1, 10):
        csp = CSP(n_components=i * 2, reg=None, log=False, norm_trace=False)

        svm = SVC(kernel='linear')
        clf = Pipeline([('CSP', csp), ('SVM', svm)])
        clf.fit(train_data, train_label)
        acc = clf.score(test_data, test_label)

        print('n_components:', i * 2, ' score:', acc)


#四分类
def save_2model():

    dataset_all = np.load('/home/handuo/test03/BCIC/preprocessed/A01T_preprocessed.npy', allow_pickle=True)
    data = dataset_all.item().get('dataset')
    labels = dataset_all.item().get('labels')
    labels=labels-7

    train_data, test_data, train_label, test_label = train_test_split(data, labels, test_size=0.3,random_state=5)
    train_data = train_data.astype(np.float64).copy()
    train_label = train_label.astype(np.float64).copy()
    test_data = test_data.astype(np.float64).copy()
    test_label = test_label.astype(np.float64).copy()

    #01模型训练
    train_data0_1 = np.concatenate(
        (train_data[np.where(train_label == 0)], train_data[np.where(train_label == 1)]),axis=0)
    train_label0_1 = np.concatenate(
        (train_label[np.where(train_label == 0)], train_label[np.where(train_label == 1)]), axis=0)

    csp = CSP(n_components=6, reg=None, log=False, norm_trace=False)
    svm = SVC(kernel='rbf')
    clf = Pipeline([('CSP', csp), ('SVM', svm)])
    clf.fit(train_data0_1, train_label0_1)
    prdict_label01=clf.predict(test_data)
    joblib.dump(clf, 'model0_1 .model')

    #02模型训练
    train_data0_2 = np.concatenate(
        (train_data[np.where(train_label == 0)], train_data[np.where(train_label == 2)]),axis=0)
    train_label0_2 = np.concatenate(
        (train_label[np.where(train_label == 0)], train_label[np.where(train_label == 2)]), axis=0)
    csp = CSP(n_components=3 * 2, reg=None, log=False, norm_trace=False)
    svm = SVC(kernel='rbf')
    clf = Pipeline([('CSP', csp), ('SVM', svm)])
    clf.fit(train_data0_2, train_label0_2)
    prdict_label02=clf.predict(test_data)
    joblib.dump(clf, 'model0_2 .model')

    #03模型训练
    train_data0_3 = np.concatenate(
        (train_data[np.where(train_label == 0)], train_data[np.where(train_label == 3)]),axis=0)
    train_label0_3 = np.concatenate(
        (train_label[np.where(train_label == 0)], train_label[np.where(train_label == 3)]), axis=0)

    csp = CSP(n_components=4, reg=None, log=False, norm_trace=False)
    svm = SVC(kernel='rbf')
    clf = Pipeline([('CSP', csp), ('SVM', svm)])
    clf.fit(train_data0_3, train_label0_3)
    # acc = clf.score(test_data0_3,test_label0_3)
    prdict_label03=clf.predict(test_data)
    joblib.dump(clf, 'model0_3 .model')

    #12模型
    train_data1_2 = np.concatenate(
        (train_data[np.where(train_label == 1)], train_data[np.where(train_label == 2)]),axis=0)
    train_label1_2 = np.concatenate(
        (train_label[np.where(train_label == 1)], train_label[np.where(train_label == 2)]), axis=0)
    
    csp = CSP(n_components=4, reg=None, log=False, norm_trace=False)
    svm = SVC(kernel='rbf')
    clf = Pipeline([('CSP', csp), ('SVM', svm)])
    clf.fit(train_data1_2, train_label1_2)
    prdict_label12=clf.predict(test_data)
    joblib.dump(clf, 'model1_2 .model')

    #13模型
    train_data1_3 = np.concatenate(
        (train_data[np.where(train_label == 1)], train_data[np.where(train_label == 3)]),
        axis=0)
    train_label1_3 = np.concatenate(
        (train_label[np.where(train_label == 1)], train_label[np.where(train_label == 3)]), axis=0)
   
    csp = CSP(n_components=4, reg=None, log=False, norm_trace=False)
    svm = SVC(kernel='rbf')
    clf = Pipeline([('CSP', csp), ('SVM', svm)])
    clf.fit(train_data1_3, train_label1_3)
    prdict_label13=clf.predict(test_data)
    joblib.dump(clf, 'model1_3 .model')

    #12模型
    train_data1_2 = np.concatenate(
        (train_data[np.where(train_label == 1)], train_data[np.where(train_label == 2)]),axis=0)
    train_label1_2 = np.concatenate(
        (train_label[np.where(train_label == 1)], train_label[np.where(train_label == 2)]), axis=0)
    csp = CSP(n_components=4, reg=None, log=False, norm_trace=False)
    svm = SVC(kernel='rbf')
    clf = Pipeline([('CSP', csp), ('SVM', svm)])
    clf.fit(train_data1_2, train_label1_2)
    prdict_label12=clf.predict(test_data)
    joblib.dump(clf, 'model1_2 .model')

    #23模型
    train_data2_3 = np.concatenate(
        (train_data[np.where(train_label == 2)], train_data[np.where(train_label == 3)]),
        axis=0)
    train_label2_3 = np.concatenate(
        (train_label[np.where(train_label == 2)], train_label[np.where(train_label == 3)]), axis=0)
    csp = CSP(n_components=4, reg=None, log=False, norm_trace=False)
    svm = SVC(kernel='rbf')
    clf = Pipeline([('CSP', csp), ('SVM', svm)])
    clf.fit(train_data2_3, train_label2_3)
    prdict_label23=clf.predict(test_data)
    joblib.dump(clf, 'model2_3 .model')

    #按列展开
    two_dimensional_vector = np.array([prdict_label01, prdict_label02, prdict_label03, prdict_label12,prdict_label13,prdict_label23])
    one_dimensional_vectors = two_dimensional_vector[:, np.newaxis]
    one_dimensional_vectors = np.hsplit(two_dimensional_vector, 87)

    # 初始化一个列表来存储最终的预测结果
    final_predictions = []

    # 遍历每个一维向量
    for vector in one_dimensional_vectors:
        # 计算每个一维向量中出现次数最多的值
        vector_int = vector.flatten().astype(int)
        mode_result = np.argmax(np.bincount(vector_int))
        final_predictions.append(mode_result)

    print(test_label)
    # 输出最终的预测结果
    print("Final Predictions:", final_predictions)
    accuracy = np.mean(final_predictions == test_label)
    accuracy_percentage = accuracy * 100

    # 输出准确率
    print(f"Accuracy: {accuracy_percentage:.2f}%")
if __name__ == '__main__':
    #standardCSP()
    save_2model()


