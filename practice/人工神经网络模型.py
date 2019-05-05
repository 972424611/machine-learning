# 具体的人工神经网络模型 RBF FNN LM神经网络(精度非常高) BP神经网络

import pandas as pda
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from numpy import *
from os import listdir


# 下面是BP神经网络的实现课程销量预测
def test1():
    file_name = "E:\pythonResult\\tree\lesson.csv"
    data_file = pda.read_csv(file_name, encoding="gbk")
    x = data_file.iloc[:, 1:5].as_matrix()
    y = data_file.iloc[:, 5].as_matrix()

    for i in range(0, len(x)):
        for j in range(0, len(x[i])):
            this_data = x[i][j]
            if this_data == '是' or this_data == "多" or this_data == "高":
                x[i][j] = 1
            else:
                x[i][j] = 0

    for i in range(0, len(y)):
        this_data = y[i]
        if this_data == "高":
            y[i] = 1
        else:
            y[i] = 0

    # 将x y转换为数据框(二维), 然后再转化为数组并指定格式
    xf = pda.DataFrame(x)
    yf = pda.DataFrame(y)

    xf = xf.as_matrix().astype(int)
    yf = yf.as_matrix().astype(int)

    model = Sequential()
    # 输入层 input_dim特征数 Activation()激活函数
    model.add(Dense(10, input_dim=len(xf[0])))
    model.add(Activation("relu"))
    # 输出层
    model.add(Dense(1, input_dim=1))
    model.add(Activation("sigmoid"))
    # 模型编译
    model.compile(loss="binary_crossentropy", optimizer="adam")
    # 训练
    model.fit(xf, yf, nb_epoch=1000, batch_size=100)
    # 预测分类
    rst = model.predict_classes(x).reshape(len(x))
    k = 0
    for i in range(0, len(xf)):
        if rst[i] != y[i]:
            k += 1
    print(1-float(k)/float(len(xf)))


# 下面是BP神经网络的实现手写体数字识别
# 加载数据
def dataToArray(fName):
    arr = []
    fh = open(fName)
    for i in range(0, 32):
        thisLine = fh.readline()
        for j in range(0, 32):
            arr.append(int(thisLine[j]))
    return arr


def seplabel(fileName):
    fileStr = fileName.split(".")[0]
    label = fileStr.split("_")[0]
    return label


# 建立训练数据
def trainData():
    labels = []
    trainFile = listdir("E:\pythonResult\knn\\traindata")
    num = len(trainFile)
    # 用一个数组存储所有训练数据, 行num: 文件总数, 列1024
    trainArr = zeros((num, 1024))
    for i in range(0, num):
        thisFileName = trainFile[i]
        thisLabel = seplabel(thisFileName)
        labels.append(thisLabel)
        trainArr[i, :] = dataToArray("E:\pythonResult\knn\\traindata\\" + thisFileName)
    return trainArr, labels


trainarr, labels = trainData()
xf = pda.DataFrame(trainarr)
yf = pda.DataFrame(labels)
xf = xf.as_matrix().astype(int)
yf = yf.as_matrix().astype(int)
model = Sequential()
# 输入层 input_dim特征数 Activation()激活函数
model.add(Dense(10, input_dim=len(xf[0])))
model.add(Activation("relu"))
# 输出层
model.add(Dense(1, input_dim=1))
model.add(Activation("sigmoid"))
# 模型编译
model.compile(loss="mean_squared_error", optimizer="adam")
# 训练
model.fit(xf, yf, nb_epoch=100, batch_size=6)
# 预测分类
rst = model.predict_classes(trainarr).reshape(len(trainarr))
k = 0
for i in range(0, len(xf)):
    if rst[i] != labels[i]:
        k += 1
print(1-float(k)/float(len(xf)))
