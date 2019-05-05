import numpy
from numpy import *
import operator
from PIL import Image
from os import listdir

class Bayes:
    def __init__(self):
        self.length = -1
        self.labelCount = dict()
        self.vectorCount = dict()

    def fit(self, data_set: list, labels: list):
        if len(data_set) != len(labels):
            raise ValueError("您输入的测试数组跟类别数组长度不一致")
        # 测试数据特征值的长度
        self.length = len(data_set[0])
        # 类别所有的数量
        labels_num = len(labels)
        # 不重复类别的数量
        no_repeat_labels = set(labels)
        for item in no_repeat_labels:
            this_label = item
            # 当前类别占类别总数的比例
            self.labelCount[this_label] = labels.count(this_label) / labels_num
        for vector, label in zip(data_set, labels):
            if label not in self.vectorCount:
                self.vectorCount[label] = []
            self.vectorCount[label].append(vector)
        print("训练结束")
        return self

    def test(self, test_data, label_set):
        if self.length == -1:
            raise ValueError("您还没进行训练, 请先训练")
        # 计算test_data分别为各个类别的概率
        lb_dict = dict()
        for this_lb in label_set:
            p = 1
            all_label = self.labelCount[str(this_lb)]
            all_vector = self.vectorCount[str(this_lb)]
            vector_num = len(all_vector)
            all_vector = numpy.array(all_vector).T
            for index in range(0, len(test_data)):
                vector = list(all_vector[index])
                p = p * vector.count(test_data[index]) / vector_num
            lb_dict[this_lb] = p * all_label
        this_label = sorted(lb_dict, key=lambda x: lb_dict[x], reverse=True)[0]
        return this_label

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
        trainArr[i,:] = dataToArray("E:\pythonResult\knn\\traindata\\" + thisFileName)
    return trainArr, labels

bys = Bayes()
train_data, labels = trainData()
# 训练
bys.fit(train_data, labels)
# 测试
this_data = dataToArray("E:\pythonResult\knn\\testdata\8_90.txt")
labels_all = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# 识别单个手写体
rst = bys.test(this_data, labels_all)
print(rst)
# 识别多个手写体
test_file_all = listdir("E:\pythonResult\knn\\testdata")
num = len(test_file_all)
for i in range(0, num):
    this_file_name = test_file_all[i]
    this_label = seplabel(this_file_name)
    this_data_array = dataToArray("E:\pythonResult\knn\\testdata\\" + this_file_name)
    label = bys.test(this_data_array, labels_all)
    print("该数字是: " + str(this_label) + ", 识别出来的数字: " + str(label))
