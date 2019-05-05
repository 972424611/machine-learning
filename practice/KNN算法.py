from numpy import *
import operator
from PIL import Image
from os import listdir


def knn(k, testData, trainData, labels):
    # 获取行数
    trainDataSize = trainData.shape[0]
    # 从列的方向扩展
    dif = tile(testData, (trainDataSize, 1)) - trainData
    sqdif = dif**2
    # 1 每一行求和, 0 是每一列求和
    sumSqdif = sqdif.sum(axis=1)
    distance = sumSqdif**0.5
    sortDistance = distance.argsort()
    count = {}
    for i in range(0, k):
        vote = labels[sortDistance[i]]
        count[vote] = count.get(vote, 0) + 1
    sortCount = sorted(count.items(), key=operator.itemgetter(1), reverse=True)
    return sortCount[0][0]


# 图片处理, 先将所有文件转为固定宽高, 比如32x32, 再转为文本
im = Image.open("E:\pythonResult\knn\\baidu.jpg")
fh = open("E:\pythonResult\knn\\baidu.txt", "a")
width = im.size[0]
height = im.size[1]
for i in range(0, width):
    for j in range(0, height):
        color = im.getpixel((i, j))
        colorAll = color[0] + color[1] + color[2]
        if(colorAll == 0):
            #黑色
            fh.write("1")
        else:
            fh.write("0")
    fh.write("\n")
fh.close()


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


# 用测试数据调用knn算法去测试, 看是否能够准确识别
def dataTest():
    trainArr, labels = trainData()
    testList = listdir("E:\pythonResult\knn\\testdata")
    testNum = len(testList)
    for i in range(0, testNum):
        thisTestFile = testList[i]
        testArr = dataToArray("E:\pythonResult\knn\\testdata\\" + thisTestFile)
        result = knn(3, testArr, trainArr, labels)
        print(result)


#dataTest()
#抽取某个文件出来进行测试
trainArr, labels = trainData()
thisTestFile = "0_56.txt"
testArr = dataToArray("E:\pythonResult\knn\\testdata\\" + thisTestFile)
result = knn(3, testArr, trainArr, labels)
print(result)
