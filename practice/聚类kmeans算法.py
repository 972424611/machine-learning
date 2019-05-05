# 聚类三法 kmeans主要是划分法
# 划分法(分裂), 层次分析法, 密度分析法, 其实还有: 网络法，模型法
# 通过程序实现录取学生的聚类
import pandas as pda
import numpy as npy
import matplotlib.pylab as pyl
from sklearn.cluster import Birch, KMeans

file_name = "E:\pythonResult\kmeans\luqu.csv"
data_file = pda.read_csv(file_name)
x = data_file.iloc[:, 1: 4].as_matrix()
kms = KMeans(n_clusters=4)
y = kms.fit_predict(x)
print(y)
s = npy.arange(0, len(y))
pyl.plot(s, y, "o")
pyl.show()
# 通过程序实现商品的聚类
