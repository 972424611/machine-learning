import pymysql
import pandas as pda
import numpy as npy
from sklearn.decomposition import PCA

# 主成分分析法(PCA)
conn = pymysql.connect(host="127.0.0.1", user="root", passwd="199710", db="hexun")
sql = "SELECT hits, comment FROM myhexun"
data1 = pda.read_sql(sql, conn)
ch = data1[u"comment"] / data1["hits"]
data1[u"评点比"] = ch
pca1 = PCA()
pca1.fit(data1)
# 返回模型中各个特征量
tz1 = pca1.components_
print(tz1)
# 各个成分中各自方差百分比, 贡献率
bfb = pca1.explained_variance_ratio_
print(bfb)

# 降到2维
pca2 = PCA(2)
pca2.fit(data1)
jf = pca2.transform(data1)
print(jf)
# 恢复原来维数
pca2 = pca2.inverse_transform(jf)
print(pca2)