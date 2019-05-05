import pymysql
import pandas as pda
import numpy as npy

conn = pymysql.connect(host="127.0.0.1", user="root", password="199710", db="csdn")
sql = "SELECT price, comment FROM taob"
data = pda.read_sql(sql, conn)

# 离差标准化
data2 = (data - data.min()) / (data.max() - data.min())
print(data2)

# 标准差标准化 std()得到标准差
data3 = (data - data.mean()) / data.std()
print(data3)

# 小数定标标准化 ceil() 4.1 -> 5.0
k = npy.ceil(npy.log10(data.abs().max()))
data4 = data / 10**k
print(data4)

# ---------------------------------------------------------------------
# 连续型数据离散化
# 等宽离散化
data5 = data[u"price"].copy()
data6 = data5.T
data7 = data6.value
k = 3
c1 = pda.cut(data7, k, labels=["便宜", "适中", "贵"])
print(c1)
# 非等宽离散化
k = [0, 50, 100, 300, 500, 2000, data7.max()]
c2 = pda.cut(data7, k, labels=["非常便宜", "便宜", "适中", "贵", "非常贵", "天价"])
print(c2)
