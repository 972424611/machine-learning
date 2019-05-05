import pymysql
import pandas as pda
import numpy as npy
conn = pymysql.connect(host="127.0.0.1", user="root", passwd="199710", db="hexun")
sql = "SELECT * FROM myhexun"
data1 = pda.read_sql(sql, conn)
# 评点比 (属性构造)
ch = data1[u"comment"] / data1["hits"]
data1[u"评点比"] = ch
