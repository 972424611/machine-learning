import pandas as pda
from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import RandomizedLogisticRegression as RLR

file_name = "E:\pythonResult\huigui\luqu.csv"
data_file = pda.read_csv(file_name)
x = data_file.iloc[:, 1:4].as_matrix()
y = data_file.iloc[:, 0:1].as_matrix()

r1 = RLR()
r1.fit(x, y)
# 特征筛选
print(r1.get_support())
# print(data_file.columns[r1.get_support()[0]])
t = data_file[data_file.columns[r1.get_support()]].as_matrix()

r2 = LR()
r2.fit(t, y)
print("训练结束")
print("模型正确率为: " + str(r2.score(x, y)))
