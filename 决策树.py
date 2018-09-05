import pandas as pda
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import numpy as npy
import pydotplus

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
            x[i][j] = -1

for i in range(0, len(y)):
    this_data = y[i]
    if this_data == "高":
        y[i] = 1
    else:
        y[i] = -1

# 将x y转换为数据框(二维), 然后再转化为数组并指定格式
xf = pda.DataFrame(x)
yf = pda.DataFrame(y)

xf = xf.as_matrix().astype(int)
yf = yf.as_matrix().astype(int)

# 建立决策树
dtc = DTC(criterion="entropy")
dtc.fit(xf, yf)

# 可视化决策树
dot_data = export_graphviz(dtc, feature_names=["combat", "num", "promotion", "datum"], out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("E:\pythonResult\\tree\\tree.pdf")

# 直接预测销量高低
x3 = npy.array([[1, -1, -1, 1], [1, 1, 1, 1], [-1, 1, -1, 1]])
rst = dtc.predict(x3)
print(rst)
