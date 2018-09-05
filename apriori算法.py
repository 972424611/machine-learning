# 计算学员购买课程的相关性
from apriori import *
import pandas as pda
file_name = "E:\pythonResult\\apriori\lesson_buy.xls"
data_file = pda.read_excel(file_name, header=None)

# 转换数据
change = lambda x: pda.Series(1, index=x[pda.notnull(x)])
map_data = map(change, data_file.as_matrix())
data = pda.DataFrame(list(map_data)).fillna(0)

# 临界支持度, 置信度设置
spt = 0.1
cfd = 0.3

# 使用apriori计算结果
find_rule(data, spt, cfd, "&&")