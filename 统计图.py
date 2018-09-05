# 散点图 和 直线图
import matplotlib.pylab as pyl
import numpy

x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 7]

x2 = [1, 9, 5, 12, 15, 17]
y2 = [2, 5, 6, 7, 8, 13]
# matplotlib.pylab.plot(x, y) #plot(x轴数据, y轴数据, 展现形式)
# matplotlib.pylab.show()
# pyl.plot(x, y, 'o')  # 'o' 代表散点图
# pyl.show()

'''
颜色的样式
c-cyan--青色
r-red--红色
m-magente--品红
g-green--绿色
b-blue--蓝色
y-yellow--黄色
k-black--黑色
w-white--白色
'''

'''
线条样式
- 直线
-- 虚线
-. -.形式
: 细小虚线
'''

'''
点的样式
s--方形
h--六角形
H--六角形
*--星形
+--加号
x--x形
d--菱形
D--菱形
p--五角形
'''
pyl.plot(x, y)
pyl.plot(x2, y2)
pyl.title("show")
pyl.xlabel("ages")  # x轴名称
pyl.ylabel("temp")  # y轴名称
pyl.show()
# 直方图
# 随机数生成
data = numpy.random.random_integers(1, 20, 10)  # (最小值, 最大值, 个数)
print(data)
data2 = numpy.random.normal(5.0, 2.0, 10)  # (均数, 西格玛, 个数)
print(data2)

#直方图hist
data3 = numpy.random.normal(10.0, 1.0, 10000)
#pyl.hist(data3)
#pyl.show()
data4 = numpy.random.random_integers(1, 25, 1000)
#pyl.hist(data4)
sty = numpy.arange(0, 30, 4)
#pyl.hist(data4, sty, histtype='stepfilled')

#pyl.subplot(2, 2, 3)  # 行, 列, 当前区域
#pyl.show()
pyl.subplot(2, 2, 1)
x1 = [1, 2, 3, 4, 5]
y1 = [5, 3, 5, 23, 5]
pyl.plot(x1, y1)

pyl.subplot(2, 2, 2)
x2 = [5, 2, 3, 8, 6]
y2 = [1, 6, 5, 8, 6]
pyl.plot(x2, y2)

pyl.subplot(2, 1, 2)
x3 = [4, 5, 6, 8, 6, 10]
y3 = [2, 4, 7, 8, 6, 7]
pyl.plot(x3, y3)

pyl.show()