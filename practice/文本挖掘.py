import jieba
import jieba.posseg
import jieba.analyse
sentence = "长沙理工大学"
w1 = jieba.cut(sentence, cut_all=True)  # 全局模式
for item in w1:
    print(item)
print("-------------------------------------------------")
w2 = jieba.cut(sentence, cut_all=False)  # 精准模式(默认模式)
for item in w2:
    print(item)
print("-------------------------------------------------")
w3 = jieba.cut_for_search(sentence)  # 搜索引擎模式
for item in w3:
    print(item)
print("-------------------------------------------------")
# 词性标注 .flag词性  .word词语
sentence = "百度"
w5 = jieba.posseg.cut(sentence)
for item in w5:
    print(item.word + "---" + item.flag)
print("-------------------------------------------------")
'''
    a: 形容词, c: 连词, d: 副词, e: 叹词, f: 方位词, i: 成语, m: 数
    n: 名词, v: 动词, p: 介词, r: 代词, t: 时间, u: 助词, w: 标点符号
    nr: 人名(专有名词), ns: 地名, nt: 机构团体, nz: 其他专有名词, vn: 动名词, un: 未知词语
'''
# 自定义词典加载 这里注意加载文件的编码
# jieba.load_userdict("这里填自定义词典的目录")
sentence = "长沙理工大学"
w6 = jieba.posseg.cut(sentence)
for item in w6:
    print(item)
print("-------------------------------------------------")
# 更改词频
sentence = "我喜欢上海东方明珠"
w7 = jieba.cut(sentence)
for item in w7:
    print(item)
print("-------------------------------------------------")
jieba.suggest_freq("上海东方", True)
w8 = jieba.cut(sentence)
for item in w8:
    print(item)
print("-------------------------------------------------")
# 提取3个关键词
tag = jieba.analyse.extract_tags(sentence, 3)
print(tag)
print("-------------------------------------------------")
# 返回词语的位置
w9 = jieba.tokenize(sentence)
for item in w9:
    print(item)
print("-------------------------------------------------")
w10 = jieba.tokenize(sentence, mode="search")  # 搜索引擎模式
for item in w10:
    print(item)
print("-------------------------------------------------")