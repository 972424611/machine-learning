import jieba
import jieba.posseg
import jieba.analyse
# 编码问题决解方案 可以通过保存为html的格式在decode
data = open("E:\pythonResult\龙王传说.txt", encoding='utf-8').read()
tag = jieba.analyse.extract_tags(data, 20)
print(tag)
