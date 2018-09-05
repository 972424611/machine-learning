'''
    1. 读取文档
    2. 对要计算的多篇文档进行分词
    3. 对文档进行整理成指定格式, 方便后续计算
    4. 计算词语的频率
    5. (可选) 对频率低的词语进行过滤
    6. 通过词料库建立词典
    7. 加载要对比的文档
    8. 将要对比的文档通过doc2bow转化为稀疏向量
    9. 对稀疏向量进行进一步处理, 得到新词料库
    10. 将新词料库通过tf-idf model进行处理, 得到tf-ide
    11. 通过token2id得到特征数
    12. 稀疏矩阵相似度, 从而建立索引
    13. 得到最终相似度结果
'''
from gensim import corpora, models, similarities
import jieba
from collections import defaultdict
# doc1 = "E:\pythonResult\老九门.txt"
doc1 = "E:\pythonResult\武动乾坤.txt"
# doc2 = "E:\pythonResult\鬼吹灯.txt"
doc2 = "E:\pythonResult\大主宰.txt"
d1 = open(doc1, encoding='utf-8').read()
d2 = open(doc2, encoding='utf-8').read()
data1 = jieba.cut(d1)
data2 = jieba.cut(d2)
data11 = ""
for item in data1:
    data11 += item + " "

data21 = ""
for item in data2:
    data21 += item + " "
documents = [data11, data21]
texts = [[word for word in document.split()] for document in documents]
frequency = defaultdict(int)
for text in texts:
    for token in text:
        # 计算词语的频率
        frequency[token] += 1

# (可选) 对频率低的词语进行过滤
texts = [[word for word in text if frequency[word] > 20] for text in texts]

# 通过词料库建立词典
dictionary = corpora.Dictionary(texts)
dictionary.save("E:\pythonResult\ciku.dict")

# 加载要对比的文档
# doc3 = "E:\pythonResult\鬼吹灯.txt"
doc3 = "E:\pythonResult\斗破苍穹.txt"
d3 = open(doc3, encoding='utf-8').read()
data3 = jieba.cut(d3)
data31 = ""
for item in data3:
    data31 += item + " "
new_doc = data31.split()

# 将要对比的文档通过doc2bow转化为稀疏向量
new_vec = dictionary.doc2bow(new_doc)

# 对稀疏向量进行进一步处理, 得到新词料库
corpus = [dictionary.doc2bow(text) for text in texts]

# 将新词料库通过tf-idf model进行处理, 得到tf-ide
tf_idf = models.TfidfModel(corpus)

# 通过token2id得到特征数
featureNum = len(dictionary.token2id.keys())

# 稀疏矩阵相似度, 从而建立索引
index = similarities.SparseMatrixSimilarity(tf_idf[corpus], num_features=featureNum)

# 得到最终相似度结果
sim = index[tf_idf[new_vec]]
print(sim)
