import numpy


class Bayes:
    def __init__(self):
        self.length = -1
        self.labelCount = dict()
        self.vectorCount = dict()

    def fit(self, data_set: list, labels: list):
        if len(data_set) != len(labels):
            raise ValueError("您输入的测试数组跟类别数组长度不一致")
        # 测试数据特征值的长度
        self.length = len(data_set[0])
        # 类别所有的数量
        labels_num = len(labels)
        # 不重复类别的数量
        no_repeat_labels = set(labels)
        for item in no_repeat_labels:
            this_label = item
            # 当前类别占类别总数的比例
            self.labelCount[this_label] = labels.count(this_label) / labels_num
        for vector, label in zip(data_set, labels):
            if label not in self.vectorCount:
                self.vectorCount[label] = []
            self.vectorCount[label].append(vector)
        print("训练结束")
        return self

    def test(self, test_data, label_set):
        if self.length == -1:
            raise ValueError("您还没进行训练, 请先训练")
        # 计算test_data分别为各个类别的概率
        lb_dict = dict()
        for this_lb in label_set:
            p = 1
            all_label = self.labelCount[this_lb]
            all_vector = self.vectorCount[this_lb]
            vector_num = len(all_vector)
            all_vector = numpy.array(all_vector)
            for index in range(0, len(test_data)):
                vector = all_vector[index]
                p = p * vector.count(test_data[index]) / vector_num
            lb_dict[this_lb] = p * all_label
        this_label = sorted(lb_dict, key=lambda x:lb_dict[x], reverse=True)[0]
        return this_label
