import numpy as np


class KNN:
    def __init__(self, k=1):
        self.k = k

    def fit(self, train_feature, train_label):

        self.train_feature = train_feature
        self.train_label = train_label
        return self

    def Knn(self, te_feature, tr_feature, tr_labels, k):
        tr_num = tr_feature.shape[0]
        a = np.tile(te_feature, (tr_num, 1)) - tr_feature
        b = a ** 2
        c = b.sum(axis=1)
        d = c ** 0.5
        sortd = d.argsort()
        classCount = dict()
        for i in range(k):
            candidateLabel = tr_labels[sortd[i]]
            classCount[candidateLabel] = classCount.get(candidateLabel, 0) + 1
        sortClassCount = sorted(classCount.iteritems(), key=classCount.get, reverse=True)
        return sortClassCount[0][0]

    def predict(self, test_feature):
        pre = list()
        for i in range(test_feature.shape[0]):
            pre.append(self.Knn(test_feature[i], self.train_feature, self.train_label, self.k))
        return pre
