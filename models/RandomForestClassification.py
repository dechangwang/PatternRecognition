# coding:utf-8
import copy
import random
import threading

import numpy as np
import pandas as pd

from preprocessing.grid import RoadGrid
from utils import distance


# from multiply_thread import threeThread

class threeThread(threading.Thread):
    def __init__(self, threadID, name, i, df, file_dir):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.i = i
        self.df = df
        self.file_dir = file_dir

    def run(self):
        print "Starting " + self.name
        bagging_data, bagginglabels = baggingDataSet(self.df)
        decision_tree = createTree(bagging_data, bagginglabels[:-1])
        import save_tree
        save_tree.store_tree(decision_tree, '%s/tree%d.txt' % (self.file_dir, self.i))
        print "Exiting " + self.name


# 最后一个属性还不能将样本完全分开，此时数量最多的label被选为最终类别
def majorClass(classList):
    classDict = {}
    for cls in classList:
        classDict[cls] = classDict.get(cls, 0) + 1
    sortClass = sorted(classDict.items(), key=lambda item: item[1])
    return sortClass[-1][0]


# 计算基尼系数
def calcGini(dataSet):
    labelCounts = {}
    # 给所有可能分类创建字典
    for dt in dataSet:
        currentLabel = dt[-1]
        labelCounts[currentLabel] = labelCounts.get(currentLabel, 0) + 1
    Gini = 1
    for key in labelCounts:
        prob = labelCounts[key] / len(dataSet)
        Gini -= prob * prob
    return Gini


# 对连续变量划分数据集
def splitDataSet(dataSet, featIndex, value):
    leftData, rightData = [], []
    for dt in dataSet:
        if dt[featIndex] <= value:
            leftData.append(dt)
        else:
            rightData.append(dt)
    return leftData, rightData


# 选择最好的数据集划分方式
def chooseBestFeature(dataSet, index=1):
    bestGini = 1
    bestFeatureIndex = -1
    bestSplitValue = None

    # 随机选择一个特征
    import random
    bestFeatureIndex = random.sample(range(len(dataSet[0])), 1)[0]

    arr = np.array(dataSet)
    # bestSplitValue = np.mean(arr[:, bestFeatureIndex])
    # min_value = np.min(arr[:, bestFeatureIndex])
    # max_value = np.max(arr[:, bestFeatureIndex])
    # bestSplitValue = random.uniform(min_value, max_value)

    if index == 0:
        featList = [dt[bestFeatureIndex] for dt in dataSet]
        # 产生候选划分点
        sortfeatList = sorted(list(set(featList)))
        splitList = []
        for j in range(len(sortfeatList) - 1):
            splitList.append((sortfeatList[j] + sortfeatList[j + 1]) / 2)

        # 第j个候选划分点，记录最佳划分点
        for splitValue in splitList:
            newGini = 0
            subDataSet0, subDataSet1 = splitDataSet(dataSet, i, splitValue)
            newGini += len(subDataSet0) / len(dataSet) * calcGini(subDataSet0)
            newGini += len(subDataSet1) / len(dataSet) * calcGini(subDataSet1)
            if newGini < bestGini:
                bestGini = newGini
                bestSplitValue = splitValue
        return bestFeatureIndex, bestSplitValue

    for i in range(len(dataSet[0]) - 1):
        featList = [dt[i] for dt in dataSet]
        # 产生候选划分点
        sortfeatList = sorted(list(set(featList)))
        splitList = []
        for j in range(len(sortfeatList) - 1):
            splitList.append((sortfeatList[j] + sortfeatList[j + 1]) / 2)

        # 第j个候选划分点，记录最佳划分点
        for splitValue in splitList:
            newGini = 0
            subDataSet0, subDataSet1 = splitDataSet(dataSet, i, splitValue)
            newGini += len(subDataSet0) / len(dataSet) * calcGini(subDataSet0)
            newGini += len(subDataSet1) / len(dataSet) * calcGini(subDataSet1)
            if newGini < bestGini:
                bestGini = newGini
                bestFeatureIndex = i
                bestSplitValue = splitValue
    return bestFeatureIndex, bestSplitValue


# 去掉第i个属性，生成新的数据集
def splitData(dataSet, featIndex, features, value):
    newFeatures = copy.deepcopy(features)
    newFeatures.remove(features[featIndex])
    leftData, rightData = [], []
    for dt in dataSet:
        temp = []
        temp.extend(dt[:featIndex])
        temp.extend(dt[featIndex + 1:])
        if dt[featIndex] <= value:
            leftData.append(temp)
        else:
            rightData.append(temp)
    return newFeatures, leftData, rightData


# 建立决策树
def createTree(dataSet, features, index=0):
    classList = [dt[-1] for dt in dataSet]

    # if len(dataSet) == 0 or len(features) == 0:
    #     return

    # label一样，全部分到一边
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 最后一个特征还不能把所有样本分到一边，则选数量最多的label

    if len(features) == 1 or len(dataSet[0]) == 1:
        return majorClass(classList)
    bestFeatureIndex, bestSplitValue = chooseBestFeature(dataSet, index)
    if bestFeatureIndex >= len(features):
        return majorClass(classList)
    bestFeature = features[bestFeatureIndex]
    # 生成新的去掉bestFeature特征的数据集
    newFeatures, leftData, rightData = splitData(dataSet, bestFeatureIndex, features, bestSplitValue)
    if len(leftData) == 0 or len(rightData) == 0:
        return majorClass(classList)
    # 左右两颗子树，左边小于等于最佳划分点，右边大于最佳划分点
    myTree = {bestFeature: {'<' + str(bestSplitValue): {}, '>' + str(bestSplitValue): {}}}
    myTree[bestFeature]['<' + str(bestSplitValue)] = createTree(leftData, newFeatures, index + 1)
    myTree[bestFeature]['>' + str(bestSplitValue)] = createTree(rightData, newFeatures, index + 1)
    return myTree


# 用生成的决策树对测试样本进行分类
def treeClassify(decisionTree, featureLabel, testDataSet):
    firstFeature = decisionTree.keys()[0]
    secondFeatDict = decisionTree[firstFeature]
    splitValue = float(secondFeatDict.keys()[0][1:])
    featureIndex = featureLabel.index(firstFeature)
    if testDataSet[featureIndex] <= splitValue:
        valueOfFeat = secondFeatDict['<' + str(splitValue)]
    else:
        valueOfFeat = secondFeatDict['>' + str(splitValue)]
    if isinstance(valueOfFeat, dict):
        pred_label = treeClassify(valueOfFeat, featureLabel, testDataSet)
    else:
        pred_label = valueOfFeat
    return pred_label


# 随机抽取样本，样本数量与原训练样本集一样，维度为m-1
def baggingDataSet(dataSet):
    n, m = dataSet.shape
    # features = random.sample(dataSet.columns.values[:-1], int(m - 1))
    features = [0, 1, 3, 6, 7, 9, 12, 13, 15, 18, 19, 20, 24, 25, 27, 30, 31, 33]
    features.append(dataSet.columns.values[-1])
    # features = dataSet.columns.values[:]
    rows = [random.randint(0, n - 1) for _ in range(n)]
    trainData = dataSet.iloc[rows][features]

    return trainData.values.tolist(), features


def load_data(filename):
    data = np.loadtxt(filename, dtype=np.float, delimiter=",", skiprows=1)
    return data


def split_train_test_data(data):
    train_set = []
    test_set = []

    m, n = data.shape
    for i in range(1, m):
        if i % 16 == 0:
            test_set.append(data[i])
        else:
            train_set.append(data[i])

    trains = np.array(train_set)
    tests = np.array(test_set)
    print(trains.shape)
    print(tests.shape)
    return trains, tests


def error_distance(predict_label, test_label):
    errors_predict = test_label - predict_label
    # 计算距离
    a = errors_predict[0:, 1] * np.pi / 180.0
    b = errors_predict[0:, 0] * np.pi / 180.0
    sin_square_half_a = (np.sin(a / 2)) ** 2
    sin_square_half_b = (np.sin(b / 2)) ** 2
    extract = (sin_square_half_a + np.cos(test_label[0:, 1] * np.pi / 180.0) * np.cos(
        predict_label[0:, 1] * np.pi / 180.0) * sin_square_half_b)
    extract = np.sqrt(extract)
    r = 6378.137
    s = 2 * np.arcsin(extract) * r
    print(s)
    print("平均距离误差（单位：km）：%f" % np.mean(s))
    print("最大距离误差（单位：km）：%f" % np.max(s))
    print("最小距离误差（单位：km）：%f" % np.min(s))
    print("中位误差（单位：km）： %f" % np.median(s))


def classify():
    data = load_data('./data/4g.csv')
    trains, tests = split_train_test_data(data)
    train_x = trains[0:, 6:]
    train_label = trains[0:, 4:6]
    test_x = tests[0:, 6:]
    test_label = tests[0:, 4:6]

    rg = RoadGrid(np.vstack((train_label, test_label)), 30)
    train_label_ = rg.transform(train_label, False)

    # from raster_data import rasterization
    # train_label, dict_raster = rasterization(trains, (4, 5))
    train_data = pd.DataFrame(np.c_[train_x[:, :], train_label_[:]])
    df = train_data
    # df = pd.read_csv('data/wine.txt', header=None)
    feature_labels = df.columns.values.tolist()
    # df = df[df[labels[-1]] != 3]

    import save_tree

    # 生成多棵决策树，放到一个list里边
    tree_counts = 50
    treeList = []
    # for i in range(tree_counts):
    #     bagging_data, bagginglabels = baggingDataSet(df)
    #     decision_tree = createTree(bagging_data, bagginglabels[:-1])
    #     print (decision_tree)
    #     save_tree.store_tree(decision_tree, 'trees50_part_attr_gini/tree%d.txt' % i)
    #     treeList.append(decision_tree)
    print (treeList)

    # threads = [threeThread('id%d' % i, 'name%d' % i, i, df, 'trees100') for i in range(tree_counts)]
    # for t in threads:
    #     t.start()
    #
    # for t in threads:
    #     t.join()

    print ("start testing ...")
    for i in range(tree_counts):
        tree = save_tree.load_tree('trees50_1/tree%d.txt' % i)
        treeList.append(tree)
    m = len(test_x)
    # predict_res = np.zeros((m, 2))
    predict_res = []
    # 对测试样本分类
    for i in range(m):
        labelPred = []
        for tree in treeList:
            if not isinstance(tree, float):
                label = treeClassify(tree, feature_labels[:-1], test_x[i])
                labelPred.append(label)
        # 投票选择最终类别
        labelDict = {}
        for label in labelPred:
            labelDict[label] = labelDict.get(label, 0) + 1
        sortClass = sorted(labelDict.items(), key=lambda item: item[1])
        predict_res.append(int(sortClass[-1][0]))
        # predict_res[i] = dict_raster[int(sortClass[-1][0])]
    pre = np.array([rg.grid_center[idx] for idx in predict_res])
    error = [distance(pt1, pt2) for pt1, pt2 in zip(pre, test_label)]
    print("平均距离误差（单位：m）：%f" % np.mean(error))
    print("最大距离误差（单位：m）：%f" % np.max(error))
    print("最小距离误差（单位：m）：%f" % np.min(error))
    print("中位误差（单位：m）： %f" % np.median(error))
    return error
    # error_distance(predict_res, test_label)

# classify()
