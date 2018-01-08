# coding:utf-8

from __future__ import division
import pandas as pd
import numpy as np
import copy
import random
import math


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


# 对连续变量划分数据集，返回数据只包括最后一列
def splitDataSet(dataSet, featIndex, value):
    left_data, right_data = [], []
    for dt in dataSet:
        if dt[featIndex] <= value:
            left_data.append(dt[-2:])
        else:
            right_data.append(dt[-2:])
    return left_data, right_data


# 选择最好的数据集划分方式，使得误差平方和最小
def chooseBestFeature(dataSet):
    best_r2 = np.array([float('inf'), float('inf')])
    best_feature_index = -1
    best_split_value = None
    # 第i个特征
    for i in range(len(dataSet[0]) - 2):
        feat_list = [dt[i] for dt in dataSet]
        # 产生候选划分点
        sortfeat_list = sorted(list(set(feat_list)))
        split_list = []
        # 如果值相同，不存在候选划分点
        if len(sortfeat_list) == 1:
            split_list.append(sortfeat_list[0])
        else:
            for j in range(len(sortfeat_list) - 1):
                split_list.append((sortfeat_list[j] + sortfeat_list[j + 1]) / 2)
        # 第j个候选划分点，记录最佳划分点
        for splitValue in split_list:
            sub_data_set0, sub_data_set1 = splitDataSet(dataSet, i, splitValue)
            len_left, len_right = len(sub_data_set0), len(sub_data_set1)
            # 防止数据集为空，mean不能计算
            if len_left == 0 and len_right != 0:
                right_mean = np.mean(sub_data_set1, axis=0)
                R2 = sum([(x - right_mean) ** 2 for x in sub_data_set1])
            elif len_left != 0 and len_right == 0:
                left_mean = np.mean(sub_data_set0, axis=0)
                R2 = sum([(x - left_mean) ** 2 for x in sub_data_set0])
            else:
                left_mean, right_mean = np.mean(sub_data_set0, axis=0), np.mean(sub_data_set1, axis=0)
                left_r2 = sum([(x - left_mean) ** 2 for x in sub_data_set0])
                right_r2 = sum([(x - right_mean) ** 2 for x in sub_data_set1])
                R2 = left_r2 + right_r2
            if R2.all() <= best_r2.all():
                best_r2 = R2
                best_feature_index = i
                best_split_value = splitValue
    return best_feature_index, best_split_value


# 去掉第i个属性，生成新的数据集
def splitData(dataSet, featIndex, features, value):
    new_features = copy.deepcopy(features)
    new_features.remove(features[featIndex])
    left_data, right_data = [], []
    for dt in dataSet:
        temp = []
        temp.extend(dt[:featIndex])
        temp.extend(dt[featIndex + 1:])
        if dt[featIndex] <= value:
            left_data.append(temp)
        else:
            right_data.append(temp)
    return new_features, left_data, right_data


# 建立决策树
def regressionTree(dataSet, features):
    class_list = [dt[-2:] for dt in dataSet]

    # label一样，全部分到一边
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    # 最后一个特征还不能把所有样本分到一边，则划分到平均值
    if len(features) == 1:
        return np.mean(class_list, axis=0)
    best_feature_index, best_split_value = chooseBestFeature(dataSet)
    best_feature = features[best_feature_index]
    # 删除root特征，生成新的去掉root特征的数据集
    new_features, left_data, right_data = splitData(dataSet, best_feature_index, features, best_split_value)

    # 左右子树有一个为空，则返回该节点下样本均值
    if len(left_data) == 0 or len(right_data) == 0:
        return np.mean(([dt[-2:] for dt in left_data] + [dt[-2:] for dt in right_data]), axis=0)
    else:
        # 左右子树不为空，则继续分裂
        myTree = {best_feature: {'<' + str(best_split_value): {}, '>' + str(best_split_value): {}}}
        myTree[best_feature]['<' + str(best_split_value)] = regressionTree(left_data, new_features)
        myTree[best_feature]['>' + str(best_split_value)] = regressionTree(right_data, new_features)
    return myTree


# 用生成的回归树对测试样本进行测试
def treeClassify(decisionTree, featureLabel, testDataSet):
    first_feature = decisionTree.keys()[0]
    second_feat_dict = decisionTree[first_feature]
    split_value = float(second_feat_dict.keys()[0][1:])
    feature_index = featureLabel.index(first_feature)
    if testDataSet[feature_index] <= split_value:
        value_of_feat = second_feat_dict['<' + str(split_value)]
    else:
        value_of_feat = second_feat_dict['>' + str(split_value)]
    if isinstance(value_of_feat, dict):
        pred_label = treeClassify(value_of_feat, featureLabel, testDataSet)
    else:
        pred_label = value_of_feat
    return pred_label


# 随机抽取样本，样本数量与原训练样本集一样，维度为m-2
def baggingDataSet(dataSet):
    n, m = dataSet.shape
    features = random.sample(dataSet.columns.values[:-2], int((m - 2)))
    features.append(dataSet.columns.values[-2])
    features.append(dataSet.columns.values[-1])
    rows = [random.randint(0, n - 1) for _ in range(n)]
    trainData = dataSet.iloc[rows][features]
    return trainData.values.tolist(), features


def rad(d):
    return d * math.pi / 180.0


def distance(true_pt, pred_pt):
    lat1 = float(true_pt[1])
    lng1 = float(true_pt[0])
    lat2 = float(pred_pt[1])
    lng2 = float(pred_pt[0])
    radLat1 = rad(lat1)
    radLat2 = rad(lat2)
    a = radLat1 - radLat2
    b = rad(lng1) - rad(lng2)
    s = 2 * math.asin(math.sqrt(math.pow(math.sin(a / 2), 2) +
                                math.cos(radLat1) * math.cos(radLat2) * math.pow(math.sin(b / 2), 2)))
    s *= 6378.137
    s = round(s * 10000) / 10
    return s


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
    s = s * 1000
    print(s)
    print("平均距离误差（单位：m）：%f" % np.mean(s))
    print("最大距离误差（单位：m）：%f" % np.max(s))
    print("最小距离误差（单位：m）：%f" % np.min(s))
    print("中位误差（单位：m）： %f" % np.median(s))


def regression():
    data = load_data('data/4g.csv')
    trains, tests = split_train_test_data(data)
    train_x = trains[:, 6:]
    train_label = trains[:, 4:6]
    test_x = tests[0:, 6:]
    test_label = tests[0:, 4:6]
    train_data = pd.DataFrame(np.c_[train_x[:, :], train_label[:, :]])

    # 用index标识feature
    feature_labels = train_data.columns.values.tolist()
    tree_counts = 50

    # tree_list 保存生成的多棵回归树
    tree_list = []
    import save_tree
    for i in range(tree_counts):
        bagging_data, bagginglabels = baggingDataSet(train_data)
        decision_tree = regressionTree(bagging_data, bagginglabels)
        save_tree.store_tree(decision_tree, 'regressTrees/regressTree%d.txt' % i)
        # tree_list.append(decision_tree)
    print (tree_list)
    for i in range(tree_counts):
        tree = save_tree.load_tree('regressTrees/regressTree%d.txt' % i)
        tree_list.append(tree)

    # 对测试样本求预测值
    m = len(test_x)
    predict_res = np.zeros((m, 2))
    for i in range(m):
        labelPred = []
        for tree in tree_list:
            if isinstance(tree,dict):
                label = treeClassify(tree, feature_labels[:-2], test_x[0])
                labelPred.append(label)
        pre = np.mean(labelPred, axis=0)
        predict_res[i] = pre

    error_distance(predict_res, test_label)


regression()

