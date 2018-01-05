#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author wangdechang
# Time 2018/1/1


import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, ExtraTreesClassifier


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
        predict_label[0:, 1] * np.pi / 180.0) * sin_square_half_b) ** (1 / 2)
    r = 6378.137
    s = 2 * np.arcsin(extract) * r

    print("平均距离误差（单位：km）：%f" % np.mean(s))
    print("最大距离误差（单位：km）：%f" % np.max(s))
    print("最小距离误差（单位：km）：%f" % np.min(s))
    print("中位误差（单位：km）： %f" % np.median(s))



def regress_predict():
    data = load_data('data/2g.csv')

    trains, tests = split_train_test_data(data)
    train_x = trains[0:, 6:]
    train_label = trains[0:, 4:6]

    test_x = tests[0:, 6:]
    test_label = tests[0:, 4:6]

    etr = ExtraTreesRegressor()
    etr.fit(train_x, train_label)
    etr_y_predict = etr.predict(test_x)
    error_distance(etr_y_predict,test_label)
    # errors_predict = test_label - etr_y_predict
    #
    # # 计算距离
    # a = errors_predict[0:, 1] * np.pi / 180.0
    # b = errors_predict[0:, 0] * np.pi / 180.0
    # sin_square_half_a = (np.sin(a / 2)) ** 2
    # sin_square_half_b = (np.sin(b / 2)) ** 2
    # extract = (sin_square_half_a + np.cos(test_label[0:, 1] * np.pi / 180.0) * np.cos(
    #     etr_y_predict[0:, 1] * np.pi / 180.0) * sin_square_half_b) ** (1 / 2)
    # r = 6378.137
    # s = 2 * np.arcsin(extract) * r
    #
    # print("平均距离误差（单位：km）：%f" % np.mean(s))


def classifer():
    data = load_data('data/4g.csv')

    trains, tests = split_train_test_data(data)
    train_x = trains[0:, 6:]
    # train_label = trains[0:, 4:6]
    from raster_data import rasterization
    train_label, dict_raster = rasterization(trains, (4, 5))

    test_x = tests[0:, 6:]
    test_label = tests[0:, 4:6]

    etc = ExtraTreesClassifier()
    etc.fit(train_x, train_label)
    etc_y_predict = etc.predict(test_x)
    m = len(test_x)
    predict_res = np.zeros((m, 2))
    for i in range(0, m):
        predict_res[i] = dict_raster[int(etc_y_predict[i])]
    print(etc_y_predict)
    error_distance(predict_res, test_label)


classifer()
regress_predict()