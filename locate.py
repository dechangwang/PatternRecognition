#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author wangdechang
# Time 2017/12/30

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor

data = np.loadtxt('data/4g.csv', dtype=np.float, delimiter=",", skiprows=1)

dataframe = pd.DataFrame(data)
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

train_x = trains[0:, 6:]
train_label = trains[0:, 4:6]

test_x = tests[0:, 6:]
test_label = tests[0:, 4:6]

#
# rft = RandomForestRegressor()
# rft.fit(train_x, train_label)
# predicted = rft.predict(test_x)
# error = test_label - predicted
# print(error)

etr = ExtraTreesRegressor()
etr.fit(train_x, train_label)
etr_y_predict = etr.predict(test_x)
errors_predict = test_label - etr_y_predict

# 计算距离
a = errors_predict[0, 1]
b = errors_predict[0, 0]
sin_square_half_a = (np.sin(a / 2)) ** 2
sin_square_half_b = (np.sin(b / 2)) ** 2
extract = (sin_square_half_a + np.cos(test_label[0, 1]) * np.cos(etr_y_predict[0, 1]) * sin_square_half_b) ** (1 / 2)
r = 6378.137
s = 2 * np.arcsin(extract) * r
print(s)
print(test_label)
print(etr_y_predict)

# gbt = GradientBoostingRegressor()
# gbt.fit(x,label)
# gbt_y_predict = gbt.predict(x)
# print(label - gbt_y_predict)


# model = LogisticRegression()
# model.fit(x, label)
# expected = label
# predicted = model.predict(x)

print(data.shape)

'''
函数
'''


def func():
    pass
