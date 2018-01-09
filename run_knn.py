import pandas as pd

from models.Knn import *
from preprocessing.grid import RoadGrid
from utils import distance


def divide_train_test(data):
    data = data.drop('EcNo_1', axis=1)

    data_num = data.shape[0]
    data_perm = np.random.permutation(range(data_num))
    data = data.iloc[data_perm]
    train_data = data[:int(data_num * 0.8)]
    test_data = data[int(data_num * 0.8):]

    train_feature = np.array(train_data.ix[:, 6:])
    test_feature = np.array(test_data.ix[:, 6:])
    train_label = train_data.ix[:, 4:6].astype(float)
    test_label = test_data.ix[:, 4:6].astype(float)

    rg = RoadGrid(np.vstack((train_label, test_label)), 30)
    train_label_ = rg.transform(np.array(train_label), False)
    test_label_ = np.array(test_label)
    return train_feature, test_feature, train_label_, test_label_, rg


def run():
    filename = './data/4g.csv'
    data_raw = pd.read_csv(filename)
    train_feature, test_feature, train_label, test_label, rg = divide_train_test(data_raw)
    knn = KNN()
    knn.fit(train_feature, train_label)
    pred = knn.predict(test_feature)
    pre = np.array([rg.grid_center[idx] for idx in pred])
    error = [distance(pt1, pt2) for pt1, pt2 in zip(pre, test_label)]
    print [np.min(error), np.max(error), np.mean(error), np.median(error)]
    return error
