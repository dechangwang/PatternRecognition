# encoding=utf-8
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from models.bayes import NaiveBayes
from preprocessing.grid import RoadGrid
from utils import distance


def split_data(raw_data):
    df_4g = shuffle(raw_data).reset_index(drop=True)
    # feature_col_name_base = ['RNCID', 'CellID', 'RSCP', 'RTT', 'UE_Rx_Tx']
    col1 = ['RNCID', 'CellID']
    col2 = ['RSCP']
    # feature_col_name = ['{}_{}'.format(base, i) for i in range(1, 4) for base in feature_col_name_base]
    n_baseStation = 3
    feature_col_name = ['RNCID_1', 'CellID_1'] + ['{}_{}'.format(base, i) for i in range(1, 1 + n_baseStation) for base
                                                  in col2]
    print '>feature_col_name:', feature_col_name
    label_col_name = ['Longitude', 'Latitude']

    proportion = int(df_4g.shape[0] * 0.80)
    train = df_4g.iloc[:proportion, :]
    test = df_4g.iloc[proportion:, :]
    # print 'train.shape:', train.shape
    # print 'test.shape:', test.shape

    train_features = train[feature_col_name].values
    train_label = train[label_col_name].values
    test_features = test[feature_col_name].values
    test_label = test[label_col_name].values
    features = df_4g[feature_col_name].values
    labels = df_4g[label_col_name].values

    return train_features, train_label, test_features, test_label


def run(filename='./data/4g.csv'):
    filename = './data/4g.csv'
    df_4g = pd.read_csv(filename)
    train_features, train_label, test_features, test_label = split_data(df_4g)
    rg = RoadGrid(np.vstack((train_label, test_label)), 30)
    train_label_ = np.array(rg.transform(train_label, False))

    nb = NaiveBayes(is_feature_continuous=[0, 0, 1, 1, 1], debug=False).fit(train_features, train_label_)
    pre_ = nb.predict(test_features)
    pre = np.array([rg.grid_center[idx] for idx in pre_])
    error = [distance(pt1, pt2) for pt1, pt2 in zip(pre, test_label)]
    print [np.min(error), np.max(error), np.mean(error), np.median(error)]
    return error
