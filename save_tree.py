#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author wangdechang
# Time 2018/1/7
import pickle


def store_tree(input_tree, filename):
    fw = open(filename, 'w')
    pickle.dump(input_tree, fw)
    fw.close()


def load_tree(filename):
    fr = open(filename)
    return pickle.load(fr)
