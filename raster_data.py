#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author wangdechang
# Time 2018/1/1

import numpy as np
import math


def rasterization(data, label_pos=(4, 5), longitude_num=70, latitude_num=60):
    labels = data[:, label_pos[0]:label_pos[1] + 1]
    m = data.shape[0]
    raster_labels = np.zeros(m)
    max_longitude = np.max(labels[:, 0])
    min_longitude = np.min(labels[:, 0])
    max_latitude = np.max(labels[:, 1])
    min_latitude = np.min(labels[:, 1])
    print("max_longitude = %f , min_longitude = %f" % (max_longitude, min_longitude))
    print("max_latitude = %f , min_latitude = %f" % (max_latitude, min_latitude))
    longtitude_space = (max_longitude - min_longitude) / longitude_num
    latitude_space = (max_latitude - min_latitude) / latitude_num

    # min_point = np.array([min_longitude, min_latitude])
    space = np.array(longtitude_space, latitude_space)
    mat_min_point = np.tile([min_longitude, min_latitude], (m, 1))
    mat_space = np.tile([longtitude_space, latitude_space], (m, 1))

    subtract_min_lables = labels - mat_min_point
    space_lables = subtract_min_lables / mat_space
    for i in range(0, m):
        if (math.ceil(space_lables[i, 1]) >= 1):
            longitude_offset = longitude_num * (math.ceil(space_lables[i, 1]) - 1) + int(space_lables[i, 0])
        else:
            longitude_offset = int(space_lables[i, 0])
        raster_labels[i] = longitude_offset

    dict = {}
    for i in range(0, latitude_num):
        for j in range(0, longitude_num):
            key = i * longitude_num + j
            lon = min_longitude + longtitude_space * (j + 0.5)
            lat = min_latitude + latitude_space * (i + 0.5)
            dict[key] = [lon, lat]

    return raster_labels, dict
