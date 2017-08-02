# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 17:26:26 2017

@author: Administrator
"""

import numpy as np
import pandas as pd


dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':float, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}
train_data = pd.read_csv('data/wk3_kc_house_train_data.csv', dtype=dtype_dict)
valid_data = pd.read_csv('data/wk3_kc_house_valid_data.csv', dtype=dtype_dict)
test_data = pd.read_csv('data/wk3_kc_house_test_data.csv', dtype=dtype_dict)


def normalizeFeatures(in_features):
    global a 
    out_features = np.array(in_features)
    a = out_features
    norm = np.linalg.norm(out_features)
    out_features = out_features/ norm
    return (out_features, norm)

def get_numpy_data(data_sframe, features, output):
    '''
    takes a data frame, extract features according to the features as the first output 
    takes the output col name and returns it as the second output
    '''
    output_features = data_sframe[features[0]]
    for idx in range(1, len(features)):
        output_features = np.vstack([output_features, data_sframe[features[idx]]])
    output_out = np.array(data_sframe[output])
    return output_features, np.reshape(output_out, (1, output_out.shape[0]))

def calDistance(in_points, target):
    diff_vec = in_points - target
    return sum(diff_vec ** 2)

def k_nearest_neighbors(k, feature_train, feature_query):
    '''
    keep a max heap map {distance: feature} nlog(m)
    argsort nlog(n)
    push out the max when the size is larger than k 
    '''
    all_dist = calDistance(feature_train, feature_query)
    result_idxs = np.argsort(all_dist)[:k]
    result_points = feature_train[:,result_idxs]
    return result_points

def predict_knn(k_points):
    return np.sum(k_points, axis = 1)/k_points.shape[1]

