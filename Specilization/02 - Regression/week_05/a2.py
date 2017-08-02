# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 20:39:32 2017

@author: Administrator
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':float, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}
train_data = pd.read_csv('data/wk3_kc_house_train_data.csv', dtype=dtype_dict)
valid_data = pd.read_csv('data/wk3_kc_house_valid_data.csv', dtype=dtype_dict)
test_data = pd.read_csv('data/wk3_kc_house_test_data.csv', dtype=dtype_dict)

# ridge regression implementation
def get_numpy_data(data_sframe, features, output):
    '''
    takes a data frame, extract features according to the features as the first output 
    takes the output col name and returns it as the second output
    '''
    output_features = np.ones((1, data_sframe.shape[0]))
    for idx in range(0, len(features)):
        output_features = np.vstack([output_features, data_sframe[features[idx]]])
    output_out = np.array(data_sframe[output])
    return output_features, np.reshape(output_out, (1, output_out.shape[0]))


def predict_output(feature_matrix, weights):
    return np.dot(weights, feature_matrix)

def normalizeFeatures(in_features):
    out_features = np.copy(in_features)
    for idx in range(in_features.shape[0]):
        cur_norm = np.linalg.norm(in_features[idx,:])
        out_features[idx, :] = in_features[idx, :]/cur_norm
    return (out_features, norm)

def lasso_weight_update(in_weights, l1_pen, in_features, out_pred, 
                        max_iter_num):
    out_weight = np.copy(in_weights)
    for cur_iter_num in range(max_iter_num):
        for idx in range(out_weight.shape[1]):
            cur_weight = np.copy(out_weight)
            cur_weight[0,idx] = 0
            cur_pred = predict_output(in_features, cur_weight)
            cur_diff = out_pred - cur_pred
            rho = np.dot(cur_diff, in_features[idx,:].transpose())
            if idx == 0:
                out_weight[0, idx] = rho
            elif rho < -l1_pen/2:
                out_weight[0, idx] = rho + l1_pen/2
            elif rho > l1_pen/2:
                out_weight[0, idx] = rho - l1_pen/2
            else:
                out_weight[0, idx] = 0
        print(out_weight)
    return out_weight

train_in, train_out = get_numpy_data(train_data, ['sqft_living', 'bedrooms'], 'price')
train_in_n, norm = normalizeFeatures(train_in)
test_weight = lasso_weight_update(np.array([0.0,0.0,0.0], ndmin = 2), 1e7,
                                  train_in_n, train_out,
                                  10)