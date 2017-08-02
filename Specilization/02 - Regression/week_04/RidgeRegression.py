# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 22:24:19 2017

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
    output_features = np.array(data_sframe[features[0]], ndmin = 2)
    for idx in range(1, len(features)):
        output_features = np.vstack([output_features, data_sframe[features[idx]]])
    output_out = np.array(data_sframe[output])
    return output_features, np.reshape(output_out, (1, output_out.shape[0]))


def predict_output(feature_matrix, weights):
    in_matrix = np.vstack([np.ones((1,feature_matrix.shape[1])), feature_matrix])
    return np.dot(weights, in_matrix)



def weight_update_ridge(feature, weight, output, l2_penalty, step_size, max_iter_num = 100):
    out_weight = np.copy(weight)
    for iter_num in range(max_iter_num):
        errors = output - predict_output(feature, out_weight) 
        feature_with_inter = np.vstack([np.ones((1,feature.shape[1])), feature])
        feature_with_inter = feature_with_inter.transpose()
        derivative_rss = -2 * np.dot(errors, feature_with_inter)
        derivative_reg = 2 * l2_penalty * out_weight
        derivative_reg[0,0] = 0
    #   neglect w0 term 
        total_derivative = derivative_reg + derivative_rss
        out_weight = out_weight - step_size * total_derivative
    return out_weight
        
    
# no regularization 
train_feature, train_out = get_numpy_data(train_data, ['sqft_living'], 'price')
test_feature, test_out = get_numpy_data(test_data, ['sqft_living'], 'price')
pen_no = 0
pen_high = 1e11

weight_no = np.array([0,0], ndmin = 2)
step_no = 1e-12
max_iter_no = 1000

def calSum(np_array):
    total = 0
    for each in np_array.transpose():
#        print(each)
        total += each
    return total

weight_no_reg = weight_update_ridge(train_feature, weight_no, train_out, pen_no, step_no, max_iter_no)
weight_high_reg = weight_update_ridge(train_feature, weight_no, train_out, pen_high, step_no, max_iter_no)
print('No reg weight: ', weight_no_reg)
print('High reg weight: ', weight_high_reg)
pred_no = predict_output(train_feature, weight_no_reg).transpose()
pred_high = predict_output(train_feature, weight_high_reg).transpose()
# on test 
pred_no_test = predict_output(test_feature, weight_no_reg)
pred_high_test = predict_output(test_feature, weight_high_reg)
# print result 
print('Rss on Test - no Reg: ', calSum((pred_no_test - test_out) ** 2))
print('Rss on Test - high Reg: ', calSum((pred_high_test - test_out) ** 2))

plt.figure(1)
plt.scatter(train_feature.transpose(), train_out.transpose(), s = 1)
plt.plot(train_feature.transpose(), pred_high ,'g-')
plt.plot(train_feature.transpose(), pred_no ,'r-')


weight_no_2 = np.array([0,0,0], ndmin = 2)
train_feature_2, train_out_2 = get_numpy_data(train_data, ['sqft_living', 'sqft_living15'], 'price')
test_feature_2, test_out_2 = get_numpy_data(train_data, ['sqft_living', 'sqft_living15'], 'price')

weight_no_reg_2 = weight_update_ridge(train_feature_2, weight_no_2, train_out_2, pen_no, step_no, max_iter_no)
weight_high_reg_2 = weight_update_ridge(train_feature_2, weight_no_2, train_out_2, pen_high, step_no, max_iter_no)
print('No reg weight: ', weight_no_reg_2)
print('High reg weight: ', weight_high_reg_2)
pred_no_test_2 = predict_output(test_feature_2, weight_no_reg_2)
pred_high_test_2 = predict_output(test_feature_2, weight_high_reg_2)


print('Rss on Test - no Reg: ', calSum((pred_no_test_2 - test_out_2) ** 2))
print('Rss on Test - high Reg: ', calSum((pred_high_test_2 - test_out_2) ** 2))
pred_1_diff_no = predict_output(test_feature_2[:,0:1], weight_no_reg_2) - test_out_2[0,0]
pred_1_diff_high = predict_output(test_feature_2[:,0:1], weight_high_reg_2) - test_out_2[0,0]
print('Pred 1st no: ', pred_1_diff_no)
print('Pred 1st high: ', pred_1_diff_high)
