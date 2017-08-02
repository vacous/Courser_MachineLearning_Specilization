# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 23:32:47 2017

@author: Administrator
"""

import pandas as pd
from sklearn import linear_model
import numpy as np 

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':float, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}

sales = pd.read_csv('data/kc_house_data.csv', dtype=dtype_dict)
sales = sales.sort(['sqft_living','price'])


def polynomialDataFrame(feature, degree):
    '''
    takes a pd.series feature 
    put the first colum with name power_1 and the original feature 
    the second with in_feature ** 2 under name power_2 
    '''
    output = pd.DataFrame()
    for cur_degree in range(degree):
        output["power" + str(cur_degree + 1)] = feature ** (cur_degree + 1)
    return output

small_penalty = 1.5e-5

poly_15_data = polynomialDataFrame(sales['sqft_living'], 15)
model = linear_model.Ridge(alpha = small_penalty , normalize = True)
model.fit(poly_15_data, sales['price'])
# Q1: coefficient of power 1
print('Coeff of power 1: ', model.coef_[0])

small_pen_2 = 1e-9
large_pen = 1.23e2
# dtype_dict same as above
set_1 = pd.read_csv('data/wk3_kc_house_set_1_data.csv', dtype=dtype_dict)
set_2 = pd.read_csv('data/wk3_kc_house_set_2_data.csv', dtype=dtype_dict)
set_3 = pd.read_csv('data/wk3_kc_house_set_3_data.csv', dtype=dtype_dict)
set_4 = pd.read_csv('data/wk3_kc_house_set_4_data.csv', dtype=dtype_dict)
sets = [set_1, set_2, set_3, set_4]

def diffPenalty(in_sets ,in_penalty):
    models = [linear_model.Ridge(alpha = in_penalty, normalize = True) for _ in range(len(in_sets))]
    coeff_1 = []
    for idx in range(len(in_sets)):
        cur_set = in_sets[idx]
        cur_data = polynomialDataFrame(cur_set['sqft_living'],15)
        cur_model = models[idx]
        cur_model.fit(cur_data, cur_set['price'])
        coeff_1.append(cur_model.coef_[0])
    print("Max coeff p1:", max(coeff_1))
    print("Min coeff p1:", min(coeff_1))

# Q2: 
print('Small')
diffPenalty(sets, small_pen_2)
print('Large')
diffPenalty(sets, large_pen)

train_valid_shuffled = pd.read_csv('data/wk3_kc_house_train_valid_shuffled.csv', dtype=dtype_dict)
test = pd.read_csv('data/wk3_kc_house_test_data.csv', dtype=dtype_dict)

n = len(train_valid_shuffled)
k = 10 # 10-fold cross-validation

def calRSS(in_model, valid_in, valid_out):
    global valid_print 
    predicts = in_model.predict(valid_in)
    RSS = sum([ (predicts[idx] - list(valid_out)[idx]) ** 2 for idx in range(len(valid_out)) ])
    return RSS
    
def k_fold_cross_validation(k, l2_penalty, data, output):
    '''
    k: number of fold validation sets 
    l2_penalty: for ridge regression
    data: in_data, x 
    output: y
    '''
    model_record = []
    valid_error_record = []
    for idx in range(k):
        cur_start = int((n * idx) / k)
        cur_end = int((n * (idx + 1))/k - 1)
#       slice the data for training set
        cur_train_in = data[0:cur_start].append(data[cur_end:])
        cur_train_out = output[0:cur_start].append(output[cur_end:])
#       slice data for validation set
        cur_valid_in = data[cur_start: cur_end]
        cur_valid_out = output[cur_start: cur_end]
#       fit model 
        cur_model = linear_model.Ridge(alpha = l2_penalty, normalize = True)
        cur_model.fit(cur_train_in, cur_train_out)
        model_record.append(cur_model)
        valid_error_record.append( calRSS(cur_model, cur_valid_in, cur_valid_out))
    return sum(valid_error_record)/len(valid_error_record)


possible_pen = np.logspace(3,9, num = 13)
result_map = {}
power_15_data = polynomialDataFrame(train_valid_shuffled['sqft_living'], 15)
price_out = train_valid_shuffled['price']
for each_pen in possible_pen:
    print(each_pen, ':')
    cur_avg_error = k_fold_cross_validation(10, each_pen, power_15_data, price_out)
    print(cur_avg_error)
    result_map[each_pen] = cur_avg_error
    
best_pen = min(result_map, key = result_map.get)    
print('Min error penalty: ', best_pen)
print('Min error: ', min(result_map.values()))


best_model = linear_model.Ridge(alpha = best_pen, normalize = True)
best_model.fit(power_15_data, price_out)
test_15_data = polynomialDataFrame(test['sqft_living'],15)
test_price_out = test['price']
test_rss = calRSS(best_model, test_15_data, test_price_out)
print('Test Rss: ', test_rss)

