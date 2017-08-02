# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 17:42:11 2017

@author: Administrator
"""

import numpy as np
import pandas as pd
from sklearn import linear_model
#import matplotlib.pyplot as plt 

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':float, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}
sales = pd.read_csv('data/kc_house_data.csv', dtype=dtype_dict)
testing = pd.read_csv('data/wk3_kc_house_test_data.csv', dtype=dtype_dict)
training = pd.read_csv('data/wk3_kc_house_train_data.csv', dtype=dtype_dict)
validation = pd.read_csv('data/wk3_kc_house_valid_data.csv', dtype=dtype_dict)
data_sets = [sales, testing, training, validation]

# create new features 
for each_data_set in data_sets:
    each_data_set['sqft_living_sqrt'] = each_data_set['sqft_living'] ** 0.5
    each_data_set['sqft_lot_sqrt'] = each_data_set['sqft_lot'] ** 0.5
    each_data_set['bedrooms_square'] = each_data_set['bedrooms'] ** 2
    each_data_set['floors_square'] = each_data_set['floors'] ** 2


sales_headers = ['bedrooms', 'bedrooms_square',
            'bathrooms',
            'sqft_living', 'sqft_living_sqrt',
            'sqft_lot', 'sqft_lot_sqrt',
            'floors', 'floors_square',
            'waterfront', 'view', 'condition', 'grade',
            'sqft_above',
            'sqft_basement',
            'yr_built', 'yr_renovated']

def findFeatureName(in_model, feature_name_array):
    coeff_with_intercept = np.append(in_model.coef_, in_model.intercept_)
    out_names = []
#   the last coeff is for the intercept 
    for idx in range(len(coeff_with_intercept)-1):
        if coeff_with_intercept[idx] != 0:
            out_names.append(feature_name_array[idx])
    return out_names

def calRSS(in_model, in_x, in_y):
    predicts = in_model.predict(in_x)
    RSS = sum( (predicts - in_y) ** 2 )
    return RSS

# test lasso
sales_model = linear_model.Lasso(alpha = 5e2, normalize = True)
sales_in = sales[sales_headers]
sales_out = sales['price']
sales_model.fit(sales_in, sales_out)
print(findFeatureName(sales_model, sales_headers))

# create possible penalites
pens = np.logspace(1,7, num = 13)    
# extract col according to the headers 
train_in = training[sales_headers]
train_out = training['price']
test_in = testing[sales_headers]
test_out = testing['price']
# find the best penalty value 
RSS_record = {}
model_record = {}
for each_pen in pens:
    cur_model = linear_model.Lasso(alpha = each_pen, normalize = True)
    cur_model.fit(train_in, train_out)
    RSS_record[each_pen] = calRSS(cur_model, test_in, test_out)
    model_record[each_pen] = cur_model
best_pen = min(RSS_record, key = RSS_record.get)
best_RSS = RSS_record[best_pen]
best_model = model_record[best_pen]
num_non_zero = np.count_nonzero(best_model.coef_) + np.count_nonzero(best_model.intercept_)
RSS_on_test = calRSS(best_model, test_in, test_out)
print('Best Penalty value: ', best_pen)
print('Lowest RSS on validation: ', best_RSS)
print('Num of Non Zero coeff: ', num_non_zero)
print('Features: ', findFeatureName(best_model, sales_headers))

rough_range = np.logspace(1,4, num = 20)
def findPenRange(in_x, in_y, possible_points, target_value):
    '''
    find the penalty value range for a given number of remaining features
    '''
    good_pens = []
    for each_pen in possible_points:
        cur_model = linear_model.Lasso(alpha = each_pen, normalize = True)
        cur_model.fit(in_x, in_y)
        cur_num_non_zero = np.count_nonzero(cur_model.coef_) + np.count_nonzero(cur_model.intercept_)
        if cur_num_non_zero == target_value:
            print(each_pen)
            good_pens.append(each_pen)
    return (min(good_pens), max(good_pens))

def refindRange(rough_min, rough_max, mesh_num, 
                train_in, train_out, valid_in, valid_out):
    ''' 
    refine the rough range by remshing the range
    '''
    RSS_record = {}
    model_record = {}
    new_pens = np.linspace(rough_min, rough_max, mesh_num)
    for each_pen in new_pens:
        cur_model = linear_model.Lasso(alpha = each_pen, normalize = True)
        cur_model.fit(train_in, train_out)
        RSS_record[each_pen] = calRSS(cur_model, valid_in, valid_out)
        model_record[each_pen] = cur_model
    best_pen = min(RSS_record, key = RSS_record.get)
    lowest_rss = RSS_record[best_pen]
    best_model = model_record[best_pen]
    return best_model, best_pen, lowest_rss


(min_1, max_1) = findPenRange(train_in, train_out, rough_range, 7)
best_model_7, best_pen_7, lowest_rss_7 = refindRange(min_1, max_1, 20, 
                                                     train_in, train_out,
                                                     test_in, test_out)
best_features = findFeatureName(best_model_7, sales_headers)
print(min_1, max_1)
print(best_pen_7, best_RSS)
print(len(best_features), best_features)
        
    


    
    