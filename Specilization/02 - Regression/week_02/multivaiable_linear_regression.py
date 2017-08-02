# -*- coding: utf-8 -*-
"""
Created on Sun Feb 05 17:11:01 2017

@author: Administrator
"""
import pandas as pd 
import numpy as np
from sklearn import linear_model

# read in data 
test_data = pd.read_csv("data/kc_house_test_data.csv")
# create four new variables 
# raw features 

def readData(current_data):
    bed_rooms = np.transpose(np.matrix(current_data['bedrooms']))
    bath_rooms = np.transpose(np.matrix(current_data['bathrooms']))
    sqft_living = np.transpose(np.matrix(current_data['sqft_living']))
    lat = np.transpose(np.matrix(current_data['lat']))
    longi = np.transpose(np.matrix(current_data['long']))
    price = np.transpose(np.matrix(current_data['price']))
    # combined features 
    bed_rooms_sq = np.square(bed_rooms)
    bed_bath_rooms = np.multiply(bath_rooms,bed_rooms)
    log_sqft_living = np.log(sqft_living)
    lat_longi = lat + longi
    return [bed_rooms, bath_rooms, sqft_living, 
            lat, longi,
            bed_rooms_sq, bed_bath_rooms, log_sqft_living, lat_longi,
            price]
    
[bed_rooms, bath_rooms, sqft_living, 
            lat, longi,
            bed_rooms_sq, bed_bath_rooms, log_sqft_living, lat_longi,
            price] = readData(test_data)
# train model: 
    # model 1: sqft_living bedrooms bathrooms lat long 
    # model 2: sqft_living bedrooms bathrooms lat long bed_bat_rooms 
    # moedl 3: sqft_living bedrooms bathrooms lat long bed_bath_rooms bed_rooms_sq
    # log_sqft_living lat_longi 
def trainMultiModel(matrixList, yValue):
    '''
    returns a trained linear model of multi-variable 
    '''
    output_model = linear_model.LinearRegression()
    concate_data = np.hstack(matrixList)
    output_model.fit(concate_data, yValue)
    return output_model

# model 1 
model_1 = trainMultiModel([sqft_living, bed_rooms, bath_rooms, lat, longi], price)
# model 2 
model_2 = trainMultiModel([sqft_living, bed_rooms, bath_rooms, lat, longi, bed_bath_rooms], price)
# model 3 
model_3 = trainMultiModel([sqft_living, bed_rooms, bath_rooms, lat, longi, 
                           bed_bath_rooms, bed_rooms_sq, log_sqft_living, lat_longi],price)
# calculate RSS for each model
calRss = lambda list1, list2: sum( (list1[idx][0] - list2[idx][0]) for idx in range(len(list1)) )
predict_1_test = model_1.predict(np.hstack([sqft_living, bed_rooms, bath_rooms, lat, longi]))
predict_2_test = model_2.predict(np.hstack([sqft_living, bed_rooms, bath_rooms, lat, longi, bed_bath_rooms]))
predict_3_test = model_3.predict(np.hstack([sqft_living, bed_rooms, bath_rooms, lat, longi, 
                           bed_bath_rooms, bed_rooms_sq, log_sqft_living, lat_longi]))
    
train_data = pd.read_csv('data/kc_house_train_data.csv')
[train_bed_rooms, train_bath_rooms, train_sqft_living, 
            train_lat, train_longi,
            train_bed_rooms_sq, train_bed_bath_rooms, train_log_sqft_living, train_lat_longi,
            train_price] = readData(train_data)
predict_1_train = model_1.predict(np.hstack([train_sqft_living, train_bed_rooms, train_bath_rooms, train_lat, train_longi]))
predict_2_train = model_2.predict(np.hstack([train_sqft_living, train_bed_rooms, train_bath_rooms,
                                             train_lat, train_longi, train_bed_bath_rooms]))
predict_3_train = model_3.predict(np.hstack([train_sqft_living, train_bed_rooms, train_bath_rooms, train_lat, train_longi, 
                           train_bed_bath_rooms, train_bed_rooms_sq, train_log_sqft_living, train_lat_longi]))
    
# question 1: mean of bedrooms_sq
mean_fun = lambda x: sum(x)/len(x)
mean_bed_sq = mean_fun(bed_rooms_sq)
print(mean_bed_sq)
# question 2: mean bed_bath_rooms 
mean_bed_bath = mean_fun(bed_bath_rooms)
print(mean_bed_bath)
# question 3: mean of log_sq_living 
mean_log = mean_fun(log_sqft_living)
print(mean_log)
# question 4 mean of lat_long 
mean_lat_long = mean_fun(lat_longi)
print(mean_lat_long)
# question 5 
print(model_1.coef_[0][2])
# question 6 
print(model_2.coef_[0][2])
# question 7 
print(model_3.coef_[0][2]) 
# question 8
print(calRss(predict_1_train, train_price))
print(calRss(predict_2_train, train_price))
print(calRss(predict_3_train, train_price))
print('---------------')
# question 9
print(calRss(predict_1_test, price))
print(calRss(predict_2_test, price))
print(calRss(predict_3_test, price))


