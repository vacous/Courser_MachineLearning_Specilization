# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 23:17:11 2017

@author: Administrator
"""

import json 
import pandas as pd 
import string 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
with open('../data/important_words.json') as json_data:
    important_words = json.load(json_data)


def remove_punctuation(in_string):
    return in_string.translate(str.maketrans('','',string.punctuation))

data_file = pd.read_csv('../data/amazon_baby_subset.csv')
data_file = data_file.fillna({'review':''})
data_file['review_clean'] = data_file['review'].apply(remove_punctuation)

for word in important_words:
    data_file[word] = data_file['review_clean'].apply(lambda s : s.split().count(word))
    
def countNum(in_frame, word):
    counter = 0
    for each in in_frame[word]:
        if each > 0:
            counter += 1
    return counter

print(countNum(data_file, 'perfect'))

def extractXY(data_frame, x_names, y_name):
    out_x = np.array(data_frame[x_names])
    ones_col = np.ones((out_x.shape[0],1))
    out_x = np.hstack([ones_col, out_x])
    out_y = np.array(data_frame[y_name])
    return out_x, out_y

train_x, train_y = extractXY(data_file, important_words, 'sentiment')

def sep(x):
    return (x == 1) * (np.ones_like(x)).transpose()
    
def predict_probability(in_x_matrix, coefficient):
    '''
    dim(in_x_matrix) = num_data * num_feature
    dim(coefficients) = num_feature * 1
    '''
    scores = in_x_matrix * coefficient 
    predictions = 1 / (1 + np.exp(-scores))
    return predictions

def cal_derivative(in_x_matrix, coefficients, in_y):
    pred_prob = predict_probability(in_x_matrix, coefficients)
    errors = sep(in_y) - pred_prob
    feature_derivatives = in_x_matrix.transpose() * errors
    return feature_derivatives 

def logistic_regression(in_x_matrix, in_coeff, in_y, step_size, max_iter):
    out_coeff = np.copy(in_coeff)
    for iter_num in range(max_iter):
        derivative = cal_derivative(in_x_matrix, out_coeff, in_y)
        out_coeff += step_size * derivative
    return out_coeff
    
    
def logistic_regression_with_pen(in_x_matrix, in_coeff, in_y, step_size, max_iter, l2_pen):
    out_coeff = np.copy(in_coeff)
    for iter_num in range(max_iter):
        derivative = cal_derivative(in_x_matrix, out_coeff, in_y) - 2 * l2_pen * out_coeff
        out_coeff += step_size * derivative
    return out_coeff

    