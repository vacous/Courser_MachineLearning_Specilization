# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 17:50:07 2017

@author: Zhaoxi Zhang
"""
import numpy as np
import pandas as pd 
from sklearn import linear_model
import matplotlib.pyplot as plt

## helper functions 
def readData(input_file_address, input_headers, output_header):
    '''
    combine the col with the name in input_headers
    return ([input1, input2, input3], output)
    '''
    all_data = pd.read_csv(input_file_address)
    input_data_matrix = [np.transpose(np.matrix(all_data[header])) 
                                        for header in input_headers]
    input_data_features = np.hstack(input_data_matrix)
    print(np.shape(input_data_features))
    constant_col = np.matrix(np.ones((np.shape(input_data_features)[0],1)))
    input_data = np.hstack((input_data_features,constant_col))
    output_data = np.transpose(np.matrix(all_data[output_header]))
    return (input_data, output_data)

predict_outcome = lambda input_data, weight: np.dot(input_data, weight)


def feature_derivative(errors, feature):
    '''
    return the gradient of the cost with respect to weight 
    '''
    return -2 * np.dot(np.transpose(feature), errors)


def regression_gradient_descent(feature_matrix, output, initial_weight, step_size, tolerance):
    '''
    calculate the sum square of the (weight(t+1) - weight(t))^2
    if within the tolerance: stop the loop
    '''
    counter = 0
    weight = np.copy(initial_weight)
    current_tolerance = float('inf')
    while current_tolerance > tolerance:
        current_predict = predict_outcome(feature_matrix, weight)
        current_error = output - current_predict
        weight_derivative = feature_derivative(feature_matrix, current_error)
        weight -= np.transpose(weight_derivative) * step_size
        current_tolerance = np.power(sum([weight_derivative[0, idx]**2 for idx in range(len(weight_derivative))]),0.5)
        counter += 1
    return weight


 
(input_features, output) = readData('data/kc_house_test_data.csv', ['sqft_living'], 'price')
(input_features_2, output_2) = readData('data/kc_house_test_data.csv', ['sqft_living','sqft_living15'], 'price')

# self-made gd
# model 1 
ini_weight = np.transpose(np.matrix(([1.,-47000.])))
decent_weight_1 = regression_gradient_descent(input_features, output, ini_weight, 7e-12, 2.5e7)
# plot to verify 
plt.scatter(input_features[:,0], output)
plt.plot(input_features[:,0], predict_outcome(input_features, decent_weight_1))
# model 2 
ini_weight_2 = np.transpose(np.matrix([1., 1, -100000]))
decent_weight_2 = regression_gradient_descent(input_features_2, output_2, ini_weight_2, 4e-12, 1e9)

predict_1_gd = predict_outcome(input_features, decent_weight_1)
predict_2_gd = predict_outcome(input_features_2, decent_weight_2)


# sklearn 
model = linear_model.LinearRegression()
model.fit(input_features, output)
print(str(model.coef_)+ " " + str(model.intercept_))
print(str(model.predict(input_features[0,])))

model2 = linear_model.LinearRegression()
model2.fit(input_features_2, output_2)
print(str(model2.coef_)+ " " + str(model2.intercept_))
print(model2.predict(input_features_2[0,]))

predict_01 = model.predict(input_features)
predict_02 = model2.predict(input_features_2)
print(sum([ (output[idx] - predict_01[idx])**2 for idx in range(len(output))]))
print(sum([ (output_2[idx] - predict_02[idx])**2 for idx in range(len(output_2))]))
