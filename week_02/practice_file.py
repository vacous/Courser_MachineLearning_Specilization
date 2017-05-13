# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 01:30:07 2017

@author: Administrator
"""
import pandas as pd 
import numpy as np 
from sklearn import linear_model
import matplotlib.pyplot as plt
#
#file_address = "data/kc_house_data.csv"
#read_in_file = pd.read_csv(file_address)
##plt.scatter(read_in_file['bedrooms'],read_in_file['price'],linewidths = 0.001,marker=".")
#test_model = linear_model.LinearRegression()
#bed_room_num = np.transpose(np.matrix(read_in_file['bedrooms']))
#price = np.transpose(np.matrix(read_in_file['price']))
#test_model.fit(bed_room_num,price)
#print(test_model.coef_)
#print(test_model.intercept_)

a = np.matrix([1,2,3])
b= np.matrix([3,4,5])
d = np.hstack([a,b])
print(d)