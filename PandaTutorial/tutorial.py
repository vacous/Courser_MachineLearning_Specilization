# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 08:06:36 2016

@author: Administrator
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn import  linear_model

file_resource  = "D:/NetDrive/OneDrive/Coursera/MachineLearning/Uwash-Specilization/02/week_01/data/kc_house_data.csv"
data = pd.read_csv(file_resource)
# get the header names for the data frame 
print "The header names are:"
print list(data)
model = linear_model.LinearRegression()
price = pd.DataFrame(data['price'])
area = pd.DataFrame(data['sqft_living'])
model.fit(area,price)
print model.predict(pd.DataFrame([1,2,3,4]))

