# -*- coding: utf-8 -*-
"""
Created on Wed Feb 01 19:39:35 2017

@author: Administrator
"""

import pandas as pd
import datetime 
import pandas.io.data as web 
import matplotlib.pyplot as plt 
from matplotlib import style
import numpy as np
from sklearn import linear_model

file_address = 'practice_data/data/kc_house_data.csv'
raw_data = pd.read_csv(file_address)
raw_data = raw_data.set_index('id')
price = np.matrix(raw_data['price'])
sqft_living = np.matrix(raw_data['sqft_living'])
test_model = linear_model.LinearRegression()
test_model.fit(np.transpose(sqft_living),np.transpose(price))
print(test_model.coef_)
print(test_model.intercept_)