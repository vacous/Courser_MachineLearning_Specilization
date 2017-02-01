# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 23:12:19 2017

@author: Administrator
"""

import pandas as pd 
import numpy as np
from sklearn import linear_model

file_source = "data/kc_house_train_data.csv"
data = pd.read_csv(file_source)
print(data['id'])