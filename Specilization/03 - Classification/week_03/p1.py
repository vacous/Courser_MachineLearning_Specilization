# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 09:47:00 2017

@author: Administrator
"""

import json 
import pandas as pd 
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

#   Load data  
loans = pd.read_csv('../data/lending-club-data.csv')
# Train and test idx 
with open('../data/module-5-assignment-1-train-idx.json') as json_data:
       train_idx = json.load(json_data)       
with open('../data/module-5-assignment-1-validation-idx.json') as json_data:
       validation_idx = json.load(json_data)
loans['safe_loans'] = loans['bad_loans'].apply(lambda x: 1 if x == 0 else -1)
features = ['grade',                     # grade of the loan
            'sub_grade',                 # sub-grade of the loan
            'short_emp',                 # one year or less of employment
            'emp_length_num',            # number of years of employment
            'home_ownership',            # home_ownership status: own, mortgage or rent
            'dti',                       # debt to income ratio
            'purpose',                   # the purpose of the loan
            'term',                      # the term of the loan
            'last_delinq_none',          # has borrower had a delinquincy
            'last_major_derog_none',     # has borrower had 90 day or worse rating
            'revol_util',                # percent of available credit being used
            'total_rec_late_fee',        # total late fees received to day
           ]

target = 'safe_loans'                    # prediction target (y) (+1 means safe, -1 is risky)

# Extract the feature columns and target column
loans = loans[features + [target]]
train_data = loans.iloc[train_idx]
validation_data = loans.iloc[validation_idx]
safe_loans_raw = loans[loans[target] == +1]
risky_loans_raw = loans[loans[target] == -1]
print( "Number of safe loans  : %s" % len(safe_loans_raw) )
print( "Number of risky loans : %s" % len(risky_loans_raw) )

percentage = len(risky_loans_raw)/float(len(safe_loans_raw))

risky_loans = risky_loans_raw
safe_loans = safe_loans_raw.sample(frac = percentage)

# Append the risky_loans with the downsampled version of safe_loans
loans_data = risky_loans.append(safe_loans)
added_col = []
for feat_name in list(loans_data):
    print(feat_name)
    if type(loans_data[feat_name][1]) == str:
#        add this name to the loans_data
        added_col.append(feat_name)

def unique(iterable):
    out = list(set(iterable))
    return out 

for each_col in added_col:
    for each_var in unique(loans_data[each_col]):
        loans_data[each_var] = 0
    for idx in loans_data.index:
        one_name = loans_data[each_col][idx]
        loans_data[one_name].set_value(idx, 1)
#        very important!!!! find the name col first then set 
train = loans_data.sample(frac = 0.8, random_state = 200)
test = loans_data.drop(train.index)

def prepareData(pd_frame, x_labels, y_label):
    out_x = np.array(pd_frame[x_labels])
    out_y = np.array(pd_frame[y_label])
    return out_x, out_y
x_labels = list(set(list(loans_data)).difference(set(added_col)))
y_label = 'safe_loans'
x_labels.remove(y_label)
train_x, train_y = prepareData(train, x_labels, y_label)
test_x, test_y = prepareData(test, x_labels, y_label)

def calAcc(raw_y, pred_y):
    counter = 0
    for idx in range(len(raw_y)):
        if raw_y[idx] == pred_y[idx]:
            counter += 1
    return counter/len(raw_y)

level_record = []
train_record = []
test_record = []

for level in range(2,30):
    print(level)
    model = DecisionTreeClassifier(max_depth= level)
    model.fit(train_x, train_y)
    train_acc = calAcc(train_y, model.predict(train_x))
    test_acc = calAcc(test_y, model.predict(test_x))
    level_record.append(level)
    train_record.append(train_acc)
    test_record.append(test_acc)
    print(train_acc)
    print(test_acc)

plt.figure(1)
plt.xlabel('Depth Level')
plt.ylabel('Acc')
plt.plot(level_record, train_record)
plt.plot(level_record, test_record)

    


        

