# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 14:33:45 2017

@author: Administrator
"""

import json 
import pandas as pd 
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt

#   Load data  
loans = pd.read_csv('../data/lending-club-data.csv')
# Train and test idx 
with open('../data/module-8-assignment-1-train-idx.json') as json_data:
       train_idx = json.load(json_data)       
with open('../data/module-8-assignment-1-validation-idx.json') as json_data:
       validation_idx = json.load(json_data)
loans['safe_loans'] = loans['bad_loans'].apply(lambda x: 1 if x == 0 else -1)
target = 'safe_loans'
features = ['grade',                     # grade of the loan (categorical)
            'sub_grade_num',             # sub-grade of the loan as a number from 0 to 1
            'short_emp',                 # one year or less of employment
            'emp_length_num',            # number of years of employment
            'home_ownership',            # home_ownership status: own, mortgage or rent
            'dti',                       # debt to income ratio
            'purpose',                   # the purpose of the loan
            'payment_inc_ratio',         # ratio of the monthly payment to income
            'delinq_2yrs',               # number of delinquincies
             'delinq_2yrs_zero',          # no delinquincies in last 2 years
            'inq_last_6mths',            # number of creditor inquiries in last 6 months
            'last_delinq_none',          # has borrower had a delinquincy
            'last_major_derog_none',     # has borrower had 90 day or worse rating
            'open_acc',                  # number of open credit accounts
            'pub_rec',                   # number of derogatory public records
            'pub_rec_zero',              # no derogatory public records
            'revol_util',                # percent of available credit being used
            'total_rec_late_fee',        # total late fees received to day
            'int_rate',                  # interest rate of the loan
            'total_rec_int',             # interest received to date
            'annual_inc',                # annual income of borrower
            'funded_amnt',               # amount committed to the loan
            'funded_amnt_inv',           # amount committed by investors for the loan
            'installment',               # monthly payment owed by the borrower
           ]

# Extract the feature columns and target column
loans = loans[features + [target]]
safe_loans_raw = loans[loans[target] == +1]
risky_loans_raw = loans[loans[target] == -1]
print( "Number of safe loans  : %s" % len(safe_loans_raw) )
print( "Number of risky loans : %s" % len(risky_loans_raw) )
loans = loans[[target] + features].dropna()

train_data = loans.iloc[train_idx]
validation_data = loans.iloc[validation_idx]

percentage = len(risky_loans_raw)/float(len(safe_loans_raw))

risky_loans = risky_loans_raw
safe_loans = safe_loans_raw.sample(frac = percentage)
loans_data = risky_loans.append(safe_loans)
print("Percentage of safe loans                 :", len(safe_loans) / float(len(loans_data)))
print("Percentage of risky loans                :", len(risky_loans) / float(len(loans_data)))
print("Total number of loans in our new dataset :", len(loans_data))

# IMPORTANT replace NA with 0
loans_data = loans_data.fillna(0)
def oneHotEncoding(all_features, original_data):
    all_added_col = []
    for each_fea in all_features:
        if type(original_data[each_fea][1]) == str:
            all_features.remove(each_fea) # remove the hot encoded feature
            unique_labels = []
            for idx in original_data.index:
                cur_label = original_data[each_fea][idx]
                if cur_label not in unique_labels:
                    unique_labels.append(cur_label)
                    original_data[cur_label] = 0
                original_data[cur_label].set_value(idx, 1)
            all_added_col.extend(unique_labels)
    all_features.extend(all_added_col)
    print(all_features)
# one hot encode the original data and returns the added cols 
oneHotEncoding(features, loans_data)        

train = loans_data.sample(frac = 0.8)
test = loans_data.drop(train.index)


def prepareData(in_pd, in_xs, in_y):
    out_xs = (in_pd[in_xs]).as_matrix()
    out_y = (in_pd[in_y]).as_matrix()
    return out_xs, out_y
train_x, train_y = prepareData(train, features, target)
test_x, test_y = prepareData(test, features, target)

model = GradientBoostingClassifier(max_depth=6, n_estimators=5)
model.fit(train_x, train_y)

probas = model.predict_proba(test_x)
print(probas)            
            
            