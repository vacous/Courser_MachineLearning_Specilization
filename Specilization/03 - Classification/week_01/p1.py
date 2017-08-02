# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 16:23:53 2017

@author: Administrator
"""
import json 
import pandas as pd 
import string 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
with open('../data/module-2-assignment-train-idx.json') as json_data:
    train_idx = json.load(json_data)

with open('../data/module-2-assignment-test-idx.json') as json_data:
    test_idx = json.load(json_data)    

def remove_punctuation(in_string):
    return in_string.translate(str.maketrans('','',string.punctuation))

data_file = pd.read_csv('../data/amazon_baby.csv')
data_file = data_file.fillna({'review':''})
data_file['review_clean'] = data_file['review'].apply(remove_punctuation)
data_file = data_file[data_file['rating'] != 3]
data_file['sentiment'] = data_file['rating'].apply(lambda rating : +1 if rating > 3 else -1)

train_data = data_file.iloc[train_idx]
test_data = data_file.iloc[test_idx]



vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
# Use this token pattern to keep single-letter words
# First, learn vocabulary from the training data and assign columns to words
# Then convert the training data into a sparse matrix
train_matrix = vectorizer.fit_transform(train_data['review_clean'])
# Second, convert the test data into a sparse matrix, using the same word-column mapping
test_matrix = vectorizer.transform(test_data['review_clean'])

train_y = train_data['sentiment']
test_y = test_data['sentiment']

model = LogisticRegression()
model.fit(train_matrix, train_y)
print('Positive Coefficients: ', len([each for each in model.coef_[0] if each >= 0]))

sample_test_data = test_data[10:13]
print(sample_test_data)

sample_test_matrix = vectorizer.transform(sample_test_data['review_clean'])
scores = model.decision_function(sample_test_matrix)
print(scores)

def calProbability(in_score):
    return 1/(1 + np.exp(-in_score))

all_test_scores = model.decision_function(test_matrix)
all_test_probs = calProbability(all_test_scores)
sorted_idxs = np.argsort(all_test_probs)
largest_idxs = sorted_idxs[-20:]
smalles_idxs = sorted_idxs[:19]

def calModelAcc(in_model, in_x_matrix, in_y):
    predictions = in_model.predict(in_x_matrix)
    model_acc = 1 - sum( np.abs(predictions - in_y)/2 )/len(in_y)
    return model_acc

train_acc = calModelAcc(model, train_matrix, train_y)
test_acc = calModelAcc(model, test_matrix, test_y)

significant_words = ['love', 'great', 'easy', 'old', 'little', 'perfect', 'loves', 
      'well', 'able', 'car', 'broke', 'less', 'even', 'waste', 'disappointed', 
      'work', 'product', 'money', 'would', 'return']
vectorizer_word_subset = CountVectorizer(vocabulary=significant_words) # limit to 20 words
train_matrix_word_subset = vectorizer_word_subset.fit_transform(train_data['review_clean'])
test_matrix_word_subset = vectorizer_word_subset.transform(test_data['review_clean'])
simple_model = LogisticRegression()
simple_model.fit(train_matrix_word_subset, train_y)
simple_acc_train = calModelAcc(simple_model, train_matrix_word_subset, train_y)
simple_acc_test = calModelAcc(simple_model, test_matrix_word_subset, test_y)