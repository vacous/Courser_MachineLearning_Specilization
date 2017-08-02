# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 17:04:33 2017

@author: Administrator
"""

import json 
import pandas as pd 
import numpy as np

#   Load data  
loans = pd.read_csv('../data/lending-club-data.csv')
# Train and test idx 
with open('../data/module-8-assignment-2-train-idx.json') as json_data:
       train_idx = json.load(json_data)       
with open('../data/module-8-assignment-2-test-idx.json') as json_data:
       validation_idx = json.load(json_data)
loans['safe_loans'] = loans['bad_loans'].apply(lambda x: 1 if x == 0 else -1)
target = 'safe_loans'
features = ['grade',              # grade of the loan
            'term',               # the term of the loan
            'home_ownership',     # home ownership status: own, mortgage or rent
            'emp_length'
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
    to_remove = []
    for each_fea in all_features:
        print(each_fea, original_data[each_fea][1])
        if type(original_data[each_fea][1]) == str:
            to_remove.append(each_fea)
            unique_labels = []
            for idx in original_data.index:
                cur_label = original_data[each_fea][idx]
                if cur_label not in unique_labels:
                    unique_labels.append(cur_label)
                    original_data[cur_label] = 0
                original_data[cur_label].set_value(idx, 1)
            all_added_col.extend(unique_labels)
    for each_fea in to_remove:
        all_features.remove(each_fea)
    all_features.extend(all_added_col)
    print('feature len: ', len(all_features))
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

def calWeightedError(raw_y, pred_y, weight_array):
    same_y = (raw_y == pred_y)
    same_weight = np.multiply(same_y, weight_array)
    weight_sum = np.sum(same_weight)
    total_weight = np.sum(weight_array)
    return 1 - weight_sum/total_weight

def intermediateNodeWeightMistakes(labels_in_node, data_weight):
    total_weight_positive = np.sum(data_weight[labels_in_node == 1])
    total_weight_negative = np.sum(data_weight[labels_in_node == -1])
    if total_weight_positive <= total_weight_negative:
        return total_weight_positive, 1
    else:
        return total_weight_negative, -1
    
def bestSplit(data_pd, feature_list, target_str, data_weight_np):
    '''
    choose the feature to split by the weighted error 
    '''
    min_we = float('inf')
    best_fea = None
    for each_fea in feature_list:
        left_idx = data_pd[each_fea] == 0
        right_idx = data_pd[each_fea] == 1
        
        left_labels = data_pd[target_str][left_idx]
        right_labels = data_pd[target_str][right_idx]
        
        left_weight = data_weight_np[left_idx]
        right_weight = data_weight_np[right_idx]
        
        left_wm, left_class = intermediateNodeWeightMistakes(left_labels, left_weight)
        right_wm, right_class = intermediateNodeWeightMistakes(right_labels, right_weight)
        
        all_we = (left_wm + right_wm)/np.sum(data_weight_np)
        
        if all_we < min_we:
            min_we = all_we
            best_fea = each_fea
    return best_fea
        
def create_leaf(target_values, data_weights):    
    # Create a leaf node
    leaf = {'splitting_feature' : None,
            'is_leaf': True}
    weighted_error, best_class = intermediateNodeWeightMistakes(target_values, data_weights)
    leaf['prediction'] = best_class
    return leaf        

def weighted_decision_tree_create(data, features, target, data_weights, current_depth = 1, max_depth = 10):
    remaining_features = features[:] # Make a copy of the features.
    target_values = data[target]
    print( "--------------------------------------------------------------------")
    print( "Subtree, depth = %s (%s data points)." % (current_depth, len(target_values)))
    # Stopping condition 1. Error is 0.
    if intermediateNodeWeightMistakes(target_values, data_weights)[0] <= 1e-15:
        print("Stopping condition 1 reached.")
        return create_leaf(target_values, data_weights)
    
    # Stopping condition 2. No more features.
    if remaining_features == []:
        print( "Stopping condition 2 reached.")
        return create_leaf(target_values, data_weights)    
    # Additional stopping condition (limit tree depth)
    if current_depth > max_depth:
        print( "Reached maximum depth. Stopping for now.")
        return create_leaf(target_values, data_weights)
    
    # If all the datapoints are the same, splitting_feature will be None. Create a leaf
    splitting_feature = bestSplit(data, features, target, data_weights)
    remaining_features.remove(splitting_feature)
        
    left_split = data[data[splitting_feature] == 0]
    right_split = data[data[splitting_feature] == 1]
    
    left_data_weights = data_weights[data[splitting_feature] == 0]
    right_data_weights = data_weights[data[splitting_feature] == 1]
    
    print( "Split on feature %s. (%s, %s)" % (\
              splitting_feature, len(left_split), len(right_split)))
    
    # Create a leaf node if the split is "perfect"
    if len(left_split) == len(data):
        print( "Creating leaf node.")
        return create_leaf(left_split[target], data_weights)
    if len(right_split) == len(data):
        print( "Creating leaf node.")
        return create_leaf(right_split[target], data_weights)
    
    # Repeat (recurse) on left and right subtrees
    left_tree = weighted_decision_tree_create(
        left_split, remaining_features, target, left_data_weights, current_depth + 1, max_depth)
    right_tree = weighted_decision_tree_create(
        right_split, remaining_features, target, right_data_weights, current_depth + 1, max_depth)
    
    return {'is_leaf'          : False, 
            'prediction'       : None,
            'splitting_feature': splitting_feature,
            'left'             : left_tree, 
            'right'            : right_tree}

def count_nodes(tree):
    if tree['is_leaf']:
        return 1
    return 1 + count_nodes(tree['left']) + count_nodes(tree['right'])    

def classify(tree, x, annotate = False):   
    # If the node is a leaf node.
    if tree['is_leaf']:
        if annotate: 
            print( "At leaf, predicting %s" % tree['prediction'])
        return tree['prediction'] 
    else:
        # Split on feature.
        split_feature_value = x[tree['splitting_feature']]
        if annotate: 
            print( "Split on %s = %s" % (tree['splitting_feature'], split_feature_value))
        if split_feature_value == 0:
            return classify(tree['left'], x, annotate)
        else:
            return classify(tree['right'], x, annotate)

def adaboostWithTreeStumps(data_pd, features_list, target_str, num_tree_stumps):
    weights = []
    tree_stumps = []
    y = data_pd[target_str]
    #    intialize weight 
    point_weight = np.ones_like(y)
    for tree_idx in range(num_tree_stumps):
        cur_tree = weighted_decision_tree_create(data_pd, features_list, target_str, point_weight)
        tree_stumps.append(cur_tree)
        cur_pred = data_pd.apply(lambda x: classify(cur_tree, x))
        cur_we = calWeightedError(y, cur_pred, point_weight)
        cur_model_coefficient = 1/2 * np.log((1-cur_we)/cur_we)
        weights.append(cur_model_coefficient)
        
        is_correct = cur_pred == y
        adjustment = is_correct.apply(lambda is_correct: np.exp(-cur_model_coefficient) if is_correct else np.exp(cur_model_coefficient))
        point_weight = np.multiply(point_weight, adjustment)
#        normalize 
        point_weight = point_weight/np.sum(point_weight)
    return weights, tree_stumps

def predictAdaboost(stump_weights, tree_stumps, in_xs):
    out_len = in_xs.shape[0]
    predictions = np.zeros(out_len)
    for idx in range(out_len):
        x = in_xs[idx,:]     
        total = 0
        for idx in range(len(stump_weights)):
            total += stump_weights * classify(tree_stumps, x)
        predictions[idx] = np.sign(total)
    return predictions
    

        



















