# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 14:58:43 2017

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

train_data = loans.iloc[train_idx]
validation_data = loans.iloc[validation_idx]

loans['safe_loans'] = loans['bad_loans'].apply(lambda x: 1 if x == 0 else -1)
target = 'safe_loans'
safe_loans_raw = loans[loans[target] == 1]
risky_loans_raw = loans[loans[target] == -1]

# Since there are less risky loans than safe loans, find the ratio of the sizes
# and use that percentage to undersample the safe loans.
percentage = len(risky_loans_raw)/float(len(safe_loans_raw))
safe_loans = safe_loans_raw.sample(frac = percentage)
risky_loans = risky_loans_raw
loans_data = risky_loans.append(safe_loans)

print("Percentage of safe loans                 :", len(safe_loans) / float(len(loans_data)))
print("Percentage of risky loans                :", len(risky_loans) / float(len(loans_data)))
print("Total number of loans in our new dataset :", len(loans_data))

added_col = ['grade',              # grade of the loan
            'term',               # the term of the loan
            'home_ownership',     # home_ownership status: own, mortgage or rent
            'emp_length']

def unique(iterable):
    out = list(set(iterable))
    return out 

fea_name = []
for each_col in added_col:
    print(each_col)
    for each_var in unique(loans_data[each_col]):
        loans_data[each_var] = 0
        fea_name.append(each_var)
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

train_x, train_y = prepareData(train, fea_name, target)
test_x, test_y = prepareData(test, fea_name, target)

# Helper functions 
def calAcc(raw_y, pred_y):
    counter = 0
    for idx in range(len(raw_y)):
        if raw_y[idx] == pred_y[idx]:
            counter += 1
    return counter/len(raw_y)

def reached_minimum_node_size(data, min_node_size):
    # Return True if the number of data points is less than or equal to the minimum node size.
    return len(data) <= min_node_size

def error_reduction(error_before_split, error_after_split):
    return error_after_split - error_before_split
    
def count_leaves(tree):
    if tree['is_leaf']:
        return 1
    return count_leaves(tree['left']) + count_leaves(tree['right'])

def intermediate_node_num_mistakes(labels_in_node):
    # Corner case: If labels_in_node is empty, return 0
    if len(labels_in_node) == 0:
        return 0    
    count_record = []
    for each in np.unique(labels_in_node):
        cur_count = sum(labels_in_node == each) 
        count_record.append(cur_count)
    return min(count_record)

def best_splitting_feature(data, features, target):
    
    target_values = data[target]
    best_feature = None # Keep track of the best feature 
    best_error = float('inf')     # Keep track of the best error so far 
    # Note: Since error is always <= 1, we should intialize it with something larger than 1.

    # Convert to float to make sure error gets computed correctly.
    num_data_points = float(len(data))  
    
    # Loop through each feature to consider splitting on that feature
    for feature in features:
        # Calculate the number of misclassified examples in the left split.
        # Remember that we implemented a function for this! (It was called intermediate_node_num_mistakes)
        # YOUR CODE HERE
        left_idx = data[feature] == 0
        left_mistakes = intermediate_node_num_mistakes(target_values[left_idx])            

        # Calculate the number of misclassified examples in the right split.
        ## YOUR CODE HERE
        right_idx = data[feature] == 1
        right_mistakes = intermediate_node_num_mistakes(target_values[right_idx])
            
        # Compute the classification error of this split.
        # Error = (# of mistakes (left) + # of mistakes (right)) / (# of data points)
        ## YOUR CODE HERE
        error = (right_mistakes + left_mistakes)/num_data_points

        # If this is the best error we have found so far, store the feature as best_feature and the error as best_error
        ## YOUR CODE HERE
        if error < best_error:
            best_error = error
            best_feature = feature
    
    return best_feature # Return the best feature we found


def create_leaf(target_values):    
    # Create a leaf node
    leaf = {'splitting_feature' : None,
            'left' : None,
            'right' : None,
            'is_leaf': True }   ## YOUR CODE HERE 
   
    # Count the number of data points that are +1 and -1 in this node.
    num_ones = len(target_values[target_values == +1])
    num_minus_ones = len(target_values[target_values == -1])    

    # For the leaf node, set the prediction to be the majority class.
    # Store the predicted class (1 or -1) in leaf['prediction']
    if num_ones > num_minus_ones:
        leaf['prediction'] = 1          ## YOUR CODE HERE
    else:
        leaf['prediction'] = -1        ## YOUR CODE HERE        

    # Return the leaf node
    return leaf 

def decision_tree_create(data, features, target, current_depth = 0, max_depth = 10):
    remaining_features = features[:] # Make a copy of the features.
    
    target_values = data[target]
    print("--------------------------------------------------------------------")
    print("Subtree, depth = %s (%s data points)." % (current_depth, len(target_values)))
    
    splitting_feature = best_splitting_feature(data, features, target)
    # Stopping condition 1
    # (Check if there are mistakes at current node.
    # Recall you wrote a function intermediate_node_num_mistakes to compute this.)
    split_fea_idx = data[splitting_feature] == 1
    if  intermediate_node_num_mistakes(target_values[split_fea_idx])== 0:  ## YOUR CODE HERE
        print ("Stopping condition 1 reached.")
        # If not mistakes at current node, make current node a leaf node
        return create_leaf(target_values)
    
    # Stopping condition 2 (check if there are remaining features to consider splitting on)
    if len(remaining_features) == 1:   ## YOUR CODE HERE
        print("Stopping condition 2 reached.")
        # If there are no remaining features to consider, make current node a leaf node
        return create_leaf(target_values)    
    
    # Additional stopping condition (limit tree depth)
    if current_depth >= max_depth:  ## YOUR CODE HERE
        print("Reached maximum depth. Stopping for now.")
        # If the max tree depth has been reached, make current node a leaf node
        return create_leaf(target_values)

    # Find the best splitting feature (recall the function best_splitting_feature implemented above)
    ## YOUR CODE HERE

    
    # Split on the best feature that we found. 
    left_split = data[data[splitting_feature] == 0]
    right_split = data[data[splitting_feature] == 1]      ## YOUR CODE HERE
    remaining_features.remove(splitting_feature)
    print("Split on feature %s. (%s, %s)" % (\
                      splitting_feature, len(left_split), len(right_split)))
    
    # Create a leaf node if the split is "perfect"
    if len(left_split) == len(data):
        print("Creating leaf node.")
        return create_leaf(left_split[target])
    if len(right_split) == len(data):
        print("Creating leaf node.")
        ## YOUR CODE HERE

        
    # Repeat (recurse) on left and right subtrees
    left_tree = decision_tree_create(left_split, remaining_features, target, current_depth + 1, max_depth)        
    ## YOUR CODE HERE
    right_tree = decision_tree_create(right_split, remaining_features, target, current_depth + 1, max_depth)        

    return {'is_leaf'          : False, 
            'prediction'       : None,
            'splitting_feature': splitting_feature,
            'left'             : left_tree, 
            'right'            : right_tree}


def classify(tree, x, annotate = False):
       # if the node is a leaf node.
    if tree['is_leaf']:
        if annotate:
             print("At leaf, predicting %s" % tree['prediction'])
        return tree['prediction']
    else:
        # split on feature.
        split_feature_value = x[tree['splitting_feature']]
        if annotate:
             print ("Split on %s = %s" % (tree['splitting_feature'], split_feature_value))
        if split_feature_value == 0:
            return classify(tree['left'], x, annotate)
        else:
            return classify(tree['right'], x, annotate)