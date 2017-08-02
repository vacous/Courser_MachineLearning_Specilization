# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
def calProbProduct(in_list):
    all_prod = []
    for each in in_list:
        cur_prod = 1/(1 + np.exp(-each))
        all_prod.append(cur_prod)
    return all_prod

prob_list = calProbProduct([2.5,0.3,2.8,0.5])
out_y = [1, -1, 1, 1]


def calDeriv(in_features, in_prob, in_out):
    total = 0
    for idx in range(len(in_features)):
        cur_prob = in_prob[idx]
        cur_fea = in_features[idx]
        cur_out = in_out[idx]
        if cur_out == 1:
            sep = 1
        else:
            sep = 0
        cur_val = (sep - cur_prob) * cur_fea
        total += cur_val
    return total
print(prob_list)
print(calDeriv([2.5,0.3,2.8,0.5],
               prob_list, out_y))

plus_list = [0.92414181997875655, 1 - 0.57444251681165903, 0.94267582410113127, 0.62245933120185459]
total_prod = 1
for each in plus_list:
    total_prod *= each
print(total_prod)
