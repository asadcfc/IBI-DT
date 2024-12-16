#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import json
import os
import shutil
import scipy.stats as stats
from scipy.stats import fisher_exact
import scipy
from datetime import datetime
import math
import time
from joblib import Parallel, delayed
import math
import warnings
from ast import literal_eval
import itertools

# Import Trees (Note: these trees generated from IBI-DT)
Tree_df = pd.read_csv('IBIDT_trees.csv', index_col=0)

# Single Node Analysis

UniqueTrees = list(Tree_df['Tree'].unique())

X = []
# Finding unique best nodes for each of the trees 
for i in UniqueTrees:
    Tree_df_temp = Tree_df.copy()
    Tree_df_temp = Tree_df_temp[(Tree_df_temp['Node']!='0') & (Tree_df_temp['Tree']==i)]
    y = list(Tree_df_temp['Node'].unique())
    X.append(y)

# Calculate 'Number of appearances as the best estimators' statistic
merged = list(itertools.chain.from_iterable(X))
values, counts = np.unique(merged, return_counts=True)
final = [values, counts]
SingleNode_df = pd.DataFrame(final)
SingleNode_df = SingleNode_df.T
SingleNode_df.columns = ['SGA', 'Number of appearances as the best estimators']
SingleNode_df = SingleNode_df.set_index('SGA')
SingleNode_df = SingleNode_df.sort_values('Number of appearances as the best estimators', ascending=False)
SingleNode_df['Number of appearances as the best estimators'] = SingleNode_df['Number of appearances as the best estimators']
SingleNode_df['Rank of number of appearances as the best estimators'] = SingleNode_df['Number of appearances as the best estimators'].rank(ascending=False)
SingleNode_df = SingleNode_df.round({'Rank of number of appearances as the best estimators': 0})

# Save Single Node analysis as csv file 
SingleNode_df.to_csv('SingleNode_analysis.csv')

# Pathway Pair Analysis

# Create temporary dataframe 
Tree_df_temp = Tree_df.copy()
Tree_df_temp = Tree_df_temp[Tree_df_temp['Node']!='0']
unique_list = Tree_df_temp['Node'].unique().tolist()

# Create matrix d for finding pathway pairs, here d is 2m x m, and m represents unique node
all_ = []
for i in unique_list:
    _i0 = i + '_0'
    all_.append(_i0)
    _i1 = i + '_1'
    all_.append(_i1)
d = pd.DataFrame(0, index=all_, columns=unique_list)

# Computes the 'Number of appearances as partners in the same branch'
UniqueTrees = Tree_df['Tree'].unique().tolist()
for k in UniqueTrees:
	remove_double = []
    Tree_df_temp = Tree_df[Tree_df['Tree']==k]
    Tree_df_temp = Tree_df_temp[Tree_df_temp['Node']!='0']
    z = np.array(Tree_df_temp)
    all_leaf_node = []
    for i in np.arange(len(z)):
        if z[i][1] != '0':
            all_leaf_node.append(i)
    final = []
    for line_number in all_leaf_node[1:]:
        _path = []
        current = z[line_number]
        current_position = z[line_number][2]
        current_gene = z[line_number][1]
        direction = z[line_number][3]
        next_search = z[line_number][2] - 1
        next_search_index = line_number
        _numbers = list(np.arange(next_search_index))
        _numbers.reverse()
        for i in _numbers:
            if z[i][1] != '0' and z[i][2] == next_search:
                prev_gene = z[i][1]
                prev_gene_dir = prev_gene + '_' + str(direction)
                current_gene_0 = current_gene + '_0'
                current_gene_1 = current_gene + '_1'
                par_child = prev_gene_dir + '___' + current_gene
                if par_child not in remove_double:
                	d.loc[prev_gene_dir,current_gene] = d.loc[prev_gene_dir,current_gene] + 1
                	remove_double.append(par_child)
                next_search = next_search - 1
                direction = z[i][3]
                
# Function to extract and format the pair data from the pathway pair (d) matrix 
def extractPairs(gene):
    direction = int(gene[-1])
    gene_name = gene[:-2]
    x = d.loc[[gene]].T
    x_filtered = x[x[gene] != 0]
    final = [
        [f"{gene}___{index}", value, direction, gene_name, index]
        for index, value in x_filtered[gene].items()
    ]
    return final
element_run = Parallel(n_jobs=-1)(delayed(extractPairs)(var) for var in d.index.tolist()[:])
element_run2 = list(itertools.chain.from_iterable(element_run))
PathwayPair_df = pd.DataFrame(element_run2)
PathwayPair_df.columns = ['Gene_pair', 'Number of appearances as partners in the same branch', 'Branch', 'Start_gene', 'Finish_gene']
PathwayPair_df['Rank of number of appearances as partners in the same branch'] = PathwayPair_df['Number of appearances as partners in the same branch'].rank(ascending=False)
PathwayPair_df = PathwayPair_df.round({'Rank of number of appearances as partners in the same branch': 0})
PathwayPair_df = PathwayPair_df.set_index('Gene_pair')
PathwayPair_df = PathwayPair_df.sort_values('Number of appearances as partners in the same branch', ascending=False)
PathwayPair_df = PathwayPair_df[['Number of appearances as partners in the same branch', 'Rank of number of appearances as partners in the same branch', 'Branch',
       'Start_gene', 'Finish_gene']]

# Save Pathway Pair analysis as csv file 
PathwayPair_df.to_csv('PathwayPair_analysis.csv')
