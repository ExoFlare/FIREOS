# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 21:35:04 2022

@author: ExoFlare
"""
import os
import pandas as pd
from scipy.stats import spearmanr

base_dir = os.getcwd()
external_validation_dir = '/results/external_validation_indices/'
internal_validation_dir = '/results/internal_validation_indices/'
ireos_java_dir = internal_validation_dir + 'ireos_java/'
aggregated_dir = '/results/aggregated/'
ranking_dir = aggregated_dir + 'rankings/'


datasets = ['complex_1', 'complex_2', 'complex_3', 'complex_4', 'complex_5', 'complex_6', 'complex_17', 'complex_18', 'complex_19', 'complex_10',
	'complex_11', 'complex_12', 'complex_13', 'complex_14', 'complex_15', 'complex_16',
    'complex_20']

complete_indices_df = None
complete_indices_ranking_df = None

for dataset_name in datasets:

     external_validation_indices_df = pd.read_csv(base_dir + external_validation_dir + dataset_name + '.csv')
     internal_validation_indices_df = pd.read_csv(base_dir + internal_validation_dir + dataset_name + '.csv')
     ireos_java_df = pd.read_csv(base_dir + ireos_java_dir + dataset_name + '.csv').T
     ireos_java_df.columns = ['ireos_java']
     
     internal_validation_selection = [('decision_tree_native', 1), ('decision_tree_sklearn', 1), ('random_forest_native', 1)
                                      , ('random_forest_sklearn', 1), ('liblinear', 1), ('xgboost_tree', 1), ('xgboost_dart', 1)
                                      , ('xgboost_linear', 1), ('libsvm', 1)]
     
     
     internal_validation_indices_df = internal_validation_indices_df[(internal_validation_indices_df['clf'].isin([x[0] for x in internal_validation_selection]))
                                    & (internal_validation_indices_df['window_ratio'].isin([x[1] for x in internal_validation_selection]))]
     
     clfs = columns=internal_validation_indices_df['clf'].values
     
     internal_validation_indices_df = pd.DataFrame(internal_validation_indices_df.iloc[:,6:].T)
     internal_validation_indices_df.index = [x.removeprefix('ireos_') for x in internal_validation_indices_df.index]
     internal_validation_indices_df.columns = clfs
     
     all_validation_indices = pd.merge(external_validation_indices_df, ireos_java_df, left_on=['Alg.'], right_index=True)
     all_validation_indices = pd.merge(all_validation_indices, internal_validation_indices_df, left_on=['Alg.'], right_index=True)
     
     all_validation_indices_ranking = all_validation_indices.copy()
     all_validation_indices_ranking.iloc[:,2:] = all_validation_indices_ranking.iloc[:,2:].rank(ascending=False)
     
     
     mean_external_ranking = external_validation_indices_df.iloc[:,2:].rank().mean(axis=1)
     internal_ranking = all_validation_indices_ranking.iloc[:,7:]
     
     result = spearmanr(mean_external_ranking, internal_ranking, axis=0)
     
     all_validation_indices.to_csv(base_dir + aggregated_dir + dataset_name + '.csv', index=False)
     all_validation_indices_ranking.to_csv(base_dir + ranking_dir + dataset_name + '.csv', index=False)
     
     complete_indices_df = pd.concat([complete_indices_df, all_validation_indices])
     complete_indices_ranking_df = pd.concat([complete_indices_ranking_df, all_validation_indices_ranking])

complete_indices_df.to_csv(base_dir + aggregated_dir + 'all_scores.csv', index=False)
complete_indices_ranking_df.to_csv(base_dir + ranking_dir + 'all_ranks.csv', index=False)