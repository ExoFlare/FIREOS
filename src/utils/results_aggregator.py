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
internal_aggregated_dir = internal_validation_dir + 'aggregated/'
ireos_java_dir = internal_validation_dir + 'ireos_java/'
aggregated_dir = '/results/aggregated/'
ranking_dir = aggregated_dir + 'rankings/'


datasets = ['complex_1', 'complex_2', 'complex_3', 'complex_4', 'complex_5', 'complex_6', 'complex_7', 'complex_8', 'complex_9', 'complex_10',
	'complex_11', 'complex_12', 'complex_13', 'complex_14', 'complex_15', 'complex_16',
    'complex_17', 'complex_18', 'complex_19', 'complex_20', 'high-noise_1', 'high-noise_2', 'high-noise_3',
	'high-noise_4', 'high-noise_5', 'high-noise_6', 'high-noise_7', 'high-noise_8', 'high-noise_9', 
	'high-noise_10', 'high-noise_11', 'high-noise_12', 'high-noise_13', 
    'high-noise_14', 'high-noise_15', 'high-noise_16', 'high-noise_17', 'high-noise_18', 'high-noise_19', 'high-noise_20',
	'low-noise_1', 'low-noise_2', 'low-noise_3', 'low-noise_4', 'low-noise_5', 'low-noise_6', 'low-noise_7', 'low-noise_8', 'low-noise_9',
	'low-noise_10', 'low-noise_11', 'low-noise_12', 'low-noise_13', 'low-noise_14', 'low-noise_15', 'low-noise_16', 
    'low-noise_17', 'low-noise_18', 'low-noise_19', 'low-noise_20']

"""datasets = ['basic2d_1.csv', 'basic2d_10.csv', 'basic2d_11.csv', 'basic2d_12.csv', 'basic2d_13.csv', 'basic2d_14.csv', 'basic2d_15.csv', 'basic2d_16.csv', 
    'basic2d_17.csv', 'basic2d_18.csv', 'basic2d_19.csv', 'basic2d_2.csv', 'basic2d_20.csv', 'basic2d_3.csv', 'basic2d_4.csv', 'basic2d_5.csv', 'basic2d_6.csv', 
    'basic2d_7.csv', 'basic2d_8.csv', 'basic2d_9.csv']"""

complete_indices_df = None
complete_indices_ranking_df = None

for dataset_name in datasets:

     # read datasets
     #external_validation_indices_df = pd.read_csv(base_dir + external_validation_dir + dataset_name)
     internal_validation_indices_df = pd.read_csv(base_dir + internal_validation_dir + dataset_name + ".csv")
     ireos_java_df = pd.read_csv(base_dir + ireos_java_dir + dataset_name + ".csv").T
     ireos_java_df.columns = ['ireos_java']
     ireos_java_df.drop('time', inplace=True)
     
     all_internal_validation_indices = internal_validation_indices_df.append(ireos_java_df.T.add_prefix('ireos_').assign(dataset=dataset_name).assign(clf='ireos').assign(window_ratio=1.0))
     all_internal_validation_indices.to_csv(base_dir + internal_aggregated_dir + dataset_name + '.csv', index=False)
     
     internal_validation_selection = [('decision_tree_native', 1), ('decision_tree_sklearn', 1), ('random_forest_native', 1)
                                      , ('random_forest_sklearn', 1), ('liblinear', 1), ('xgboost_tree', 1), ('xgboost_dart', 1)
                                      , ('xgboost_linear', 1), ('libsvm', 1)]
     
     
     internal_validation_indices_df = internal_validation_indices_df[(internal_validation_indices_df['clf'].isin([x[0] for x in internal_validation_selection]))
                                    & (internal_validation_indices_df['window_ratio'].isin([x[1] for x in internal_validation_selection]))]
     
     clfs = columns=internal_validation_indices_df['clf'].values
     
     internal_validation_indices_df = pd.DataFrame(internal_validation_indices_df.iloc[:,6:].T)
     internal_validation_indices_df.index = [x.removeprefix('ireos_') for x in internal_validation_indices_df.index]
     internal_validation_indices_df.columns = clfs
     
     all_validation_indices = pd.merge(internal_validation_indices_df, ireos_java_df, left_index=True, right_index=True)
     #all_validation_indices = pd.merge(external_validation_indices_df, all_validation_indices, left_on=['Alg.'], right_index=True)
     
     all_validation_indices_ranking = all_validation_indices.copy()
     all_validation_indices_ranking.iloc[:,2:] = all_validation_indices_ranking.iloc[:,2:].rank(ascending=False)
     
     
     #mean_external_ranking = external_validation_indices_df.iloc[:,2:].rank().mean(axis=1)
     internal_ranking = all_validation_indices_ranking.iloc[:,7:]
     
     #result = spearmanr(mean_external_ranking, internal_ranking, axis=0)
     
     
     all_validation_indices.to_csv(base_dir + aggregated_dir + dataset_name + '.csv', index=False)
     all_validation_indices_ranking.to_csv(base_dir + ranking_dir + dataset_name + '.csv', index=False)
     
     complete_indices_df = pd.concat([complete_indices_df, all_validation_indices])
     complete_indices_ranking_df = pd.concat([complete_indices_ranking_df, all_validation_indices_ranking])

complete_indices_df.to_csv(base_dir + aggregated_dir + 'all_scores.csv', index=False)
complete_indices_ranking_df.to_csv(base_dir + ranking_dir + 'all_ranks.csv', index=False)