# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 15:47:41 2022

@author: ExoFlare
"""

import os
import pandas as pd

base_dir = os.getcwd()

fireos_dir = '/results/internal_validation_indices/'
ireos_java_dir = fireos_dir + 'ireos_java/'
timings_dir = 'timings/'

result_file_name = 'complex-high_low-noise-timings.csv'

datasets = ['basic2d_1.csv', 'basic2d_10.csv', 'basic2d_11.csv', 'basic2d_12.csv', 'basic2d_13.csv', 'basic2d_14.csv', 'basic2d_15.csv', 'basic2d_16.csv', 
    'basic2d_17.csv', 'basic2d_18.csv', 'basic2d_19.csv', 'basic2d_2.csv', 'basic2d_20.csv', 'basic2d_3.csv', 'basic2d_4.csv', 'basic2d_5.csv', 'basic2d_6.csv', 
    'basic2d_7.csv', 'basic2d_8.csv', 'basic2d_9.csv']

datasets = ["complex_1.csv", "complex_2.csv", "complex_3.csv", "complex_4.csv", "complex_5.csv", "complex_6.csv", "complex_7.csv", "complex_8.csv", "complex_9.csv", "complex_10.csv",
	"complex_11.csv", "complex_12.csv", "complex_13.csv", "complex_14.csv", "complex_15.csv", "complex_16.csv",
    "complex_17.csv", "complex_18.csv", "complex_19.csv", "complex_20.csv", "high-noise_1.csv", "high-noise_2.csv", "high-noise_3.csv",
	"high-noise_4.csv", "high-noise_5.csv", "high-noise_6.csv", "high-noise_7.csv", "high-noise_8.csv", "high-noise_9.csv", 
	"high-noise_10.csv", "high-noise_11.csv", "high-noise_12.csv", "high-noise_13.csv", 
    "high-noise_14.csv", "high-noise_15.csv", "high-noise_16.csv", "high-noise_17.csv", "high-noise_18.csv", "high-noise_19.csv", "high-noise_20.csv",
	"low-noise_1.csv", "low-noise_2.csv", "low-noise_3.csv", "low-noise_4.csv", "low-noise_5.csv", "low-noise_6.csv", "low-noise_7.csv", "low-noise_8.csv", "low-noise_9.csv",
	"low-noise_10.csv", "low-noise_11.csv", "low-noise_12.csv", "low-noise_13.csv", "low-noise_14.csv", "low-noise_15.csv", "low-noise_16.csv", 
    "low-noise_17.csv", "low-noise_18.csv", "low-noise_19.csv", "low-noise_20.csv"]

times = pd.DataFrame()
for dataset_name in datasets:
     
     fireos_df = pd.read_csv(base_dir + fireos_dir + dataset_name)
     
     clfs = fireos_df['clf'].map(str) + '-' + fireos_df['window_ratio'].map(str) + "-" + ["sequential" if x == False else 'parallel' for x in fireos_df['is_parallel']]
     
     row = pd.DataFrame(fireos_df['time']).T
     row.columns=clfs
     
     ireos_java_df = pd.read_csv(base_dir + ireos_java_dir + dataset_name)
     row['ireos_java'] = ireos_java_df['time'].values
     row.index = [dataset_name]
     
     times = times.append(row)
     
times.index.names = ['Dataset']
times.to_csv(base_dir + fireos_dir + timings_dir + result_file_name)