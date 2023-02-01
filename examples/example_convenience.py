from seim import convenience

from seim import prediction, analyses
import pandas as pd

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

import os

from catboost import CatBoostRegressor
import logging
import itertools
from functools import partial

# Setup base directories ########################

results_dir = "/home/eduardo.falbel/scratch/Github/bike-science/spatial_econometrics/results/"
base_dir = f"{results_dir}8nn_minimal_normalize_intra_log_distance/"
# base_dir = f"{results_dir}log_poi_census_intra_log_distance/"

coeff_dir = f"{base_dir}coefficients/"

data_dir = f"{base_dir}data/"
train_data_dir = f"{data_dir}train/"
test_data_dir = f"{data_dir}test/"

predictions_dir = f"{base_dir}predictions/"
residuals_dir = f"{base_dir}residuals/"

comparison_dir = f"{base_dir}comparison/"
graphs_dir = f"{comparison_dir}graphs/"

error_dir = f"{graphs_dir}error_observed/"
pred_dir = f"{graphs_dir}pred_observed/"
moran_dir = f"{graphs_dir}moran/"

latex_dir = f"{comparison_dir}latex_tables/"

graph_types = [error_dir, pred_dir, moran_dir]

stats_file = f"{comparison_dir}stats.txt"

os.makedirs(graphs_dir, exist_ok=True)
os.makedirs(latex_dir, exist_ok=True)

for dir in graph_types:
    try:
        os.mkdir(dir)
    except(FileExistsError):
        print(f"File already exists")

logged_flows = True

predicted = {}

# Spatial econometrics ########################

analyses.add_pred(predicted, "test", "Trend-corrected", convenience.estimate_predict("trip_counts", f"{train_data_dir}node.csv", f"{train_data_dir}pair.csv", f"{test_data_dir}node.csv", f"{test_data_dir}pair.csv", prediction.tc))