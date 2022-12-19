from seim import prediction, analyses
import pandas as pd

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

import os

from catboost import CatBoostRegressor

import itertools


data_dir = "/mnt/c/Users/ebobrow/Downloads/data/"

node_path = f"{data_dir}shapefile/node_data.shp"
pair_path = f"{data_dir}pair_data.csv"
temp_data_dir = "/tmp/"

# Setup base directories ########################

results_dir = "/home/eduardo.falbel/scratch/Github/bike-science/spatial_econometrics/results/"
base_dir = f"{results_dir}8nn_minimal_normalize_intra_log_distance/"

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

train_node_path = f"{train_data_dir}node.csv"
train_pre_path = f"{train_data_dir}pair.csv"

test_node_path = f"{test_data_dir}node.csv"
test_pre_path = f"{test_data_dir}pair.csv"

predicted = {}

flows_MSP = prediction.multi_stage_predict(prediction.tc, prediction.tc, "trip_counts", train_node_path, train_pre_path, test_node_path, test_pre_path, temp_data_dir)

analyses.add_pred(predicted, "test", "Trend-corrected", flows_MSP) 

###############################

def transform_predicted(predicted, transformation):
    transformed = {}
    for pred_type, y_true in zip(predicted, [y_train, y_test]):
        transformed[pred_type] = {}
        for model in predicted[pred_type]:
            print(f"{model}")
            transformed[pred_type][model] = transformation(predicted[pred_type][model], y_true)
    return transformed

residuals = transform_predicted(predicted, lambda x, y: y.to_numpy().flatten() - x)

weights_list = [np.loadtxt(f"{directory}weights.txt") for directory in [train_data_dir, test_data_dir]]

analyses.moran_plot(residuals, weights=weights_list, moran_dir=moran_dir)

logged_flows = True

if logged_flows:
    y_train, y_test = np.exp(y_train) - 1, np.exp(y_test) - 1

    for pred_type, model_pred in predicted.items():
        for model, pred in model_pred.items():
            predicted[pred_type][model] = np.exp(pred) - 1
            # print(predicted[pred_type][model])

errors = transform_predicted(predicted, lambda x, y: mean_squared_error(y, x, squared=False))

model_names = set(itertools.chain.from_iterable(predicted.values()))


col_names = {model : [errors[pred_type][model] if model in errors[pred_type] else None for pred_type in errors.keys()] for model in model_names}

headers = ([""] + list(errors.keys()))

analyses.RMSE_table(col_names, headers, stats_file, latex_dir)

analyses.analysis_plots(predicted, [y_train, y_test], line_45=True, graph_dir=pred_dir, y_name="predicted flows")

diff = {}
for pred_type, y_true in zip(predicted, [y_train, y_test]):
    diff[pred_type] = {}
    for model in predicted[pred_type]:
        diff[pred_type][model] = y_true - predicted[pred_type][model]

analyses.analysis_plots(diff, [y_train, y_test], line_45=True, graph_dir=error_dir, y_name="residuals")

quantiles = {}

for pred_type, y_true in zip(predicted, [y_train, y_test]):
    quantiles[pred_type] = {}
    for model in predicted[pred_type]:
        print(type(predicted[pred_type][model]))
        quantiles[pred_type][model] = analyses.separate_into_tiers(pd.DataFrame(predicted[pred_type][model]).rename({0 : "trip_counts"}, axis=1))

y_train_quart = analyses.separate_into_tiers(y_train)
y_test_quart = analyses.separate_into_tiers(y_test)

analyses.quantile_cross_table(quantiles, list(map(analyses.separate_into_tiers, [y_train, y_test])), stats_file, latex_dir)
