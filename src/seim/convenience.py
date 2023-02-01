import logging

import numpy as np
import pandas as pd
import re

import os


from functools import partial

from seim.model_estimation_wrapper import estimate_model_params as estimate, weights_matrix_list_vector_builder as weights_builder, read_and_write_data
from seim import prediction, analyses

logging.basicConfig(encoding='utf-8', level=logging.DEBUG)
logging.debug("test")



def estimate_predict(dependent_var: str, train_node_path: str, train_pair_path: str, test_node_path: str, test_pair_path: str, prediction_method, weights_spec: dict = None, slx: bool = False, temp_data_dir="/tmp", *args, **kwargs) -> pd.DataFrame:
    """
    
    """
    weights_ro = weights_builder(weights_spec)

    logging.info("Built weights matrix")

    coef = estimate(train_node_path, train_pair_path, weights_ro, *args, **kwargs)

    logging.info("Estimated model parameters")

    read_and_write_data(test_node_path, test_pair_path, weights_ro, temp_data_dir)

    logging.info("Wrote R-parsed testing data")

    node_data = pd.read_csv(f"{temp_data_dir}/node.csv")
    pair_data = pd.read_csv(f"{temp_data_dir}/pair.csv").drop(columns=[dependent_var])
    
    weights_matrix = np.loadtxt(f"{temp_data_dir}weights.txt")

    logging.info("Read testing data")

    return prediction_method(coefficients=coef, node_data=node_data, pair_data=pair_data, weights_matrix=weights_matrix)