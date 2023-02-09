import logging

import numpy as np
import pandas as pd

import rpy2.robjects.vectors as ro_vectors

from functools import partial

from seim.model_estimation_wrapper import estimate_model_params as estimate, read_and_write_data
from seim import prediction, analyses

logging.basicConfig(encoding='utf-8', level=logging.DEBUG)
logging.debug("test")

def default_weights_matrix():
    weights_spec = {
        "neighbour" : "contiguity",
        "queen/k" : True,
        "weight" : "standard",
        "style" : "W"}

    return weights_spec

def estimate_predict(dependent_var: str, train_node_path: str, train_pair_path: str, test_node_path: str, test_pair_path: str, prediction_method, weights_spec: dict = None, temp_data_dir="/tmp", *args, **kwargs) -> pd.DataFrame:
    """
    
    """
    if weights_spec is None:
        weights_spec = default_weights_matrix()
    weights_ro = ro_vectors.ListVector(weights_spec)

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