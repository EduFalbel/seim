import logging

import numpy as np
import pandas as pd
import re

import rpy2.robjects as ro
from rpy2.robjects.packages import STAP
from functools import partial

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.debug("test")

def split_coefficients(coefficients: dict) -> tuple[dict]: 
    '''Internal use function: splits the given coefficient dictionary into spatial and aspatial components, sorting them in a specific way for later use'''

    spatial_coef = dict(sorted({key : [value] for (key,value) in coefficients.items() if "rho_" in key}.items())) #Sort rho_ d o w

    const_coef = dict(sorted({key: [value] for (key, value) in coefficients.items() if re.search("Intercept|Intra", key)}.items())) # Sort Inter Intra
    node_coef = dict(sorted({key : [value] for (key,value) in coefficients.items() if re.search("ORIG_|DEST_|INTRA_", key)}.items())) # Sort DEST_ INTRA_ ORIG_
    pair_coef = dict(sorted({key: [value] for (key, value) in coefficients.items() if not re.search("ORIG_|DEST_|INTRA_|Intercept|Intra|rho", key)}.items())) # Sort alphab

    node_lags = dict(sorted({key : [value] for (key,value) in coefficients.items() if re.search("\.lag1", key)}.items())) # Sort DEST_ INTRA_ ORIG_

    # Remove lagged coefficients from node_coef
    [node_coef.pop(key, None) for key in node_lags]

    # We must do this explicitly to make sure all the lagged coefficients come after the regular ones in the dictionary
    node_keys = [*node_coef, *node_lags]

    # Join all dictionaries sequentially (maintaining their previous sorting)
    aspatial_coefficients = const_coef | {key : (node_coef | node_lags)[key] for key in node_keys} | pair_coef

    logging.debug(f"Spatial: {spatial_coef}\nAspatial: {aspatial_coefficients}")

    return (spatial_coef, aspatial_coefficients)

def calc_weights_matrices(weights_matrix: np.ndarray) -> list[np.ndarray, np.ndarray, np.ndarray]:
    '''Internal use function: uses Kronecker products to calculate the three n^2 X n^2 weights matrices (w_d, w_o, w_w)'''
    
    w_d = np.kron(np.identity(weights_matrix.shape[0]), weights_matrix)
    w_o = np.kron(weights_matrix, np.identity(weights_matrix.shape[0]))
    w_w = np.kron(weights_matrix, weights_matrix)

    weights = [w_d, w_o, w_w]

    assert len(set([i.shape[0] for i in weights])) == 1, f"{set([i.shape[0] for i in weights])}"

    logging.info("Done weights")

    return weights

def calc_constants(n: int) -> tuple[np.ndarray]:
    '''Internal use function: builds the constant vectors i_n, i_N, i_I'''

    constant_n = np.ones((n, 1))
    constant_N = np.kron(constant_n, constant_n)
    constant_I = np.identity(n).flatten()

    return (constant_n, constant_N, constant_I)

def calc_explanatory_var(constant_n, constant_I, node_matrix):
    '''Internal use function: uses Kronecker products to calculate the three independent variable matrices (x_d, x_o, x_i)'''

    x_d = np.kron(constant_n, node_matrix)
    x_o = np.kron(node_matrix, constant_n)
    x_i = np.matmul(np.diag(constant_I), np.kron(node_matrix, constant_n))

    # We must do d, i, o since we sorted the variables and coefficients alphabetically (so INTRA comes before ORIG)
    return [x_d, x_i, x_o]

def calc_spatial_filter(spatial_coef, N, weights_matrix, inv_matrix_path):
    '''Internal use function: calculates the spatial filter matrix (inverse of identity matrix minus the sum of the (weights matrices times their respective spatial coefficients)'''

    # assert len(spatial_coef) == len(weights), f"Spatial coefficients and weights have different length"
    if (len(spatial_coef) != 0):
        try:
            spatial_filter = np.load(inv_matrix_path)
        except(FileNotFoundError):
            weights = calc_weights_matrices(weights_matrix)

            spatial_filter =  np.linalg.inv(np.identity(N) - sum([rho * w for (rho, w) in zip(spatial_coef.values(), weights)]))
            # np.save(inv_matrix_path, spatial_filter)
    else: spatial_filter = np.identity(N)

    logging.info("Done filter")

    return spatial_filter

def tc(coefficients: dict, node_data: pd.DataFrame, pair_data: pd.DataFrame, weights_matrix: np.ndarray = np.empty((1, 1)), inv_matrix_path: str = "./tmp/matrix.npy", slx: bool = False) -> pd.DataFrame:
    '''Given a set of coefficients, weights matrix, and node and pair data for a Spatial Econometric Interaction Model, predict the values for the dependent variable based on the model's expected value equation.'''
    
    spatial_coef, aspatial_coefficients = split_coefficients(coefficients)

    logging.info("Done coef")

    node_data = node_data.reindex(sorted(node_data.columns), axis=1)
    pair_data = pair_data.reindex(sorted(pair_data.columns), axis=1)

    logging.debug(node_data)
    logging.debug(pair_data)

    n = node_data.shape[0]

    spatial_filter = calc_spatial_filter(spatial_coef, n**2, weights_matrix, inv_matrix_path)

    constant_n, constant_N, constant_I = calc_constants(n)

    assert constant_I.ndim == 1 and constant_I.shape[0] == n ** 2, f"Constant I ndim = {constant_I.ndim}, shape = {constant_I.shape}"

    logging.info("Done constants")

    node_matrix = node_data.drop(columns=["ID"]).to_numpy()
    if slx:
        node_matrix = np.hstack((node_matrix, np.matmul(weights_matrix, node_matrix)))
    
    explanatory = calc_explanatory_var(constant_n, constant_I, node_matrix) 

    assert len(set([i.shape[0] for i in explanatory])) == 1, f"{set([i.shape[0] for i in explanatory])}"

    logging.info("Done explanatory")

    g = pair_data.drop(columns=["ID_ORIG", "ID_DEST"]).to_numpy()

    logging.debug(g)

    assert len(set([i.shape[0] for i in [constant_N, constant_I, *explanatory]])) == 1, f"{set([i.shape[0] for i in [constant_N, constant_I, *explanatory]])}"

    Z = np.column_stack((constant_N, constant_I, *explanatory, g))

    logging.info("Done Z")
    logging.debug(Z)

    delta = np.array(list(aspatial_coefficients.values())).flatten()

    assert(delta.shape[0] == Z.shape[1]), f"Delta: {delta.shape[0]}\nZ: {Z.shape[1]}"

    logging.info("Done delta")
    logging.debug(delta)

    mult = np.multiply(Z, delta)
    sum_mult = np.sum(mult, axis=1)

    logging.debug(mult)
    logging.debug(sum_mult)

    assert spatial_filter.shape[1] == sum_mult.shape[0], f"Spatial shape: {spatial_filter.shape}, var sum shape: {sum_mult.shape}"

    return spatial_filter @ np.sum(np.multiply(Z, delta), axis=1)

def trend_signal(flows: np.ndarray, coefficients: dict, node_data: pd.DataFrame, pair_data: pd.DataFrame, weights_matrix: np.ndarray, slx: bool = False) -> pd.DataFrame:
    spatial_coef, aspatial_coefficients = split_coefficients(coefficients)

    logging.info("Done coef")

    node_data = node_data.reindex(sorted(node_data.columns), axis=1)
    pair_data = pair_data.reindex(sorted(pair_data.columns), axis=1)

    logging.debug(node_data)
    logging.debug(pair_data)

    n = node_data.shape[0]

    constant_n, constant_N, constant_I = calc_constants(n)

    assert constant_I.ndim == 1 and constant_I.shape[0] == n ** 2, f"Constant I ndim = {constant_I.ndim}, shape = {constant_I.shape}"

    logging.info("Done constants")

    node_matrix = node_data.drop(columns=["ID"]).to_numpy()
    if slx:
        node_matrix = np.hstack((node_matrix, np.matmul(weights_matrix, node_matrix)))
    
    explanatory = calc_explanatory_var(constant_n, constant_I, node_matrix) 

    assert len(set([i.shape[0] for i in explanatory])) == 1, f"{set([i.shape[0] for i in explanatory])}"

    logging.info("Done explanatory")

    g = pair_data.drop(columns=["ID_ORIG", "ID_DEST"]).to_numpy()

    logging.debug(g)

    assert len(set([i.shape[0] for i in [constant_N, constant_I, *explanatory]])) == 1, f"{set([i.shape[0] for i in [constant_N, constant_I, *explanatory]])}"

    Z = np.column_stack((constant_N, constant_I, *explanatory, g))

    logging.info("Done Z")
    logging.debug(Z)

    delta = np.array(list(aspatial_coefficients.values())).flatten()

    assert(delta.shape[0] == Z.shape[1]), f"Delta: {delta.shape[0]}\nZ: {Z.shape[1]}"

    logging.info("Done delta")
    logging.debug(delta)

    mult = np.multiply(Z, delta)
    sum_mult = np.sum(mult, axis=1)

    logging.debug(mult)
    logging.debug(sum_mult)

    return np.sum(np.multiply(Z, delta), axis=1) + (sum([rho * w for (rho, w) in zip(spatial_coef.values(), calc_weights_matrices(weights_matrix))]) @ flows)

def bp(flows, coefficients: dict, node_data: pd.DataFrame, pair_data: pd.DataFrame, weights_matrix: np.ndarray = np.empty((1, 1)), inv_matrix_path: str = "./tmp/matrix.npy", slx: bool = False) -> pd.DataFrame:
    '''Given a set of coefficients, weights matrix, and node and pair data for a Spatial Econometric Interaction Model, predict the values for the dependent variable based on the model's expected value equation.'''
    
    spatial_coef, aspatial_coefficients = split_coefficients(coefficients)

    logging.info("Done coef")

    node_data = node_data.reindex(sorted(node_data.columns), axis=1)
    pair_data = pair_data.reindex(sorted(pair_data.columns), axis=1)

    logging.debug(node_data)
    logging.debug(pair_data)

    n = node_data.shape[0]

    spatial_filter = calc_spatial_filter(spatial_coef, n**2, weights_matrix, inv_matrix_path)

    constant_n, constant_N, constant_I = calc_constants(n)

    assert constant_I.ndim == 1 and constant_I.shape[0] == n ** 2, f"Constant I ndim = {constant_I.ndim}, shape = {constant_I.shape}"

    logging.info("Done constants")

    node_matrix = node_data.drop(columns=["ID"]).to_numpy()
    if slx:
        node_matrix = np.hstack((node_matrix, np.matmul(weights_matrix, node_matrix)))
    
    explanatory = calc_explanatory_var(constant_n, constant_I, node_matrix) 

    assert len(set([i.shape[0] for i in explanatory])) == 1, f"{set([i.shape[0] for i in explanatory])}"

    logging.info("Done explanatory")

    g = pair_data.drop(columns=["ID_ORIG", "ID_DEST"]).to_numpy()

    logging.debug(g)

    assert len(set([i.shape[0] for i in [constant_N, constant_I, *explanatory]])) == 1, f"{set([i.shape[0] for i in [constant_N, constant_I, *explanatory]])}"

    Z = np.column_stack((constant_N, constant_I, *explanatory, g))

    logging.info("Done Z")
    logging.debug(Z)

    delta = np.array(list(aspatial_coefficients.values())).flatten()

    assert(delta.shape[0] == Z.shape[1]), f"Delta: {delta.shape[0]}\nZ: {Z.shape[1]}"

    logging.info("Done delta")
    logging.debug(delta)

    mult = np.multiply(Z, delta)
    sum_mult = np.sum(mult, axis=1)

    logging.debug(mult)
    logging.debug(sum_mult)

    assert spatial_filter.shape[1] == sum_mult.shape[0], f"Spatial shape: {spatial_filter.shape}, var sum shape: {sum_mult.shape}"

    return #(spatial_filter @ np.sum(np.multiply(Z, delta), axis=1)) - 

def two_stage(tc_coeff: dict, ts_coeff: dict, node_data: pd.DataFrame, pair_data: pd.DataFrame, weights_matrix: np.ndarray = np.empty((1, 1)), inv_matrix_path: str = "./tmp/matrix.npy", slx: bool = False) -> pd.DataFrame:
    flows = tc(tc_coeff, node_data, pair_data, weights_matrix, inv_matrix_path, slx)
    return trend_signal(flows, ts_coeff, node_data, pair_data, weights_matrix, slx)

def iterative(flows, estimate, predict, n = 2):
    
    for i in range(n):
        coef = {key : value for (key, value) in estimate(flows).items()}
        flows = predict(coef)
    return flows
    
    


# TODO: Properly define predict method type. Probably using Union[Callable[...], Callable[...]] so it can take methods with different parameters

def multi_stage_predict(stage1_prediction_method, stage2_prediction_method, train_node_path: str, train_pair_path: str, test_node_path: str, test_pair_path: str, temp_data_dir: str, stage1_estimation_method = None, stage2_estimation_method = None, weights_spec: dict = None, slx: bool = False, **kwargs) -> pd.DataFrame:
    """
    stage1_prediction_method: Callable(coefficients, node_data, pair_data, weights_matrix, *args, **kwargs)
    stage1_estimation_method: one of "SLA", "SLX", "SDM", "Aspatial"

    stage2_prediction_method: Callable(flows, coefficients, node_data, pair_data, weights_matrix, *args, **kwargs)
    stage2_estimation_method: one of "SLA", "SLX", "SDM", "Aspatial"

    weights_spec: dictionary describing the weights matrix
    """
    
    # Read model_estimation R script and parse it for use
    with open('./model_estimation.r', 'r') as f:
        string = f.read()
    model_estimation = STAP(string, "model_estimation")

    logging.info("Parsed R script")

    # Default weights matrix specification is row-standardized queen contiguity without distance-decay
    if (weights_spec is None):
        weights_spec = {
            "neighbour" : "contiguity",
            "queen/k" : True,
            "weight" : "standard",
            "style" : "W"}

    weights_spec = ro.vectors.ListVector(weights_spec) 

    logging.info("Created weights matrix ListVector")

    # Do the first stage estimation and prediction ####################################

    coef = model_estimation.estimate_model_params(f"{train_node_path}", f"{train_pair_path}", f"{temp_data_dir}", "boston", "trip_counts", weights_spec)

    logging.info("Estimated parameters")

    coef = {key : value for (key, value) in coef.items()}

    weights_matrix = np.loadtxt(f"{temp_data_dir}weights.txt")

    logging.info("Got weights matrix")

    # Writes the testing data to disk so that we can read it with pandas
    model_estimation.read_and_write_data(test_node_path, test_pair_path, weights_spec, temp_data_dir)

    logging.info("Wrote R parsed testing data")

    node_data = pd.read_csv(f"{temp_data_dir}/node.csv")
    pair_data = pd.read_csv(f"{temp_data_dir}/pair.csv")
    
    weights_matrix = np.loadtxt(f"{temp_data_dir}weights.txt")

    logging.info("Read testing data")
    logging.debug(f"Node data: {node_data}")
    logging.debug(f"Pair data: {pair_data}")
    logging.debug(f"Weights matrix: {weights_matrix}")

    flows = stage1_prediction_method(coefficients=coef, node_data=node_data, pair_data=pair_data, weights_matrix=weights_matrix)

    # Do the second stage estimation and prediction one time (to get the transformed node and pair data and weights matrix) ####################################

    partial_stage2_prediction_method = partial(stage2_prediction_method, node_data=node_data, pair_data=pair_data, weights_matrix=weights_matrix)
    
    partial_stage2_estimation_method = partial(stage2_estimation_method, node_path=test_node_path, pair_path=test_pair_path, wm_options=weights_spec, **kwargs)

    return iterative(flows, partial_stage2_estimation_method, partial_stage2_prediction_method)