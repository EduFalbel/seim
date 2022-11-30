import logging

import numpy as np
import pandas as pd
import re

logger = logging.getLogger()
logger.setLevel(logging.INFO)
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
            np.save(inv_matrix_path, spatial_filter)
    else: spatial_filter = np.identity(N)

    logging.info("Done filter")

    return spatial_filter
