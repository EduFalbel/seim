import rpy2.robjects as ro
from rpy2.robjects.packages import STAP

model_estimation = None

def parse_model_estimation():
    with open('./model_estimation.r', 'r') as f:
        string = f.read()
    global model_estimation
    model_estimation = STAP(string, "model_estimation")

def verify_model_estimation_exists():
    """Check if model_estimation.r has already been parsed. If it hasn't, call parser. Allows for 'lazy loading' of the R module."""
    if model_estimation is None:
        parse_model_estimation()

def estimate_model_params(train_node_path, train_pair_path, temp_data_dir, *args, **kwargs):
    """Basically a wrapper function for the R spflow package"""
    verify_model_estimation_exists()

    coef = model_estimation.estimate_model_params(f"{train_node_path}", f"{train_pair_path}", f"{temp_data_dir}", *args, **kwargs)

    return {key : value for (key, value) in coef.items()}

def read_and_write_data(node_path, pair_path, weights_spec, data_save_path):
    """Writes the testing data and weights matrix to disk so that we can read it with pandas"""
    model_estimation.read_and_write_data(node_path, pair_path, data_save_path)
    model_estimation.create_and_write_weights_matrix(node_path, data_save_path, weights_spec)
    pass