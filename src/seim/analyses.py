import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde#, rankdata

import matplotlib.pyplot as plt
from tabulate import tabulate

from typing import Literal
import logging

from prediction import calc_weights_matrices

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.debug("test")

# TODO: Perhaps make it so that the y_test and y_true belong to their own dictionary just like the "predicted" one. This way they can have the same keys so it would be easy to reference them simultaneously.

# TODO: Debate whether I want the predicted (and potentially associated dicts) to be global variables.

def add_pred(predicted : dict, pred_type : Literal["train", "test"], key: str, value):
    """Add a new set of model predictions to the given dictionary to be used for later analyses."""
    
    train_key = "In-sample prediction"
    test_key = "Out-of-sample prediction"
    
    if pred_type == "train":
        predicted.setdefault(train_key, {key : value})[key] = value
        
        logging.info(f"Added model {key} to {train_key} type")
        logging.debug(predicted[train_key])
    elif pred_type == "test":
        predicted.setdefault(test_key, {key : value})[key] = value
        
        logging.info(f"Added model {key} to {test_key} type")
        logging.debug(predicted[test_key])
    
    return predicted

def transform_predicted(predicted, transformation, list_y_true: list[np.ndarray]):
    """Convenience method to apply a transformation to the predicted dictionary."""
    transformed = {}
    for pred_type, y_true in zip(predicted, list_y_true):
        transformed[pred_type] = {}
        for model in predicted[pred_type]:
            transformed[pred_type][model] = transformation(predicted[pred_type][model], y_true)
    return transformed

def RMSE_table(col_names, headers, stats_file):
    """Use tabulate to create a table with the models' RMSE and save it to a .txt file."""

    # TODO: Make it so it accepts predicted and use a lambda function to calculate the RMSE
    with open(stats_file, "w") as sf:
        sf.write("RMSE values:\n\n")
        print(tabulate([(model, *rmse) for model, rmse in col_names.items()], headers=headers), file=sf)
        print("\n\n\n", file=sf)

def moran_plot(names_residuals: dict[str, dict[str, np.ndarray]], weights: list[np.ndarray], moran_dir : str):
    """Create Moran scatterplots for the given residuals and weights matrices and save it to the given directory."""

    assert len(weights) == len(names_residuals), "Amount of weights matrices and prediction types do not match"

    for (name_resid, weights_matrix) in zip(names_residuals.items(), weights):

        pred_type, model_resid = name_resid

        logging.info(f"Prediction type: {pred_type}")

        weights = calc_weights_matrices(weights_matrix)
        titles = [f"W_{r}" for r in ["d", "o", "w"]]

        title_weights = {title : weight for (title, weight) in zip(titles, weights)}

        fig, ax = plt.subplots(nrows=len(model_resid), ncols=3, sharex=True, sharey=True)
        ax = ax.flatten()

        for (model, i) in zip(model_resid.keys(), range(2, len(ax), 3)):

            logging.info(f"Model for Axes modification: {model}")

            ax[i].yaxis.set_label_position('right')
            ax[i].set_ylabel(model)

        i = 0 # Counter to interact with Axes objects in ax

        for model, resid in model_resid.items():

            logging.info(f"Model: {model}")

            for title, weight in title_weights.items():

                logging.info(f"Weight: {title}")


                lagged_res = np.matmul(weight, resid)

                if i < 3:
                    ax[i].set_title(f"$W_r={title}$") 
                

                # https://stackoverflow.com/questions/26447191/how-to-add-trendline-in-python-matplotlib-dot-scatter-graphs
                z = np.polyfit(resid, lagged_res, 1)
                p = np.poly1d(z)
                ax[i].plot(resid, p(resid), "r--")

                # https://stackoverflow.com/questions/20105364/how-can-i-make-a-scatter-plot-colored-by-density-in-matplotlib
                xy = np.vstack([resid, lagged_res])
                z = gaussian_kde(xy)(xy)
                idx = z.argsort()
                x, y, z = resid[idx], lagged_res[idx], z[idx]

                ax[i].scatter(x, y, c=z)

                i = i + 1


        file_path = f"{moran_dir}{pred_type}.png"
        epsilon = "\\varepsilon"
        fig.supxlabel(f"${epsilon}$")
        fig.supylabel(f"$W_r{epsilon}$")
        fig.suptitle(f"Moran scatter plot of residuals against spatial lags of residuals ({pred_type})")
        fig.tight_layout()
        fig.savefig(file_path, bbox_inches='tight')
        plt.close(fig)

    
def analysis_plots(predicted: dict[str, dict[str, np.ndarray]], ground_truth: list[np.ndarray], line_45: bool, graph_dir: str, y_name: str) -> None:
    """Create scatterplots of values in predicted against the observed values and save them to png files in the given directory."""

    assert len(ground_truth) == len(predicted)

    for pred_type, y_true in zip(predicted, ground_truth):
        fig, axs = plt.subplots(nrows=1, ncols=len(predicted[pred_type]), sharex=True, sharey=True)
        
        i = 0

        for model, pred in predicted[pred_type].items():

            axs[i].set_title(f"{model}") 
            
            axs[i].scatter(y_true, predicted[pred_type][model])

            # https://stackoverflow.com/questions/26447191/how-to-add-trendline-in-python-matplotlib-dot-scatter-graphs
            z = np.polyfit(y_true, predicted[pred_type][model], 1)
            p = np.poly1d(z)
            axs[i].plot(y_true, p(y_true), "r--")

            if(line_45): axs[i].plot(y_true, y_true, "b--")

            i += 1
        
        fig.supxlabel(f"Observed flows")
        fig.supylabel(f"{y_name.lower().capitalize()}")
        fig.suptitle(f"Scatter plot of observed flows vs. {y_name.lower()} ({pred_type})")
        fig.tight_layout()

        fig.savefig(f"{graph_dir}{pred_type}.png", bbox_inches='tight')
        plt.close(fig)
    
    
    return

def quantile_cross_table(quartiles: dict[str, dict[str, np.ndarray]], ground_truth: list[np.ndarray], stats_file):
    "Generate quantile comparison tables of predicted vs. observed values and append them to a .txt file."
    
    for pred_type, y_true in zip(quartiles, ground_truth):
        for model in quartiles[pred_type]:
            with open(stats_file, "a") as sf:
                sf.write(f"{pred_type}: {model} quartiles vs. observed quartiles:\n\n")
                lines = [line.split() for line in pd.crosstab(y_true, quartiles[pred_type][model], margins=True, rownames=['Observed'], colnames=['Predicted'])\
                    .to_string().split("\n")]
                print(tabulate(lines[1:], headers=lines[0]), file=sf)
                print("\n\n\n", file=sf)