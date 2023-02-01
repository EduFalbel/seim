import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde#, rankdata

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from tabulate import tabulate

import os
from typing import Literal
import logging

from seim.prediction import calc_weights_matrices

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

def RMSE_table(col_names, headers, stats_file, latex_dir = None):
    """Use tabulate to create a table with the models' RMSE and save it to a .txt file."""

    # TODO: Make it so it accepts predicted and use a lambda function to calculate the RMSE
    with open(stats_file, "w") as sf:
        sf.write("RMSE values:\n\n")
        print(tabulate([(model, *rmse) for model, rmse in col_names.items()], headers=headers), file=sf)
        print("\n\n\n", file=sf)

    if latex_dir is not None:
        with open(f"{latex_dir}RMSE.tex", "w") as rmse_tex:
            rmse_tex.write("RMSE values:\n\n")
            print(tabulate([(model, *rmse) for model, rmse in col_names.items()], headers=headers, tablefmt='latex'), file=rmse_tex)
            print("\n\n\n", file=rmse_tex)

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
                


                # https://stackoverflow.com/questions/20105364/how-can-i-make-a-scatter-plot-colored-by-density-in-matplotlib
                xy = np.vstack([resid, lagged_res])
                z = gaussian_kde(xy)(xy)
                idx = z.argsort()
                x, y, z = resid[idx], lagged_res[idx], z[idx]

                ax[i].scatter(x, y, c=z)

                # https://stackoverflow.com/questions/26447191/how-to-add-trendline-in-python-matplotlib-dot-scatter-graphs
                z = np.polyfit(resid, lagged_res, 1)
                p = np.poly1d(z)
                ax[i].plot([min(resid), max(resid)], [p(min(resid)), p(max(resid))], "r--")

                # https://stackoverflow.com/questions/21603585/how-to-add-equation-of-a-line-onto-a-plot-in-python
                ax[i].text(0.5, 0, "y=%.2fx%+.2f"%(z[0], z[1]), ha="center", va="bottom", transform=ax[i].transAxes, color='orange', fontsize='large', fontweight='semibold')

                ax[i].axline([0, 0], [1, 0], color="black", linewidth=0.6, alpha=0.6)
                
                i = i + 1

        trendline_legend = mlines.Line2D([], [], color="red", linestyle="--", label="Trendline")
        fig.legend(handles=[trendline_legend], loc="lower right")
        line_0_legend = mlines.Line2D([], [], color="black", linewidth=0.6, alpha=0.6, label="$y=0$")
        fig.legend(handles=[line_0_legend], loc="lower left")

        file_path = f"{moran_dir}{pred_type}.png"
        epsilon = "\\varepsilon"
        fig.supxlabel(f"${epsilon}$")
        fig.supylabel(f"$W_r{epsilon}$")
        fig.suptitle(f"Moran scatter plot of residuals against spatial lags of residuals ({pred_type})")
        fig.tight_layout(pad=0.5)
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
            
            # axs[i].scatter(y_true, predicted[pred_type][model])

            # https://stackoverflow.com/questions/20105364/how-can-i-make-a-scatter-plot-colored-by-density-in-matplotlib
            xy = np.vstack([y_true, predicted[pred_type][model]])
            z = gaussian_kde(xy)(xy)
            idx = z.argsort()
            x, y, z = y_true[idx], predicted[pred_type][model][idx], z[idx]

            axs[i].scatter(x, y, c=z)

            # https://stackoverflow.com/questions/26447191/how-to-add-trendline-in-python-matplotlib-dot-scatter-graphs
            z = np.polyfit(y_true, predicted[pred_type][model], 1)
            p = np.poly1d(z)

            axs[i].plot([min(y_true), max(y_true)], [p(min(y_true)), p(max(y_true))], "r--")

            # https://stackoverflow.com/questions/21603585/how-to-add-equation-of-a-line-onto-a-plot-in-python
            axs[i].text(0.5, 0.75, "y=%.2fx%+.2f"%(z[0], z[1]), ha="center", va="top", transform=axs[i].transAxes, color='black', fontsize='large', fontweight='medium')

            # https://stackoverflow.com/questions/25118628/add-x-y-45-degree-line-within-matplotlib-axis-limits
            if(line_45): axs[i].axline([0, 0], [1, 1], color="black", linewidth=0.6, alpha=0.6)

            i += 1
        
        trendline_legend = mlines.Line2D([], [], color="red", linestyle="--", label="Trendline")
        fig.legend(handles=[trendline_legend], loc="lower right")
        if (line_45):
            line_45_legend = mlines.Line2D([], [], color="black", linewidth=0.6, alpha=0.6, label="$y=x$")
            fig.legend(handles=[line_45_legend], loc="lower left")

        fig.supxlabel(f"Observed flows")
        fig.supylabel(f"{y_name.lower().capitalize()}")
        fig.suptitle(f"Scatter plot of observed flows vs. {y_name.lower()} ({pred_type})")
        fig.tight_layout()

        fig.savefig(f"{graph_dir}{pred_type}.png", bbox_inches='tight')
        plt.close(fig)
    
    
    return

def separate_into_tiers(flows: pd.DataFrame, num_tiers: int = 4):
    """Helper function to separate flows into n tiers, each tier's flows concentrating 1/n of the total trip volume, in descending order."""
    flows_tier = pd.DataFrame(flows.copy(deep=True)).sort_values(by="trip_counts", ascending=False).clip(lower=0)
    flows_tier["cumulative_prop"] = flows_tier["trip_counts"].cumsum()/flows_tier["trip_counts"].sum()

    bins = np.linspace(0, max(flows_tier["cumulative_prop"]), num_tiers + 1, endpoint=True)
    labels = [i for i in range(num_tiers)]

    flows_tier["tier"] = pd.cut(flows_tier["cumulative_prop"], bins=bins, labels=labels, right=True)

    return flows_tier.sort_index()["tier"]

def quantile_cross_table(quartiles: dict[str, dict[str, np.ndarray]], ground_truth: list[np.ndarray], stats_file, latex_dir = None):
    """Generate quantile comparison tables of predicted vs. observed values and append them to a .txt file."""
    
    for pred_type, y_true in zip(quartiles, ground_truth):
        for model in quartiles[pred_type]:
            with open(stats_file, "a") as sf:
                sf.write(f"{pred_type}: {model} quartiles vs. observed quartiles:\n\n")
                lines = [line.split() for line in pd.crosstab(y_true, quartiles[pred_type][model], margins=True, rownames=['Observed'], colnames=['Predicted'])\
                    .to_string().split("\n")]
                print(tabulate(lines[1:], headers=lines[0]), file=sf)
                print("\n\n\n", file=sf)
            if (latex_dir is not None):
                try:
                    os.mkdir(f"{latex_dir}{pred_type}/")
                except FileExistsError:
                    pass
                with open(f"{latex_dir}{pred_type}/{model}.tex", "w") as lx:
                    print(pd.crosstab(y_true, quartiles[pred_type][model], margins=True, rownames=['Observed'], colnames=['Predicted']).style.to_latex(caption=f"{pred_type} flow tier analysis for {model} model. Lower tier numbers indicate higher flow volume.", position="H", label=f"table:{pred_type.split('-')[0]}-{model}"), file=lx)