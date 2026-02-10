#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Track parameter analysis script.
This script analyzes track parameters from the TrackML dataset.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import plotly.graph_objects as go
from scipy.stats import norm, scoreatpercentile
from scipy.optimize import curve_fit

# Import project-specific modules
from src.datasets.datamodules import DataModule
from src.my_model.transformer import TrackFormer


def get_model_checkpoint_path(version_dir: str):
    """Get the checkpoint path from a version directory."""
    version_dir = Path(version_dir)
    checkpoint_dir = version_dir / "checkpoints"
    checkpoint_files = list(checkpoint_dir.glob("model*.ckpt"))
    assert (
        len(checkpoint_files) == 1
    ), f"Found {len(checkpoint_files)} checkpoints in {checkpoint_dir}"
    return checkpoint_files[0]


def get_config(version_dir: str):
    """Load configuration from a version directory."""
    version_dir = Path(version_dir)
    config_file = version_dir / "config.yaml"
    import yaml

    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def validate_model_config(
    model_config,
    config,
    dataset_dir=None,
    criterion="mse",
    allow_other_datasets=False,
):
    """Validate that model config matches the current config."""
    # Check that the dataset_dir is the same as the one used in the config
    assert (
        Path(model_config["data"]["init_args"]["dataset_dir"]) == dataset_dir
        or allow_other_datasets
    ), f"Dataset dir is {model_config['data']['init_args']['dataset_dir']} instead of {dataset_dir}"

    # Check that the config is the same as the one used in the config
    for key in model_config["data"]["init_args"]["kwargs"]:
        if not key in config:
            if key in ["dataset_suffix", "wrapper_workers"]:
                continue
            print(
                f"Key {key}: {model_config['data']['init_args']['kwargs'][key]} is not in the config"
            )
            continue

        assert (
            model_config["data"]["init_args"]["kwargs"][key] == config[key]
        ), f"Key {key} is {model_config['data']['init_args']['kwargs'][key]} instead of {config[key]}"

    assert (
        model_config["model"]["init_args"]["criterion"] == criterion
    ), f"Criterion is {model_config['model']['init_args']['criterion']} instead of {criterion}"


def var_to_pT(values, from_var):
    """Convert variables to pT."""
    if from_var == "pT":
        return values
    if from_var == "qopT":
        return 1 / np.abs(values)
    if from_var == "qpT":
        return np.abs(values)
    raise ValueError(f"Invalid from_var: {from_var}")


def compute_likelyhood(data, model, neglog=False):
    """Compute likelihood of data given model."""
    likelyhood = model(data)
    if neglog:
        return np.sum(-2 * np.log(likelyhood))
    else:
        return np.prod(likelyhood)


def compute_track_resolution(true_values, measured_values, max_iterations=100):
    """
    Compute the resolution of track parameters by iteratively removing outliers beyond 3 standard deviations.

    Parameters:
        true_values (numpy array): Array of true track parameters from simulation.
        measured_values (numpy array): Array of reconstructed track parameters.
        max_iterations (int): Maximum number of iterations to achieve convergence.

    Returns:
        tuple: (Filtered track residuals, removed outliers)
    """
    residuals = np.array(measured_values) - np.array(true_values)
    removed_outliers = np.array([])

    for iteration in range(max_iterations):
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)

        # Identify inliers within 3 standard deviations
        inlier_mask = np.abs(residuals - mean_residual) <= 3 * std_residual
        filtered_residuals = residuals[inlier_mask]
        removed_outliers = np.concatenate((removed_outliers, residuals[~inlier_mask]))

        # Check for convergence
        if len(filtered_residuals) == len(residuals):
            break

        residuals = filtered_residuals

    if iteration == max_iterations - 1:
        print(f"Warning: Maximum iterations reached before convergence.")

    return filtered_residuals, removed_outliers


def view_trajectory(inputs, mask=None):
    """
    Plots the trajectory of a particle in 3D.
    """
    if mask is None:
        # Filter out zero rows
        inputs = inputs[(inputs != 0).any(dim=1)]
    else:
        inputs = inputs[mask]

    x = inputs[:, 0].numpy()
    y = inputs[:, 1].numpy()
    z = inputs[:, 2].numpy()

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines+markers",
                marker=dict(size=4, color="blue", opacity=0.8),
                line=dict(color="blue", width=2),
            )
        ]
    )

    fig.update_layout(
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        title="Particle Trajectory",
    )

    fig.show()


def sync_plot_limits():
    """Synchronize plot limits across subplots."""
    # Exit if the figure has only one subplot
    if len(plt.gcf().axes) == 1:
        return
    # Use the same axis limits for both plots
    left, right = None, None
    bottom, top = None, None

    for i in range(1, 3):
        plt.subplot(1, 2, i)
        left_, right_ = plt.xlim()
        bottom_, top_ = plt.ylim()

        if left is None or left_ < left:
            left = left_
        if right is None or right_ > right:
            right = right_
        if bottom is None or bottom_ < bottom:
            bottom = bottom_
        if top is None or top_ > top:
            top = top_

    # Apply the same limits to the subplots
    for i in range(1, 3):
        plt.subplot(1, 2, i)
        plt.xlim(left, right)
        plt.ylim(bottom, top)


def plot_err_vs_n_hits(
    target_labels,
    p_true_list,
    n_hits_list,
    models,
    config,
    title_suffix,
    output_dir,
    variable_filenames,
    variable_labels,
    low_pt=False,
):
    """Plot error vs number of hits."""
    for var_index, var in enumerate(target_labels):
        # Get the true values for the current variable.
        pi_true_values = np.array(p_true_list)[:, var_index]
        n_hits_values = np.array(n_hits_list)
        if var == "p_T":
            pi_true_values = var_to_pT(
                values=pi_true_values, from_var=config["output_variables"][var_index]
            )

        title = f"${var}^{{true}}$ vs ${var}^{{pred}}$"
        if low_pt:
            title += " ($p_T$ < 10 GeV)"
        title += title_suffix

        true_label = f"${var}^{{true}}$"
        pred_label = f"${var}^{{pred}}$"
        xlabel = "number of hits"
        ylabel = (
            r"$\frac{"
            + f"{pred_label} - {true_label}".replace("$", "")
            + "}{"
            + f"{true_label}".replace("$", "")
            + "}$"
            + " [%]"
        )
        filename = f"{variable_filenames[var]}_true_vs_{variable_filenames[var]}_pred_square_number_of_hits.png"
        if low_pt:
            filename = filename.replace(".", "_low_pt.")
        filename = output_dir / filename

        if low_pt:
            assert any(v in config["output_variables"] for v in ["pT", "qopT", "qpT"])
            if "pT" in config["output_variables"]:
                pt_true = np.array(p_true_list)[
                    :, config["output_variables"].index("pT")
                ]
            elif "qopT" in config["output_variables"]:
                qopt_true = np.array(p_true_list)[
                    :, config["output_variables"].index("qopT")
                ]
                pt_true = 1 / np.abs(qopt_true)
            elif "qpT" in config["output_variables"]:
                qpt_true = np.array(p_true_list)[
                    :, config["output_variables"].index("qpT")
                ]
                pt_true = np.abs(qpt_true)

            # Select only events with p_T < 10 GeV.
            mask = pt_true < 10

            # Apply the pT mask.
            pi_true_values = pi_true_values[mask]

        # Plot pt_true and pt_pred
        plt.figure(figsize=(6 * len(models), 6))
        plt.suptitle(title, fontsize=16)

        min_pi_true = min(pi_true_values)
        max_pi_true = max(pi_true_values)

        # Loop over each model.
        for model_index, model_info in enumerate(models.values(), start=1):
            pi_pred_values = np.array(model_info["data_list"])[:, var_index]
            if var == "p_T":
                pi_pred_values = var_to_pT(
                    values=pi_pred_values,
                    from_var=config["output_variables"][var_index],
                )

            if low_pt:
                # Apply the same pT mask.
                pi_pred_values = pi_pred_values[mask]

            err_values = pi_pred_values - pi_true_values
            relative_err_values = err_values / pi_true_values

            plt.subplot(1, len(models), model_index)
            plt.plot(
                n_hits_values,
                relative_err_values,
                alpha=0.7,
                label=f"{model_info['label']} Model",
                color=f"{model_info['color']}",
                marker="o",
                linestyle="None",
            )

            plt.xlabel(xlabel, fontsize=12)
            plt.ylabel(ylabel, fontsize=12)
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.legend()
            plt.ylim(-10, 10)

        sync_plot_limits()

        if len(models) > 1:
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        else:
            plt.tight_layout()
        plt.savefig(filename, format="png")
        plt.close()


def plot_pi_true_vs_pred(
    target_labels,
    p_true_list,
    models,
    config,
    title_suffix,
    output_dir,
    variable_filenames,
    variable_labels,
    low_pt=False,
):
    """Plot true values vs predicted values."""
    for var_index, var in enumerate(target_labels):
        # Get the true values for the current variable.
        pi_true_values = np.array(p_true_list)[:, var_index]
        if var == "p_T":
            pi_true_values = var_to_pT(
                values=pi_true_values, from_var=config["output_variables"][var_index]
            )

        title = f"${var}^{{true}}$ vs ${var}^{{pred}}$"
        if low_pt:
            title += " ($p_T$ < 10 GeV)"
        title += title_suffix

        true_label = f"${var}^{{true}}$"
        pred_label = f"${var}^{{pred}}$"
        xlabel = true_label + (" [GeV]" if var.startswith("p") else "")
        ylabel = pred_label + (" [GeV]" if var.startswith("p") else "")
        filename = f"{variable_filenames[var]}_true_vs_{variable_filenames[var]}_pred_square.png"
        if low_pt:
            filename = filename.replace(".", "_low_pt.")
        filename = output_dir / filename

        if low_pt:
            assert any(v in config["output_variables"] for v in ["pT", "qopT", "qpT"])
            if "pT" in config["output_variables"]:
                pt_true = np.array(p_true_list)[
                    :, config["output_variables"].index("pT")
                ]
            elif "qopT" in config["output_variables"]:
                qopt_true = np.array(p_true_list)[
                    :, config["output_variables"].index("qopT")
                ]
                pt_true = 1 / np.abs(qopt_true)
            elif "qpT" in config["output_variables"]:
                qpt_true = np.array(p_true_list)[
                    :, config["output_variables"].index("qpT")
                ]
                pt_true = np.abs(qpt_true)

            # Select only events with p_T < 10 GeV.
            mask = pt_true < 10

            # Apply the pT mask.
            pi_true_values = pi_true_values[mask]

        # Plot pt_true and pt_pred
        plt.figure(figsize=(6 * len(models), 6))
        plt.suptitle(title, fontsize=16)

        min_pi_true = min(pi_true_values)
        max_pi_true = max(pi_true_values)

        # Loop over each model.
        for model_index, model_info in enumerate(models.values(), start=1):
            pi_pred_values = np.array(model_info["data_list"])[:, var_index]
            if var == "p_T":
                pi_pred_values = var_to_pT(
                    values=pi_pred_values,
                    from_var=config["output_variables"][var_index],
                )

            if low_pt:
                # Apply the same pT mask.
                pi_pred_values = pi_pred_values[mask]

            plt.subplot(1, len(models), model_index)
            plt.plot(
                pi_true_values,
                pi_pred_values,
                alpha=0.7,
                label=f"{model_info['label']} Model",
                color=f"{model_info['color']}",
                marker="o",
                linestyle="None",
            )

            # Add x = y line
            plt.plot(
                [min_pi_true, max_pi_true],
                [min_pi_true, max_pi_true],
                color="black",
                linestyle="--",
                label=f"{true_label} = {pred_label}",
            )

            plt.xlabel(xlabel, fontsize=12)
            plt.ylabel(ylabel, fontsize=12)
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.legend()
            if var == "p_T" and config["output_variables"][var_index] == "qopT":
                plt.ylim(0, 10)

        sync_plot_limits()

        if len(models) > 1:
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        else:
            plt.tight_layout()
        plt.savefig(filename, format="png")
        plt.close()


def plot_pi_rel_resolutions(
    target_labels,
    p_true_list,
    models,
    config,
    title_suffix,
    output_dir,
    variable_filenames,
    variable_labels,
    pruning=False,
    low_pt=False,
):
    """Plot relative resolutions."""
    for var_index, var in enumerate(target_labels):
        pi_true_values = np.array(p_true_list)[:, var_index]
        if var == "p_T":
            pi_true_values = var_to_pT(
                values=pi_true_values, from_var=config["output_variables"][var_index]
            )

        title = f"Relative resolution for ${var}$"
        if low_pt:
            title += " ($p_T$ < 10 GeV)"
        title += title_suffix

        true_label = f"${var}^{{true}}$"
        pred_label = f"${var}^{{pred}}$"
        xlabel = true_label + (" [GeV]" if var.startswith("p") else "")
        ylabel = (
            r"$\frac{"
            + f"{pred_label} - {true_label}".replace("$", "")
            + "}{"
            + f"{true_label}".replace("$", "")
            + "}$"
            + " [%]"
        )
        filename = f"rel_resolution_{variable_filenames[var]}.png"
        if low_pt:
            filename = filename.replace(".", "_low_pt.")
        if pruning:
            filename = filename.replace(".", "_pruning.")
        filename = output_dir / filename

        if low_pt:
            assert any(v in config["output_variables"] for v in ["pT", "qopT", "qpT"])
            if "pT" in config["output_variables"]:
                pt_true = np.array(p_true_list)[
                    :, config["output_variables"].index("pT")
                ]
            elif "qopT" in config["output_variables"]:
                qopt_true = np.array(p_true_list)[
                    :, config["output_variables"].index("qopT")
                ]
                pt_true = 1 / np.abs(qopt_true)
            elif "qpT" in config["output_variables"]:
                qpt_true = np.array(p_true_list)[
                    :, config["output_variables"].index("qpT")
                ]
                pt_true = np.abs(qpt_true)

            # Select only events with p_T < 10 GeV.
            mask = pt_true < 10

            # Apply the pT mask.
            pi_true_values = pi_true_values[mask]

        # Plot pt_true and pt_pred
        plt.figure(figsize=(6 * len(models), 6))
        plt.suptitle(title, fontsize=16)

        bins = np.linspace(min(pi_true_values), max(pi_true_values), 10)
        if var == "p_T":
            bins = np.logspace(
                np.log10(min(pi_true_values)), np.log10(max(pi_true_values)), 15
            )
            bins = np.logspace(np.log10(min(pi_true_values)), np.log10(50), 10)
            bins = np.array([0, 1, 2, 4, 6, 8, 12, 16, 20, 24, 28, 34, 40, 50])
            bins = np.array([0, 1, 2, 4, 6, 8, 10])

        # Loop over each model.
        model_data = {}
        for model_name, model_info in models.items():
            pi_pred_values = np.array(model_info["data_list"])[:, var_index]
            if var == "p_T":
                pi_pred_values = var_to_pT(
                    values=pi_pred_values,
                    from_var=config["output_variables"][var_index],
                )

            if low_pt:
                # Apply the same pT mask.
                pi_pred_values = pi_pred_values[mask]

            errors = pi_pred_values - pi_true_values
            relative_errors = errors / pi_true_values

            # Make it percentage
            relative_errors_per = 100 * relative_errors
            model_data[model_name] = relative_errors_per

        # Loop over each model.
        model_bins = {}
        for bin_index in range(len(bins)):
            if bin_index == 0:
                lower_bound = -np.inf
                continue
            else:
                lower_bound = bins[bin_index - 1]
            upper_bound = bins[bin_index]

            sel = (lower_bound < pi_true_values) & (pi_true_values <= upper_bound)
            for model_name, model_info in models.items():
                if not pruning:
                    relative_errors_bin = model_data[model_name][sel]
                    model_bin_mean, model_bin_std = np.mean(
                        relative_errors_bin
                    ), np.std(relative_errors_bin)
                else:
                    pi_pred_values = np.array(model_info["data_list"])[:, var_index]
                    if var == "p_T":
                        pi_pred_values = var_to_pT(
                            values=pi_pred_values,
                            from_var=config["output_variables"][var_index],
                        )

                    if low_pt:
                        # Apply the same pT mask.
                        pi_pred_values = pi_pred_values[mask]

                    relative_errors_bin_pruned, removed_outliers_bin = (
                        compute_track_resolution(
                            (pi_true_values[sel] / pi_true_values[sel]),
                            pi_pred_values[sel] / pi_true_values[sel],
                        )
                    )
                    # Make it percentage
                    relative_errors_bin_pruned = 100 * relative_errors_bin_pruned

                    if len(pi_true_values[sel]) == 0:
                        outlier_ratio = np.inf
                    else:
                        outlier_ratio = len(removed_outliers_bin) / len(
                            pi_true_values[sel]
                        )

                    print(
                        f"Fraction of tracks identified as outliers by model {model_name}: {outlier_ratio:.4%} for bin {(lower_bound, upper_bound)}"
                    )

                    model_bin_mean, model_bin_std = np.mean(
                        relative_errors_bin_pruned
                    ), np.std(relative_errors_bin_pruned)

                model_bins.setdefault(model_name, []).append(
                    (model_bin_mean, model_bin_std)
                )

        bin_centers = (bins[1:] + bins[:-1]) / 2

        # Loop over each model.
        for model_index, (model_name, model_info) in enumerate(models.items(), start=1):
            bin_counts = np.array(model_bins[model_name])[..., 0]
            bin_std = np.array(model_bins[model_name])[..., 1]

            plt.subplot(1, len(models), model_index)

            plt.errorbar(
                bin_centers,
                bin_counts,
                yerr=bin_std,
                xerr=(bins[1:] - bins[:-1]) / 2,
                alpha=0.7,
                label=f"{model_info['label']} Model",
                color=f"{model_info['color']}",
                marker=model_info["marker"],
                linestyle="None",
            )

            plt.xlabel(xlabel, fontsize=12)
            plt.ylabel(ylabel, fontsize=12)
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.legend()
            if var == "p_T":
                if config["output_variables"][0] == "qopT":
                    plt.ylim(-30, 30)
                else:
                    plt.ylim(-10, 10)
            if var == "p_z" and not pruning:
                plt.ylim(-1.5 * 100, 1.5 * 100)

        sync_plot_limits()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(filename, format="png")
        plt.close()

        # One figure
        filename = f"rel_resolution_{variable_filenames[var]}_one_plot.png"
        if low_pt:
            filename = filename.replace(".", "_low_pt.")
        if pruning:
            filename = filename.replace(".", "_pruning.")
        filename = output_dir / filename
        bin_sizes = bins[1:] - bins[:-1]
        plt.figure(figsize=(12, 6))

        for model_index, (model_name, model_info) in enumerate(models.items(), start=1):
            bin_counts = np.array(model_bins[model_name])[..., 0]
            bin_std = np.array(model_bins[model_name])[..., 1]
            if model_index == 1:
                xerr = (bins[1:] - bins[:-1]) / 2
                x = bin_centers
            elif model_index == 2:
                shift = bin_sizes * 0.1
                xerr_low = bin_sizes / 2 + shift
                xerr_high = bin_sizes / 2 - shift
                xerr_high[xerr_high < 0] = 0
                xerr = [xerr_low, xerr_high]
                x = bin_centers + shift

            plt.errorbar(
                x,
                bin_counts,
                yerr=bin_std,
                xerr=xerr,
                alpha=0.7,
                label=f"{model_info['label']} Model",
                color=f"{model_info['color']}",
                marker=model_info["marker"],
                linestyle="None",
            )

        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()
        plt.title(title)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if var == "p_T":
            if config["output_variables"][0] == "qopT" and False:
                plt.ylim(-30, 30)
            else:
                plt.ylim(-10, 10)
        if var == "p_z":
            plt.ylim(-1.5 * 100, 1.5 * 100)
            if pruning:
                plt.ylim(-20, 20)
        plt.savefig(filename, format="png")
        plt.close()


def plot_2d_histogram(
    target_labels,
    p_true_list,
    models,
    config,
    title_suffix,
    output_dir,
    variable_filenames,
    variable_labels,
):
    """Plot 2D histograms of relative errors vs true values."""
    for var_index, var in enumerate(target_labels):
        pi_true_values = np.array(p_true_list)[:, var_index]
        if var == "p_T":
            pi_true_values = var_to_pT(
                values=pi_true_values, from_var=config["output_variables"][var_index]
            )

        title = f"Relative error resolution for ${var}$"
        title += title_suffix

        true_label = f"${var}^{{true}}$"
        pred_label = f"${var}^{{pred}}$"
        xlabel = true_label + (" [GeV]" if var.startswith("p") else "")
        ylabel = (
            r"$\frac{"
            + f"{pred_label} - {true_label}".replace("$", "")
            + "}{"
            + f"{true_label}".replace("$", "")
            + "}$"
        )

        filename = f"rel_error_histogram_{variable_filenames[var]}.png"
        filename = output_dir / filename

        plt.figure(figsize=(6 * len(models), 6))
        plt.suptitle(title, fontsize=16)

        bins = (50, 50)
        from matplotlib import colors

        # Loop over each model.
        y_range = (-1, 1)
        for model_index, model_info in enumerate(models.values(), start=1):
            pi_pred_values = np.array(model_info["data_list"])[:, var_index]
            if var == "p_T":
                pi_pred_values = var_to_pT(
                    values=pi_pred_values,
                    from_var=config["output_variables"][var_index],
                )

            errors = pi_pred_values - pi_true_values
            relative_errors = errors / pi_true_values

            plt.subplot(1, len(models), model_index)
            plt.hist2d(
                pi_true_values,
                relative_errors,
                bins=bins,
                cmap=model_info["cmap"],
                density=False,
                norm=colors.LogNorm(),
                range=[(min(pi_true_values), max(pi_true_values)), y_range],
            )
            plt.colorbar(label="Density")
            plt.xlabel(xlabel, fontsize=12)
            plt.ylabel(ylabel, fontsize=12)
            plt.title(f"{model_info['label']} Model")
            plt.grid(True, linestyle="--", alpha=0.7)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(filename, format="png")
        plt.close()


def plot_pi_relative_error_distributions_low_pt_1_2(
    target_labels,
    p_true_list,
    models,
    config,
    title_suffix,
    output_dir,
    variable_filenames,
    variable_labels,
):
    """Plot relative error distributions for low pT tracks."""
    import ROOT

    ROOT.gROOT.SetBatch(True)

    # Try to load and set ATLAS style (make sure AtlasStyle.C is in your working directory or adjust the path).
    try:
        ROOT.gROOT.LoadMacro("AtlasStyle.C")
        ROOT.SetAtlasStyle()
    except Exception as e:
        print("AtlasStyle.C not found. Proceeding with default ROOT style.")
        ROOT.gROOT.SetStyle("ATLAS")

    for var_index, var in enumerate(target_labels):
        title = f"Relative Error Distributions for ${var}$ (1 GeV < $p_T$ < 2 GeV)"
        title += title_suffix

        true_label = f"${var}^{{true}}$"
        pred_label = f"${var}^{{pred}}$"
        xlabel = f"$\\frac{{ {pred_label.replace('$', '')} - {true_label.replace('$', '')} }}{{ {true_label.replace('$', '')} }}$"
        ylabel = "Density"
        filename = (
            f"relative_error_distributions_{variable_filenames[var]}_low_pt_1_2.png"
        )
        filename = output_dir / filename

        # Get the transverse momentum values (assumed to be the first column).
        assert config["output_variables"][0] in ["pT", "qopT", "qpT"]
        pt_true = np.array(p_true_list)[:, 0]
        # Convert the transverse momentum values to the current variable.
        pt_true = var_to_pT(values=pt_true, from_var=config["output_variables"][0])

        # Apply the pT mask: only select events with 1 < p_T < 2 GeV.
        mask = (pt_true > 1) & (pt_true < 2)

        # Get the true values for the current variable.
        pi_true_values = np.array(p_true_list)[:, var_index]
        if var == "p_T":
            pi_true_values = var_to_pT(
                values=pi_true_values, from_var=config["output_variables"][var_index]
            )

        # Add a gaussian with mean and std of the distribution
        from scipy.stats import norm

        model_root_fits = {}
        norm_fits = {}
        model_quantiles = {}
        model_data = {}
        model_bins_dict = {}
        # Loop over each model.
        for model_name, model_info in models.items():
            # Compute the relative error: (prediction - true)/true.
            pi_pred_values = np.array(model_info["data_list"])[:, var_index]
            if var == "p_T":
                pi_pred_values = var_to_pT(
                    values=pi_pred_values,
                    from_var=config["output_variables"][var_index],
                )
            errors = pi_pred_values - pi_true_values
            relative_errors = errors / pi_true_values
            # Apply the same pT mask.
            relative_errors = relative_errors[mask]

            model_data[model_name] = relative_errors

            model_mean, model_std = norm.fit(
                relative_errors
            )  # that's a global fit, i.e. of all the data with MLE.

            norm_fits[model_name] = (model_mean, model_std)

            # Get the 82% quantile
            from scipy.stats import scoreatpercentile

            high_quantile_target = (1 - (1 / 1744278)) * 100
            q_high = scoreatpercentile(relative_errors, high_quantile_target)

            print(f"{high_quantile_target}% quantile for {model_name}: {q_high}")

            # Same with lower quantile
            low_quantile_target = 100 - high_quantile_target
            q_low = scoreatpercentile(relative_errors, low_quantile_target)
            print(f"{low_quantile_target}% quantile for {model_name}: {q_low}")

            model_quantiles[model_name] = (q_low, q_high)

            from scipy.optimize import curve_fit

            n_sig = 5
            q_low = max(q_low, -n_sig * model_std + model_mean)
            q_high = min(q_high, n_sig * model_std + model_mean)

            if var == "p_z":
                q_low = -2
                q_high = 2

            n_bins = 100

            fit_range = (q_low, q_high)
            # Prune the data
            relative_errors = relative_errors[
                (relative_errors > q_low) & (relative_errors < q_high)
            ]
            # Print norm fit of pruned data
            print(f"Pruned fit for {model_name}: {norm.fit(relative_errors)}")

            n, model_bins = np.histogram(
                relative_errors, bins=n_bins, density=True, range=fit_range
            )
            model_bins_dict[model_name] = model_bins
            param_model, cov_model = curve_fit(
                norm.pdf,
                model_bins[:-1],
                n,
                [model_mean, model_std],
                bounds=([-np.inf, 0], [np.inf, np.inf]),
            )
            perr_model = np.sqrt(np.diag(cov_model))
            model_mean, model_std = param_model
            model_mean_err, model_std_err = perr_model
            norm_fits[model_name] = (
                (model_mean, model_mean_err),
                (model_std, model_std_err),
            )

            # Create the histogram.
            # Use model_bins_dict for the bins
            hist = ROOT.TH1F(
                f"hist_{model_name}_{var}",
                f"{model_name} Model: Relative Error Distribution for {var} (1 < p_T < 2 GeV)",
                len(model_bins) - 1,
                model_bins,
            )

            # Fill the histogram.
            for err in relative_errors:
                hist.Fill(err)
            # Normalize the histogram if there are entries.
            if hist.Integral() > 0:
                hist.Scale(1.0 / hist.Integral())

            # Fit a Gaussian to the histogram.
            fit = ROOT.TF1(
                f"fit_{model_name}_{var}",
                "gaus",
                np.min(relative_errors),
                np.max(relative_errors),
            )

            # Set the initial parameters (mean, std)
            fit.SetParameters(1e-2, 1e-1)

            # Use strategy 2 for the fit
            hist.Fit(fit, "R")

            # Retrieve the fit parameters.
            mean_fit = fit.GetParameter(1)
            sigma_fit = fit.GetParameter(2)
            mean_fit_err = fit.GetParError(1)
            sigma_fit_err = fit.GetParError(2)
            print(
                f"{model_name} fit: {mean_fit = } +/- {mean_fit_err = }",
                f"{sigma_fit = } +/- {sigma_fit_err = }",
            )

            model_root_fits[model_name] = (
                (mean_fit, mean_fit_err),
                (sigma_fit, sigma_fit_err),
            )

        # Plot pt_true and pt_pred
        plt.figure(figsize=(6 * len(models), 6))
        plt.suptitle(title, fontsize=16)

        for model_index, (model_name, model_info) in enumerate(models.items(), start=1):
            # MSE Model
            plt.subplot(1, len(models), model_index)
            bins = model_bins_dict[model_name]
            plt.hist(
                model_data[model_name],
                bins=bins,
                alpha=0.7,
                label=f"{model_info['label']} Model",
                color=f"{model_info['color']}",
                edgecolor="black",
                density=True,
            )

            model_mean, model_std = norm_fits[model_name]
            model_mean, model_mean_err = model_mean
            model_std, model_std_err = model_std

            model_root_mean, model_root_std = model_root_fits[model_name]
            model_root_mean, model_root_mean_err = model_root_mean
            model_root_std, model_root_std_err = model_root_std

            # Limit at 5 sigma
            plt.xlim(*model_quantiles[model_name])
            xmin, xmax = plt.xlim()
            x = np.linspace(xmin, xmax, 100)
            p_curve_fit = norm.pdf(x, model_mean, model_std)
            p_root_fit = norm.pdf(x, model_root_mean, model_root_std)
            plt.plot(
                x,
                p_curve_fit,
                "r",
                linewidth=2,
                linestyle="--",
                label=f"Curve Fit Gaussian\nMean: {model_mean:5.3f} +/- {model_mean_err:5.3f}, Std: {model_std:5.3f} +/- {model_std_err:5.3f}",
            )
            plt.plot(
                x,
                p_root_fit,
                "g",
                linewidth=2,
                linestyle="--",
                label=f"ROOT Fit Gaussian\nMean: {model_root_mean:5.3f} +/- {model_root_mean_err:5.3f}, Std: {model_root_std:5.3f} +/- {model_root_std_err:5.3f}",
            )

            plt.xlabel(xlabel, fontsize=12)
            plt.ylabel(ylabel, fontsize=12)
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(filename, format="png")
        plt.close()

        for model_name, model_info in models.items():
            model_mean, model_std = norm_fits[model_name]
            model_mean, model_mean_err = model_mean
            model_std, model_std_err = model_std
            print(
                f"Likely hood Curve Fit {model_name}: {compute_likelyhood(model_data[model_name], lambda x: norm.pdf(x, model_mean, model_std))}"
            )
            model_root_mean, model_root_std = model_root_fits[model_name]
            model_root_mean, model_root_mean_err = model_root_mean
            model_root_std, model_root_std_err = model_root_std
            print(
                f"Likely hood ROOT Fit {model_name}: {compute_likelyhood(model_data[model_name], lambda x: norm.pdf(x, model_root_mean, model_root_std))}"
            )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Track parameter analysis script.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="TrackML_zenodo",
        choices=["TrackML_kaggle", "TrackML_zenodo", "TrackML_zenodo_full"],
        help="Dataset to use",
    )
    parser.add_argument(
        "--input-vars",
        type=str,
        nargs="+",
        default=["tr", "dphi", "tz"],
        help="Input variables",
    )
    parser.add_argument(
        "--output-vars",
        type=str,
        nargs="+",
        default=["qopT", "pz"],
        help="Output variables",
    )
    parser.add_argument(
        "--min-hits", type=int, default=3, help="Minimum number of hits"
    )
    parser.add_argument("--min-pt", type=float, default=0.5, help="Minimum pT")
    parser.add_argument("--max-pt", type=float, default=10, help="Maximum pT")
    parser.add_argument(
        "--max-abs-eta", type=float, default=1, help="Maximum absolute eta"
    )
    parser.add_argument(
        "--output-dir", type=str, default="Results/TrackML/", help="Output directory"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="Models/TrackML/version_487",
        help="Model directory",
    )
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Set up configuration
    config = {
        "min_hits": args.min_hits,
        "min_pt": args.min_pt,
        "max_pt": args.max_pt,
        "keep_secondaries": False,
        "max_abs_eta": args.max_abs_eta,
        "sort_by_radius": True,
        "cut_scattered": True,
        "input_variables": args.input_vars,
        "output_variables": args.output_vars,
    }

    # Set up dataset
    dataset_name = args.dataset
    if dataset_name == "TrackML_kaggle":
        dataset_dir = Path("Data/Tml/train_1")
    elif dataset_name == "TrackML_zenodo":
        dataset_dir = Path("Data/TrackML/training_part01")
    else:
        dataset_dir = Path("Data/TrackML/full_dataset")

    # Set up variable labels
    variable_labels = {
        "tx": "tx",
        "ty": "ty",
        "tz": "tz",
        "tr": "tr",
        "tphi": r"t\varphi",
        "dphi": r"d\varphi",
        "pT": "p_T",
        "qopT": "q/p_T",
        "qpT": "q*p_T",
        "pz": "p_z",
        "pT_circle_estimate": "p_T (circle estimate)",
    }

    # variable_filenames is a reverse mapping of variable_labels
    variable_filenames = {v: k for k, v in variable_labels.items()}

    # Create model dataframe
    model_data = [
        (
            ["tr", "dphi", "tz"],
            dataset_name,
            ["qopT", "pz"],
            args.model_dir,
            None,
            None,
        ),
    ]

    model_df = pd.DataFrame(
        model_data,
        columns=[
            "input_variables",
            "dataset_name",
            "output_variables",
            "mse_dir",
            "mse_inv_dir",
            "qloss_dir",
        ],
    )
    model_df["input_variables"] = model_df["input_variables"].apply(tuple)
    model_df["output_variables"] = model_df["output_variables"].apply(tuple)

    # Function to retrieve available models
    def get_available_models(model_df, config_input, dataset_name, config_output):
        config_input = tuple(config_input)
        config_output = tuple(config_output)

        filtered_df = model_df[
            (model_df["input_variables"] == config_input)
            & (model_df["dataset_name"] == dataset_name)
            & (model_df["output_variables"] == config_output)
        ]

        if not filtered_df.empty:
            model_dict = {
                "mse_dir": filtered_df.iloc[0]["mse_dir"],
                "mse_inv_dir": filtered_df.iloc[0]["mse_inv_dir"],
                "qloss_dir": filtered_df.iloc[0]["qloss_dir"],
            }
            return {
                k: v for k, v in model_dict.items() if v is not None
            }  # Remove None values
        else:
            return {}

    available_models = get_available_models(
        model_df, config["input_variables"], dataset_name, config["output_variables"]
    )
    print(available_models)

    # Set up model directories
    models_dir = {
        "MSE": Path(available_models["mse_dir"]),
    }
    if "qloss_dir" in available_models:
        models_dir["QLoss"] = Path(available_models["qloss_dir"])
    if "mse_inv_dir" in available_models:
        models_dir["MSE_inv"] = Path(available_models["mse_inv_dir"])

    # Define models
    models = {
        "MSE": {
            "color": "skyblue",
            "cmap": "Blues",
            "label": "MSE",
            "marker": "o",
        },
    }
    if "QLoss" in models_dir:
        models["QLoss"] = {
            "color": "coral",
            "cmap": "Reds",
            "label": "Quantile Loss",
            "marker": "^",
        }

    # Set up title suffix
    title_suffix = ""

    # Add input and output variables to the title suffix
    title_suffix += f" ((${'$, $'.join([variable_labels[var] for var in config['input_variables']])}$)"
    title_suffix += r" $ \to $ "
    title_suffix += f"(${'$, $'.join([variable_labels[var] for var in config['output_variables']])}$))"

    # Set up output directory
    output_dir = Path(args.output_dir)

    input_variables = "".join(config["input_variables"]).replace("t", "")

    output_variables = "_".join(config["output_variables"])

    experiment_name = f"{input_variables}_n_hits_{config['min_hits']}_{output_variables}_POC_{dataset_name}_full_dataset"
    experiment_name += "_bacth_size_2048"

    output_dir = output_dir / experiment_name

    # Create the output directory if it does not exist
    output_dir.mkdir(exist_ok=True, parents=True)

    # Set up data module
    dataset_wrapper = DataModule(
        dataset_type="tml",
        dataset_dir=dataset_dir,
        use_wrapper=False,
        kwargs=config,
    )

    dataset_wrapper.setup(stage="test")
    tl = dataset_wrapper.test_dataloader()

    # Validate model configs
    mse_config = get_config(models_dir["MSE"])
    validate_model_config(
        mse_config,
        config,
        dataset_dir=dataset_dir,
        criterion="mse",
        allow_other_datasets=dataset_name == "TrackML_zenodo_full",
    )
    if "QLoss" in models_dir:
        qloss_config = get_config(models_dir["QLoss"])
        validate_model_config(
            qloss_config, config, dataset_dir=dataset_dir, criterion="qloss-0.5"
        )

    # Load models
    for model in models_dir:
        if model not in models:
            print(f"Skipping {model} model (not defined in models)")
            continue
        print(f"Loading {model} model")
        ml_model_path = get_model_checkpoint_path(models_dir[model])
        # _instantiator is a function that is called to instantiate the model
        # Here the models have been trained using cli raising an error if _instantiator is not None
        ml_model = TrackFormer.load_from_checkpoint(
            checkpoint_path=ml_model_path,
            _instantiator=None,
            metric="sign",
        )
        ml_model.eval().to("cpu")
        models[model]["model"] = ml_model

    # Process data
    p_true_list = []
    n_hits_list = []

    bad_tracks_q = []
    bad_tracks_mse = []
    bad_tracks_mse_pT = []
    bad_tracks_mse_pT_sign = []
    worst_tracks_mse_pT = []

    abs_relative_error_model_max = {model: 0 for model in models}
    abs_relative_error_model_max_pT = {model: 0 for model in models}

    for model in models:
        models[model]["data_list"] = []

    print("Processing track data...")
    with torch.no_grad():
        for i, (input, mask, target) in enumerate(tqdm(tl)):

            # Transformers
            p_true = target
            p_true_list.extend(p_true.tolist())
            n_hits_list.extend(mask.sum(dim=1).tolist())

            for model in models:
                ml_model = models[model]["model"]
                p_pred_model = ml_model(input, mask=mask)
                models[model]["data_list"].extend(p_pred_model.tolist())

                error_model = p_pred_model - p_true
                relative_error_model = error_model / p_true
                abs_relative_error_model = np.abs(relative_error_model)

                if model != "MSE":
                    continue

                if (abs_relative_error_model > 10).any():
                    mask_ = (abs_relative_error_model > 10).any(axis=1)
                    selected = input[mask_][0]

                    relative_error_values = relative_error_model[mask_].tolist()
                    p_values = p_true[mask_].tolist()
                    bad_tracks_mse.extend(
                        ((error, p), selected)
                        for error, p in zip(relative_error_values, p_values)
                    )

                abs_relative_error_model_max[model] = max(
                    abs_relative_error_model[:, 0].max(),
                    abs_relative_error_model_max[model],
                )

                if (
                    abs_relative_error_model[:, 0].max()
                    == abs_relative_error_model_max[model]
                ):
                    mask_ = (
                        abs_relative_error_model[:, 0]
                        == abs_relative_error_model_max[model]
                    )
                    selected = input[mask_][0]

                    relative_error_values = relative_error_model[mask_].tolist()
                    p_values = p_true[mask_].tolist()
                    worst_tracks_mse_pT.extend(
                        ((error, p), selected)
                        for error, p in zip(relative_error_values, p_values)
                    )

                if abs_relative_error_model[:, 0].max() > 1:
                    mask_ = abs_relative_error_model[:, 0] > 1
                    selected = input[mask_][0]

                    relative_error_values = relative_error_model[mask_].tolist()
                    p_values = p_true[mask_].tolist()
                    bad_tracks_mse_pT.extend(
                        ((error, p), selected)
                        for error, p in zip(relative_error_values, p_values)
                    )

                if (np.sign(p_pred_model[:, 0]) != np.sign(p_true[:, 0])).any():
                    mask_ = np.sign(p_pred_model[:, 0]) != np.sign(p_true[:, 0])
                    selected = input[mask_][0]

                    relative_error_values = relative_error_model[mask_].tolist()
                    p_values = p_true[mask_].tolist()
                    bad_tracks_mse_pT_sign.extend(
                        ((error, p), selected)
                        for error, p in zip(relative_error_values, p_values)
                    )

    # Generate plots
    print("Generating plots...")

    # Plot relative error vs number of hits
    plot_err_vs_n_hits(
        target_labels=[variable_labels[var] for var in config["output_variables"]],
        p_true_list=p_true_list,
        n_hits_list=n_hits_list,
        models=models,
        config=config,
        title_suffix=title_suffix,
        output_dir=output_dir,
        variable_filenames=variable_filenames,
        variable_labels=variable_labels,
    )

    if config["output_variables"][0] in ["qopT", "qpT"]:
        plot_err_vs_n_hits(
            target_labels=["p_T"],
            p_true_list=p_true_list,
            n_hits_list=n_hits_list,
            models=models,
            config=config,
            title_suffix=title_suffix,
            output_dir=output_dir,
            variable_filenames=variable_filenames,
            variable_labels=variable_labels,
        )

    # Plot true vs predicted values
    plot_pi_true_vs_pred(
        target_labels=[variable_labels[var] for var in config["output_variables"]],
        p_true_list=p_true_list,
        models=models,
        config=config,
        title_suffix=title_suffix,
        output_dir=output_dir,
        variable_filenames=variable_filenames,
        variable_labels=variable_labels,
    )

    if config["output_variables"][0] in ["qopT", "qpT"]:
        plot_pi_true_vs_pred(
            target_labels=["p_T"],
            p_true_list=p_true_list,
            models=models,
            config=config,
            title_suffix=title_suffix,
            output_dir=output_dir,
            variable_filenames=variable_filenames,
            variable_labels=variable_labels,
        )

    # Plot true vs predicted values for low pT
    plot_pi_true_vs_pred(
        target_labels=[variable_labels[var] for var in config["output_variables"]],
        p_true_list=p_true_list,
        models=models,
        config=config,
        title_suffix=title_suffix,
        output_dir=output_dir,
        variable_filenames=variable_filenames,
        variable_labels=variable_labels,
        low_pt=True,
    )

    if config["output_variables"][0] in ["qopT", "qpT"]:
        plot_pi_true_vs_pred(
            target_labels=["p_T"],
            p_true_list=p_true_list,
            models=models,
            config=config,
            title_suffix=title_suffix,
            output_dir=output_dir,
            variable_filenames=variable_filenames,
            variable_labels=variable_labels,
            low_pt=True,
        )

    # Plot relative resolutions
    plot_pi_rel_resolutions(
        target_labels=[variable_labels[var] for var in config["output_variables"]],
        p_true_list=p_true_list,
        models=models,
        config=config,
        title_suffix=title_suffix,
        output_dir=output_dir,
        variable_filenames=variable_filenames,
        variable_labels=variable_labels,
    )

    if config["output_variables"][0] in ["qopT", "qpT"]:
        plot_pi_rel_resolutions(
            target_labels=["p_T"],
            p_true_list=p_true_list,
            models=models,
            config=config,
            title_suffix=title_suffix,
            output_dir=output_dir,
            variable_filenames=variable_filenames,
            variable_labels=variable_labels,
        )

    # Plot relative resolutions with pruning
    plot_pi_rel_resolutions(
        target_labels=[variable_labels[var] for var in config["output_variables"]],
        p_true_list=p_true_list,
        models=models,
        config=config,
        title_suffix=title_suffix,
        output_dir=output_dir,
        variable_filenames=variable_filenames,
        variable_labels=variable_labels,
        pruning=True,
    )

    if config["output_variables"][0] in ["qopT", "qpT"]:
        plot_pi_rel_resolutions(
            target_labels=["p_T"],
            p_true_list=p_true_list,
            models=models,
            config=config,
            title_suffix=title_suffix,
            output_dir=output_dir,
            variable_filenames=variable_filenames,
            variable_labels=variable_labels,
            pruning=True,
        )

    # Plot relative resolutions for low pT
    plot_pi_rel_resolutions(
        target_labels=[variable_labels[var] for var in config["output_variables"]],
        p_true_list=p_true_list,
        models=models,
        config=config,
        title_suffix=title_suffix,
        output_dir=output_dir,
        variable_filenames=variable_filenames,
        variable_labels=variable_labels,
        low_pt=True,
    )

    if config["output_variables"][0] in ["qopT", "qpT"]:
        plot_pi_rel_resolutions(
            target_labels=["p_T"],
            p_true_list=p_true_list,
            models=models,
            config=config,
            title_suffix=title_suffix,
            output_dir=output_dir,
            variable_filenames=variable_filenames,
            variable_labels=variable_labels,
            low_pt=True,
        )

    # Plot 2D histograms
    plot_2d_histogram(
        target_labels=[variable_labels[var] for var in config["output_variables"]],
        p_true_list=p_true_list,
        models=models,
        config=config,
        title_suffix=title_suffix,
        output_dir=output_dir,
        variable_filenames=variable_filenames,
        variable_labels=variable_labels,
    )

    if config["output_variables"][0] in ["qopT", "qpT"]:
        plot_2d_histogram(
            target_labels=["p_T"],
            p_true_list=p_true_list,
            models=models,
            config=config,
            title_suffix=title_suffix,
            output_dir=output_dir,
            variable_filenames=variable_filenames,
            variable_labels=variable_labels,
        )

    # Plot relative error distributions for low pT
    plot_pi_relative_error_distributions_low_pt_1_2(
        target_labels=[variable_labels[var] for var in config["output_variables"]],
        p_true_list=p_true_list,
        models=models,
        config=config,
        title_suffix=title_suffix,
        output_dir=output_dir,
        variable_filenames=variable_filenames,
        variable_labels=variable_labels,
    )

    if config["output_variables"][0] in ["qopT", "qpT"]:
        plot_pi_relative_error_distributions_low_pt_1_2(
            target_labels=["p_T"],
            p_true_list=p_true_list,
            models=models,
            config=config,
            title_suffix=title_suffix,
            output_dir=output_dir,
            variable_filenames=variable_filenames,
            variable_labels=variable_labels,
        )

    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
