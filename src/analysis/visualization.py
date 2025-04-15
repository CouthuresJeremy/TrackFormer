#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualization functions for track parameter analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.stats import norm, scoreatpercentile
from scipy.optimize import curve_fit

from src.analysis.utils import var_to_pT, compute_track_resolution, compute_likelyhood


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
    # variable_labels,
    low_pt=False,
    show=True,
    save=True,
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

            # Make it percentage
            relative_err_values = 100 * relative_err_values

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

        if save:
            plt.savefig(filename, format="png", bbox_inches="tight")
        if show:
            plt.show()
        plt.close()


def plot_pi_true_vs_pred(
    target_labels,
    p_true_list,
    models,
    config,
    title_suffix,
    output_dir,
    variable_filenames,
    # variable_labels,
    low_pt=False,
    show=True,
    save=True,
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

        if save:
            plt.savefig(filename, format="png", bbox_inches="tight")
        if show:
            plt.show()
        plt.close()


def plot_pi_error_distributions(
    target_labels,
    p_true_list,
    models,
    config,
    title_suffix,
    output_dir,
    variable_filenames,
    # variable_labels,
    show=True,
    save=True,
):
    """Plot error distributions."""
    for var_index, var in enumerate(target_labels):
        pi_true_values = np.array(p_true_list)[:, var_index]
        if var == "p_T":
            pi_true_values = var_to_pT(
                values=pi_true_values, from_var=config["output_variables"][var_index]
            )

        title = f"Error Distributions for ${var}$"
        title += title_suffix

        true_label = f"${var}^{{true}}$"
        pred_label = f"${var}^{{pred}}$"
        xlabel = true_label + (" [GeV]" if var.startswith("p") else "")
        ylabel = f"{pred_label} - {true_label}" + (
            " [GeV]" if var.startswith("p") else ""
        )
        filename = f"error_distributions_{variable_filenames[var]}.png"
        filename = output_dir / filename

        # Plot pt_true and pt_pred
        plt.figure(figsize=(6 * len(models), 6))
        plt.suptitle(title, fontsize=16)

        # Loop over each model.
        for model_index, model_info in enumerate(models.values(), start=1):
            pi_pred_values = np.array(model_info["data_list"])[:, var_index]
            if var == "p_T":
                pi_pred_values = var_to_pT(
                    values=pi_pred_values,
                    from_var=config["output_variables"][var_index],
                )

            errors = pi_pred_values - pi_true_values

            plt.subplot(1, len(models), model_index)
            plt.plot(
                pi_true_values,
                errors,
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
            if var == "p_T" and config["output_variables"][0] == "qopT":
                plt.ylim(-2, 10)

        sync_plot_limits()

        if save:
            plt.savefig(filename, format="png", bbox_inches="tight")
        if show:
            plt.show()
        plt.close()


def plot_pi_rel_resolutions(
    target_labels,
    p_true_list,
    models,
    config,
    title_suffix,
    output_dir,
    variable_filenames,
    # variable_labels,
    pruning=False,
    low_pt=False,
    show=True,
    save=True,
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

        if save:
            plt.savefig(filename, format="png", bbox_inches="tight")
        if show:
            plt.show()
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
        if var == "p_T":
            if config["output_variables"][0] == "qopT":
                plt.ylim(-30, 30)
            else:
                plt.ylim(-10, 10)
        if var == "p_z":
            plt.ylim(-1.5 * 100, 1.5 * 100)
            if pruning:
                plt.ylim(-20, 20)
        if save:
            plt.savefig(filename, format="png", bbox_inches="tight")
        if show:
            plt.show()
        plt.close()


def plot_2d_histogram(
    target_labels,
    p_true_list,
    models,
    config,
    title_suffix,
    output_dir,
    variable_filenames,
    # variable_labels,
    show=True,
    save=True,
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

        if save:
            plt.savefig(filename, format="png", bbox_inches="tight")
        if show:
            plt.show()
        plt.close()


def plot_pi_relative_error_distributions(
    target_labels,
    p_true_list,
    models,
    config,
    title_suffix,
    output_dir,
    variable_filenames,
    # variable_labels,
    abs=False,
    low_pt=False,
    show=True,
    save=True,
):
    """
    Plot distributions of the relative errors for the predicted variables.
    """
    for var_index, var in enumerate(target_labels):
        pi_true_values = np.array(p_true_list)[:, var_index]
        if var == "p_T":
            pi_true_values = var_to_pT(
                values=pi_true_values, from_var=config["output_variables"][var_index]
            )

        title = f"Relative Error Distributions for ${var}$"
        if abs:
            title = "Absolute " + title
        if low_pt:
            title += " ($p_T$ < 10 GeV)"
        title += title_suffix

        true_label = f"${var}^{{true}}$"
        pred_label = f"${var}^{{pred}}$"
        xlabel = f"$\\frac{{ {pred_label.replace('$', '')} - {true_label.replace('$', '')} }}{{ {true_label.replace('$', '')} }}$"
        if abs:
            xlabel = f"$\\left|\\frac{{ {pred_label.replace('$', '')} - {true_label.replace('$', '')} }}{{ {true_label.replace('$', '')} }}\\right|$"
        ylabel = "Counts"
        filename = f"relative_error_distributions_{variable_filenames[var]}.png"
        if abs:
            filename = "absolute_" + filename
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
            if abs:
                # Make it absolute
                relative_errors = np.abs(relative_errors)
            model_data[model_name] = relative_errors

        bins = np.linspace(
            min([min(v) for v in model_data.values()]),
            max([max(v) for v in model_data.values()]),
            50,
        )
        if abs:
            bins = np.linspace(0, max([max(v) for v in model_data.values()]), 50)

        for model_index, (model_name, model_info) in enumerate(models.items(), start=1):
            relative_errors = model_data[model_name]

            plt.subplot(1, len(models), model_index)
            plt.hist(
                relative_errors,
                bins=bins,
                alpha=0.7,
                label=f"{model_info['label']} Model",
                color=model_info["color"],
                edgecolor="black",
            )
            plt.xlabel(xlabel, fontsize=12)
            plt.ylabel(ylabel, fontsize=12)
            plt.yscale("log")
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.legend()

        sync_plot_limits()

        if save:
            plt.savefig(filename, format="png", bbox_inches="tight")
        if show:
            plt.show()
        plt.close()


def plot_pi_error_distributions_p(
    target_labels,
    p_true_list,
    models,
    config,
    title_suffix,
    output_dir,
    variable_filenames,
    # variable_labels,
    show=True,
    save=False,
):
    """
    Plot distributions of the errors for the predicted variables, with special handling for q/p_T.
    """
    for var_index, var in enumerate(target_labels):
        pi_true_values = np.array(p_true_list)[:, var_index]
        if var == "p_T":
            pi_true_values = var_to_pT(
                values=pi_true_values, from_var=config["output_variables"][var_index]
            )

        title = f"Relative Error Distributions for ${var}$"
        title += title_suffix

        true_label = f"${var}^{{true}}$"
        pred_label = f"${var}^{{pred}}$"
        xlabel = f"${pred_label.replace('$', '')} - {true_label.replace('$', '')}$"
        if var == "q/p_T":
            xlabel = f"$({pred_label.replace('$', '')} - {true_label.replace('$', '')}) \\times p_T$"
        ylabel = "Counts"
        filename = f"relative_error_distributions_p_{variable_filenames[var]}.png"
        filename = output_dir / filename

        # Plot pt_true and pt_pred
        plt.figure(figsize=(6 * len(models), 6))
        plt.suptitle(title, fontsize=16)

        # Loop over each model.
        model_data = {}
        for model_name, model_info in models.items():
            pi_pred_values = np.array(model_info["data_list"])[:, var_index]
            if var == "p_T":
                pi_pred_values = var_to_pT(
                    values=pi_pred_values,
                    from_var=config["output_variables"][var_index],
                )

            errors = pi_pred_values - pi_true_values
            if var == "q/p_T":
                pt_true_values = var_to_pT(
                    values=pi_true_values,
                    from_var=config["output_variables"][var_index],
                )
                errors = errors * pt_true_values
            model_data[model_name] = errors

        bins = np.linspace(
            min([min(v) for v in model_data.values()]),
            max([max(v) for v in model_data.values()]),
            50,
        )

        for model_index, (model_name, model_info) in enumerate(models.items(), start=1):
            model_errors = model_data[model_name]

            plt.subplot(1, len(models), model_index)
            print(
                f"{model_name}: {np.mean(model_errors):.4f} +- {np.std(model_errors):.4f}"
            )
            plt.hist(
                model_errors,
                bins=bins,
                alpha=0.7,
                label=f"{model_info['label']} Model",
                color=model_info["color"],
                edgecolor="black",
            )
            plt.xlabel(xlabel, fontsize=12)
            plt.ylabel(ylabel, fontsize=12)
            # plt.yscale("log")
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.legend()

        sync_plot_limits()

        if save:
            plt.savefig(filename, format="png", bbox_inches="tight")
        if show:
            plt.show()
        plt.close()


def plot_pi_relative_error_distributions_low_pt_1_2(
    target_labels,
    p_true_list,
    models,
    config,
    title_suffix,
    output_dir,
    variable_filenames,
    # variable_labels,
    show=True,
    save=True,
):
    """
    Plot detailed distributions of the relative errors for predicted variables,
    specifically for tracks with 1 < pT < 2 GeV.
    Includes Gaussian fits and statistical analysis.
    """
    # Import needed modules
    from scipy.stats import norm
    from scipy.optimize import curve_fit

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

        # Get the transverse momentum values
        assert config["output_variables"][0] in ["pT", "qopT", "qpT"]
        pt_true = np.array(p_true_list)[:, 0]
        # Convert the transverse momentum values to the current variable
        pt_true = var_to_pT(values=pt_true, from_var=config["output_variables"][0])

        # Apply the pT mask: only select events with 1 < p_T < 2 GeV
        mask = (pt_true > 1) & (pt_true < 2)

        # Get the true values for the current variable
        pi_true_values = np.array(p_true_list)[:, var_index]
        if var == "p_T":
            pi_true_values = var_to_pT(
                values=pi_true_values, from_var=config["output_variables"][var_index]
            )
        pi_true_values = pi_true_values[mask]

        # Add a gaussian with mean and std of the distribution
        model_root_fits = {}
        norm_fits = {}
        model_quantiles = {}
        model_data = {}
        model_bins_dict = {}

        # Loop over each model
        for model_name, model_info in models.items():
            # Compute the relative error: (prediction - true)/true
            pi_pred_values = np.array(model_info["data_list"])[:, var_index]
            if var == "p_T":
                pi_pred_values = var_to_pT(
                    values=pi_pred_values,
                    from_var=config["output_variables"][var_index],
                )
            # Apply the same pT mask
            pi_pred_values = pi_pred_values[mask]

            errors = pi_pred_values - pi_true_values
            relative_errors = errors / pi_true_values

            model_data[model_name] = relative_errors

            # Global fit of all the data with MLE
            model_mean, model_std = norm.fit(relative_errors)
            norm_fits[model_name] = (model_mean, model_std)

            # Get the 99.9999% quantile (approximately 5 sigma for normal distribution)
            from scipy.stats import scoreatpercentile

            high_quantile_target = 99.9999
            q_high = scoreatpercentile(relative_errors, high_quantile_target)

            print(f"{high_quantile_target}% quantile for {model_name}: {q_high}")

            # Same with lower quantile
            low_quantile_target = 100 - high_quantile_target
            q_low = scoreatpercentile(relative_errors, low_quantile_target)
            print(f"{low_quantile_target}% quantile for {model_name}: {q_low}")

            model_quantiles[model_name] = (q_low, q_high)

            n_sig = 5
            q_low = max(q_low, -n_sig * model_std + model_mean)
            q_high = min(q_high, n_sig * model_std + model_mean)

            if var == "p_z":
                q_low = max(-2, q_low)
                q_high = min(2, q_high)

            n_bins = 100

            fit_range = (q_low, q_high)
            # Prune the data
            relative_errors_pruned = relative_errors[
                (relative_errors > q_low) & (relative_errors < q_high)
            ]
            # Print norm fit of pruned data
            print(f"Pruned fit for {model_name}: {norm.fit(relative_errors_pruned)}")

            n, model_bins = np.histogram(
                relative_errors_pruned, bins=n_bins, density=True, range=fit_range
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

            # Create ROOT histogram for more advanced fitting if ROOT is available
            try:
                import ROOT

                # Create the histogram
                hist = ROOT.TH1F(
                    f"hist_{model_name}_{var}",
                    f"{model_name} Model: Relative Error Distribution for {var} (1 < p_T < 2 GeV)",
                    len(model_bins) - 1,
                    model_bins,
                )

                # Fill the histogram
                for err in relative_errors_pruned:
                    hist.Fill(err)
                # Normalize the histogram if there are entries
                if hist.Integral() > 0:
                    hist.Scale(1.0 / hist.Integral())

                # Fit a Gaussian to the histogram
                fit = ROOT.TF1(
                    f"fit_{model_name}_{var}",
                    "gaus",
                    np.min(relative_errors_pruned),
                    np.max(relative_errors_pruned),
                )

                # Set the initial parameters (mean, std)
                fit.SetParameters(1e-2, 1e-1)

                hist.Fit(fit, "R")

                # Retrieve the fit parameters
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
            except ImportError:
                # ROOT is not available, use normal distribution fit instead
                model_root_fits[model_name] = norm_fits[model_name]
            except Exception as e:
                # Handle any other exceptions that may occur
                print(f"Error fitting ROOT histogram for {model_name}: {e}")
                model_root_fits[model_name] = norm_fits[model_name]

        # Plot histograms and fits
        plt.figure(figsize=(6 * len(models), 6))
        plt.suptitle(title, fontsize=16)

        for model_index, (model_name, model_info) in enumerate(models.items(), start=1):
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

            # Get fit parameters and errors
            model_mean, model_std = norm_fits[model_name]
            model_mean, model_mean_err = model_mean
            model_std, model_std_err = model_std

            model_root_mean, model_root_std = model_root_fits[model_name]
            model_root_mean, model_root_mean_err = model_root_mean
            model_root_std, model_root_std_err = model_root_std

            # Plot fitted curves
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

        if save:
            plt.savefig(filename, format="png", bbox_inches="tight")
        if show:
            plt.show()
        plt.close()

        # Calculate likelihood for the fits
        for model_name, model_info in models.items():
            model_mean, model_std = norm_fits[model_name]
            model_mean, model_mean_err = model_mean
            model_std, model_std_err = model_std

            from src.analysis.utils import compute_likelyhood

            print(
                f"Likelihood Curve Fit {model_name}: {compute_likelyhood(model_data[model_name], lambda x: norm.pdf(x, model_mean, model_std))}"
            )
            model_root_mean, model_root_std = model_root_fits[model_name]
            model_root_mean, model_root_mean_err = model_root_mean
            model_root_std, model_root_std_err = model_root_std
            print(
                f"Likelihood ROOT Fit {model_name}: {compute_likelyhood(model_data[model_name], lambda x: norm.pdf(x, model_root_mean, model_root_std))}"
            )
