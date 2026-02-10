#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utility functions for loading models and data processing.
"""

import numpy as np
from pathlib import Path


def get_model_checkpoint_path(version_dir: str):
    """Get the checkpoint path from a version directory."""
    version_dir = Path(version_dir)
    checkpoint_dir = version_dir / "checkpoints"
    checkpoint_files = list(checkpoint_dir.glob("model*.ckpt"))
    if len(checkpoint_files) == 0:
        raise FileNotFoundError(
            f"No checkpoints found in {checkpoint_dir}. Please ensure the model has been trained."
        )
    print(f"Found {len(checkpoint_files)} checkpoints in {checkpoint_dir}")
    return checkpoint_files


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


def get_smallest_confidence_interval(data, alpha=0.6827):
    """
    Find the smallest interval containing alpha percentage of the data.

    Parameters:
        data (numpy array): Input data array
        alpha (float): Confidence level (default: 0.6827 corresponds to 1 sigma)

    Returns:
        tuple: (lower_bound, upper_bound, interval_center)
    """
    data = np.sort(data)
    n = len(data)
    n_in_interval = int(alpha * n)
    smallest_interval = np.inf
    for i in range(n - n_in_interval):
        interval = data[i + n_in_interval] - data[i]
        if interval < smallest_interval:
            smallest_interval = interval
            lower_bound = data[i]
            upper_bound = data[i + n_in_interval]
    # Determine the center of the confidence interval
    lower_bound_index = np.where(data == lower_bound)[0][0]
    upper_bound_index = np.where(data == upper_bound)[0][0]
    interval_center = data[(lower_bound_index + upper_bound_index) // 2]
    return lower_bound, upper_bound, interval_center
