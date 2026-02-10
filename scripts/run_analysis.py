#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to run track parameter analysis and generate visualizations.
This is a commandline version of the dataanalysis.ipynb notebook.
"""

# import os
import sys

# import json
import argparse
from pathlib import Path
import numpy as np

# import pandas as pd
# import matplotlib.pyplot as plt
import torch

# from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Import local modules
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.datasets.datamodules import DataModule
from src.my_model.transformer import TrackFormer
from src.analysis.utils import (
    get_model_checkpoint_path,
    get_config,
    validate_model_config,
    # var_to_pT,
    # compute_likelyhood,
    # compute_track_resolution,
    # get_smallest_confidence_interval,
)
from src.analysis.visualization import (
    # view_trajectory,
    # sync_plot_limits,
    plot_err_vs_n_hits,
    plot_binned_confusion_matrix,
    plot_pi_error_distributions,
    plot_pi_rel_resolutions,
    plot_2d_histogram,
    plot_pi_relative_error_distributions,
    plot_pi_error_distributions_p,
    plot_pi_relative_error_distributions_low_pt_1_2,
)


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Run track parameter analysis")

    # Dataset configuration
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="TrackML_zenodo",
        choices=["TrackML_kaggle", "TrackML_zenodo", "TrackML_zenodo_full"],
        help="Name of dataset to use",
    )
    parser.add_argument(
        "--min-hits", type=int, default=3, help="Minimum number of hits required"
    )
    parser.add_argument("--min-pt", type=float, default=0.5, help="Minimum pT value")
    parser.add_argument("--max-pt", type=float, default=10, help="Maximum pT value")
    parser.add_argument(
        "--max-abs-eta", type=float, default=1, help="Maximum absolute eta value"
    )

    # Input/output variables
    parser.add_argument(
        "--input-variables",
        type=str,
        default="tr,dphi,tz",
        help="Comma-separated list of input variables",
    )
    parser.add_argument(
        "--output-variables",
        type=str,
        default="qopT,pz",
        help="Comma-separated list of output variables",
    )

    # Model configuration
    parser.add_argument(
        "--model-name",
        type=str,
        default="MSE",
        help="Name of the model (e.g., MSE, QLoss)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Directory containing the model to use",
    )

    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save output files (default: auto-generated)",
    )
    parser.add_argument(
        "--show", action="store_true", help="Display plots interactively"
    )

    # Plot selection
    parser.add_argument(
        "--plots",
        type=str,
        default="all",
        help="Comma-separated list of plots to generate (or 'all' for all plots)",
    )

    # Publication argument
    parser.add_argument(
        "--publication",
        action="store_true",
        help="Generate publication-ready plots",
    )

    return parser.parse_args()


def setup_config(args):
    """Set up configuration dict from arguments"""
    # Basic config
    config = {
        "min_hits": args.min_hits,
        "min_pt": args.min_pt,
        "max_pt": args.max_pt,
        "keep_secondaries": False,
        "max_abs_eta": args.max_abs_eta,
        "sort_by_radius": True,
        "cut_scattered": True,
        "input_variables": args.input_variables.split(","),
        "output_variables": args.output_variables.split(","),
    }

    return config


def setup_dataset(dataset_name, config):
    """Set up dataset and data directories"""
    if dataset_name == "TrackML_kaggle":
        dataset_dir = Path("Data/Tml/train_1")
    elif dataset_name == "TrackML_zenodo":
        dataset_dir = Path("Data/TrackML/training_part01")
    else:  # TrackML_zenodo_full
        dataset_dir = Path("Data/TrackML/full_dataset")

    return dataset_dir


def setup_variable_mappings():
    """Set up variable mappings for labels and units"""
    # Variable label mappings
    variable_labels = {
        "tx": r"x_{truth}",
        "ty": r"y_{truth}",
        "tz": r"z_{truth}",
        "tr": r"r_{truth}",
        "tphi": r"\varphi_{truth}",
        "dphi": r"\Delta\varphi",
        "pT": "p_T",
        # "qopT": r"\frac{q}{p_T}",
        "qopT": "q/p_T",
        "qpT": "q*p_T",
        "pz": "p_z",
        "pT_circle_estimate": "p_T (circle estimate)",
    }

    # Variable filenames is a reverse mapping of variable_labels
    variable_filenames = {v: k for k, v in variable_labels.items()}

    # Variable units
    variable_units = {
        "tx": "mm",
        "ty": "mm",
        "tz": "mm",
        "tr": "mm",
        "tphi": "rad",
        "dphi": "rad",
        "pT": "GeV",
        "qopT": "1/GeV",
        "qpT": "GeV",
        "pz": "GeV",
    }

    return variable_labels, variable_filenames, variable_units


def setup_output_dir(args, config, dataset_name):
    """Set up output directory"""
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Auto-generate output dir based on configuration
        base_output_dir = Path("Results/TrackML/")

        input_variables = "".join(config["input_variables"]).replace("t", "")
        output_variables = "_".join(config["output_variables"])

        experiment_name = f"{input_variables}_n_hits_{config['min_hits']}_{output_variables}_POC_{dataset_name}"
        output_dir = base_output_dir / experiment_name

    # Create the output directory if it does not exist
    output_dir.mkdir(exist_ok=True, parents=True)
    return output_dir


def setup_model(model_dir, model_name, dataset_dir, config, dataset_name):
    """Set up the model"""
    # Define model attributes
    models = {
        model_name: {
            "color": "skyblue" if model_name == "MSE" else "coral",
            "cmap": "Blues" if model_name == "MSE" else "Reds",
            "label": "MSE" if model_name == "MSE" else "Quantile Loss",
            "marker": "o" if model_name == "MSE" else "^",
        }
    }

    models_dir = {model_name: Path(model_dir)}

    # Load model configuration and validate
    model_config = get_config(models_dir[model_name])
    try:
        criterion = "mse" if model_name == "MSE" else "qloss-0.5"
        validate_model_config(
            model_config,
            config,
            dataset_dir=dataset_dir,
            criterion=criterion,
            allow_other_datasets=(dataset_name == "TrackML_zenodo_full"),
        )
    except AssertionError as e:
        print(f"Warning: Model configuration validation failed: {e}")

    # Load the model
    ml_model_path = get_model_checkpoint_path(models_dir[model_name])
    ml_model = TrackFormer.load_from_checkpoint(
        checkpoint_path=ml_model_path,
        _instantiator=None,
        metric="sign",
    )
    # Get available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # ml_model = ml_model.to(device)
    ml_model.eval().to(device)
    models[model_name]["model"] = ml_model

    return models, models_dir


def run_inference(dataset_wrapper, models):
    """Run inference on the dataset"""
    tl = dataset_wrapper.test_dataloader()
    p_true_list = []
    n_hits_list = []

    # bad_tracks_q = []
    bad_tracks_mse = []
    bad_tracks_mse_pT = []
    bad_tracks_mse_pT_sign = []
    worst_tracks_mse_pT = []

    abs_relative_error_model_max = {model: 0 for model in models}
    # abs_relative_error_model_max_pT = {model: 0 for model in models}

    for model in models:
        models[model]["data_list"] = []

    # Get available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Running inference...")
    with torch.no_grad():
        for i, (input, mask, target) in enumerate(tqdm(tl)):
            # Move data to device
            input = input.to(device)
            mask = mask.to(device)
            target = target.to(device)
            # Transformers
            p_true = target.cpu().numpy()
            p_true_list.extend(p_true.tolist())
            n_hits_list.extend(mask.sum(dim=1).tolist())

            for model in models:
                ml_model = models[model]["model"]
                p_pred_model = ml_model(input, mask=mask)
                models[model]["data_list"].extend(p_pred_model.tolist())

                # Move data to CPU for further processing
                p_pred_model = p_pred_model.cpu().numpy()
                input = input.cpu().numpy()
                mask = mask.cpu().numpy()
                # Calculate relative error
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

    print(f"Max relative error for pT:")
    for model in models:
        print(f"{model}: {abs_relative_error_model_max[model]}")

    result_data = {
        "p_true_list": p_true_list,
        "n_hits_list": n_hits_list,
        "bad_tracks_mse": bad_tracks_mse,
        "bad_tracks_mse_pT": bad_tracks_mse_pT,
        "bad_tracks_mse_pT_sign": bad_tracks_mse_pT_sign,
        "worst_tracks_mse_pT": worst_tracks_mse_pT,
    }

    return result_data


def generate_plots(
    args,
    config,
    result_data,
    models,
    output_dir,
    dataset_name,
    variable_labels,
    variable_filenames,
):
    """Generate plots based on configuration"""
    p_true_list = result_data["p_true_list"]
    n_hits_list = result_data["n_hits_list"]

    publication = args.publication

    # Generate title suffix
    title_suffix = ""
    if not publication:
        title_suffix += f" ((${'$, $'.join([variable_labels[var] for var in config['input_variables']])}$)"
        title_suffix += r" $ \to $ "
        title_suffix += f"(${'$, $'.join([variable_labels[var] for var in config['output_variables']])}$))"

    # Determine which plots to generate
    all_plots = [
        "err_vs_n_hits",
        "true_vs_pred",
        "error_distributions",
        "rel_resolutions",
        "rel_resolutions_pruned",
        "rel_resolutions_low_pt",
        "2d_histogram",
        "relative_error_distributions",
        "relative_error_distributions_abs",
        "relative_error_distributions_low_pt",
        "error_distributions_p",
        "relative_error_distributions_low_pt_1_2",
    ]

    if args.plots.lower() == "all":
        plots_to_generate = all_plots
    else:
        plots_to_generate = args.plots.split(",")

    show = args.show
    if publication:
        show = False
        output_dir = output_dir / "publication"
        output_dir.mkdir(exist_ok=True, parents=True)
        print(f"Publication output will be saved to: {output_dir}")
    else:
        output_dir = output_dir / "plots"
        output_dir.mkdir(exist_ok=True, parents=True)
        print(f"Plots will be saved to: {output_dir}")

    target_labels_list = [[variable_labels[var] for var in config["output_variables"]]]
    if config["output_variables"][0] in ["qopT", "qpT"]:
        target_labels_list.append(["p_T"])

    # Generate selected plots
    for plot_type in plots_to_generate:
        print(f"Generating {plot_type} plots...")

        for target_labels in target_labels_list:

            if plot_type == "err_vs_n_hits":
                plot_err_vs_n_hits(
                    target_labels=target_labels,
                    p_true_list=p_true_list,
                    n_hits_list=n_hits_list,
                    models=models,
                    config=config,
                    title_suffix=title_suffix,
                    output_dir=output_dir,
                    variable_filenames=variable_filenames,
                    low_pt=False,
                    show=show,
                    publication=False,
                )
            elif plot_type == "true_vs_pred":
                plot_binned_confusion_matrix(
                    target_labels=target_labels,
                    p_true_list=p_true_list,
                    models=models,
                    config=config,
                    title_suffix=title_suffix,
                    output_dir=output_dir,
                    variable_filenames=variable_filenames,
                    show=show,
                    publication=publication,
                )

                # Also generate low pT version
                plot_binned_confusion_matrix(
                    target_labels=target_labels,
                    p_true_list=p_true_list,
                    models=models,
                    config=config,
                    title_suffix=title_suffix,
                    output_dir=output_dir,
                    variable_filenames=variable_filenames,
                    low_pt=True,
                    show=show,
                    publication=publication,
                )

            elif plot_type == "error_distributions":
                plot_pi_error_distributions(
                    target_labels=target_labels,
                    p_true_list=p_true_list,
                    models=models,
                    config=config,
                    title_suffix=title_suffix,
                    output_dir=output_dir,
                    variable_filenames=variable_filenames,
                    show=show,
                    publication=False,
                )

            elif plot_type == "rel_resolutions":
                plot_pi_rel_resolutions(
                    target_labels=target_labels,
                    p_true_list=p_true_list,
                    models=models,
                    config=config,
                    title_suffix=title_suffix,
                    output_dir=output_dir,
                    variable_filenames=variable_filenames,
                    pruning=False,
                    low_pt=False,
                    show=show,
                    publication=publication,
                )

            elif plot_type == "rel_resolutions_pruned":
                plot_pi_rel_resolutions(
                    target_labels=target_labels,
                    p_true_list=p_true_list,
                    models=models,
                    config=config,
                    title_suffix=title_suffix,
                    output_dir=output_dir,
                    variable_filenames=variable_filenames,
                    pruning=True,
                    low_pt=False,
                    show=show,
                    publication=publication,
                )

            elif plot_type == "rel_resolutions_low_pt":
                plot_pi_rel_resolutions(
                    target_labels=target_labels,
                    p_true_list=p_true_list,
                    models=models,
                    config=config,
                    title_suffix=title_suffix,
                    output_dir=output_dir,
                    variable_filenames=variable_filenames,
                    pruning=False,
                    low_pt=True,
                    show=show,
                    publication=publication,
                )

            elif plot_type == "2d_histogram":
                plot_2d_histogram(
                    target_labels=target_labels,
                    p_true_list=p_true_list,
                    models=models,
                    config=config,
                    title_suffix=title_suffix,
                    output_dir=output_dir,
                    variable_filenames=variable_filenames,
                    show=show,
                    publication=publication,
                )

            elif plot_type == "relative_error_distributions":
                plot_pi_relative_error_distributions(
                    target_labels=target_labels,
                    p_true_list=p_true_list,
                    models=models,
                    config=config,
                    title_suffix=title_suffix,
                    output_dir=output_dir,
                    variable_filenames=variable_filenames,
                    abs=False,
                    low_pt=False,
                    show=show,
                    publication=publication,
                )

            elif plot_type == "relative_error_distributions_abs":
                plot_pi_relative_error_distributions(
                    target_labels=target_labels,
                    p_true_list=p_true_list,
                    models=models,
                    config=config,
                    title_suffix=title_suffix,
                    output_dir=output_dir,
                    variable_filenames=variable_filenames,
                    abs=True,
                    low_pt=False,
                    show=show,
                    publication=publication,
                )

            elif plot_type == "relative_error_distributions_low_pt":
                plot_pi_relative_error_distributions(
                    target_labels=target_labels,
                    p_true_list=p_true_list,
                    models=models,
                    config=config,
                    title_suffix=title_suffix,
                    output_dir=output_dir,
                    variable_filenames=variable_filenames,
                    abs=False,
                    low_pt=True,
                    show=show,
                    publication=publication,
                )

            elif plot_type == "error_distributions_p":
                plot_pi_error_distributions_p(
                    target_labels=target_labels,
                    p_true_list=p_true_list,
                    models=models,
                    config=config,
                    title_suffix=title_suffix,
                    output_dir=output_dir,
                    variable_filenames=variable_filenames,
                    show=show,
                    publication=publication,
                )

            elif plot_type == "relative_error_distributions_low_pt_1_2":
                plot_pi_relative_error_distributions_low_pt_1_2(
                    target_labels=target_labels,
                    p_true_list=p_true_list,
                    models=models,
                    config=config,
                    title_suffix=title_suffix,
                    output_dir=output_dir,
                    variable_filenames=variable_filenames,
                    show=show,
                    publication=publication,
                )

                # Get the maximum eta from the configuration
                max_abs_eta = config["max_abs_eta"]
                for min_eta in range(0, int(max_abs_eta)):
                    plot_pi_relative_error_distributions_low_pt_1_2(
                        target_labels=target_labels,
                        p_true_list=p_true_list,
                        models=models,
                        config=config,
                        title_suffix=title_suffix,
                        output_dir=output_dir,
                        variable_filenames=variable_filenames,
                        eta_range=(min_eta, min_eta + 1),
                        show=show,
                        publication=publication,
                    )
            else:
                print(f"Warning: Unknown plot type '{plot_type}'")


def main():
    """Main function"""
    # Parse command-line arguments
    args = parse_args()

    # Set up configuration
    config = setup_config(args)

    # Set up dataset and directories
    dataset_name = args.dataset_name
    dataset_dir = setup_dataset(dataset_name, config)

    # Set up variable mappings
    variable_labels, variable_filenames, variable_units = setup_variable_mappings()

    # Set up output directory
    output_dir = setup_output_dir(args, config, dataset_name)
    print(f"Output will be saved to: {output_dir}")

    # Set up models
    models, models_dir = setup_model(
        args.model_dir, args.model_name, dataset_dir, config, dataset_name
    )

    # Instantiate the DataModule
    print("Setting up dataset...")
    dataset_wrapper = DataModule(
        dataset_type="tml",
        dataset_dir=dataset_dir,
        use_wrapper=False,
        kwargs=config,
    )
    dataset_wrapper.setup(stage="test")

    # Run inference
    result_data = run_inference(dataset_wrapper, models)

    # Generate plots
    generate_plots(
        args,
        config,
        result_data,
        models,
        output_dir,
        dataset_name,
        variable_labels,
        variable_filenames,
    )

    print("Analysis complete!")


if __name__ == "__main__":
    main()
