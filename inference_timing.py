# %%
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import plotly.graph_objects as go
import json

# %%
from src.datasets.datamodules import DataModule
from src.my_model.transformer import *

# %%
publication = False
# if publication:
#     plt.style.use('seaborn-whitegrid')
#     plt.rcParams['font.family'] = 'serif'
#     plt.rcParams['font.serif'] = ['Times New Roman']
#     plt.rcParams['font.size'] = 12
#     plt.rcParams['axes.titlesize'] = 12
#     plt.rcParams['axes.labelsize'] = 12
#     plt.rcParams['xtick.labelsize'] = 12
#     plt.rcParams['ytick.labelsize'] = 12
#     plt.rcParams['legend.fontsize'] = 12
#     plt.rcParams['figure.titlesize'] = 12
#     plt.rcParams['figure.figsize'] = (6, 4)
#     plt.rcParams['lines.linewidth'] = 1.5
#     plt.rcParams['lines.markersize'] = 6
#     plt.rcParams['axes.linewidth'] = 1.5
#     plt.rcParams['legend.frameon'] = False
#     plt.rcParams['legend.handlelength'] = 1.5
#     plt.rcParams['legend.handleheight'] = 0.5
#     plt.rcParams['legend.borderpad'] = 0.5
#     plt.rcParams['legend.borderaxespad'] = 0.5
#     plt.rcParams['legend.labelspacing'] = 0.5
#     plt.rcParams['legend.columnspacing'] = 1.0
#     plt.rcParams['legend.loc'] = 'best'
#     plt.rcParams['legend.framealpha'] = 0.5

# %%
config = {}

# %%
# test_dir = "odd_output_muons_1"
# test_dir = "odd_output_muons_1_single_charge"
# test_dir = "test"
# test_dir = "Test"
# test_dir = "geant4_test_2"
# test_dir = "odd_output_muons_2_splits"
# test_dir = "ACAT_2025/" + "odd_output_muons_2_splits_eta_1"
# test_dir = "odd_output_muons_2_splits_eta_1"
test_dir = "odd_output_muons_2_splits_eta_1_1mm"
# test_dir = "odd_output_muons_2_splits_eta_1_test"
# test_dir = "odd_output_muons_10_splits_eta_1"
# test_dir = "odd_output_muons_100_splits_eta_1"
# test_dir = "odd_output_muons_1_splits_eta_1"

# %%
dataset_name = "Acts_muons"
dataset_name = "Acts_muons_root"
if "TrackML" in dataset_name:
    dataset_type = "tml"
elif "Acts" in dataset_name:
    dataset_type = "acts"
    if dataset_name.startswith("Acts_muons_root"):
        dataset_type = "acts_root"
if dataset_name.startswith("Acts_muons"):
    dataset_dir = Path("Data/Acts/Muons")

# %%
if dataset_name.startswith("Acts_muons"):
    config = {
        "min_hits": 7,
        "particle_types": [13, -13],
        "max_abs_eta": 1,
        # "max_abs_eta": 3,
    }
    # config["max_abs_eta"] = 3
    config["sort_by_radius"] = True
    # config["truth_position"] = True
    config["truth_position"] = False
    # config["particle_file"] = "particles_hits_helix"
    # config["particle_file"] = "particles_hits"
    # config["verbose"] = True

# %%
config["input_variables"] = ["tx", "ty", "tz"]
# config["input_variables"] = ["tr", "tphi", "tz"]
config["input_variables"] = ["tr", "dphi", "tz"]
config["input_variables"] = ["r", "dphi", "z"]
# config["input_variables"] = ["tx", "ty", "tz", "pT_circle_estimate"]


config["output_variables"] = ["pT", "pz"]
config["output_variables"] = ["qopT", "pz"]
# config["output_variables"] = ["q_over_p", "ptheta", "phi0", "d0", "z0"]
config["output_variables"] = ["q_over_p", "ptheta", "dphi0", "d0", "z0"]
# config["output_variables"] = ["qpT", "pz"]

# %%
import pandas as pd

# Creating a structured dataframe for the models
model_data = [
    # (["tr", "dphi", "tz"], "Acts_muons", ['qopT', 'pz'], "Models/TrackML/version_533", None, None),
    # (["tr", "dphi", "tz"], "Acts_muons", ['qopT', 'pz'], "Models/TrackML/version_648", None, None), # eta 1 # no z symmetry
    # (["tr", "dphi", "tz"], "Acts_muons", ['qopT', 'pz'], "Models/TrackML/version_652", None, None), # eta 1 # z symmetry
    # (["tr", "dphi", "tz"], "Acts_muons", ['qopT', 'pz'], "Models/ActsODD/version_5", None, None),
    # (["tr", "dphi", "tz"], "Acts_muons", ['qopT', 'pz'], "Models/ActsODD/version_4", None, None), # eta 1
    # (["tr", "dphi", "tz"], "Acts_muons", ['qopT', 'pz'], "Models/ActsODD/version_26", None, None), # eta 3; min 7 hits
    # (["tr", "dphi", "tz"], "Acts_muons", ['qopT', 'pz'], "Models/ActsODD/version_28", None, None), # eta 3; min 7 hits
    # (["tr", "dphi", "tz"], "Acts_muons", ['qopT', 'pz'], "Models/ActsODD/version_63", None, None), # eta 3; min 7 hits
    # (["tr", "dphi", "tz"], "Acts_muons", ['qopT', 'pz'], "Models/ActsODD/version_69", None, None), # eta 3; min 7 hits
    # (["tr", "dphi", "tz"], "Acts_muons", ['qopT', 'pz'], "Models/ActsODD/version_72", None, None), # eta 3; min 7 hits; padding issue
    # (["tr", "dphi", "tz"], "Acts_muons", ['qopT', 'pz'], "Models/ActsODD/version_121", None, None), # eta 3; min 7 hits
    # (["tr", "dphi", "tz"], "Acts_muons_root", ['qopT', 'pz'], "Models/ActsODD/version_132", None, None), # eta 3; min 7 hits # 128 - 2
    # (["tr", "dphi", "tz"], "Acts_muons_root", ['qopT', 'pz'], "Models/ActsODD/version_257", None, None), # eta 3; min 7 hits # 32 - 2
    # (["r", "dphi", "z"], "Acts_muons_root", ['qopT', 'pz'], "Models/ActsODD/version_323", None, None), # eta 1; min 7 hits # 128 - 2 # measurements
    # (["r", "dphi", "z"], "Acts_muons_root", ['qopT', 'pz'], "Models/ActsODD/version_344"), # eta 1; min 7 hits # 32 - 16 # measurements
    # (["r", "dphi", "z"], "Acts_muons_root", ['qopT', 'pz'], "Models/ActsODD/version_517"), # eta 1; min 7 hits # 32 - 16 # measurements # trained muons
    (
        ["r", "dphi", "z"],
        "Acts_muons_root",
        ["qopT", "pz"],
        "Models/ActsODD/version_471",
    ),  # eta 1; min 7 hits # 32 - 16 # measurements # trained tt # mean
    # (["tr", "dphi", "tz"], "Acts_muons_root", ['qopT', 'pz'], "Models/ActsODD/version_113", None, None), # eta 1; min 7 hits
    # (["tr", "dphi", "tz"], "Acts_muons_root", ["q_over_p", "ptheta", "phi0", "d0", "z0"], "Models/ActsODD/version_232", None, None), # eta 1; min 7 hits
    # (["tr", "dphi", "tz"], "Acts_muons_root", ["q_over_p", "ptheta", "phi0", "d0", "z0"], "Models/ActsODD/version_228", None, None), # eta 1; min 7 hits
    # (["tr", "dphi", "tz"], "Acts_muons_root", ["q_over_p", "ptheta", "phi0", "d0", "z0"], "Models/ActsODD/version_234", None, None), # eta 1; min 7 hits
    (
        ["tr", "dphi", "tz"],
        "Acts_muons_root",
        ["q_over_p", "ptheta", "phi0", "d0", "z0"],
        "Models/ActsODD/version_395",
    ),  # eta 1; min 7 hits # 32 - 2 # norm # geo mean
    # (["tr", "dphi", "tz"], "Acts_muons_root", ["q_over_p", "ptheta", "dphi0", "d0", "z0"], "Models/ActsODD/version_236", None, None), # eta 1; min 7 hits
    # (["tr", "dphi", "tz"], "Acts_muons_root", ["q_over_p", "ptheta", "dphi0", "d0", "z0"], "Models/ActsODD/version_243", None, None), # eta 1; min 7 hits
    # (["tr", "dphi", "tz"], "Acts_muons_root", ["q_over_p", "ptheta", "dphi0", "d0", "z0"], "Models/ActsODD/version_355", None, None), # eta 1; min 7 hits # 32 - 16
    (
        ["tr", "dphi", "tz"],
        "Acts_muons_root",
        ["q_over_p", "ptheta", "dphi0", "d0", "z0"],
        "Models/ActsODD/version_414",
    ),  # eta 1; min 7 hits # 32 - 16 # geo mean # angles
    # (["r", "dphi", "z"], "Acts_muons_root", ["q_over_p", "ptheta", "dphi0", "d0", "z0"], "Models/ActsODD/version_444/checkpoints/model-epoch=999-val_loss=1.067e-04.ckpt"), # eta 1; min 7 hits # 128 - 2 # geo mean
    # (["r", "dphi", "z"], "Acts_muons_root", ["q_over_p", "ptheta", "dphi0", "d0", "z0"], "Models/ActsODD/version_444/checkpoints/model-epoch=999-val_loss=1.060e-04.ckpt"), # eta 1; min 7 hits # 128 - 2 # geo mean # angles
    # (["r", "dphi", "z"], "Acts_muons_root", ["q_over_p", "ptheta", "dphi0", "d0", "z0"], "Models/ActsODD/version_479/checkpoints/model-epoch=999-val_loss=9.456e-05.ckpt"), # eta 1; min 7 hits # 32 - 16 # geo mean
    # (["r", "dphi", "z"], "Acts_muons_root", ["q_over_p", "ptheta", "dphi0", "d0", "z0"], "/home/couthures/Bureau/HashingWork/athena/TrackParameters/TrackFormer/Models/ActsODD/lightning_logs/version_503/checkpoints/model-epoch=1026-val_loss=1.219e-03.ckpt"), # eta 1; min 7 hits # 32 - 16 # geo mean # finetuned muons
    # (["r", "dphi", "z"], "Acts_muons_root", ["q_over_p", "ptheta", "dphi0", "d0", "z0"], "/home/couthures/Bureau/HashingWork/athena/TrackParameters/TrackFormer/Models/ActsODD/lightning_logs/version_503/checkpoints/model-epoch=1041-val_loss=6.649e-04.ckpt"), # eta 1; min 7 hits # 32 - 16 # geo mean # finetuned muons
    # (["r", "dphi", "z"], "Acts_muons_root", ["q_over_p", "ptheta", "dphi0", "d0", "z0"], "/home/couthures/Bureau/HashingWork/athena/TrackParameters/TrackFormer/Models/ActsODD/lightning_logs/version_503/checkpoints/model-epoch=1060-val_loss=4.479e-04.ckpt"), # eta 1; min 7 hits # 32 - 16 # geo mean # finetuned muons
    # (["r", "dphi", "z"], "Acts_muons_root", ["q_over_p", "ptheta", "dphi0", "d0", "z0"], "/home/couthures/Bureau/HashingWork/athena/TrackParameters/TrackFormer/Models/ActsODD/lightning_logs/version_503/checkpoints/model-epoch=1093-val_loss=3.148e-04.ckpt"), # eta 1; min 7 hits # 32 - 16 # geo mean # finetuned muons
    # (["r", "dphi", "z"], "Acts_muons_root", ["q_over_p", "ptheta", "dphi0", "d0", "z0"], "/home/couthures/Bureau/HashingWork/athena/TrackParameters/TrackFormer/Models/ActsODD/lightning_logs/version_505"), # eta 1; min 7 hits # 32 - 2 # geo mean # trained muons
    # (["r", "dphi", "z"], "Acts_muons_root", ["q_over_p", "ptheta", "dphi0", "d0", "z0"], "/home/couthures/Bureau/HashingWork/athena/TrackParameters/TrackFormer/Models/ActsODD/lightning_logs/version_514"), # eta 1; min 7 hits # 32 - 16 # geo mean # finetuned muons # epoch 1600
    (
        ["r", "dphi", "z"],
        "Acts_muons_root",
        ["q_over_p", "ptheta", "dphi0", "d0", "z0"],
        "Models/ActsODD/version_529",
    ),  # eta 1; min 7 hits # 32 - 16 # geo mean # finetuned muons # lr 1e-6
    # (["r", "dphi", "z"], "Acts_muons_root", ["q_over_p", "ptheta", "dphi0", "d0", "z0"], "Models/ActsODD/version_522"), # eta 1; min 7 hits # 32 - 16 # geo mean # trained muons
    (
        ["r", "dphi", "z"],
        "Acts_muons_root",
        ["q_over_p", "ptheta", "phi0", "d0", "z0"],
        "Models/ActsODD/version_444/checkpoints/model-epoch=999-val_loss=2.287e-03.ckpt",
    ),  # eta 1; min 7 hits # 32 - 2 # geo mean # angles
    # (["tr", "dphi", "tz"], "Acts_muons_root", ["q_over_p", "ptheta", "dphi0", "d0", "z0"], "Models/ActsODD/version_391", None, None), # eta 1; min 7 hits # 32 - 2 # norm # geo mean
    # (["tr", "dphi", "tz"], "Acts_muons", ['qopT', 'pz'], "Models/ActsODD/version_85", None, None), # eta 1; min 7 hits
    # (["tr", "dphi", "tz"], "Acts_muons", ['qopT', 'pz'], "Models/ActsODD/version_89", None, None), # eta 1; min 7 hits; with scattered
]

# Creating a DataFrame
model_df = pd.DataFrame(
    model_data,
    columns=[
        "input_variables",
        "dataset_name",
        "output_variables",
        "mse_dir",
    ],
)
model_df["input_variables"] = model_df["input_variables"].apply(tuple)
model_df["output_variables"] = model_df["output_variables"].apply(tuple)


# %%
# Function to retrieve available models based on input configuration, dataset name, and DataFrame
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

# %%
models_dir = {
    "MSE": Path(available_models["mse_dir"]),
}

# %%
# Define models. (Assumes that p_pred_mse_list and p_pred_qloss_list are defined externally.)
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

# %%
from src.analysis.utils import (
    get_model_checkpoint_path,
    get_config,
    validate_model_config,
    compute_track_resolution,
)

# %%
# Load models
for model in models_dir:
    if model not in models:
        print(f"Skipping {model} model (not defined in models)")
        continue
    print(f"Loading {model} model")
    if models_dir[model].is_dir():
        ml_model_path = get_model_checkpoint_path(models_dir[model])[0]
    elif models_dir[model].is_file():
        ml_model_path = models_dir[model]
    else:
        raise ValueError(
            f"Model path {models_dir[model]} is neither a file nor a directory"
        )
    # _instantiator is a function that is called to instantiate the model
    # Here the models have been trained using cli raising an error if _instantiator is not None
    ml_model = TrackFormer.load_from_checkpoint(
        checkpoint_path=ml_model_path,
        _instantiator=None,
        metric="sign",
    )
    ml_model.eval().to("cpu")
    ml_model_ = torch.load(ml_model_path, map_location="cpu")
    print(list(ml_model_["callbacks"].values())[0])

    # ml_model_ = torch.load(Path("/".join(str(ml_model_path).split("/")[:-1]) + "/last-v1.ckpt"), map_location="cpu")
    # ml_model_ = torch.load(Path("/".join(str(ml_model_path).split("/")[:-1]) + "/last.ckpt"), map_location="cpu")
    # print(list(ml_model_['callbacks'].values())[0])
    # print(ml_model_)
    print(ml_model)
    models[model]["model"] = ml_model

# %%
output_dir = Path("Results/TrackML/")
if dataset_type == "acts" or True:
    output_dir = Path("Results/ActsODD/")

input_variables = "".join(config["input_variables"]).replace("t", "")

output_variables = "_".join(config["output_variables"])

experiment_name = f"{input_variables}_n_hits_{config['min_hits']}_{output_variables}_POC_{dataset_name}"
# experiment_name = f"{input_variables}_n_hits_{config['min_hits']}_{output_variables}_POC_{dataset_name}_full_dataset"
# experiment_name += "_batch_size_2048"
# experiment_name += "_eta_4"
# experiment_name += "_eta_3"
experiment_name += f"_eta_{config['max_abs_eta']}"
# experiment_name += "_eta_1"
# experiment_name += "_trained_eta_1"
# experiment_name += "_trained_eta_3"
# experiment_name += "_trained_TrackML"
# experiment_name += "_trained_TrackML_first"
# experiment_name += "_z_symmetry"
experiment_name += "_trained_ODD_full"
# experiment_name += "_" + "_".join(test_dir.split("_")[-2:])  # Use the last two parts of the test_dir name
experiment_name += "_" + test_dir.replace(
    "/", "_"
)  # Use the last two parts of the test_dir name
# experiment_name += "_e_32_h_8"
# experiment_name += "_e_128_h_2"
# experiment_name += "_trained_7_hits"
# experiment_name += "_huber"
# experiment_name += "_trained_with_scat"
# experiment_name += "_huber_resolution_e_32_h_2"
# experiment_name += "_huber_resolution_huber_e_32_h_2"
# experiment_name = f"{input_variables}_n_hits_{config['min_hits']}_{output_variables}_POC_no_scat_{dataset_name}_e_128_h_2_no_droupout"
if not any("t" in var for var in config["input_variables"]):
    experiment_name += "_measurements"

# Add model version to the output directory
model_version = str(models_dir["MSE"]).split("version_")[-1]
model_version = model_version.split("/")[0]
experiment_name += f"_version_{model_version}"

output_dir = output_dir / experiment_name

# Create the output directory if it does not exist
output_dir.mkdir(exist_ok=True, parents=True)

# %%
config["input_variables"] = ["tr", "dphi", "tz"]

# %%
# Instantiate the DataModule num_workers=16)
# dataset_wrapper = DataModule(dataset_type="tml", dataset_dir="Data/Tml/trackml_100_events_splits") # 20 000 particles
dataset_wrapper = DataModule(
    dataset_type=dataset_type,
    dataset_dir=dataset_dir,
    use_wrapper=False,
    kwargs=config,
)  # 1 404 273 particles

# dataset_wrapper.setup(stage="test")
dataset_wrapper.test_dataset = dataset_wrapper._create_dataset(test_dir)
tl = dataset_wrapper.test_dataloader()
# 1 min 41.4 s


# %%
def p_to_pt(p, ptheta):
    return p * np.sin(ptheta)


def p_to_pz(p, ptheta):
    return p * np.cos(ptheta)


# %% [markdown]
# # Tracks

# %%
from copy import deepcopy

all_events_tracks_list = []
all_events_processed = []
all_hits_list = []
dataset_wrapper.test_dataset.truth_tracks = False
for i in range(len(dataset_wrapper.test_dataset.available_events)):
    event = dataset_wrapper.test_dataset.available_events[i]
    # print(f"Loading event {event}")
    event_files = dataset_wrapper.test_dataset._load_event(event, n_events_split=50000)
    # event_files = dataset_wrapper.test_dataset._load_event(event, n_events_split=100)
    processed_event = dataset_wrapper.test_dataset._preprocessor(
        event_files=deepcopy(event_files)
    )
    # print(f"{list(processed_event) = }")
    hits, tracks, particles = event_files
    # print(hits.head())
    # print(tracks.head())
    # print(particles.head())
    # Add Hit_ID to the hits dataframe (index)
    hits["hit_id"] = hits.index
    # Save number of hits
    number_of_hits = len(hits)
    # print(f"{hits['event_id'].unique() = }")
    # print(f"{particles['event_id'].unique() = }")
    # print(f"{tracks['event_id'].unique() = }")

    # Add "_truth" suffix to the columns of the particles dataframe
    particles = particles.rename(
        columns=lambda x: f"{x}".replace("t_", "") if x.startswith("t_") else x
    )

    # Add "_truth" suffix to the columns of the particles dataframe
    particles = particles.rename(
        columns=lambda x: (
            f"{x}_truth"
            if x not in ["event_id", "particle_id", "particle_type", "process"]
            else x
        )
    )
    # Count the number of hits per particle
    hit_particle_counts = (
        hits.groupby(["event_id", "particle_id"])
        .size()
        .to_frame(name="n_particle_hits")
    )
    # This must be an integer
    assert (
        hit_particle_counts["n_particle_hits"].dtype == "int64"
    ), f"n_particle_hits is not an integer, it is {hit_particle_counts['n_particle_hits'].dtype}"
    # Add index as a column
    hit_particle_counts = hit_particle_counts.reset_index()
    # Merge the hit counts with the particles dataframe
    particles = particles.merge(
        hit_particle_counts, on=["event_id", "particle_id"], how="left"
    )

    # print(f"{particles.columns = }")
    # Match the tracks with the particles
    tracks = tracks.merge(
        particles, on=["event_id", "particle_id"], how="left", suffixes=("", "_truth")
    )
    # processed_event = list(processed_event)

    # Keep only the particles that are in the specified particle types
    particle_types = config["particle_types"]
    tracks = tracks[tracks["particle_type"].isin(particle_types)]

    # Add the event number to the tracks and hits dataframes
    tracks["event"] = event
    hits["event"] = event

    # Add a new column with the total number of hits in the track
    tracks["n_hits_track"] = tracks["nMeasurements"]
    # # Count the number of hits of the truth particle for each track
    # # Group by the track id and particle id then count the number of hits
    # # and select the value of the particle id
    # # # Temporarily explode the tracks dataframe to count the hits
    # # tracks["Hits_ID"] = tracks["Hits_ID"].str.strip("[]").str.split(",")
    # # # Convert to int
    # # tracks["Hits_ID"] = tracks["Hits_ID"].apply(
    # #     lambda x: [int(i) for i in x if i]
    # # )
    # # Explode the dataframe
    # # tracks_exploded = tracks.explode("Hits_ID")
    # Keep only Hits_ID, particle_id and track_id
    tracks_exploded = tracks[["event_id", "particle_id", "track_id"]].copy()
    # Rename Hits_ID to hit_id
    # tracks_exploded.rename(columns={"Hits_ID": "hit_id"}, inplace=True)

    # Rename the particle_id column to particle_id_track (to avoid confusion with the truth particle_id)
    tracks_exploded.rename(columns={"particle_id": "particle_id_track"}, inplace=True)

    # # Merge with hits dataframe
    hits = pd.merge(
        hits[hits["stateType"] == 0],
        tracks_exploded,
        on=["event_id", "track_id"],
        how="right",
        # validate="one_to_many",
    )
    # print(f"{hits.columns = }")
    # print(f"{len(hits) = }")
    # Make sure there is no NaN in the tx, ty, tz columns of the hits dataframe
    assert (
        hits["tx"].notna().all()
    ), "There are NaN values in the tx column of the hits dataframe"
    assert (
        hits["ty"].notna().all()
    ), "There are NaN values in the ty column of the hits dataframe"
    assert (
        hits["tz"].notna().all()
    ), "There are NaN values in the tz column of the hits dataframe"
    # Count the number of duplicates in the hits dataframe
    duplicates = hits.duplicated(subset=["hit_id"], keep="first")
    n_duplicates = duplicates.sum()
    if n_duplicates > 0:
        print(
            f"Warning: There are {n_duplicates} duplicates in the hits dataframe for event {event}"
        )
        # Print the duplicates
        all_duplicates = hits.duplicated(subset=["hit_id"], keep=False)
        duplicates_hits = hits[all_duplicates]
        # Print the duplicates grouped by hit_id
        duplicates_grouped = duplicates_hits.groupby("hit_id")
        # Show ["hit_id", "particle_id", "particle_id_track", "track_id"] for each group
        # Set the index to hit_id
        duplicates_summary = duplicates_grouped[
            ["particle_id", "particle_id_track", "track_id"]
        ].apply(lambda x: x)
        # Print the summary
        print(duplicates_summary)

        print(f"Number of duplicates: {n_duplicates}")

    # Make sure that the hits dataframe has the same number of hits as the exploded tracks dataframe
    # assert len(hits) == len(tracks_exploded), f"Number of hits ({len(hits)}) does not match the number of exploded tracks ({len(tracks_exploded)}) for event {event}"

    tracks["n_hits_truth"] = (
        hits.groupby(["event_id", "track_id"])["particle_id"]
        .apply(lambda x: x.value_counts().get(x.iloc[0], 0))
        .reset_index(drop=True)
    )
    # print(f"{tracks['n_hits_truth'].unique() = }")

    # Remove the tracks with less than config["min_hits"] hits
    print(f"Number of tracks before filtering: {len(tracks)}")
    tracks = tracks[tracks["n_hits_truth"] >= config["min_hits"]]
    print(f"Number of tracks after filtering: {len(tracks)}")
    tracks = tracks[np.abs(tracks["eta_truth"]) <= config["max_abs_eta"]]
    print(f"Number of tracks after filtering: {len(tracks)}")
    print(f"{tracks['event_id'].unique() = }")

    # Remove hits with no matching tracks tuple [("event_id", "track_id")]
    sel = (
        hits[["event_id", "track_id"]]
        .apply(tuple, axis=1)
        .isin(tracks[["event_id", "track_id"]].apply(tuple, axis=1))
    )
    # sel = hits.groupby(["event_id", "track_id"]).ngroup().isin(tracks.groupby(["event_id", "track_id"]).ngroup())
    # sel = hits[["event_id", "track_id"]].isin(tracks[["event_id", "track_id"]]).all(axis=1)
    # print(sel.head())
    hits = hits[sel]

    # Compare the number of tracks with the size of the processed event
    # assert len(tracks) == len(processed_event), f"Number of tracks ({len(tracks)}) does not match the size of the processed event ({len(processed_event)}) for event {event}"
    # assert len(hits) == number_of_hits, f"Number of hits ({len(hits)}) does not match the number of hits in the event files ({number_of_hits}) for event {event}"
    # Add the event to the list
    all_events_tracks_list.append(tracks)
    all_events_processed.append(processed_event)
    all_hits_list.append(hits)
    # break
    # if i > 10:
    #     break

# Concatenate all the events
all_events_tracks = pd.concat(all_events_tracks_list, ignore_index=True)
all_events_hits = pd.concat(all_hits_list, ignore_index=True)

# %%
# Remove duplicate tracks
all_events_tracks = all_events_tracks.drop_duplicates(
    subset=["event_id", "track_id"], ignore_index=True
)

# %%
# Remove duplicated columns
all_events_tracks = all_events_tracks.loc[:, ~all_events_tracks.columns.duplicated()]


# %%
# Load the model predictions from the file (if it exists)
# only [event, track_id, q/p_T_pred, pz_pred] are saved
def load_model_prediction(model_predictions_file, all_events_tracks):
    if model_predictions_file.exists():
        print(f"Loading model predictions from {model_predictions_file}")
        model_predictions = pd.read_csv(
            model_predictions_file, dtype={"event_id": int, "track_id": int}
        )
        print(f"{model_predictions.columns = }")
        # Merge the model predictions with the all_events_tracks dataframe
        all_events_tracks = all_events_tracks.merge(
            model_predictions,
            on=["event_id", "track_id"],
            how="left",  # Preserve original shape of dataframe
            # suffixes=("", "_pred_save"),
            validate="one_to_one",
        )
        # There should be no missing values in the predictions
        for variable in model_predictions.columns:
            if variable in ["phi_offset"]:
                continue
            assert (
                all_events_tracks[variable].notna().all()
            ), f"There are missing values in the {variable}. {all_events_tracks[variable].isna().sum()} missing values."
        print(f"Model predictions loaded and merged with all_events_tracks.")
    return all_events_tracks


if output_dir.exists():
    model_predictions_file = output_dir / "model_track_predictions_hits.csv"
    all_events_tracks = load_model_prediction(model_predictions_file, all_events_tracks)
    model_predictions_file = output_dir / "model_track_predictions_measurements.csv"
    all_events_tracks = load_model_prediction(model_predictions_file, all_events_tracks)

# %%
from torch.nn.utils.rnn import pad_sequence


def process_batch(
    hits_tensor_batch,
    mask_tensor_batch,
    batch_row_data,
    ml_model,
    config,
    rows,
    suffix="_pred",
):
    """
    Processes a batch of tracks, performs model prediction, and appends results to `rows`.

    Parameters:
    - hits_tensor_batch: List of tensors containing input data for the batch.
    - mask_tensor_batch: List of tensors containing mask data for the batch.
    - batch_row_data: List containing the row data for each track in the batch.
    - ml_model: The machine learning model to use for predictions.
    - config: Configuration dictionary with output variables.
    - rows: List to append prediction results.

    Returns:
    - Updated `rows` list containing prediction results for the batch.
    """
    # Pad the sequences to ensure they have the same length in the second dimension (number of hits)
    hits_tensor_batch_padded = pad_sequence(hits_tensor_batch, batch_first=True)
    mask_tensor_batch_padded = pad_sequence(
        mask_tensor_batch, batch_first=True, padding_value=0
    )
    # Get available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")
    if device.type == "cuda":
        # Move data to device
        hits_tensor_batch_padded = hits_tensor_batch_padded.to(device)
        mask_tensor_batch_padded = mask_tensor_batch_padded.to(device)

    # Model prediction
    pred = ml_model(hits_tensor_batch_padded, mask=mask_tensor_batch_padded)

    # Process the predictions
    for i, row_data in enumerate(batch_row_data):
        for j, col in enumerate(config["output_variables"]):
            row_data[f"{col}{suffix}"] = pred[i, j].item()

        if "dphi0" in config["output_variables"]:
            row_data[f"phi0{suffix}"] = (
                row_data[f"dphi0{suffix}"] + row_data["phi_offset"]
            )

        rows.append(row_data)  # Append processed row

    return rows


# %%
from time import time


def model_inference(
    ml_model,
    all_events_hits,
    all_events_tracks,
    input_variables,
    config,
    batch_size=64,
    suffix="_pred_measurement",
    phi_offset_column="g_phi_offset",
    output_dir=output_dir,
    output_filename="model_track_predictions_measurements.csv",
):
    start_time = time()

    # Create empty dataframe for track predictions: {"event_id", "track_id", "q/p_T_pred", "pz_pred"}
    track_predictions_measurements = pd.DataFrame(
        columns=["event_id", "track_id"]
        + [f"{col}{suffix}" for col in config["output_variables"]]
        + ([f"phi0{suffix}"] if "dphi0" in config["output_variables"] else [])
    )

    # Set dtypes to the ones of all_events_tracks for matching columns
    matching_columns = [
        col
        for col in track_predictions_measurements.columns
        if col in all_events_tracks.columns
    ]
    track_predictions_measurements = track_predictions_measurements.astype(
        all_events_tracks.dtypes[matching_columns].to_dict()
    )

    last_log = 0
    hits_groups = all_events_hits.copy().groupby(["event_id", "track_id"])
    # track_groups = all_events_tracks.groupby(["event_id", "track_id"])
    # print(all_events_hits.head())

    processed_tracks = 0
    progress_interval = 1000

    batch_hits_tensor = []
    batch_mask_tensor = []
    batch_row_data = []
    rows = []

    with torch.no_grad():
        for group_key, hits_sel in hits_groups:
            event, track_id = group_key

            # if not group_key in track_groups.groups:
            #     print(f"Track {track_id} in event {event} not found in event tracks.")
            #     continue

            # track_sel = track_groups.get_group(group_key)
            # # print(f"{hits_sel}")

            # # Select the hits for this track
            # assert len(hits_sel) == track_sel["n_hits_track"].iloc[0], f"Number of hits for track {track_id} does not match the number of hits in the event tracks. {len(hits_sel)} vs {track_sel['n_hits_track'].iloc[0]}"

            # Convert the hits to a tensor
            hits_tensor = torch.tensor(
                hits_sel[input_variables].values, dtype=torch.float32
            )
            # Convert the mask to a tensor
            mask_tensor = torch.ones(hits_tensor.shape[0], dtype=torch.bool)

            # Accumulate tensors for batch processing
            batch_hits_tensor.append(hits_tensor)
            batch_mask_tensor.append(mask_tensor)

            # Prepare row data
            row_data = {
                "event_id": event,
                "track_id": track_id,
            }
            if "dphi0" in config["output_variables"]:
                row_data["phi_offset"] = hits_sel.iloc[0][phi_offset_column]
                assert (
                    hits_sel.iloc[0]["dphi"] == 0
                ), "First hit dphi is not zero after calculation."
            batch_row_data.append(row_data)

            # If batch is full, process the batch
            if len(batch_hits_tensor) >= batch_size:
                rows = process_batch(
                    batch_hits_tensor,
                    batch_mask_tensor,
                    batch_row_data,
                    ml_model,
                    config,
                    rows,
                    suffix=suffix,
                )

                # Clear batch lists after processing
                batch_hits_tensor = []
                batch_mask_tensor = []
                batch_row_data = []

            processed_tracks += 1
            if processed_tracks - last_log >= progress_interval:
                last_log = processed_tracks
                elapsed = time() - start_time
                print(f"Progress: {processed_tracks} / {len(all_events_tracks)}")
                print(
                    f"Time elapsed: {elapsed / processed_tracks:.6f} seconds per track ({processed_tracks / elapsed:.3f} tracks per second)"
                )

        # If any remaining tracks in the last batch (less than batch_size)
        if len(batch_hits_tensor) > 0:
            rows = process_batch(
                batch_hits_tensor,
                batch_mask_tensor,
                batch_row_data,
                ml_model,
                config,
                rows,
                suffix=suffix,
            )

        end_time = time()
        total_elapsed = end_time - start_time
        print(f"Total inference time: {total_elapsed:.6f} seconds")

    # Create the DataFrame from the rows collected during batching
    track_predictions_measurements = pd.DataFrame(rows)

    # Save predictions
    if output_dir.exists():
        model_predictions_file = (
            output_dir
            / f"model_track{suffix.replace('pred', 'predictions').replace('measurement', 'measurements')}.csv"
        )
        print(f"Saving model predictions to {model_predictions_file}")
        track_predictions_measurements.to_csv(model_predictions_file, index=False)
        # Save throughput information
        throughput_info_file = (
            output_dir
            / f"throughput_info{suffix.replace('pred', 'predictions').replace('measurement', 'measurements')}.txt"
        )
        with open(throughput_info_file, "w") as f:
            f.write(f"Total time elapsed: {total_elapsed:.6f} seconds\n")
            f.write(f"Total tracks processed: {processed_tracks}\n")
            f.write(
                f"Average time per track: {total_elapsed / processed_tracks:.6f} seconds\n"
            )
            f.write(
                f"Average tracks per second: {processed_tracks / total_elapsed:.3f}\n"
            )

    return track_predictions_measurements


# %%
# %%time
import torch
import numpy as np
from time import time

# Main loop for batching
batch_size = 64  # Adjust batch size based on available memory

all_events_hits["tr"] = np.sqrt(all_events_hits["tx"] ** 2 + all_events_hits["ty"] ** 2)
all_events_hits["tphi"] = np.arctan2(all_events_hits["ty"], all_events_hits["tx"])

# Sort for grouping
all_events_hits = all_events_hits.sort_values(
    ["event_id", "track_id", "tr"], ascending=True
)


# Get the first hit for each track
first_hits = all_events_hits.groupby(["event_id", "track_id"]).first().reset_index()

# Add a column "dphi" by subtracting the phi of the first hit of the track to the phi of the hits of the same track
all_events_hits = pd.merge(
    all_events_hits,
    first_hits[["event_id", "track_id", "tphi"]].rename(
        columns={"tphi": "tphi_offset"}
    ),
    on=["event_id", "track_id"],
    how="left",
)

if "dphi" in config["input_variables"]:
    all_events_hits["dphi"] = all_events_hits["tphi"] - all_events_hits["tphi_offset"]
    # Correct for periodicity
    all_events_hits["dphi"] = np.where(
        all_events_hits["dphi"] > np.pi,
        all_events_hits["dphi"] - 2 * np.pi,
        all_events_hits["dphi"],
    )
    all_events_hits["dphi"] = np.where(
        all_events_hits["dphi"] < -np.pi,
        all_events_hits["dphi"] + 2 * np.pi,
        all_events_hits["dphi"],
    )

suffix = "_pred"
if not all(
    f"{col}{suffix}" in all_events_tracks.columns for col in config["output_variables"]
):
    print("No model predictions found, running the model on the test dataset.")
    ml_model = models["MSE"]["model"]
    ml_model.eval()
    start_time = time()

    rows = []

    # Create empty dataframe for track predictions: {"event_id", "track_id", "q/p_T_pred", "pz_pred"}
    track_predictions = pd.DataFrame(
        columns=["event_id", "track_id"]
        + [f"{col}{suffix}" for col in config["output_variables"]]
        + ([f"phi0{suffix}"] if "dphi0" in config["output_variables"] else [])
    )

    # Set dtypes to the ones of all_events_tracks for matching columns
    matching_columns = [
        col for col in track_predictions.columns if col in all_events_tracks.columns
    ]
    track_predictions = track_predictions.astype(
        all_events_tracks.dtypes[matching_columns].to_dict()
    )

    last_log = 0
    hits_groups = all_events_hits.copy().groupby(["event_id", "track_id"])
    track_groups = all_events_tracks.groupby(["event_id", "track_id"])
    # print(all_events_hits.head())

    processed_tracks = 0
    progress_interval = 1000

    batch_hits_tensor = []
    batch_mask_tensor = []
    batch_row_data = []

    with torch.no_grad():
        for group_key, hits_sel in hits_groups:
            event, track_id = group_key

            # if not group_key in track_groups.groups:
            #     print(f"Track {track_id} in event {event} not found in event tracks.")
            #     continue

            # track_sel = track_groups.get_group(group_key)
            # # print(f"{hits_sel}")

            # # Select the hits for this track
            # assert len(hits_sel) == track_sel["n_hits_track"].iloc[0], f"Number of hits for track {track_id} does not match the number of hits in the event tracks. {len(hits_sel)} vs {track_sel['n_hits_track'].iloc[0]}"

            # Convert the hits to a tensor
            hits_tensor = torch.tensor(
                hits_sel[dataset_wrapper.test_dataset.input_variables].values,
                dtype=torch.float32,
            )
            # Convert the mask to a tensor
            mask_tensor = torch.ones(hits_tensor.shape[0], dtype=torch.bool)

            # Accumulate tensors for batch processing
            batch_hits_tensor.append(hits_tensor)
            batch_mask_tensor.append(mask_tensor)

            # Prepare row data
            row_data = {
                "event_id": event,
                "track_id": track_id,
            }
            if "dphi0" in config["output_variables"]:
                row_data["phi_offset"] = hits_sel.iloc[0]["tphi_offset"]
                assert (
                    hits_sel.iloc[0]["dphi"] == 0
                ), "First hit dphi is not zero after calculation."
            batch_row_data.append(row_data)

            # If batch is full, process the batch
            if len(batch_hits_tensor) >= batch_size:
                rows = process_batch(
                    batch_hits_tensor,
                    batch_mask_tensor,
                    batch_row_data,
                    ml_model,
                    config,
                    rows,
                    suffix=suffix,
                )

                # Clear batch lists after processing
                batch_hits_tensor = []
                batch_mask_tensor = []
                batch_row_data = []

            processed_tracks += 1
            if processed_tracks - last_log >= progress_interval:
                last_log = processed_tracks
                elapsed = time() - start_time
                print(f"Progress: {processed_tracks} / {len(all_events_tracks)}")
                print(
                    f"Time elapsed: {elapsed / processed_tracks:.6f} seconds per track ({processed_tracks / elapsed:.3f} tracks per second)"
                )

        # If any remaining tracks in the last batch (less than batch_size)
        if len(batch_hits_tensor) > 0:
            rows = process_batch(
                batch_hits_tensor,
                batch_mask_tensor,
                batch_row_data,
                ml_model,
                config,
                rows,
                suffix=suffix,
            )

    # Create the DataFrame from the rows collected during batching
    track_predictions = pd.DataFrame(rows)

    # Save predictions
    if output_dir.exists():
        model_predictions_file = output_dir / "model_track_predictions_hits.csv"
        print(f"Saving model predictions to {model_predictions_file}")
        track_predictions.to_csv(model_predictions_file, index=False)

    # Merge the predictions with the original dataset
    all_events_tracks = all_events_tracks.merge(
        track_predictions,
        on=["event_id", "track_id"],
        how="left",
        validate="one_to_one",
    )
# 10 min 38.2 seconds
# 9 min 0.4 seconds
# 8 min 51.0 seconds
# 5 min 53.3 seconds
# 3 min 43.3 seconds
# CPU times: user 16min 39s, sys: 10.9 s, total: 16min 49s
# Wall time: 4min 16s

# Time elapsed: 0.000627 seconds per track (1595.145 tracks per second)
# CPU times: user 8min 11s, sys: 4.56 s, total: 8min 16s
# Wall time: 2min 8s

# %%
# %%time
import torch
import numpy as np

# Main loop for batching
batch_size = 64  # Adjust batch size based on available memory

all_events_hits["g_r"] = np.sqrt(
    all_events_hits["g_x_hit"] ** 2 + all_events_hits["g_y_hit"] ** 2
)
all_events_hits["g_phi"] = np.arctan2(
    all_events_hits["g_y_hit"], all_events_hits["g_x_hit"]
)

# Sort for grouping
all_events_hits = all_events_hits.sort_values(
    ["event_id", "track_id", "g_r"], ascending=True
)

input_variables = []
# Replace "tX" variables with "g_X" variables for measurement-based prediction
for var in dataset_wrapper.test_dataset.input_variables:
    if var.startswith("t"):
        input_variables.append(
            "g_"
            + var[1:]
            + ("_hit" if not f"g_{var[1:]}" in all_events_hits.columns else "")
        )
    else:
        input_variables.append(var)

suffix = "_pred_measurement"

# Get the first hit for each track
first_hits = all_events_hits.groupby(["event_id", "track_id"]).first().reset_index()

# Add a column "dphi" by subtracting the phi of the first hit of the track to the phi of the hits of the same track
all_events_hits = pd.merge(
    all_events_hits,
    first_hits[["event_id", "track_id", "g_phi"]].rename(
        columns={"g_phi": "g_phi_offset"}
    ),
    on=["event_id", "track_id"],
    how="left",
)

if "dphi" in config["input_variables"]:
    all_events_hits["dphi"] = all_events_hits["g_phi"] - all_events_hits["g_phi_offset"]
    # Correct for periodicity
    all_events_hits["dphi"] = np.where(
        all_events_hits["dphi"] > np.pi,
        all_events_hits["dphi"] - 2 * np.pi,
        all_events_hits["dphi"],
    )
    all_events_hits["dphi"] = np.where(
        all_events_hits["dphi"] < -np.pi,
        all_events_hits["dphi"] + 2 * np.pi,
        all_events_hits["dphi"],
    )

if (
    not all(
        f"{col}{suffix}" in all_events_tracks.columns
        for col in config["output_variables"]
    )
    or True
):
    print("No model predictions found, running the model on the test dataset.")
    ml_model = models["MSE"]["model"]
    ml_model.eval()
    # Get available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        ml_model = ml_model.to(device)

    track_predictions_measurements = model_inference(
        ml_model,
        all_events_hits,
        all_events_tracks,
        input_variables,
        config,
        batch_size=batch_size,
        suffix=suffix,
    )

    # Merge the predictions with the original dataset
    all_events_tracks = all_events_tracks.merge(
        track_predictions_measurements,
        on=["event_id", "track_id"],
        how="left",
        validate="one_to_one",
    )
# 10 min 38.2 seconds
# 9 min 0.4 seconds
# 8 min 51.0 seconds
# 5 min 53.3 seconds
# CPU times: user 22min 56s, sys: 16.7 s, total: 23min 13s
# Wall time: 5min 55s
# CPU times: user 9min 1s, sys: 5.53 s, total: 9min 7s
# Wall time: 2min 21s
# Time elapsed: 0.000635 seconds per track (1575.095 tracks per second)
# CPU times: user 8min 19s, sys: 4.62 s, total: 8min 23s
# Wall time: 2min 10s
