from abc import ABC, abstractmethod
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info
import torch
import math
import os
from tqdm.auto import tqdm
from rich.console import Console
from rich.progress import track

console = Console()
from src.datasets.utils import ParticleGun, Detector, EventGenerator
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import lightning as L
from pathlib import Path
import trackml.dataset
import pandas as pd


##################################################################

#################################
#           TOY TRACK           #
#################################


class ToyTrackDataset(IterableDataset):
    """
    Generates track data on the fly using ToyTrack module.
    See https://github.com/ryanliu30
    """

    def __init__(
        self, hole_inefficiency=0, d0=0.1, noise=0, lambda_=50, pt_dist=[1, 5]
    ):
        super().__init__()
        self.hole_inefficiency = hole_inefficiency
        self.d0 = d0
        self.noise = noise
        self.pt_dist = pt_dist
        self.detector = self._create_detector()
        self.particle_gun = self._create_particle_gun()

    def _create_detector(self):
        return Detector(
            dimension=2, hole_inefficiency=self.hole_inefficiency
        ).add_from_template("barrel", min_radius=0.5, max_radius=3, number_of_layers=10)

    def _create_particle_gun(self):
        return ParticleGun(
            dimension=2,
            num_particles=1,
            pt=self.pt_dist,
            pphi=[-np.pi, np.pi],
            vx=[0, self.d0 * 0.5**0.5, "normal"],
            vy=[0, self.d0 * 0.5**0.5, "normal"],
        )

    def __iter__(self):
        self.event_gen = EventGenerator(self.particle_gun, self.detector, self.noise)
        return self

    def __next__(self):
        # an event
        event = self.event_gen.generate_event()
        x = torch.tensor([event.hits.x, event.hits.y], dtype=torch.float).T.contiguous()
        return (
            x,
            torch.ones(x.shape[0], dtype=bool),
            torch.tensor([event.particles.pt], dtype=torch.float),
        )


class ToytrackDataModule(L.LightningDataModule):
    """ToyTrack Lightning Data Module"""

    def __init__(
        self,
        batch_size: int = 20,
        num_workers: int = 10,
        persistence: bool = False,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["_class_path"])
        self.dataset = ToyTrackDataset()
        console.rule("Streaming ToyTrack")

    def train_dataloader(self):
        return self._create_dataloader(self.dataset)

    def val_dataloader(self):
        return self._create_dataloader(self.dataset)

    def test_dataloader(self):
        return self._create_dataloader(self.dataset)

    def _create_dataloader(self, dataset):
        """Helper method to create a DataLoader."""
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self.collate_fn,
            persistent_workers=self.hparams.num_workers > 0
            and self.hparams.persistence,
            pin_memory=self.hparams.pin_memory,
        )

    @staticmethod
    def collate_fn(ls):
        """Batch maker"""
        x, mask, pt = zip(*ls)
        return (
            pad_sequence(x, batch_first=True),
            pad_sequence(mask, batch_first=True),
            torch.cat(pt).squeeze(),
        )


##################################################################

######################################
#       BASE Realistic Datasets       #
#######################################


class IterBase(IterableDataset, ABC):
    """Iterable Base class for TrackML and ACTS datasets.

    Attributes:
        folder (Path): directory containing dataset.
    """

    def __init__(self, dataset_dir, folder="train", dataset=None, **kwargs):
        self.path = Path(dataset_dir) / folder
        self.available_events = self._event_range()

        # Add kwargs to the class
        for key, value in kwargs.items():
            setattr(self, key, value)

    def _event_range(self):

        event_numbers = []
        for file in self.path.glob("*"):
            event_numbers.append(file.stem.split("-")[0])

        if not event_numbers:
            raise FileNotFoundError(
                "Uh-oh! Looks like there data files are missing ..."
            )

        return sorted(list(set(event_numbers)))

    @abstractmethod
    def _preprocessor(self, event: str):
        """preprocessing logic."""
        raise NotImplementedError

    @abstractmethod
    def _load_event(self, eventfiles):
        """loading logic."""
        raise NotImplementedError

    def __iter__(self):
        worker_info = get_worker_info()
        total_events = len(self.available_events)
        if worker_info is None:  # Single-process
            iter_start = 0
            iter_end = total_events
        else:
            # Split workload among workers
            per_worker = int(math.ceil((total_events) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, total_events)

        for i in range(iter_start, iter_end):
            event_files = self._load_event(self.available_events[i])
            processed_data = self._preprocessor(event_files)
            yield from processed_data


########################################### streamline datasets:


class TrackMLDataset(IterBase):
    """Iterable class for TrackML"""

    def _load_event(self, event_prefix):
        self.event = event_prefix

        def get_file_path(filename):
            # Check if the compressed version exists; otherwise, fall back to uncompressed
            gz_path = self.path / f"{filename}.csv.gz"
            csv_path = self.path / f"{filename}.csv"
            return gz_path if gz_path.exists() else csv_path

        hits = get_file_path(f"{event_prefix}-hits")
        particles = get_file_path(f"{event_prefix}-particles")
        cells = get_file_path(f"{event_prefix}-cells")
        truth = get_file_path(f"{event_prefix}-truth")
        return (
            pd.read_csv(hits),
            pd.read_csv(cells),
            pd.read_csv(particles),
            pd.read_csv(truth),
        )

    def _preprocessor(self, eventfiles):
        # Get kwargs min_hits if available
        min_hits = getattr(self, "min_hits", 5)

        hits, _, particles, truth = eventfiles
        particles = particles[particles["nhits"] >= min_hits]

        merged_df = pd.merge(truth, particles, on="particle_id")
        merged_df = pd.merge(merged_df, hits, on="hit_id")

        merged_df["pT"] = np.sqrt(merged_df["px"] ** 2 + merged_df["py"] ** 2)

        # Get kwargs min_pt and max_pt if available
        min_pt = getattr(self, "min_pt", 0)
        max_pt = getattr(self, "max_pt", np.inf)

        merged_df = merged_df[(merged_df["pT"] >= min_pt) & (merged_df["pT"] <= max_pt)]

        # Get kwargs keep_secondaries if available
        keep_secondaries = getattr(self, "keep_secondaries", True)
        if not keep_secondaries:
            secondary_selection = (np.abs(merged_df["vx"]) >= 1) | (
                np.abs(merged_df["vy"]) >= 1
            )
            merged_df = merged_df[~secondary_selection]

        p = np.sqrt(merged_df["px"] ** 2 + merged_df["py"] ** 2 + merged_df["pz"] ** 2)
        merged_df["peta"] = np.arctanh(merged_df["pz"] / p)

        # Get kwargs min_abs_eta and max_abs_eta if available
        min_abs_eta = getattr(self, "min_abs_eta", 0)
        max_abs_eta = getattr(self, "max_abs_eta", np.inf)

        merged_df = merged_df[
            (np.abs(merged_df["peta"]) >= min_abs_eta)
            & (np.abs(merged_df["peta"]) <= max_abs_eta)
        ]

        # Get kwargs input_variables if available
        input_variables = getattr(self, "input_variables", ["tx", "ty", "tz"])

        if any([var not in merged_df.columns for var in input_variables]):
            # Add other coordinate system
            merged_df["tr"] = np.sqrt(merged_df["tx"] ** 2 + merged_df["ty"] ** 2)
            merged_df["tphi"] = np.arctan2(merged_df["ty"], merged_df["tx"])

        # Get kwargs output_variables if available
        output_variables = getattr(self, "output_variables", ["pT", "pz"])

        if any([var not in merged_df.columns for var in output_variables]):
            # Add other track parameters
            merged_df["qopT"] = merged_df["q"] / merged_df["pT"]
            merged_df["qpT"] = merged_df["q"] * merged_df["pT"]

        grouped = merged_df.groupby("particle_id")

        for _, group in grouped:
            # Add custom features
            if "dphi" in input_variables:
                # Remove phi of the first hit
                group["dphi"] = group["tphi"] - group["tphi"].iloc[0]
                # Correct for periodicity
                group["dphi"] = np.where(
                    group["dphi"] > np.pi, group["dphi"] - 2 * np.pi, group["dphi"]
                )
                group["dphi"] = np.where(
                    group["dphi"] < -np.pi, group["dphi"] + 2 * np.pi, group["dphi"]
                )

            inputs = group[input_variables].values
            target = group[output_variables].values[0]

            zxy = torch.tensor(inputs, dtype=torch.float32)
            target_tensor = torch.tensor(target, dtype=torch.float32)

            mask = torch.ones(zxy.shape[0], dtype=torch.bool)
            yield zxy, mask, target_tensor


class ActsDataset(IterBase):

    def _load_event(self, event_prefix):
        self.event = event_prefix
        parameters = self.path / f"{event_prefix}-parameters.csv"
        particles = self.path / f"{event_prefix}-particles.csv"
        spacepoints = self.path / f"{event_prefix}-spacepoint.csv"
        tracks = self.path / f"{event_prefix}-tracks.csv"
        return (
            pd.read_csv(spacepoints),
            pd.read_csv(tracks),
            pd.read_csv(particles),
            pd.read_csv(parameters),
        )

    def _preprocessor(self, event_files):
        """Preprocesses data for the specified event.

        Args:
            event_files (tuple): Tuple containing the loaded event data files.
        """

        spacepoints, _, particles, _ = event_files

        # Convert to a tensor
        xyz = torch.tensor(
            spacepoints[["x", "y", "z"]].values, dtype=torch.float32
        ).squeeze()
        pt = torch.sqrt(
            torch.tensor(particles.px.values, dtype=torch.float32) ** 2
            + torch.tensor(particles.py.values, dtype=torch.float32) ** 2
        ).squeeze()
        yield xyz, torch.ones(xyz.shape[0], dtype=torch.bool), pt


class DatasetWrapper(Dataset):
    """
    Traditional torch dataloading from saved object.

    Attributes:
        data_file (Path): Path to save/load preprocessed data.
        dataset_dir (Path): Directory containing dataset.
        folder (str): (train, test, val) to load.
    """

    def __init__(self, dataset_dir, folder, dataset="tml", **kwargs):
        self.dataset_dir = Path(dataset_dir)
        self.dataset_type = dataset.lower()
        self.folder = folder
        dataset_suffix = kwargs.pop("dataset_suffix", "")
        # Add kwargs input_variables and output_variables if available
        input_variables = kwargs.get("input_variables", ["tx", "ty", "tz"])
        output_variables = kwargs.get("output_variables", ["pT", "pz"])
        variable_suffix = (
            f"_i_{'_'.join(input_variables)}_o_{'_'.join(output_variables)}"
        )
        dataset_suffix = variable_suffix + (
            "_" + dataset_suffix if dataset_suffix else ""
        )
        self.data_file = (
            self.dataset_dir / f"preprocessed_{self.folder}{dataset_suffix}.pt"
        )
        self.wrapper_workers = kwargs.pop("wrapper_workers", int(os.cpu_count()))
        self.datalist = None

        # Check if dataset is valid
        if self.dataset_type not in ("tml", "acts"):
            raise ValueError(
                f"Invalid dataset type '{dataset}'. Expected 'tml' or 'acts'."
            )

        # Set the dataset class
        if self.dataset_type == "tml":
            self.ds_class = TrackMLDataset
        elif self.dataset_type == "acts":
            self.ds_class = ActsDataset

        # Add kwargs to the class
        self.ds_class_kwargs = kwargs

        self.__setup()

    def __setup(self):
        """Sets up the dataset by loading from preprocessed data if available, or processing and saving it."""
        if self.data_file.is_file():
            console.print(f"Loading data from {self.data_file}", style="cyan")
            self.datalist = torch.load(self.data_file)
        else:
            console.print(
                "Preprocessed data not found. Processing and saving data...",
                style="cyan",
            )
            ds = self.ds_class(self.dataset_dir, self.folder, **self.ds_class_kwargs)
            ds_loader = DataLoader(ds, num_workers=self.wrapper_workers)
            self.datalist = []
            # Unpack the data and save it
            for particle_index, variables in enumerate(ds_loader):
                print(f"Processing particle {particle_index}")
                self.datalist.append([var.squeeze() for var in variables])
            torch.save(self.datalist, self.data_file)
            console.print(f"Data saved to {self.data_file}")

    def __getitem__(self, index):
        return self.datalist[index]

    def __len__(self):
        return len(self.datalist)


class DataModule(L.LightningDataModule):
    """
    Lightning DataModule for managing TrackML or ACTS datasets.
    Args:
        dataset_type (str): Type of dataset ('tml' or 'acts').
        dataset_dir (path): the path to where the dataset file is
    """

    def __init__(
        self,
        dataset_type,
        dataset_dir,
        batch_size=32,
        num_workers=os.cpu_count() - 2,
        use_wrapper=True,
        persistance=False,
        pin_memory=False,
        kwargs={},  # kwargs for the dataset class
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["_class_path"])
        dataset = self.hparams.dataset_type.lower()

        # Check if dataset is valid
        if dataset not in ("tml", "acts"):
            raise ValueError(
                f"Invalid dataset_type '{dataset}'. Expected 'tml' or 'acts'."
            )

        # Set the dataset class
        if use_wrapper:
            self.dataset_class = DatasetWrapper
        elif dataset == "tml":
            self.dataset_class = TrackMLDataset
        elif dataset == "acts":
            self.dataset_class = ActsDataset

        # Add kwargs to the class
        self.dataset_class_kwargs = kwargs

    def setup(self, stage=None):
        """Setup datasets for training, validation, and testing."""
        console.rule(f"{self.hparams.dataset_type.capitalize()} Dataset")

        if stage in ("fit", None):
            self.train_dataset = self._create_dataset("train")
            self.val_dataset = self._create_dataset("val")

        if stage in ("test", None):
            self.test_dataset = self._create_dataset("test")

    def train_dataloader(self):
        return self._create_dataloader(self.train_dataset)

    def val_dataloader(self):
        return self._create_dataloader(self.val_dataset)

    def test_dataloader(self):
        return self._create_dataloader(self.test_dataset)

    def _create_dataloader(self, dataset):
        """Helper function to initialize data loaders."""
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self.collate_fn,
            persistent_workers=bool(self.hparams.num_workers)
            and self.hparams.persistance,
            pin_memory=self.hparams.pin_memory,
        )

    def _create_dataset(self, folder):
        """Helper method to create dataset for the given folder"""
        return self.dataset_class(
            dataset_dir=self.hparams.dataset_dir,
            folder=folder,
            dataset=self.hparams.dataset_type,
            **self.dataset_class_kwargs,
        )

    @staticmethod
    def collate_fn(batch):
        """Generic collate function for padding sequences."""
        inputs, masks, targets = zip(*batch)
        inputs = pad_sequence(inputs, batch_first=True)
        masks = pad_sequence(masks, batch_first=True, padding_value=0)
        return inputs, masks, torch.stack(targets, dim=0)
