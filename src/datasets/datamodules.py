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


def compute_signed_curvature(pT, q, B):
    """
    Compute the signed curvature of a track in a uniform magnetic field
    using Lorentz force.

    Parameters:
      pT : Transverse momentum (e.g. in GeV)
      q  : Charge of the particle (in units of elementary charge)
      B  : Magnetic field strength (in Tesla)

    Returns:
        signed_kappa: Signed curvature of the track (in m^-1)
    """
    signed_kappa = (q / (pT * 1e9)) * (B * 299_792_458)
    # R = pT * 1e9 / (np.abs(q) * B * 299_792_458)
    return signed_kappa


def compute_circle_parameters(x_v, y_v, phi0, signed_kappa):
    """
    Compute the parameters of
    the circle in the transverse plane.

    Parameters:
      x_v        : x-coordinate of the production vertex
      y_v        : y-coordinate of the production vertex
      phi0       : Initial azimuthal angle of the particle (radians)
      signed_kappa: Signed curvature of the track

    Returns:
        x_c, y_c, R: Center and radius of the circle
    """
    # Compute the radius of the circle
    R = 1.0 / np.abs(signed_kappa)
    # Compute the center of the circle
    x_c = x_v + (1.0 / signed_kappa) * np.sin(phi0)
    y_c = y_v - (1.0 / signed_kappa) * np.cos(phi0)

    return x_c, y_c, R


def compute_perigee(x_c, y_c, R):
    """
    Compute the coordinates of the perigee (point of closest approach)
    in the transverse plane.

    Parameters:
      x_c : x-coordinate of the circle center
      y_c : y-coordinate of the circle center
      R   : Radius of the circle

    Returns:
        x_perigee, y_perigee: Coordinates of the perigee
    """
    # The perigee in the transverse plane is reached when the azimuth of the circle equals that of its center.
    phi_c = np.arctan2(y_c, x_c)
    x_perigee = x_c - R * np.cos(phi_c)
    y_perigee = y_c - R * np.sin(phi_c)

    return x_perigee, y_perigee


def compute_impact_parameters(
    p_x, p_y, p_z, q, B, x_v, y_v, z_v, reference_point=(0, 0, 0)
):
    """
    Compute the transverse impact parameter d0 and the longitudinal impact parameter z0
    for a truth track in a uniform field.

    Parameters:
        p_x, p_y, p_z : Momentum components of the particle (in GeV/c).
        q             : Charge of the particle (in elementary charge units).
        B             : Magnetic field strength (in Tesla).
        x_v, y_v, z_v : Production vertex coordinates (in mm).
        reference_point: Reference point for the perigee calculation (default: origin).

    Returns:
        d0 : Signed transverse impact parameter (in mm).
        z0 : Longitudinal impact parameter (in mm).
        perigee_coords: (x, y, z) coordinates of the perigee (point of closest approach).
    """
    # === Step 0: Express the vertex in the reference frame ===
    x_ref, y_ref, z_ref = reference_point
    x_v = x_v - x_ref
    y_v = y_v - y_ref
    z_v = z_v - z_ref

    # === Step 1: Calculate curvature and radius ===
    pT = np.sqrt(p_x**2 + p_y**2)
    signed_kappa = compute_signed_curvature(pT=pT, q=q, B=B)
    # Convert to mm^-1
    signed_kappa = signed_kappa / 1000

    # === Step 2: Compute the circle center in the transverse plane ===
    phi0 = np.arctan2(p_y, p_x)
    x_c, y_c, R = compute_circle_parameters(
        x_v=x_v, y_v=y_v, phi0=phi0, signed_kappa=signed_kappa
    )

    # === Step 3: Find the perigee (closest approach to the origin) ===
    x_perigee, y_perigee = compute_perigee(x_c=x_c, y_c=y_c, R=R)
    # Check that the perigee is on the circle
    assert np.allclose((x_perigee - x_c) ** 2 + (y_perigee - y_c) ** 2, R**2)

    # === Step 4: Compute the (unsigned) distance from the origin to the perigee ===
    d0_unsigned = np.sqrt(x_perigee**2 + y_perigee**2)

    # === Step 5: Assign a sign to d0 ===
    # Sign is given by the z-component of the cross product of the perigee position and the momentum (angular momentum).
    cross_z = x_perigee * p_y - y_perigee * p_x
    sign = np.sign(cross_z)
    # default to +1 if zero
    # sign = ((sign + 1)/2) - 1
    # sign[sign == 0] = 1

    d0 = sign * d0_unsigned

    # === Step 6: Compute the z-coordinate at the perigee ===
    # Compute the path length from the vertex to the perigee
    # The angle between the vertex and the perigee with respect to the center of the circle
    # is twice the angle between the vertex and the midpoint of the perigee and the center of the circle.
    # Compute the distance between the vertex and the perigee
    d_vertex_perigee = np.sqrt((x_v - x_perigee) ** 2 + (y_v - y_perigee) ** 2)
    # Compute the distance between the vertex and the midpoint
    d_vertex_midpoint = d_vertex_perigee / 2
    # Compute the angle between the vertex and the midpoint
    angle_vertex_midpoint = np.arctan2(d_vertex_midpoint, R)
    # Compute the angle between the vertex and the perigee
    angle_vertex_perigee = 2 * angle_vertex_midpoint

    # Determine the sign of the angle difference
    # = sign of inner product of the vector from the vertex to the perigee and the momentum
    propagation_vector_sign = np.sign((x_perigee - x_v) * p_x + (y_perigee - y_v) * p_y)
    assert (
        propagation_vector_sign.shape == angle_vertex_perigee.shape
    ), f"{propagation_vector_sign.shape} != {angle_vertex_perigee.shape}"
    # Compute the signed angle between the vertex and the perigee
    angle_vertex_perigee = propagation_vector_sign * angle_vertex_perigee

    # Compute the path length between the vertex and the perigee
    s_vertex_perigee = R * angle_vertex_perigee

    # Compute the z-coordinate at the perigee using the linear propagation along z
    z_perigee = z_v + s_vertex_perigee * (p_z / pT)
    z0 = z_perigee

    return d0, z0, (x_perigee, y_perigee, z_perigee)


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
        # print(f"Loading event {event_prefix}")

        # Handle empty csv files
        hits_df = pd.read_csv(hits) if hits.stat().st_size > 0 else pd.DataFrame()
        try:
            cells_df = (
                pd.read_csv(cells) if cells.stat().st_size > 0 else pd.DataFrame()
            )
            # This raises an OSError and EmptyDataError for some non empty files
        except OSError:
            cells_df = pd.DataFrame()
        except pd.errors.EmptyDataError:
            cells_df = pd.DataFrame()
        particles_df = (
            pd.read_csv(particles) if particles.stat().st_size > 0 else pd.DataFrame()
        )
        truth_df = pd.read_csv(truth) if truth.stat().st_size > 0 else pd.DataFrame()

        return (
            hits_df,
            cells_df,
            particles_df,
            truth_df,
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

        # Get kwargs truth_position if available
        truth_position = getattr(self, "truth_position", True)
        if truth_position:
            # Override reconstructed position by truth position
            merged_df["x"] = merged_df["tx"]
            merged_df["y"] = merged_df["ty"]
            merged_df["z"] = merged_df["tz"]

        # Get kwargs input_variables if available
        default_inputs = ["x", "y", "z"]
        input_variables = getattr(self, "input_variables", default_inputs)

        if (
            any([var not in merged_df.columns for var in input_variables])
            or getattr(self, "sort_by_radius", False)
            or getattr(self, "cut_scattered", False)
        ):
            # Add other coordinate system
            merged_df["tr"] = np.sqrt(merged_df["tx"] ** 2 + merged_df["ty"] ** 2)
            merged_df["tphi"] = np.arctan2(merged_df["ty"], merged_df["tx"])
            merged_df["r"] = np.sqrt(merged_df["x"] ** 2 + merged_df["y"] ** 2)
            merged_df["phi"] = np.arctan2(merged_df["y"], merged_df["x"])

        # Get kwargs output_variables if available
        output_variables = getattr(self, "output_variables", ["pT", "pz"])

        if any([var not in merged_df.columns for var in output_variables]):
            # Add other track parameters
            merged_df["qopT"] = merged_df["q"] / merged_df["pT"]
            merged_df["qpT"] = merged_df["q"] * merged_df["pT"]
            merged_df["phi0"] = np.arctan2(merged_df["py"], merged_df["px"])
            if any(
                [
                    var in output_variables
                    for var in ["d0", "z0", "x_perigee", "y_perigee", "z_perigee"]
                ]
            ):
                (
                    merged_df["d0"],
                    merged_df["z0"],
                    (
                        merged_df["x_perigee"],
                        merged_df["y_perigee"],
                        merged_df["z_perigee"],
                    ),
                ) = compute_impact_parameters(
                    p_x=merged_df["px"],
                    p_y=merged_df["py"],
                    p_z=merged_df["pz"],
                    q=merged_df["q"],
                    B=2,
                    x_v=merged_df["vx"],
                    y_v=merged_df["vy"],
                    z_v=merged_df["vz"],
                    reference_point=(0, 0, 0),
                )
            merged_df["ptheta"] = np.arctan2(merged_df["pT"], merged_df["pz"])

        grouped = merged_df.groupby("particle_id")

        for _, group in grouped:
            # Cut scattered tracks
            if getattr(self, "cut_scattered", False):
                # This is a truth particle cut

                # The track must be ordered by radius
                if not "tr" in group:
                    group["tr"] = np.sqrt(group["tx"] ** 2 + group["ty"] ** 2)
                group_sorted = group.sort_values("tr")
                # Compute the angle between the hits
                if not "tphi" in group:
                    group_sorted["tphi"] = np.arctan2(
                        group_sorted["ty"], group_sorted["tx"]
                    )
                group_sorted["dphi"] = (
                    group_sorted["tphi"] - group_sorted["tphi"].iloc[0]
                )
                # Correct for periodicity
                group_sorted["dphi"] = np.where(
                    group_sorted["dphi"] > np.pi,
                    group_sorted["dphi"] - 2 * np.pi,
                    group_sorted["dphi"],
                )
                group_sorted["dphi"] = np.where(
                    group_sorted["dphi"] < -np.pi,
                    group_sorted["dphi"] + 2 * np.pi,
                    group_sorted["dphi"],
                )
                # Check if the angle is monotonically increasing
                scattered = (
                    (group_sorted["dphi"].shift(-1) - group_sorted["dphi"])
                    * (group_sorted["dphi"].shift(-2) - group_sorted["dphi"].shift(-1))
                    < 0
                ).any()
                if scattered:
                    continue

            # Sort by the hits by radius
            if getattr(self, "sort_by_radius", False):
                if not "r" in group:
                    group["r"] = np.sqrt(group["x"] ** 2 + group["y"] ** 2)
                group = group.sort_values("r")

            # Add custom features
            if "dphi" in input_variables:
                # Remove phi of the first hit
                group["dphi"] = group["phi"] - group["phi"].iloc[0]
                # Correct for periodicity
                group["dphi"] = np.where(
                    group["dphi"] > np.pi, group["dphi"] - 2 * np.pi, group["dphi"]
                )
                group["dphi"] = np.where(
                    group["dphi"] < -np.pi, group["dphi"] + 2 * np.pi, group["dphi"]
                )

            if (
                "pT_circle_estimate" in input_variables
                or "pT_circle_estimate_inv" in input_variables
            ):
                # Estimate pT from the circle fit
                from src.my_model.benchmarks import CircleFit

                cf = CircleFit()
                points = group[["x", "y"]].values
                points = torch.tensor(points, dtype=torch.float32)

                # Make it a batch of 1 2D list of points
                points = points.unsqueeze(0)
                r = cf.fit(points).tolist()
                pt_fit = np.array(r) * 1.0 * 2 * 299_792_458 / 1e9 / 1000
                group["pT_circle_estimate"] = np.full(group.shape[0], pt_fit)
                group["pT_circle_estimate_inv"] = 1 / np.full(group.shape[0], pt_fit)

            inputs = group[input_variables].values
            target = group[output_variables].values[0]

            zxy = torch.tensor(inputs, dtype=torch.float32)
            target_tensor = torch.tensor(target, dtype=torch.float32)

            mask = torch.ones(zxy.shape[0], dtype=torch.bool)
            yield zxy, mask, target_tensor


class ActsDataset(IterBase):

    def _load_event(self, event_prefix):
        self.event = event_prefix
        particles = self.path / f"{event_prefix}-particles_simulated.csv"
        hits = self.path / f"{event_prefix}-hits.csv"
        tracks = self.path / f"{event_prefix}-tracks_ambi.csv"
        return (
            pd.read_csv(hits),
            pd.read_csv(tracks),
            pd.read_csv(particles),
        )

    def _preprocessor(self, event_files):
        """Preprocesses data for the specified event.

        Args:
            event_files (tuple): Tuple containing the loaded event data files.
        """

        hits, _, particles = event_files

        # Get kwargs min_hits if available
        min_hits = getattr(self, "min_hits", 5)
        # particles = particles[particles["nhits"] >= min_hits]

        n_hits = hits.shape[0]
        merged_df = pd.merge(hits, particles, on="particle_id", validate="many_to_one")
        # merged_df = pd.merge(merged_df, hits, on="hit_id")

        # Verify that the number of hits is the same
        if n_hits != merged_df.shape[0]:
            raise ValueError(
                f"Number of hits in {self.event} does not match the number of hits in the merged dataframe."
            )

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

        # Get kwargs truth_position if available
        truth_position = getattr(self, "truth_position", True)
        if truth_position:
            # Override reconstructed position by truth position
            merged_df["x"] = merged_df["tx"]
            merged_df["y"] = merged_df["ty"]
            merged_df["z"] = merged_df["tz"]

        # Get kwargs input_variables if available
        default_inputs = ["x", "y", "z"]
        input_variables = getattr(self, "input_variables", default_inputs)

        if (
            any([var not in merged_df.columns for var in input_variables])
            or getattr(self, "sort_by_radius", False)
            or getattr(self, "cut_scattered", False)
        ):
            # Add other coordinate system
            merged_df["tr"] = np.sqrt(merged_df["tx"] ** 2 + merged_df["ty"] ** 2)
            merged_df["tphi"] = np.arctan2(merged_df["ty"], merged_df["tx"])
            merged_df["r"] = np.sqrt(merged_df["x"] ** 2 + merged_df["y"] ** 2)
            merged_df["phi"] = np.arctan2(merged_df["y"], merged_df["x"])

        # Get kwargs output_variables if available
        output_variables = getattr(self, "output_variables", ["pT", "pz"])

        if any([var not in merged_df.columns for var in output_variables]):
            # Add other track parameters
            merged_df["qopT"] = merged_df["q"] / merged_df["pT"]
            merged_df["qpT"] = merged_df["q"] * merged_df["pT"]
            merged_df["phi0"] = np.arctan2(merged_df["py"], merged_df["px"])
            if any(
                [
                    var in output_variables
                    for var in ["d0", "z0", "x_perigee", "y_perigee", "z_perigee"]
                ]
            ):
                (
                    merged_df["d0"],
                    merged_df["z0"],
                    (
                        merged_df["x_perigee"],
                        merged_df["y_perigee"],
                        merged_df["z_perigee"],
                    ),
                ) = compute_impact_parameters(
                    p_x=merged_df["px"],
                    p_y=merged_df["py"],
                    p_z=merged_df["pz"],
                    q=merged_df["q"],
                    B=2,
                    x_v=merged_df["vx"],
                    y_v=merged_df["vy"],
                    z_v=merged_df["vz"],
                    reference_point=(0, 0, 0),
                )
            merged_df["ptheta"] = np.arctan2(merged_df["pT"], merged_df["pz"])

        grouped = merged_df.groupby("particle_id")

        for _, group in grouped:
            # Cut tracks with too few hits
            if group.shape[0] < min_hits:
                continue

            # Cut scattered tracks
            if getattr(self, "cut_scattered", False):
                # This is a truth particle cut

                # The track must be ordered by radius
                if not "tr" in group:
                    group["tr"] = np.sqrt(group["tx"] ** 2 + group["ty"] ** 2)
                group_sorted = group.sort_values("tr")
                # Compute the angle between the hits
                if not "tphi" in group:
                    group_sorted["tphi"] = np.arctan2(
                        group_sorted["ty"], group_sorted["tx"]
                    )
                group_sorted["dphi"] = (
                    group_sorted["tphi"] - group_sorted["tphi"].iloc[0]
                )
                # Correct for periodicity
                group_sorted["dphi"] = np.where(
                    group_sorted["dphi"] > np.pi,
                    group_sorted["dphi"] - 2 * np.pi,
                    group_sorted["dphi"],
                )
                group_sorted["dphi"] = np.where(
                    group_sorted["dphi"] < -np.pi,
                    group_sorted["dphi"] + 2 * np.pi,
                    group_sorted["dphi"],
                )
                # Check if the angle is monotonically increasing
                scattered = (
                    (group_sorted["dphi"].shift(-1) - group_sorted["dphi"])
                    * (group_sorted["dphi"].shift(-2) - group_sorted["dphi"].shift(-1))
                    < 0
                ).any()
                if scattered:
                    continue

            # Sort by the hits by radius
            if getattr(self, "sort_by_radius", False):
                if not "r" in group:
                    group["r"] = np.sqrt(group["x"] ** 2 + group["y"] ** 2)
                group = group.sort_values("r")
            elif getattr(self, "sort_by_dz", False):
                # Compute the mean z of the hits
                mean_z = group["z"].mean()
                group["dz"] = (group["z"] - mean_z) * mean_z
                # Sort by absolute distance to the mean z
                # group["dz"] = np.abs(group["dz"])
                group = group.sort_values("dz")
            elif getattr(self, "sort_by_distance", False):
                group["distance"] = np.sqrt(
                    group["x"] ** 2 + group["y"] ** 2 + group["z"] ** 2
                )
                group = group.sort_values("distance")

            # Add custom features
            if "dphi" in input_variables:
                # Remove phi of the first hit
                group["dphi"] = group["phi"] - group["phi"].iloc[0]
                # Correct for periodicity
                group["dphi"] = np.where(
                    group["dphi"] > np.pi, group["dphi"] - 2 * np.pi, group["dphi"]
                )
                group["dphi"] = np.where(
                    group["dphi"] < -np.pi, group["dphi"] + 2 * np.pi, group["dphi"]
                )

            if (
                "pT_circle_estimate" in input_variables
                or "pT_circle_estimate_inv" in input_variables
            ):
                # Estimate pT from the circle fit
                from src.my_model.benchmarks import CircleFit

                cf = CircleFit()
                points = group[["x", "y"]].values
                points = torch.tensor(points, dtype=torch.float32)

                # Make it a batch of 1 2D list of points
                points = points.unsqueeze(0)
                r = cf.fit(points).tolist()
                pt_fit = np.array(r) * 1.0 * 2 * 299_792_458 / 1e9 / 1000
                group["pT_circle_estimate"] = np.full(group.shape[0], pt_fit)
                group["pT_circle_estimate_inv"] = 1 / np.full(group.shape[0], pt_fit)

            inputs = group[input_variables].values
            target = group[output_variables].values[0]

            zxy = torch.tensor(inputs, dtype=torch.float32)
            target_tensor = torch.tensor(target, dtype=torch.float32)

            mask = torch.ones(zxy.shape[0], dtype=torch.bool)
            yield zxy, mask, target_tensor


class DatasetWrapper(Dataset):
    """
    Traditional torch dataloading from saved object.

    Attributes:
        data_file (Path): Path to save/load preprocessed data.
        dataset_dir (Path): Directory containing dataset.
        folder (str): (train, test, val) to load.
    """

    def __init__(
        self, dataset_dir, folder, dataset="tml", split_size=1_000_000, **kwargs
    ):
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
        self.data_file_suffix = dataset_suffix
        self.data_file = (
            self.dataset_dir / f"preprocessed_{self.folder}{self.data_file_suffix}.pt"
        )
        self.split_size = split_size  # Number of samples per split
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
        # Preprocess the data if not already done
        # if not self._is_preprocessed():
        self._preprocess_data()

        # Load the data from the preprocessed file
        if self.data_file.is_file():
            console.print(f"Loading data from {self.data_file}", style="cyan")
            self.datalist = torch.load(self.data_file)
        else:
            self.datalist = self._load_split_data()

    def _is_preprocessed(self):
        """Check if the dataset has been preprocessed."""
        return self.data_file.is_file() or self._is_split_data()

    def _is_split_data(self):
        """Check if the dataset is split into multiple files."""
        first_split_filename = self.data_file.with_name(
            f"preprocessed_{self.folder}{self.data_file_suffix}_chunk_{0}{self.data_file.suffix}"
        )
        return first_split_filename.is_file()

    def _load_split_data(self):
        """Loads the split data from multiple files."""
        datalist = []
        i = 0
        while True:
            split_filename = self.data_file.with_name(
                f"preprocessed_{self.folder}{self.data_file_suffix}_chunk_{i}{self.data_file.suffix}"
            )
            if split_filename.is_file():
                console.print(f"Loading split data from {split_filename}", style="cyan")
                split_data = torch.load(split_filename)
                datalist.extend(split_data)
                i += 1
            else:
                break
        return datalist

    def _load_full_data(self):
        """Loads the entire dataset from a single file."""
        if self.data_file.is_file():
            console.print(f"Loading data from {self.data_file}", style="cyan")
            return torch.load(self.data_file)

    def _preprocess_data(self):
        """Preprocesses the dataset if not already done."""
        console.print(
            "Processing and saving data...",
            style="cyan",
        )
        already_preprocessed = self.split_size * self._get_next_split_index()
        if already_preprocessed > 0:
            console.print(
                f"Already preprocessed {already_preprocessed} samples. Resuming from there...",
                style="yellow",
            )
        else:
            console.print(
                "No preprocessed data found. Starting from scratch.", style="red"
            )
        ds = self.ds_class(self.dataset_dir, self.folder, **self.ds_class_kwargs)
        ds_loader = DataLoader(ds, num_workers=self.wrapper_workers)
        chunk_data = []
        for particle_index, variables in enumerate(ds_loader):
            if particle_index < already_preprocessed:
                continue
            if particle_index % 1000 == 0:
                print(f"Processing particle {particle_index}")

            # Add the current batch of data to the chunk
            chunk_data.append([var.squeeze() for var in variables])

            # If the chunk reaches the split_size, save it and clear the chunk
            if len(chunk_data) >= self.split_size:
                self._save_data(chunk_data)
                chunk_data.clear()  # Clear the chunk after saving

        # Save any remaining data after the loop ends
        if chunk_data:
            self._save_data(chunk_data)

        print(f"Processed {particle_index+1} particles")

    def _save_data(self, data):
        """Saves the dataset chunk, splitting it into parts if necessary based on split_size."""
        # Save the chunk to a split file
        split_filename = self.data_file.with_name(
            f"preprocessed_{self.folder}{self.data_file_suffix}_chunk_{self._get_next_split_index()}{self.data_file.suffix}"
        )
        torch.save(data, split_filename)
        print(f"Chunk dataset saved to {split_filename}")

    def _get_next_split_index(self):
        """Get the next index for the split dataset file."""
        # Check how many files already exist
        i = 0
        while (
            self.data_file.with_name(
                f"preprocessed_{self.folder}{self.data_file_suffix}_chunk_{i}{self.data_file.suffix}"
            )
        ).is_file():
            i += 1
        return i

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
