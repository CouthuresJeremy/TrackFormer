#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Run TrackFormer dataset preprocessing without starting training."""

import argparse
from pathlib import Path
import sys

import yaml

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.datasets.datamodules import DataModule


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess TrackFormer datasets and write cached files only"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a dataset config file such as configs/Acts/dataset.yaml",
    )
    parser.add_argument(
        "--folders",
        nargs="+",
        default=["train", "val"],
        help="Dataset folders to preprocess",
    )
    return parser.parse_args()


def load_data_init_args(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if "data" not in config:
        raise KeyError(f"No 'data' section found in {config_path}")

    data_cfg = config["data"]
    class_path = data_cfg.get("class_path")
    if class_path != "src.datasets.datamodules.DataModule":
        raise ValueError(
            "The preprocessing script expects a DataModule config, got "
            f"{class_path!r} in {config_path}"
        )

    init_args = dict(data_cfg.get("init_args", {}))
    dataset_dir = Path(init_args["dataset_dir"])

    init_args["dataset_dir"] = dataset_dir
    return init_args


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    init_args = load_data_init_args(config_path)
    datamodule = DataModule(**init_args)
    datamodule.preprocess_data(tuple(args.folders))


if __name__ == "__main__":
    main()