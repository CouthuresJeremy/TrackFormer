# TrackFormer: Particle Trackfitting with Transformer 

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python](https://img.shields.io/badge/Python-3.7%2B-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange.svg)](https://pytorch.org/)
[![PyTorch Lightning](https://img.shields.io/badge/Lightning-2.0%2B-blue.svg)](https://www.pytorchlightning.ai/)

## Overview

TrackFormer is a unique solution to particle trajectory reconstruction that uses transformer-inspired design. Using Transformers' self-attention mechanism, this model performs track fitting, resulting in better accuracy and efficiency.

## Features

- **Transformer-based Architecture**: For fast efficient and accurate particle track fitting.
- **Built with Lightning Integration**:
- **Modular Design**: 
- **Logging and CLI integration**:

## Getting Started

1. **Clone the repository**:
2. download datasets to specific directory
3. cd to appropriate script to train, test split dataset: 
 `./split_dataset.sh /path/to/downloaded/dataset 80 10 10`
4. train the model using the following command:
 `python main.py fit --config configs/tformer.yaml`
5. train the model with wandb logging:
 `python main.py fit --config configs/tformer.yaml --config configs/trainer.yaml`

## Reproducing the results

To reproduce the results, you can use the following commands.

Step 1: Clone the repository

```bash
git clone https://github.com/soot-bit/TrackFormer.git
```

Step 2: Create an environment using conda and install the dependencies

```bash
conda create --name TrackFormer python=3.10 
conda activate TrackFormer
pip install -r requirements.txt
```

Step 3: Download `train_1.zip` from the [TrackML dataset](https://www.kaggle.com/c/trackml-particle-identification/data) and extract it to the `Data` directory.

```bash
unzip train_1.zip -d Data/Tml/train_1
```

Step 4: Split the dataset into train, validation, and test sets

```bash
cd Data/Tml
bash split_tml_inplace.sh train_1 80 10 10
cd ../..
```

Step 5: Train the models

```bash
bash run_TrackML_example.sh
```

Step 6: To evaluate the models, update model paths in the `dataanalysis.ipynb` cells and run the notebook.

## Other datasets

You can use the following datasets to train the model:

- [TrackML Particle Tracking Challenge](https://www.kaggle.com/c/trackml-particle-identification/data)
- [TrackML Troughput Phase](https://zenodo.org/records/4730157)


## Important links

- [slides](https://docs.google.com/presentation/d/1YvFFSKoVI4W0tpCQEvG8J3SybidqZtn08r3DU9UGw_E/edit?usp=sharing)

- [report](https://drive.google.com/file/d/1a9syXUZ6ibbufsp7mMXBDCmBRUCzhfMg/view?usp=sharing)
