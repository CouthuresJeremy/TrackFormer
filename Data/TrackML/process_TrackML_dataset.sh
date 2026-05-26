#!/bin/bash

OUTPUT_DIR="full_dataset"

bash download_dataset.sh

bash extract_all_dataset.sh $OUTPUT_DIR

bash split_dataset_inplace.sh $OUTPUT_DIR 80 10 10