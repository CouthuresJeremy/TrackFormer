#!/bin/bash

# Check if at least one argument is provided
if [ "$#" -gt 1 ]; then
    echo "Usage: $0 [output-directory]"
    exit 1
fi

# If the argument is provided, use it as the output directory; otherwise, use default name
if [ -n "$1" ]; then
    OUTPUT_DIR=$1
else
    # Define a unique output directory using a timestamp
    OUTPUT_DIR="extracted_$(date +%Y%m%d_%H%M%S)"
fi

echo "Extracting dataset tar files into: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Loop through files matching the "training_partXX.tar" pattern
for TAR_FILE in training_part[0-9][0-9].tar; do
    # Ensure the file exists (in case no matches were found)
    if [ ! -f "$TAR_FILE" ]; then
        echo "No matching .tar files found. Exiting."
        exit 1
    fi

    echo "Extracting $TAR_FILE..."
    tar -xvf "$TAR_FILE" -C "$OUTPUT_DIR"
done

echo "Extraction completed. Files are stored in: $OUTPUT_DIR"
