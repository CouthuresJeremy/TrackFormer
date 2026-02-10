#!/bin/bash

# Check if correct arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <data_dir> <output-directory>"
    exit 1
fi

DATA_DIR=$1
OUTPUT_DIR=$2

echo "Output directory: $OUTPUT_DIR"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Find and extract all CSV files from .gz archives
echo "Finding .gz files..."
find "$DATA_DIR" -type f -name "*.gz" | while read -r gz_file; do
    echo "Processing: $gz_file"
    OUTPUT_CSV="$OUTPUT_DIR/$(basename "${gz_file%.gz}")"
    
    # Extract and print output
    gzip -dc "$gz_file" > "$OUTPUT_CSV"
    
    echo "Extracted: $OUTPUT_CSV"
done

echo "Extraction completed. CSV files are stored in: $OUTPUT_DIR"


