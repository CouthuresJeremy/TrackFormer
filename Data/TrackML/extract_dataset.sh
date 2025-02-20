#!/bin/bash

# Check if at least one argument is provided
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "Usage: $0 <tar-file> [output-directory]"
    exit 1
fi

TAR_FILE=$1

# If the second argument is provided, use it as the output directory; otherwise, derive from the TAR file name
if [ -n "$2" ]; then
    OUTPUT_DIR=$2
else
    OUTPUT_DIR="${TAR_FILE%.tar}"
fi

echo "Extracting from TAR file: $TAR_FILE"
echo "Output directory: $OUTPUT_DIR"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Extract the tar archive
echo "Extracting .tar file..."
tar -xvf "$TAR_FILE" -C "$OUTPUT_DIR"

echo "Extraction completed. Files are stored in: $OUTPUT_DIR"