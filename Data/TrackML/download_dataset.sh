#!/bin/bash

base_url="https://zenodo.org/records/4730157/files"

for i in $(seq -w 1 9); do
    file_name="training_part$(printf "%02d" $i).tar"
    
    echo "Processing ${file_name}..."
    
    # Check if the file exists
    if [ -f "${file_name}" ]; then
        echo "Resuming ${file_name}..."
    else
        echo "Downloading ${file_name}..."
    fi
    
    # Download with resume support
    curl -C - -o "${file_name}" "${base_url}/${file_name}?download=1"
done