#!/bin/bash
# curl https://zenodo.org/records/4730157/files/training_part01.tar?download=1 --output training_part01.tar
# curl https://zenodo.org/api/records/4730157/files-archive --output TrackML.zip
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