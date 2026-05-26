#!/bin/bash

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <dataset_dir> <train_percentage> <val_percentage> <test_percentage>"
    exit 1
else
    DATASET_DIR=$1
    TRAIN_PERCENT=$2 
    VAL_PERCENT=$3
    TEST_PERCENT=$4

    SUM=$((TRAIN_PERCENT + VAL_PERCENT + TEST_PERCENT))
    if [ "$SUM" -ne 100 ]; then
        echo "Error: The sum of percentages must equal 100."
        exit 1
    fi
fi

EVENTS=$(ls "$DATASET_DIR" | grep -o 'event[0-9]\{9\}' | sort -u)

# Check events
if [ -z "$EVENTS" ]; then
    echo "Error: No events found in the dataset directory."
    exit 1
fi

cd $DATASET_DIR

mkdir -p train val test

EVENT_ARRAY=($EVENTS)
total_events=${#EVENT_ARRAY[@]}

# events for each split
train_count=$((total_events * TRAIN_PERCENT / 100))
val_count=$((total_events * VAL_PERCENT / 100))
test_count=$((total_events - train_count - val_count))

total_success=0
total_files=$((total_events * 4))

echo "Starting the copy process..."
echo "Total events to process: ${#EVENT_ARRAY[@]}"
echo "Total expected files: $total_files"
echo ""

for i in "${!EVENT_ARRAY[@]}"; do
    event=${EVENT_ARRAY[$i]}
    
    if [ "$i" -lt "$train_count" ]; then
        dest_dir="train"
    elif [ "$i" -lt "$((train_count + val_count))" ]; then
        dest_dir="val"
    else
        dest_dir="test"
    fi

    # Verify the presence of all the files of the event
    all_files_exist=true
    for file_type in particles truth hits cells; do
        src_file="$event-$file_type.csv.gz"
        if [ ! -f "$src_file" ]; then
            all_files_exist=false
            break
        fi
    done

    # Skip the event if any of the files are missing
    if [ "$all_files_exist" = false ]; then
        echo "Skipping event $event due to missing files."
        echo "Removing the files that were supposed to be moved..."
        echo "Following files are missing:"
        # Remove the files that were supposed to be moved
        for file_type in particles truth hits cells; do
            src_file="$event-$file_type.csv.gz"
            if [ -f "$src_file" ]; then
                rm "$src_file"
            else
                echo "$src_file (missing)"
            fi
        done
        continue
    fi

    for file_type in particles truth hits cells; do
        src_file="$event-$file_type.csv.gz"
        dest_file="$dest_dir/$event-$file_type.csv.gz"
        
        if [ -f "$src_file" ]; then
            mv "$src_file" "$dest_file" && ((total_success++))
           
            printf "[%3d%%] \t Destination: %s \t Copied: %s-%s.csv.gz \r" $((total_success * 100 / total_files)) "$dest_dir" "$event" "$file_type"
        else
            echo "Warning: Source file $src_file does not exist."
        fi
    done
done

echo ""
echo ""

echo "============================"
echo "         Summary            "
echo "============================"
echo "Total Copies: $total_success / $total_files" 
echo "Total Events: $((total_success / 4)) / $total_events"
echo "============================"
