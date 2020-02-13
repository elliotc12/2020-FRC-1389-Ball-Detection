#!/usr/bin/env bash

#this script will:
    # take raw supervisely tar.gz
    # untar it, convert to the data/train data/test structure
    # convert JSON to CSV, then CSV to .record
    # create .pbtxt file in training/sdfsd (or data not sure)
    # create training/ folder
    # 

# Setup data directories
# tar -xf 'Images Tagged as Valid'.tar
# mv 'Images Tagged as Valid' data

mkdir -p training

python3 json_to_csv.py

python3 generate_tfrecord.py --csv_input=training/train.csv  --output_path=training/train.record
python3 generate_tfrecord.py --csv_input=training/eval.csv   --output_path=training/test.record
