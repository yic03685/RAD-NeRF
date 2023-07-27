#!/bin/bash

DATA_FOLDER="data"
OUTPUT_FOLDER="trained"

if [ -z "$1" ]; then
  echo "Error: Dataset name not provided."
  echo "Usage: $0 <dataset_name>"
  exit 1
fi

DATASET_NAME=$(basename "$1")
DATASET_FOLDER=$DATA_FOLDER/$DATASET_NAME

# data processing
mkdir -p $DATASET_FOLDER
conda run -n pytorch3d python data_utils/process.py $DATASET_FOLDER/$DATASET_NAME.mp4

# train
CUDA_VISIBLE_DEVICES=0 conda run -n pytorch3d python main.py $DATASET_FOLDER/ --workspace $OUTPUT_FOLDER/$DATASET_NAME/${DATA_SET_NAME}_eo/ -O --iters 200000
CUDA_VISIBLE_DEVICES=0 conda run -n pytorch3d python main.py $DATASET_FOLDER/ --workspace $OUTPUT_FOLDER/$DATASET_NAME/${DATA_SET_NAME}_eo/ -O --finetune_lips --iters 250000

CUDA_VISIBLE_DEVICES=0 conda run -n pytorch3d python main.py $DATASET_FOLDER/ --workspace $OUTPUT_FOLDER/$DATASET_NAME/${DATA_SET_NAME}_eo_torso/ -O --torso --iters 200000 --head_ckpt $OUTPUT_FOLDER/$DATASET_NAME/${DATA_SET_NAME}_eo/checkpoints/ngp.pth

# test
CUDA_VISIBLE_DEVICES=0 conda run -n pytorch3d python main.py $OUTPUT_FOLDER/$DATA_SET_NAME/ --workspace $OUTPUT_FOLDER/$DATASET_NAME/${DATA_SET_NAME}_eo_torso/ -O --torso --test
