#!/bin/bash

DATA_FOLDER="data"
TRAINED_MODEL_FOLDER="trained"
DEFAULT_ITERATION=200000

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
CUDA_VISIBLE_DEVICES=0 conda run -n pytorch3d python main.py $DATASET_FOLDER/ --workspace $TRAINED_MODEL_FOLDER/$DATASET_NAME/${DATA_SET_NAME}_eo/ -O --iters 10000
CUDA_VISIBLE_DEVICES=0 conda run -n pytorch3d python main.py $DATASET_FOLDER/ --workspace $TRAINED_MODEL_FOLDER/$DATASET_NAME/${DATA_SET_NAME}_eo/ -O --finetune_lips --iters 15000

CUDA_VISIBLE_DEVICES=0 conda run -n pytorch3d python main.py $DATASET_FOLDER/ --workspace $TRAINED_MODEL_FOLDER/$DATASET_NAME/${DATA_SET_NAME}_eo_torso/ -O --torso --iters 10000 --head_ckpt $TRAINED_MODEL_FOLDER/$DATASET_NAME/${DATA_SET_NAME}_eo/checkpoints/ngp.pth

# test
CUDA_VISIBLE_DEVICES=0 conda run -n pytorch3d python main.py $DATASET_FOLDER/ --workspace $TRAINED_MODEL_FOLDER/$DATASET_NAME/${DATA_SET_NAME}_eo_torso/ -O --torso --test
