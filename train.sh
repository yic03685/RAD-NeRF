#!/bin/bash

DATA_INPUT_FOLDER="data"
DATA_TEMP_FOLDER="temp_data" # we dont work inplace because it will slow down from sync
TRAINED_MODEL_FOLDER="trained"
DEFAULT_ITERATION=200000

if [ -z "$1" ]; then
  echo "Error: Dataset name not provided."
  echo "Usage: $0 <dataset_name>"
  exit 1
fi

DATASET_NAME=$(basename "$1")
DATASET_FOLDER=$DATA_TEMP_FOLDER/$DATASET_NAME

# data processing
mkdir -p $DATASET_FOLDER
cp $DATA_INPUT_FOLDER/$DATASET_NAME/${DATASET_NAME}.mp4 $DATASET_FOLDER/${DATASET_NAME}.mp4
echo ==Start Pre-process==
conda run -n pytorch3d python data_utils/process.py $DATASET_FOLDER/$DATASET_NAME.mp4

# train
echo ==Start Overll Training==
CUDA_VISIBLE_DEVICES=0 conda run -n pytorch3d python main.py $DATASET_FOLDER/ --workspace $TRAINED_MODEL_FOLDER/$DATASET_NAME/${DATA_SET_NAME}_eo/ -O --iters 10000
echo ==Start Lips Finetuning==
CUDA_VISIBLE_DEVICES=0 conda run -n pytorch3d python main.py $DATASET_FOLDER/ --workspace $TRAINED_MODEL_FOLDER/$DATASET_NAME/${DATA_SET_NAME}_eo/ -O --finetune_lips --iters 15000
echo ==Start Torso Training==
CUDA_VISIBLE_DEVICES=0 conda run -n pytorch3d python main.py $DATASET_FOLDER/ --workspace $TRAINED_MODEL_FOLDER/$DATASET_NAME/${DATA_SET_NAME}_eo_torso/ -O --torso --iters 10000 --head_ckpt $TRAINED_MODEL_FOLDER/$DATASET_NAME/${DATA_SET_NAME}_eo/checkpoints/ngp.pth

# test
echo ==Running Tests==
CUDA_VISIBLE_DEVICES=0 conda run -n pytorch3d python main.py $DATASET_FOLDER/ --workspace $TRAINED_MODEL_FOLDER/$DATASET_NAME/${DATA_SET_NAME}_eo_torso/ -O --torso --test

# remove temp files
rm -rf $DATA_TEMP_FOLDER