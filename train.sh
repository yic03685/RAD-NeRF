#!/bin/bash

DATA_INPUT_FOLDER="data"
DATA_TEMP_FOLDER="temp_data" # we dont work inplace because it will slow down from sync
TRAINED_MODEL_FOLDER="trained"
DEFAULT_ITERATION=200000
DEFAULT_LIPS_FINTUE_ITERATION=250000

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
python data_utils/process.py $DATASET_FOLDER/$DATASET_NAME.mp4

# Move some metadata beside the final model. These are needed for inference too
cp $DATASET_FOLDER/transforms_val.json $TRAINED_MODEL_FOLDER/$DATASET_NAME/
cp $DATASET_FOLDER/bc.jpg $TRAINED_MODEL_FOLDER/$DATASET_NAME/
cp -R $DATASET_FOLDER/gt_imgs $TRAINED_MODEL_FOLDER/$DATASET_NAME/
cp -R $DATASET_FOLDER/ori_imgs $TRAINED_MODEL_FOLDER/$DATASET_NAME/
cp -R $DATASET_FOLDER/parsing $TRAINED_MODEL_FOLDER/$DATASET_NAME/
cp -R $DATASET_FOLDER/torso_imgs $TRAINED_MODEL_FOLDER/$DATASET_NAME/

# train
echo ==Start Overll Training==
CUDA_VISIBLE_DEVICES=0 python main.py $DATASET_FOLDER/ --workspace $TRAINED_MODEL_FOLDER/$DATASET_NAME/${DATASET_NAME}_eo/ -O --iters $DEFAULT_ITERATION --lambda_amb 1.5
echo ==Start Lips Finetuning==
CUDA_VISIBLE_DEVICES=0 python main.py $DATASET_FOLDER/ --workspace $TRAINED_MODEL_FOLDER/$DATASET_NAME/${DATASET_NAME}_eo/ -O --finetune_lips --iters $DEFAULT_LIPS_FINTUE_ITERATION --lambda_amb 1.5
echo ==Start Torso Training==
CUDA_VISIBLE_DEVICES=0 python main.py $DATASET_FOLDER/ --workspace $TRAINED_MODEL_FOLDER/$DATASET_NAME/${DATASET_NAME}_eo_torso/ -O --torso --iters $DEFAULT_ITERATION --head_ckpt $TRAINED_MODEL_FOLDER/$DATASET_NAME/${DATASET_NAME}_eo/checkpoints/ngp.pth --lambda_amb 1.5

# test
echo ==Running Tests==
CUDA_VISIBLE_DEVICES=0 python main.py $DATASET_FOLDER/ --workspace $TRAINED_MODEL_FOLDER/$DATASET_NAME/${DATASET_NAME}_eo_torso/ -O --torso --test

# remove temp files
rm -rf $DATA_TEMP_FOLDER