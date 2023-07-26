#!/bin/bash

DATA_SET_NAME="kenny"

# data processing
conda run -n pytorch3d python data_utils/process.py data/$DATA_SET_NAME/$DATA_SET_NAME.mp4

# train
CUDA_VISIBLE_DEVICES=0 conda run -n pytorch3d python main.py data/${DATA_SET_NAME}/ --workspace trial_${DATA_SET_NAME}_eo/ -O --iters 200000
CUDA_VISIBLE_DEVICES=0 conda run -n pytorch3d python main.py data/${DATA_SET_NAME}/ --workspace trial_${DATA_SET_NAME}_eo/ -O --finetune_lips --iters 250000

CUDA_VISIBLE_DEVICES=0 conda run -n pytorch3d python main.py data/${DATA_SET_NAME}/ --workspace trial_${DATA_SET_NAME}_eo_torso/ -O --torso --iters 200000 --head_ckpt trial_${DATA_SET_NAME}_eo/checkpoints/ngp.pth

# test
CUDA_VISIBLE_DEVICES=0 conda run -n pytorch3d python main.py data/${DATA_SET_NAME}/ --workspace trial_${DATA_SET_NAME}_eo_torso/ -O --torso --test