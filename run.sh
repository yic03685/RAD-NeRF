#!/bin/bash

DATA_SET_NAME="kenny_surf"

# python test.py --pose data/kenny/transforms_val.json \
#     --ckpt trial_kenny_torso/checkpoints/ngp.pth \
#     --aud data/test/ian_16sec_eo.npy \
#     --workspace trial_kenny_test \
#     -O \
#     --torso

# python main.py --pose data/kenny/transforms_val.json \
#     --ckpt trial_kenny_torso/checkpoints/ngp.pth \
#     --aud data/test/ian_16sec_eo.npy \
#     --workspace trial_kenny_test \
#     -O \
#     --torso


python main.py data/$DATA_SET_NAME \
    --workspace trial_${DATA_SET_NAME}_eo_torso \
    --aud data/test/ian_16sec_eo.npy \
    --torso \
    --smooth_path_window 50 \
    --data_range 0 -2 \
    -O \
    --test 