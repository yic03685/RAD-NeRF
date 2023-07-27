#!/bin/bash
# Example usage. Put wav files under
# input
#  -dataset1
#   -test.wave
# and run ./run.sh model_name dataset1

INPUT_FOLDER="input"
TRAINED_MODEL_FOLDER="trained"

if [ -z "$1" ]; then
  echo "Error: Dataset name not provided."
  echo "Usage: $0 <model_name> <input_dataset_name>"
  exit 1
fi

MODEL_NAME=$1
DATASET_FOLDER=$INPUT_FOLDER/$2

# Check if the folder exists
if [ ! -d "$DATASET_FOLDER" ]; then
  echo "Error: Folder '$DATASET_FOLDER' not found."
  exit 1
fi

# Find all WAV files in the folder
WAV_FILES=$(find "$DATASET_FOLDER" -type f -iname "*.wav")
echo $WAV_FILES

# Process each WAV file
for file_path in $WAV_FILES; do
  # Get the file name without the extension
  file_name=$(basename "$file_path" .wav)
  echo $file_name
  
  # Create a folder with the same name as the file (if it doesn't exist)
  folder_name="$DATASET_FOLDER/$file_name"
  if [ ! -d "$folder_name" ]; then
    mkdir "$folder_name"
  fi
  echo $folder_name

  # Move the file into its corresponding folder
  mv "$file_path" "$folder_name/"

  # Create npy file
  conda run -n pytorch3d python nerf/asr.py --wav $folder_name/${file_name}.wav --save_feats

#   # Run the synthesis
#   conda run -n pytorch3d python main.py \
#     --workspace $TRAINED_MODEL_FOLDER/$MODEL_NAME/${MODEL_NAME}_eo_torso \
#     --aud $folder_name/${file_name}.npy \
#     --torso \
#     --smooth_path_window 50 \
#     --data_range 0 -2 \
#     -O \
#     --test 

    # Run the synthesis
    conda run -n pytorch3 python test.py --pose $TRAINED_MODEL_FOLDER/$MODEL_NAME/transforms_val.json \
        --ckpt $TRAINED_MODEL_FOLDER/$MODEL_NAME/${MODEL_NAME}_eo_torso/checkpoints/ngp.pth \
        --aud $folder_name/${file_name}.npy \
        --workspace $folder_name \
        --smooth_path_window 50 \
        --data_range 0 -2 \
        -O \
        --torso
done





