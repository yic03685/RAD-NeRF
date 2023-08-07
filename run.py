import os
import shutil
import subprocess
from LearningBuddyCommon.integration.integration_utils import get_config_file

INPUT_FOLDER = "input"
OUTPUT_FOLDER= "output"
TRAINED_MODEL_FOLDER = "trained"
TEMP_FOLDER = "temp"

def main():
    config = get_config_file("input/config.json")
    input_audio_path = os.path.join(INPUT_FOLDER, config["input_audio"])
    input_audio_name, _extension = os.path.splitext(os.path.basename(input_audio_path))
    output_audio_path = config["output_path"]
    model_name = config["model_name"]


    # in order not to pollute input, we copy to a temp file
    os.makedirs(TEMP_FOLDER, exist_ok=True)
    input_audio_workspace_path = os.path.join(TEMP_FOLDER, config["input_audio"])
    shutil.copy(input_audio_path, input_audio_workspace_path)
    
    # There are some metadata are byproduct of training
    trained_metadata_path = os.path.join(TRAINED_MODEL_FOLDER, f"{model_name}")
    trained_model_path = os.path.join(TRAINED_MODEL_FOLDER, f"{model_name}/{model_name}_eo_torso")
    
    # Create npy file
    subprocess.run(["python", "nerf/asr.py", "--wav", input_audio_workspace_path, "--save_feats"])

    # Create video
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    subprocess.run(["python", "test.py", 
                    "--pose", os.path.join(trained_metadata_path, "transforms_val.json"),
                    "--ckpt", os.path.join(trained_model_path, "checkpoints/ngp.pth"),
                    "--workspace", TEMP_FOLDER,
                    "--bg_img", os.path.join(trained_metadata_path, "bc.jpg"),
                    "--aud", os.path.join(TEMP_FOLDER, f"{input_audio_name}_eo.npy"),
                    "-O", "--torso", "--data_range", "0", "100"])
    
    # Filter files with the specified prefix
    file_list = os.listdir(os.path.join(TEMP_FOLDER, "results"))
    generated_video_file = [file for file in file_list if file.startswith("ngp_")]

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", 
                    "-i", os.path.join(TEMP_FOLDER, f"results/{generated_video_file[0]}"), 
                    "-i", input_audio_workspace_path, 
                    "-c:v", "copy", "-c:a", "aac", os.path.join(OUTPUT_FOLDER, output_audio_path)])

if __name__ == "__main__":
    main()
