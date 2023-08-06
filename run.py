import os
import shutil
import subprocess
from LearningBuddyCommon.integration.integration_utils import get_config_file

INPUT_FOLDER = "input"
OUTPUT_FOLDER= "output"
TRAINED_MODEL_FOLDER = "trained"
TEMO_FOLDER = "temp"

def main():
    config = get_config_file("input/config.json")
    input_audio_path = os.path.join(INPUT_FOLDER, config["input_audio"])
    output_audio_path = config["output_path"]
    model_name = config["model_name"]


    # in order not to pollute input, we copy to a temp file
    os.makedirs(TEMO_FOLDER, exist_ok=True)
    input_audio_workspace_path = os.path.join(TEMO_FOLDER, config["input_audio"])
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
                    "--workspace", TEMO_FOLDER,
                    "--bg_img", os.path.join(trained_metadata_path, "bc.jpg"),
                    "--aud", os.path.join(TEMO_FOLDER, "input_audio_eo.npy"),
                    "-O", "--torso", "--data_range", "0", "100"])
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", 
                    "-i", os.path.join(TEMO_FOLDER, "results/ngp_ep0009.mp4"), 
                    "-i", input_audio_workspace_path, 
                    "-c:v", "copy", "-c:a", "aac", os.path.join(OUTPUT_FOLDER, output_audio_path)])

if __name__ == "__main__":
    main()
