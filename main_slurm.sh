#!/bin/bash
    
# --- SLURM Settings ---
# Job name: A descriptive name for the long job
#SBATCH --job-name=AE_Baseline_GPU
# Partition: (Using your confirmed partition)
#SBATCH --partition=studentkillable 
# Account: (Using your confirmed account)
#SBATCH --account=gpu-students 
# CRITICAL: Request one GPU resource (If the link is fixed later, it will use this)
#SBATCH --gpus=1
#SBATCH --gres=gpu:1 
# Output file (monitor this for your loss results)
#SBATCH --output=training_output.txt
#SBATCH --error=training_error.txt
# Time limit for the job (2 hours)
#SBATCH --time=02:00:00 

# More settings
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4



# --- Environment Setup ---
ENV_PATH="/home/ML_courses/03683533_2025/yonatan_uri_reshit/venv"

# Activate the Conda environment
source /home/ML_courses/03683533_2025/miniconda3/bin/activate $ENV_PATH

# --- Run the Python Script ---
MODEL_NAME_ARG=${1:-}

if [ -z "$MODEL_NAME_ARG" ]; then
    exec python main.py
else
    exec python main.py --model-name "$MODEL_NAME_ARG"
fi
