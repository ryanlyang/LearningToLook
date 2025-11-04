#!/bin/bash -l
#
#SBATCH --job-name=psuedo_NICO
#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=4-22:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logs/%x_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logs/%x_%j.err


# ——————— Load your environment ———————
spack env activate default-ml-x86_64-25050601
spack load opencv 

# (Or, if you’re using Conda instead:)
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate my_env

# ——————— Go to your code directory ———————
cd /home/ryreu/guided_cnn/code/LearningToLook/code/WeCLIPPlus    # replace <your_repo_name> with your actual folder

# ——————— Run the hyperparameter script ———————
python generate_pseudo_masks_NICO.py
