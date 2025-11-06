#!/bin/bash -l
#SBATCH --job-name=pseudo_NICO
#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=4-22:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logs/%x_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logs/%x_%j.err
#SBATCH --signal=TERM@120

set -euo pipefail
mkdir -p /home/ryreu/guided_cnn/logs

# ---- Env ----
spack env activate default-ml-x86_64-25050601
spack load opencv
# spack load py-pytorch
# spack load py-torchvision
# spack load py-omegaconf
# spack load py-hydra-core  # if you use Hydra configs

# ---- Run dir ----
cd /home/ryreu/guided_cnn/code/LearningToLook/code/WeCLIPPlus

# ---- Imports + threads ----
export PYTHONPATH="$PWD:$PYTHONPATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# ---- Graceful timeout hook (optional) ----
trap 'pkill -SIGUSR1 -f generate_psuedo_masks_NICO.py || true' TERM

# ---- Launch ----
srun --unbuffered python -u generate_psuedo_masks_NICO.py \
    --num_workers "$(( ${SLURM_CPUS_PER_TASK:-24} - 1 ))"