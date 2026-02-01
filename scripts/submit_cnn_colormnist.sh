#!/bin/bash -l
#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsMNIST/guided_cnn_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsMNIST/guided_cnn_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail
mkdir -p /home/ryreu/guided_cnn/logsMNIST

REPO_ROOT="/home/ryreu/guided_cnn/MNIST_AGAIN/LearningToLook"
CONDA_ENV="learntolook"

DATA_PATH="${REPO_ROOT}/data/saved/ColorMNIST_images/digit"
GT_PATH="${REPO_ROOT}/code/WeCLIPPlus/results_mnist/val/prediction_cmap"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONNOUSERSITE=1

cd "${REPO_ROOT}"

echo "[$(date)] Host: $(hostname)"
which python

echo "Running invert CNN on ColorMNIST"
echo "Data: ${DATA_PATH}"
echo "GT:   ${GT_PATH}"

srun --unbuffered python -u scripts/run_guided_CNN.py \
  "${DATA_PATH}" \
  "${GT_PATH}" \
  --attention_epoch 15 \
  --kl_lambda 160
