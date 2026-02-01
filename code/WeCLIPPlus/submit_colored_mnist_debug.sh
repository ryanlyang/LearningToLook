#!/bin/bash -l
#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsMNIST/colored_mnist_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsMNIST/colored_mnist_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail
mkdir -p /home/ryreu/guided_cnn/logsMNIST

REPO_ROOT="/home/ryreu/guided_cnn/MNIST_AGAIN/ColorGen/LearningToLook"
WECLIP_ROOT="${REPO_ROOT}/code/WeCLIPPlus"

CONDA_ENV="learntolook"
CLASS_NAME="digit"
RESULTS_DIR="results_mnist"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

export CLIP_BACKEND="openai"
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONNOUSERSITE=1

echo "[$(date)] Host: $(hostname)"
which python

cd "${WECLIP_ROOT}"
export PYTHONPATH="${WECLIP_ROOT}:${PYTHONPATH:-}"

python -c "import open_clip" 2>/dev/null || {
  echo "Installing open_clip_torch..."
  pip install -q open_clip_torch
}

ARGS=(--repo-root "${REPO_ROOT}"
      --class-name "${CLASS_NAME}"
      --results-dir "${RESULTS_DIR}")

echo "Running generate_pseudo_masks_ColoredMNIST.py ${ARGS[*]}"

srun --unbuffered python -u generate_pseudo_masks_ColoredMNIST.py "${ARGS[@]}"
