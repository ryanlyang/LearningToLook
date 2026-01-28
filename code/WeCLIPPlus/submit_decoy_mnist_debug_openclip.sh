#!/bin/bash -l
#SBATCH --account=reu-aisocial
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsMNIST/decoy_mnist_openclip_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsMNIST/decoy_mnist_openclip_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail
mkdir -p /home/ryreu/guided_cnn/logsMNIST

REPO_ROOT="/home/ryreu/guided_cnn/MNIST_AGAIN/LearningToLook"
WECLIP_ROOT="${REPO_ROOT}/code/WeCLIPPlus"

CONDA_ENV="learntolook"
SRC_IMG_DIR="${REPO_ROOT}/data/saved/DecoyMNIST_images/digit"
SPLIT="train"
CLASS_NAME="digit"
SETUP_DATA=1
SORT_BY_LABEL=0
RESULTS_DIR="results_decoy_mnist_openclip"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

export CLIP_BACKEND="openclip"
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONNOUSERSITE=1

echo "[$(date)] Host: $(hostname)"
which python

echo "Generating DecoyMNIST dataset..."
srun --unbuffered python -u "${REPO_ROOT}/data/decoy_mnist.py"

cd "${WECLIP_ROOT}"
export PYTHONPATH="${WECLIP_ROOT}:${PYTHONPATH:-}"

python -c "import open_clip" 2>/dev/null || {
  echo "Installing open_clip_torch..."
  pip install -q open_clip_torch
}

ARGS=(--repo-root "${REPO_ROOT}"
      --src-img-dir "${SRC_IMG_DIR}"
      --split "${SPLIT}"
      --class-name "${CLASS_NAME}"
      --results-dir "${RESULTS_DIR}")

if [[ "${SETUP_DATA}" -eq 1 ]]; then
  ARGS+=(--setup-data)
else
  ARGS+=(--no-setup-data)
fi

if [[ "${SORT_BY_LABEL}" -eq 1 ]]; then
  ARGS+=(--sort-by-label)
fi

echo "Running generate_pseudo_masks_DecoyMNIST.py ${ARGS[*]}"

srun --unbuffered python -u generate_pseudo_masks_DecoyMNIST.py "${ARGS[@]}"
