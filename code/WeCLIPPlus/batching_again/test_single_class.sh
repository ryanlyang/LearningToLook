#!/bin/bash -l
#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --time=0-02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsDINO/test_bear_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsDINO/test_bear_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail
mkdir -p /home/ryreu/guided_cnn/logsDINO

source ~/miniconda3/etc/profile.d/conda.sh
conda activate learntolook

export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONNOUSERSITE=1

cd /home/ryreu/guided_cnn/code/SwitchDINO/LearningToLook/code/WeCLIPPlus
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

echo "[$(date)] Host: $(hostname)"
which python

# Install required packages
python -c "import open_clip" 2>/dev/null || {
  echo "Installing open_clip_torch..."
  pip install -q open_clip_torch
}

python -c "import timm" 2>/dev/null || {
  echo "Installing timm (latest version)..."
  pip install -q --upgrade timm
}

# Fix DINOv1 cache issue
echo "Cleaning DINOv1 cache..."
CACHE_DIR="$HOME/.cache/torch/hub/facebookresearch_dino_main"
if [ -d "$CACHE_DIR" ]; then
    echo "Removing corrupted cache at $CACHE_DIR"
    rm -rf "$CACHE_DIR"
fi

# Test with a single class (bear)
echo "Testing with class: bear"
srun --unbuffered env CLIP_TEXT_VERSION="bear" python -u generate_pseudo_masks_NICO.py

echo "[$(date)] Test completed successfully!"
