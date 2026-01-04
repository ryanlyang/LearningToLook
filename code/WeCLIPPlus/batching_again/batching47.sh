#!/bin/bash -l
#SBATCH --job-name=pseudo_NICO
#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=4-22:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logs2/%x_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logs2/%x_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail
mkdir -p /home/ryreu/guided_cnn/logs2

source ~/miniconda3/etc/profile.d/conda.sh
conda activate learntolook

# Quiet TensorFlow logs (and save a bit of stdout spam)
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0

# Threading hints
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONNOUSERSITE=1

cd /home/ryreu/guided_cnn/code/HaveNicoLearn/LearningToLook/code/WeCLIPPlus
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# Sanity print
echo "[$(date)] Host: $(hostname)"
which python
python - <<'PY'
import sys, torch
print("Python:", sys.version.split()[0])
print("Torch:", getattr(torch,'__version__','missing'),
      "CUDA:", getattr(torch.version,'cuda','n/a'),
      "CUDA available:", torch.cuda.is_available() if hasattr(torch,'cuda') else 'n/a')
PY

# Ensure entrypoint exists
test -f generate_pseudo_masks_NICO.py || { echo "Missing generate_pseudo_masks_NICO.py" >&2; exit 2; }

# Run (no --num_workers since the script doesn't accept it)
srun --unbuffered env CLIP_TEXT_VERSION=wheat python -u generate_pseudo_masks_NICO.py
