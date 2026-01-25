#!/bin/bash -l
#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --time=3-12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterSwitch/waterbirds_100_seg_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterSwitch/waterbirds_100_seg_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail
mkdir -p /home/ryreu/guided_cnn/logsWaterSwitch

source ~/miniconda3/etc/profile.d/conda.sh
conda activate learntolook

export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONNOUSERSITE=1

cd /home/ryreu/guided_cnn/waterbirds/New_Teach/L100/LearningToLook/code/WeCLIPPlus
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

echo "[$(date)] Host: $(hostname)"
which python

# Install open_clip if not present
python -c "import open_clip" 2>/dev/null || {
  echo "Installing open_clip_torch..."
  pip install -q open_clip_torch
}

srun --unbuffered python -u generate_pseudo_masks_waterbirds.py \
  --setup-data \
  --repo-root "/home/ryreu/guided_cnn/waterbirds/New_Teach/L100/LearningToLook" \
  --src-img-dir "/home/ryreu/guided_cnn/waterbirds/waterbird_1.0_forest2water2"\
  --class-name "bird"
