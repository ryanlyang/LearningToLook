#!/bin/bash -l
#SBATCH --account=reu-aisocial
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterSwitch/siglip2_waterbirds_95_seg_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterSwitch/siglip2_waterbirds_95_seg_%j.err
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

cd /home/ryreu/guided_cnn/waterbirds/newCLIP/LearningToLook/code/WeCLIPPlus
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

echo "[$(date)] Host: $(hostname)"
which python

python -c "import open_clip" 2>/dev/null || {
  echo "Installing open_clip_torch..."
  pip install -q open_clip_torch
}

srun --unbuffered python -u generate_pseudo_masks_waterbirds.py \
  --setup-data \
  --repo-root "/home/ryreu/guided_cnn/waterbirds/newCLIP/LearningToLook" \
  --src-img-dir "/home/ryreu/guided_cnn/waterbirds/waterbird_complete95_forest2water2" \
  --class-name "bird" \
  --clip-backend siglip2 \
  --clip-model ViT-B-16-SigLIP2 \
  --clip-pretrained webli \
  --results-dir "results_siglip2"
