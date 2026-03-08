#!/bin/bash -l
#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsWaterSwitch/waterbirds100_openclip_laion_dinovit_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsWaterSwitch/waterbirds100_openclip_laion_dinovit_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail
mkdir -p /home/ryreu/guided_cnn/logsWaterSwitch

REPO_ROOT="/home/ryreu/guided_cnn/Food101/LearningToLook"
WECLIP_ROOT="${REPO_ROOT}/code/WeCLIPPlus"
SRC_IMG_DIR="/home/ryreu/guided_cnn/waterbirds/waterbird_1.0_forest2water2"

CONDA_ENV="learntolook"
CLASS_NAME="bird"
RESULTS_DIR="results_waterbirds100_openclip_laion_dinovit"

# OpenCLIP LAION settings (override via env if needed).
CLIP_BACKEND="${CLIP_BACKEND:-openclip}"
CLIP_MODEL_NAME="${CLIP_MODEL_NAME:-}"                 # Optional, e.g. ViT-B-16
CLIP_PRETRAINED="${CLIP_PRETRAINED:-laion2b_s34b_b88k}"

# ViT DINO settings.
DINO_MODEL="${DINO_MODEL:-dinov2_vitb14_reg}"
DINO_FTS_DIM="${DINO_FTS_DIM:-768}"
DINO_DECODER_LAYERS="${DINO_DECODER_LAYERS:-3}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

export CLIP_BACKEND
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

rm -f "${WECLIP_ROOT}/configs/voc_attn_reg_runtime.yaml"

ARGS=(--setup-data
      --repo-root "${REPO_ROOT}"
      --src-img-dir "${SRC_IMG_DIR}"
      --class-name "${CLASS_NAME}"
      --results-dir "${RESULTS_DIR}"
      --clip-backend "${CLIP_BACKEND}"
      --clip-pretrained "${CLIP_PRETRAINED}"
      --dino-model "${DINO_MODEL}"
      --dino-fts-dim "${DINO_FTS_DIM}"
      --dino-decoder-layers "${DINO_DECODER_LAYERS}")

if [[ -n "${CLIP_MODEL_NAME}" ]]; then
  export CLIP_MODEL_NAME
  ARGS+=(--clip-model "${CLIP_MODEL_NAME}")
fi

echo "Running generate_pseudo_masks_waterbirds.py ${ARGS[*]}"

srun --unbuffered python -u generate_pseudo_masks_waterbirds.py "${ARGS[@]}"
