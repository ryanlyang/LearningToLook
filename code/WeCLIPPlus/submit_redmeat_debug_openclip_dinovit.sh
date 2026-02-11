#!/bin/bash -l
#SBATCH --account=reu-aisocial
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsMeat/redmeat_openclip_dinovit_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsMeat/redmeat_openclip_dinovit_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail
mkdir -p /home/ryreu/guided_cnn/logsMeat

REPO_ROOT="/home/ryreu/guided_cnn/Food101/LearningToLook"
WECLIP_ROOT="${REPO_ROOT}/code/WeCLIPPlus"
SPLIT_IMAGES_DIR="/home/ryreu/guided_cnn/Food101/data/food-101-redmeat/split_images"

CONDA_ENV="learntolook"
CLASS_NAME="meat"
RESULTS_DIR="results_redmeat_openclip_dinovit"

# ViT DINO
DINO_MODEL="dinov2_vitb14_reg"
DINO_FTS_DIM=768
DINO_DECODER_LAYERS=3

# Optional OpenCLIP overrides:
# CLIP_MODEL_NAME="ViT-B-16-quickgelu"
# CLIP_PRETRAINED="openai"
CLIP_MODEL_NAME="${CLIP_MODEL_NAME:-}"
CLIP_PRETRAINED="${CLIP_PRETRAINED:-}"

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

cd "${WECLIP_ROOT}"
export PYTHONPATH="${WECLIP_ROOT}:${PYTHONPATH:-}"

python -c "import open_clip" 2>/dev/null || {
  echo "Installing open_clip_torch..."
  pip install -q open_clip_torch
}

rm -f "${WECLIP_ROOT}/configs/voc_attn_reg_runtime.yaml"

ARGS=(--repo-root "${REPO_ROOT}"
      --split-images-dir "${SPLIT_IMAGES_DIR}"
      --class-name "${CLASS_NAME}"
      --results-dir "${RESULTS_DIR}"
      --clip-backend "openclip"
      --dino-model "${DINO_MODEL}"
      --dino-fts-dim "${DINO_FTS_DIM}"
      --dino-decoder-layers "${DINO_DECODER_LAYERS}"
      --no-setup-data)

if [[ -n "${CLIP_MODEL_NAME}" ]]; then
  export CLIP_MODEL_NAME
  ARGS+=(--clip-model "${CLIP_MODEL_NAME}")
fi
if [[ -n "${CLIP_PRETRAINED}" ]]; then
  export CLIP_PRETRAINED
  ARGS+=(--clip-pretrained "${CLIP_PRETRAINED}")
fi

echo "Running generate_pseudo_masks_redMeat.py ${ARGS[*]}"

srun --unbuffered python -u generate_pseudo_masks_redMeat.py "${ARGS[@]}"
