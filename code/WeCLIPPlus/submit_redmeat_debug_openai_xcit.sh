#!/bin/bash -l
#SBATCH --account=reu-aisocial
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsMeat/redmeat_openai_xcit_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsMeat/redmeat_openai_xcit_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail
mkdir -p /home/ryreu/guided_cnn/logsMeat

REPO_ROOT="/home/ryreu/guided_cnn/Food101/LearningToLook"
WECLIP_ROOT="${REPO_ROOT}/code/WeCLIPPlus"
SPLIT_IMAGES_DIR="/home/ryreu/guided_cnn/Food101/data/food-101-redmeat/split_images"

CONDA_ENV="learntolook"
CLASS_NAME="meat"
RESULTS_DIR="results_redmeat_openai_xcit"

# XCiT DINO (OpenAI CLIP stays the same)
DINO_MODEL="xcit_medium_24_p16"
DINO_FTS_DIM=512
DINO_DECODER_LAYERS=5
XCIT_WEIGHTS="${WECLIP_ROOT}/pretrained/dino_xcit_medium_24_p16_pretrain.pth"

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

if [[ ! -f "${XCIT_WEIGHTS}" ]]; then
  echo "Missing XCiT pretrained weights: ${XCIT_WEIGHTS}"
  echo "Place dino_xcit_medium_24_p16_pretrain.pth under ${WECLIP_ROOT}/pretrained/"
  exit 1
fi

rm -f "${WECLIP_ROOT}/configs/voc_attn_reg_runtime.yaml"

ARGS=(--repo-root "${REPO_ROOT}"
      --split-images-dir "${SPLIT_IMAGES_DIR}"
      --class-name "${CLASS_NAME}"
      --results-dir "${RESULTS_DIR}"
      --clip-backend "openai"
      --dino-model "${DINO_MODEL}"
      --dino-fts-dim "${DINO_FTS_DIM}"
      --dino-decoder-layers "${DINO_DECODER_LAYERS}"
      --no-setup-data)

echo "Running generate_pseudo_masks_redmeat.py ${ARGS[*]}"

srun --unbuffered python -u generate_pseudo_masks_redmeat.py "${ARGS[@]}"
