#!/bin/bash
# Submit jobs for specific classes with spaces in names
# Usage: ./submit_specific_classes.sh

set -e

cd "$(dirname "$0")/.."

# Define the specific classes we want to run
# Use the ORIGINAL names with spaces
CLASSES=("fishing rod" "hot air balloon")

echo "Submitting jobs for specific classes:"
for class in "${CLASSES[@]}"; do
    echo "  - $class"
done
echo ""

# Create the job template
TEMPLATE_FILE=$(mktemp)
cat > "$TEMPLATE_FILE" <<'TEMPLATE'
#!/bin/bash -l
#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --time=0-22:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsSwitch/SAFE_CLASS_NAME_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsSwitch/SAFE_CLASS_NAME_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail
mkdir -p /home/ryreu/guided_cnn/logsSwitch

source ~/miniconda3/etc/profile.d/conda.sh
conda activate learntolook

export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONNOUSERSITE=1

cd /home/ryreu/guided_cnn/code/SwitchCLIP/LearningToLook/code/WeCLIPPlus
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

echo "[$(date)] Host: $(hostname)"
which python

# Install open_clip if not present
python -c "import open_clip" 2>/dev/null || {
  echo "Installing open_clip_torch..."
  pip install -q open_clip_torch
}

# Run training for this specific class
# ACTUAL_CLASS_NAME will be replaced with the original class name (with spaces)
srun --unbuffered env CLIP_TEXT_VERSION="ACTUAL_CLASS_NAME" python -u generate_pseudo_masks_NICO.py
TEMPLATE

# Submit jobs for each class
JOB_IDS=()
FAILED_CLASSES=()
CLASS_NUM=0
DELAY=2  # seconds between submissions

for CLASS_NAME in "${CLASSES[@]}"; do
    CLASS_NUM=$((CLASS_NUM + 1))

    # Create a safe version for filenames (replace spaces with underscores)
    SAFE_CLASS_NAME="${CLASS_NAME// /_}"

    # Create class-specific job script
    JOB_SCRIPT=$(mktemp --suffix=.sh)

    # Replace BOTH the safe name (for filenames) and actual name (for env var)
    sed "s/SAFE_CLASS_NAME/$SAFE_CLASS_NAME/g" "$TEMPLATE_FILE" | \
        sed "s/ACTUAL_CLASS_NAME/$CLASS_NAME/g" > "$JOB_SCRIPT"

    # Submit the job with retry logic
    echo "[$CLASS_NUM/${#CLASSES[@]}] Submitting job for class: '$CLASS_NAME'"
    MAX_RETRIES=3
    RETRY=0
    SUCCESS=0

    while [ $RETRY -lt $MAX_RETRIES ]; do
        if JOB_ID=$(sbatch --parsable --job-name="nico_$SAFE_CLASS_NAME" "$JOB_SCRIPT" 2>&1); then
            JOB_IDS+=($JOB_ID)
            echo "    ✓ Job ID: $JOB_ID"
            SUCCESS=1
            break
        else
            RETRY=$((RETRY + 1))
            if [ $RETRY -lt $MAX_RETRIES ]; then
                echo "    ⚠ Submission failed (attempt $RETRY/$MAX_RETRIES), retrying in 5s..."
                sleep 5
            else
                echo "    ✗ Submission failed after $MAX_RETRIES attempts: $JOB_ID"
                FAILED_CLASSES+=("$CLASS_NAME")
            fi
        fi
    done

    # Clean up temp script
    rm "$JOB_SCRIPT"

    # Add delay between submissions (except after last one)
    if [ $CLASS_NUM -lt ${#CLASSES[@]} ] && [ $SUCCESS -eq 1 ]; then
        sleep $DELAY
    fi
done

# Clean up template
rm "$TEMPLATE_FILE"

echo ""
echo "========================================"
echo "Submission Complete!"
echo "========================================"
echo "Successfully submitted: ${#JOB_IDS[@]} jobs"
if [ ${#FAILED_CLASSES[@]} -gt 0 ]; then
    echo "Failed to submit: ${#FAILED_CLASSES[@]} jobs"
    echo "Failed classes: ${FAILED_CLASSES[@]}"
fi
echo ""
echo "Job IDs: ${JOB_IDS[@]}"
echo ""
echo "Monitor with: squeue --me"
echo "Cancel all with: scancel ${JOB_IDS[@]}"
echo ""
echo "Logs will be in: /home/ryreu/guided_cnn/logsSwitch/"
echo "  Format: <class_name_with_underscores>_<jobid>.out"
echo "========================================"
