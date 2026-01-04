#!/usr/bin/env python3
"""Generate batching{num}.sh files for each class in clip_text._all_class_names"""

import os
import ast

# Get the WeCLIPPlus root directory
script_dir = os.path.dirname(os.path.abspath(__file__))
weclip_root = os.path.dirname(script_dir)

# Read clip_text.py and extract _all_class_names
clip_text_path = os.path.join(weclip_root, 'clip', 'clip_text.py')
with open(clip_text_path, 'r') as f:
    content = f.read()

# Parse the Python file to find _all_class_names
tree = ast.parse(content)
_all_class_names = None

for node in ast.walk(tree):
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == '_all_class_names':
                # Extract the list value
                if isinstance(node.value, ast.List):
                    _all_class_names = [
                        elt.value for elt in node.value.elts
                        if isinstance(elt, ast.Constant)
                    ]
                break

if _all_class_names is None:
    raise ValueError("Could not find _all_class_names in clip_text.py")

# Template for batching script
BATCHING_TEMPLATE = """#!/bin/bash -l
#SBATCH --job-name=pseudo_NICO
#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --time=1-22:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --output=/home/ryreu/guided_cnn/logsSwitch/%x_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsSwitch/%x_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail
mkdir -p /home/ryreu/guided_cnn/logs2

source ~/miniconda3/etc/profile.d/conda.sh
conda activate learntolook

# Quiet TensorFlow logs (and save a bit of stdout spam)
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0

# Threading hints
export OMP_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-1}}"
export MKL_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-1}}"
export NUMEXPR_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-1}}"
export PYTHONNOUSERSITE=1

cd /home/ryreu/guided_cnn/code/SwitchCLIP/LearningToLook/code/WeCLIPPlus
export PYTHONPATH="$PWD:${{PYTHONPATH:-}}"

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
test -f generate_pseudo_masks_NICO.py || {{ echo "Missing generate_pseudo_masks_NICO.py" >&2; exit 2; }}

# Run (no --num_workers since the script doesn't accept it)
srun --unbuffered env CLIP_TEXT_VERSION={class_name} python -u generate_pseudo_masks_NICO.py
"""

def main():
    print(f"Found {len(_all_class_names)} classes in _all_class_names")
    print("Classes:", _all_class_names)
    print()

    # Generate batching files
    for num, class_name in enumerate(_all_class_names):
        filename = os.path.join(script_dir, f"batching{num}.sh")
        content = BATCHING_TEMPLATE.format(class_name=class_name)

        with open(filename, 'w') as f:
            f.write(content)

        # Make executable
        os.chmod(filename, 0o755)

        print(f"Created {filename} (CLIP_TEXT_VERSION={class_name})")

    print()
    print(f"Successfully created {len(_all_class_names)} batching files!")

if __name__ == "__main__":
    main()
