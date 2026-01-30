#!/bin/bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

FIRST_JOB="${SCRIPT_DIR}/submit_decoy_mnist_debug.sh"
OPENCLIP_JOB="${SCRIPT_DIR}/submit_decoy_mnist_debug_openclip.sh"
OPENAI_XCIT_JOB="${SCRIPT_DIR}/submit_decoy_mnist_debug_openai_xcit.sh"

JOB_ID="$(sbatch --parsable "${FIRST_JOB}")"
echo "Submitted first DecoyMNIST job: ${JOB_ID}"

DEP="afterok:${JOB_ID}"

JOB_ID_OPENCLIP="$(sbatch --parsable --dependency="${DEP}" "${OPENCLIP_JOB}")"
echo "Submitted OpenCLIP job (afterok): ${JOB_ID_OPENCLIP}"

JOB_ID_OPENAI_XCIT="$(sbatch --parsable --dependency="${DEP}" "${OPENAI_XCIT_JOB}")"
echo "Submitted OpenAI+XCiT job (afterok): ${JOB_ID_OPENAI_XCIT}"
