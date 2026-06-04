#!/bin/bash
#SBATCH --job-name=parakeet-unseen-transcribe
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${HOME}/eng-sports-parakeet-emotion/parakeet_emotion_ft}"
RUN_ROOT="${RUN_ROOT:-${HOME}/runs/parakeet-emotion}"
VENV_DIR="${VENV_DIR:-${HOME}/.venvs/parakeet-emotion-2503}"
FULL_DATASET_PARQUET="${FULL_DATASET_PARQUET:-${HOME}/eng-sports-parakeet-emotion/data/full_iu_parakeet_labeled_dataset.parquet}"
MODEL_DIR="${MODEL_DIR:-${RUN_ROOT}/train}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_ROOT}/unseen_transcription}"

mkdir -p ${HOME}/logs/parakeet \
  ${HOME}/cache/hf \
  ${HOME}/cache/hf/hub \
  ${HOME}/cache/torch \
  ${HOME}/tmp \
  "${OUTPUT_DIR}"

export HF_HOME="${HF_HOME:-${HOME}/cache/hf}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HOME}/cache/hf/hub}"
export TORCH_HOME="${TORCH_HOME:-${HOME}/cache/torch}"
export TMPDIR="${TMPDIR:-${HOME}/tmp}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

if [[ -z "${HF_TOKEN:-}" && -f ${HOME}/.cache/huggingface/token ]]; then
  export HF_TOKEN
  HF_TOKEN="$(tr -d "[:space:]" < ${HOME}/.cache/huggingface/token)"
fi
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN}}"
fi

IMAGE="nvcr.io#nvidia/pytorch:25.03-py3"

srun --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" \
  --container-image="${IMAGE}" \
  --container-mounts="${HOME}:${HOME}" \
  bash -lc '
    set -euo pipefail
    cd "'"${PROJECT_DIR}"'"
    source "'"${VENV_DIR}"'"/bin/activate
    export RANK="${RANK:-0}"
    export LOCAL_RANK="${LOCAL_RANK:-0}"
    export WORLD_SIZE="${WORLD_SIZE:-1}"
    export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
    export MASTER_PORT="${MASTER_PORT:-29501}"

    echo "model_dir='"${MODEL_DIR}"'"
    echo "dataset_parquet='"${FULL_DATASET_PARQUET}"'"
    nvidia-smi || true

    python transcribe_unseen_samples.py \
      --model-dir "'"${MODEL_DIR}"'" \
      --dataset-parquet "'"${FULL_DATASET_PARQUET}"'" \
      --output-dir "'"${OUTPUT_DIR}"'" \
      --split test \
      --samples-per-label 3 \
      --batch-size 2
  '
