#!/bin/bash
#SBATCH --job-name=parakeet-emotion-head
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

MODE="${1:-smoke}"
PROJECT_DIR="${PROJECT_DIR:-${HOME}/eng-sports-parakeet-emotion/parakeet_emotion_ft}"
RUN_ROOT="${RUN_ROOT:-${HOME}/runs/parakeet-emotion}"
VENV_DIR="${VENV_DIR:-${HOME}/.venvs/parakeet-emotion-2503}"
ASR_MODEL_DIR="${ASR_MODEL_DIR:-${RUN_ROOT}/mixture_train}"
EMOTION_MIXTURE_PARQUET="${EMOTION_MIXTURE_PARQUET:-${HOME}/eng-sports-parakeet-emotion/data/emotion_mixture_train}"

mkdir -p ${HOME}/logs/parakeet \
  ${HOME}/cache/hf \
  ${HOME}/cache/hf/hub \
  ${HOME}/cache/torch \
  ${HOME}/tmp \
  "${RUN_ROOT}"

export HF_HOME="${HF_HOME:-${HOME}/cache/hf}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HOME}/cache/hf/hub}"
export TORCH_HOME="${TORCH_HOME:-${HOME}/cache/torch}"
export TMPDIR="${TMPDIR:-${HOME}/tmp}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}"

if [[ -z "${HF_TOKEN:-}" && -f ${HOME}/.cache/huggingface/token ]]; then
  export HF_TOKEN
  HF_TOKEN="$(tr -d "[:space:]" < ${HOME}/.cache/huggingface/token)"
fi
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN}}"
fi

IMAGE="nvcr.io#nvidia/pytorch:25.03-py3"

srun --cpus-per-task="${SLURM_CPUS_PER_TASK:-12}" \
  --container-image="${IMAGE}" \
  --container-mounts="${HOME}:${HOME}" \
  bash -lc '
    set -euo pipefail
    cd "'"${PROJECT_DIR}"'"

    python -m venv --system-site-packages "'"${VENV_DIR}"'"
    source "'"${VENV_DIR}"'"/bin/activate
    unset PIP_CONSTRAINT
    unset PIP_REQUIRE_VIRTUALENV
    export PIP_CONFIG_FILE=/dev/null
    export RANK="${RANK:-0}"
    export LOCAL_RANK="${LOCAL_RANK:-0}"
    export WORLD_SIZE="${WORLD_SIZE:-1}"
    export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
    export MASTER_PORT="${MASTER_PORT:-29502}"
    REQ_HASH="$(sha256sum requirements.txt | awk "{print \$1}")"
    INSTALL_MARKER="'"${VENV_DIR}"'/.requirements-${REQ_HASH}"
    python -m pip install --upgrade pip
    if [[ ! -f "${INSTALL_MARKER}" ]]; then
      python -m pip install --upgrade -r requirements.txt
      rm -f "'"${VENV_DIR}"'"/.requirements-*
      touch "${INSTALL_MARKER}"
    fi

    echo "mode='"${MODE}"'"
    echo "asr_model_dir='"${ASR_MODEL_DIR}"'"
    echo "dataset_parquet='"${EMOTION_MIXTURE_PARQUET}"'"
    nvidia-smi || true
    python - <<PY
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available(), torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY

    if [[ "'"${MODE}"'" == "smoke" ]]; then
      python train_emotion_classifier_head.py \
        --model-dir "'"${ASR_MODEL_DIR}"'" \
        --dataset-parquet "'"${EMOTION_MIXTURE_PARQUET}"'" \
        --output-dir "'"${RUN_ROOT}"'"/emotion_head_smoke \
        --max-train-samples 4096 \
        --max-eval-samples 512 \
        --max-steps 300 \
        --eval-steps 100 \
        --batch-size 8 \
        --eval-batch-size 8 \
        --required-val-accuracy 0.20 \
        --required-val-macro-f1 0.12
    elif [[ "'"${MODE}"'" == "full" ]]; then
      python train_emotion_classifier_head.py \
        --model-dir "'"${ASR_MODEL_DIR}"'" \
        --dataset-parquet "'"${EMOTION_MIXTURE_PARQUET}"'" \
        --output-dir "'"${RUN_ROOT}"'"/emotion_head \
        --hub-model-id NathanRoll/parakeet-tdt-0.6b-v3-eng-sports-emotion-iu-mixture-emotion-head \
        --max-steps 3000 \
        --eval-steps 250 \
        --batch-size 8 \
        --eval-batch-size 8 \
        --required-val-accuracy 0.25 \
        --required-val-macro-f1 0.16 \
        --push-to-hub
    else
      echo "unknown mode: '"${MODE}"'" >&2
      exit 2
    fi
  '
