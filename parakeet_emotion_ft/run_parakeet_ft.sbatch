#!/bin/bash
#SBATCH --job-name=parakeet-emotion-ft
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

MODE="${1:-smoke}"
PROJECT_DIR="${PROJECT_DIR:-${HOME}/eng-sports-parakeet-emotion/parakeet_emotion_ft}"
RUN_ROOT="${RUN_ROOT:-${HOME}/runs/parakeet-emotion}"
VENV_DIR="${VENV_DIR:-${HOME}/.venvs/parakeet-emotion-2503}"
FULL_DATASET_PARQUET="${FULL_DATASET_PARQUET:-${HOME}/eng-sports-parakeet-emotion/data/full_iu_parakeet_labeled_dataset.parquet}"
EMOTION_MIXTURE_PARQUET="${EMOTION_MIXTURE_PARQUET:-${HOME}/eng-sports-parakeet-emotion/data/emotion_mixture_train}"
CONCAT_IU_PARQUET="${CONCAT_IU_PARQUET:-${HOME}/eng-sports-parakeet-emotion/data/emotion_mixture_concat_iu}"

mkdir -p ${HOME}/logs/parakeet \
  ${HOME}/cache/hf \
  ${HOME}/cache/hf/hub \
  ${HOME}/cache/torch \
  ${HOME}/cache/wandb \
  ${HOME}/tmp \
  "${RUN_ROOT}"

export HF_HOME="${HF_HOME:-${HOME}/cache/hf}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HOME}/cache/hf/hub}"
export TORCH_HOME="${TORCH_HOME:-${HOME}/cache/torch}"
export WANDB_DIR="${WANDB_DIR:-${HOME}/cache/wandb}"
export TMPDIR="${TMPDIR:-${HOME}/tmp}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"

if [[ -z "${HF_TOKEN:-}" && -f ${HOME}/.cache/huggingface/token ]]; then
  export HF_TOKEN
  HF_TOKEN="$(tr -d "[:space:]" < ${HOME}/.cache/huggingface/token)"
fi
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN}}"
fi

IMAGE="nvcr.io#nvidia/pytorch:25.03-py3"

srun --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" \
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
    export MASTER_PORT="${MASTER_PORT:-29500}"
    REQ_HASH="$(sha256sum requirements.txt | awk "{print \$1}")"
    INSTALL_MARKER="'"${VENV_DIR}"'/.requirements-${REQ_HASH}"
    python -m pip install --upgrade pip
    if [[ ! -f "${INSTALL_MARKER}" ]]; then
      python -m pip install --upgrade -r requirements.txt
      rm -f "'"${VENV_DIR}"'"/.requirements-*
      touch "${INSTALL_MARKER}"
    fi

    python - <<PY
from pathlib import Path
import site

old = """if _is_torch_greater_or_equal_than_2_6:
    from torch._dynamo._trace_wrapped_higher_order_op import TransformGetItemToIndex
"""
new = """if _is_torch_greater_or_equal_than_2_6:
    try:
        from torch._dynamo._trace_wrapped_higher_order_op import TransformGetItemToIndex
    except (ImportError, AttributeError):
        _is_torch_greater_or_equal_than_2_6 = False
"""
for base in site.getsitepackages():
    path = Path(base) / "transformers" / "masking_utils.py"
    if not path.exists():
        continue
    text = path.read_text()
    if old in text:
        path.write_text(text.replace(old, new))
        print(f"patched optional torch dynamo helper in {path}")
    elif "TransformGetItemToIndex" in text and "except (ImportError, AttributeError)" in text:
        print(f"optional torch dynamo helper already patched in {path}")
    else:
        print(f"no dynamo helper patch needed in {path}")

for base in site.getsitepackages():
    path = Path(base) / "transformers" / "loss" / "loss_tdt.py"
    if not path.exists():
        continue
    text = path.read_text()
    patched = text
    patched = patched.replace("targets=labels.to(device).int(),", "targets=labels.to(device).long(),")
    patched = patched.replace("logit_lengths=logit_lengths.to(device).int(),", "logit_lengths=logit_lengths.to(device).long(),")
    patched = patched.replace("target_lengths=label_lengths.to(device).int(),", "target_lengths=label_lengths.to(device).long(),")
    if patched != text:
        path.write_text(patched)
        print(f"patched TDT loss index dtypes in {path}")
    elif "targets=labels.to(device).long()" in text:
        print(f"TDT loss index dtype patch already present in {path}")
    else:
        print(f"no TDT loss dtype patch applied in {path}")
PY

    echo "mode='"${MODE}"'"
    nvidia-smi || true
    python - <<PY
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available(), torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY

    if [[ "'"${MODE}"'" == "env" ]]; then
      python train_parakeet_emotion_tokens.py --mode env
    elif [[ "'"${MODE}"'" == "smoke" ]]; then
      python train_parakeet_emotion_tokens.py \
        --mode train \
        --output-dir "'"${RUN_ROOT}"'"/smoke \
        --max-steps 20 \
        --eval-steps 10 \
        --save-steps 10 \
        --max-train-samples 128 \
        --max-eval-samples 64 \
        --per-device-train-batch-size 1 \
        --per-device-eval-batch-size 1 \
        --gradient-accumulation-steps 4 \
        --no-push-to-hub
    elif [[ "'"${MODE}"'" == "smoke_mixture" ]]; then
      python train_parakeet_emotion_tokens.py \
        --mode train \
        --output-dir "'"${RUN_ROOT}"'"/smoke_mixture \
        --dataset-parquet "'"${EMOTION_MIXTURE_PARQUET}"'" \
        --max-steps 20 \
        --eval-steps 10 \
        --save-steps 10 \
        --max-train-samples 128 \
        --max-eval-samples 64 \
        --per-device-train-batch-size 1 \
        --per-device-eval-batch-size 1 \
        --gradient-accumulation-steps 4 \
        --no-push-to-hub
    elif [[ "'"${MODE}"'" == "smoke_mixture_suffix" ]]; then
      python train_parakeet_emotion_tokens.py \
        --mode train \
        --output-dir "'"${RUN_ROOT}"'"/mixture_suffix_smoke \
        --dataset-parquet "'"${EMOTION_MIXTURE_PARQUET}"'" \
        --emotion-token-position suffix \
        --freeze-mode full \
        --learning-rate 1e-5 \
        --max-steps 300 \
        --eval-steps 100 \
        --save-steps 100 \
        --max-train-samples 4096 \
        --max-eval-samples 256 \
        --per-device-train-batch-size 2 \
        --per-device-eval-batch-size 2 \
        --gradient-accumulation-steps 8 \
        --required-token-present-rate 0.02 \
        --no-push-to-hub
    elif [[ "'"${MODE}"'" == "smoke_mixture_suffix_repeat" ]]; then
      python train_parakeet_emotion_tokens.py \
        --mode train \
        --output-dir "'"${RUN_ROOT}"'"/mixture_suffix_repeat_smoke \
        --dataset-parquet "'"${EMOTION_MIXTURE_PARQUET}"'" \
        --emotion-token-position suffix \
        --emotion-token-repeat 5 \
        --freeze-mode full \
        --learning-rate 1e-5 \
        --max-steps 300 \
        --eval-steps 100 \
        --save-steps 100 \
        --max-train-samples 4096 \
        --max-eval-samples 256 \
        --per-device-train-batch-size 2 \
        --per-device-eval-batch-size 2 \
        --gradient-accumulation-steps 8 \
        --required-token-present-rate 0.02 \
        --no-push-to-hub
    elif [[ "'"${MODE}"'" == "smoke_concat_iu" ]]; then
      python train_parakeet_emotion_tokens.py \
        --mode train \
        --model-id "'"${RUN_ROOT}"'"/mixture_train \
        --output-dir "'"${RUN_ROOT}"'"/concat_iu_smoke \
        --dataset-parquet "'"${CONCAT_IU_PARQUET}"'" \
        --hub-model-id NathanRoll/parakeet-tdt-0.6b-v3-eng-sports-emotion-iu-boundary \
        --freeze-mode full \
        --learning-rate 5e-6 \
        --max-duration-seconds 60 \
        --max-steps 300 \
        --eval-steps 100 \
        --save-steps 100 \
        --max-train-samples 4096 \
        --max-eval-samples 256 \
        --per-device-train-batch-size 1 \
        --per-device-eval-batch-size 1 \
        --gradient-accumulation-steps 8 \
        --required-token-present-rate 0.02 \
        --no-push-to-hub
    elif [[ "'"${MODE}"'" == "calibrate" ]]; then
      python train_parakeet_emotion_tokens.py \
        --mode train \
        --output-dir "'"${RUN_ROOT}"'"/calibrate \
        --max-steps 200 \
        --eval-steps 50 \
        --save-steps 50 \
        --max-eval-samples 256 \
        --no-push-to-hub
    elif [[ "'"${MODE}"'" == "train" ]]; then
      python train_parakeet_emotion_tokens.py \
        --mode train \
        --output-dir "'"${RUN_ROOT}"'"/train \
        --dataset-parquet "'"${FULL_DATASET_PARQUET}"'" \
        --max-steps 1200 \
        --eval-steps 100 \
        --save-steps 100 \
        --push-to-hub
    elif [[ "'"${MODE}"'" == "train_mixture" ]]; then
      python train_parakeet_emotion_tokens.py \
        --mode train \
        --output-dir "'"${RUN_ROOT}"'"/mixture_train \
        --dataset-parquet "'"${EMOTION_MIXTURE_PARQUET}"'" \
        --hub-model-id NathanRoll/parakeet-tdt-0.6b-v3-eng-sports-emotion-iu-mixture \
        --freeze-mode full \
        --learning-rate 1e-5 \
        --max-steps 4000 \
        --eval-steps 250 \
        --save-steps 250 \
        --early-stopping-patience 8 \
        --push-to-hub
    elif [[ "'"${MODE}"'" == "train_mixture_suffix" ]]; then
      python train_parakeet_emotion_tokens.py \
        --mode train \
        --output-dir "'"${RUN_ROOT}"'"/mixture_suffix_train \
        --dataset-parquet "'"${EMOTION_MIXTURE_PARQUET}"'" \
        --emotion-token-position suffix \
        --hub-model-id NathanRoll/parakeet-tdt-0.6b-v3-eng-sports-emotion-iu-mixture-suffix \
        --freeze-mode full \
        --learning-rate 1e-5 \
        --max-steps 4000 \
        --eval-steps 250 \
        --save-steps 250 \
        --early-stopping-patience 8 \
        --required-token-present-rate 0.02 \
        --push-to-hub
    elif [[ "'"${MODE}"'" == "train_mixture_suffix_repeat" ]]; then
      python train_parakeet_emotion_tokens.py \
        --mode train \
        --output-dir "'"${RUN_ROOT}"'"/mixture_suffix_repeat_train \
        --dataset-parquet "'"${EMOTION_MIXTURE_PARQUET}"'" \
        --emotion-token-position suffix \
        --emotion-token-repeat 5 \
        --hub-model-id NathanRoll/parakeet-tdt-0.6b-v3-eng-sports-emotion-iu-mixture-suffix \
        --freeze-mode full \
        --learning-rate 1e-5 \
        --max-steps 4000 \
        --eval-steps 250 \
        --save-steps 250 \
        --early-stopping-patience 8 \
        --required-token-present-rate 0.02 \
        --push-to-hub
    elif [[ "'"${MODE}"'" == "train_concat_iu" ]]; then
      python train_parakeet_emotion_tokens.py \
        --mode train \
        --model-id "'"${RUN_ROOT}"'"/mixture_train \
        --output-dir "'"${RUN_ROOT}"'"/concat_iu_train \
        --dataset-parquet "'"${CONCAT_IU_PARQUET}"'" \
        --hub-model-id NathanRoll/parakeet-tdt-0.6b-v3-eng-sports-emotion-iu-boundary \
        --freeze-mode full \
        --learning-rate 5e-6 \
        --max-duration-seconds 60 \
        --max-steps 4000 \
        --eval-steps 250 \
        --save-steps 250 \
        --per-device-train-batch-size 1 \
        --per-device-eval-batch-size 1 \
        --gradient-accumulation-steps 8 \
        --early-stopping-patience 8 \
        --required-token-present-rate 0.02 \
        --push-to-hub
    elif [[ "'"${MODE}"'" == "eval" ]]; then
      python train_parakeet_emotion_tokens.py \
        --mode eval \
        --output-dir "'"${RUN_ROOT}"'"/train \
        --dataset-parquet "'"${FULL_DATASET_PARQUET}"'"
    elif [[ "'"${MODE}"'" == "eval_mixture" ]]; then
      python train_parakeet_emotion_tokens.py \
        --mode eval \
        --output-dir "'"${RUN_ROOT}"'"/mixture_train \
        --dataset-parquet "'"${EMOTION_MIXTURE_PARQUET}"'"
    elif [[ "'"${MODE}"'" == "eval_mixture_suffix" ]]; then
      python train_parakeet_emotion_tokens.py \
        --mode eval \
        --output-dir "'"${RUN_ROOT}"'"/mixture_suffix_train \
        --dataset-parquet "'"${EMOTION_MIXTURE_PARQUET}"'" \
        --emotion-token-position suffix
    elif [[ "'"${MODE}"'" == "eval_concat_iu" ]]; then
      python train_parakeet_emotion_tokens.py \
        --mode eval \
        --output-dir "'"${RUN_ROOT}"'"/concat_iu_train \
        --dataset-parquet "'"${CONCAT_IU_PARQUET}"'" \
        --max-duration-seconds 60
    else
      echo "unknown mode: '"${MODE}"'" >&2
      exit 2
    fi
  '
