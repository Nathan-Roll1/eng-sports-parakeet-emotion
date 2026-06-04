#!/bin/bash
#SBATCH --job-name=parakeet-concat-iu-data
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${HOME}/eng-sports-parakeet-emotion/parakeet_emotion_ft}"
VENV_DIR="${VENV_DIR:-${HOME}/.venvs/parakeet-emotion-data-host}"
EMOTION_MIXTURE_PARQUET="${EMOTION_MIXTURE_PARQUET:-${HOME}/eng-sports-parakeet-emotion/data/emotion_mixture_train}"
CONCAT_IU_PARQUET="${CONCAT_IU_PARQUET:-${HOME}/eng-sports-parakeet-emotion/data/emotion_mixture_concat_iu}"

mkdir -p ${HOME}/logs/parakeet \
  ${HOME}/cache/hf \
  ${HOME}/cache/hf/hub \
  ${HOME}/tmp \
  "${CONCAT_IU_PARQUET}"

export HF_HOME="${HF_HOME:-${HOME}/cache/hf}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HOME}/cache/hf/hub}"
export TMPDIR="${TMPDIR:-${HOME}/tmp}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

srun --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" \
  bash -lc '
    set -euo pipefail
    cd "'"${PROJECT_DIR}"'"

    python3 -m venv --system-site-packages "'"${VENV_DIR}"'"
    source "'"${VENV_DIR}"'"/bin/activate
    unset PIP_CONSTRAINT
    unset PIP_REQUIRE_VIRTUALENV
    export PIP_CONFIG_FILE=/dev/null
    INSTALL_MARKER="'"${VENV_DIR}"'/.data-requirements-v2"
    python -m pip install --upgrade pip
    if [[ ! -f "${INSTALL_MARKER}" ]]; then
      python -m pip install --upgrade "pyarrow>=15.0.0" "numpy>=1.23.0" "soundfile>=0.12.1" "librosa>=0.10.0"
      rm -f "'"${VENV_DIR}"'"/.data-requirements-*
      touch "${INSTALL_MARKER}"
    fi

    python - <<PY
import pyarrow, numpy, soundfile, librosa
print("data prep imports", pyarrow.__version__, numpy.__version__, soundfile.__version__, librosa.__version__)
PY

    python prepare_concat_iu_dataset.py \
      --input-parquet "'"${EMOTION_MIXTURE_PARQUET}"'" \
      --output-dir "'"${CONCAT_IU_PARQUET}"'" \
      --min-ius 2 \
      --max-ius 6 \
      --target-duration-seconds 28 \
      --max-duration-seconds 55 \
      --rows-per-shard 2500
  '
