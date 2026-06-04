#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ -f ".venv_eng_sports/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv_eng_sports/bin/activate"
fi

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"

python -m dataset_builder.eng_sports_prosody_dataset process \
  --manifest-root "${MANIFEST_ROOT:-runs/eng_sports_live/segments}" \
  --output-dir "${OUTPUT_DIR:-runs/eng_sports_processing}" \
  --parakeet-model "${PARAKEET_MODEL:-nvidia/parakeet-tdt-0.6b-v3}" \
  --psst-model "${PSST_MODEL:-NathanRoll/psst-medium-en}" \
  --device "${DEVICE:-auto}" \
  --local-first \
  --gcs-timeout "${GCS_TIMEOUT:-180}" \
  --package \
  --package-every "${PACKAGE_EVERY:-100}" \
  --max-rows-per-shard "${MAX_ROWS_PER_SHARD:-200}" \
  ${LIMIT:+--limit "$LIMIT"} \
  ${UPLOAD_REPO:+--upload-repo "$UPLOAD_REPO"}
