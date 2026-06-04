#!/usr/bin/env bash
set -euo pipefail

USER_NAME="${USER_NAME:-${USER:-}}"
LOCAL_ROOT="${LOCAL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/${USER_NAME}/eng-sports-parakeet-emotion}"
ARCHIVE="/tmp/eng-sports-parakeet-emotion-ft.tgz"

command -v kubectl >/dev/null || {
  echo "missing kubectl on PATH" >&2
  exit 1
}

POD="$(kubectl get pod -n slurm -l "stanford/user=${USER_NAME}" -o jsonpath='{.items[0].metadata.name}')"
if [[ -z "${POD}" ]]; then
  echo "no login pod found for stanford/user=${USER_NAME}" >&2
  exit 1
fi

COPYFILE_DISABLE=1 tar -C "${LOCAL_ROOT}" -czf "${ARCHIVE}" parakeet_emotion_ft
kubectl cp "${ARCHIVE}" "slurm/${POD}:/home/${USER_NAME}/eng-sports-parakeet-emotion-ft.tgz" -c login
kubectl exec -n slurm "${POD}" -c login -- runuser -l "${USER_NAME}" -c "
  set -euo pipefail
  mkdir -p '${REMOTE_ROOT}'
  tar -C '${REMOTE_ROOT}' -xzf /home/${USER_NAME}/eng-sports-parakeet-emotion-ft.tgz
  find '${REMOTE_ROOT}/parakeet_emotion_ft' -name '._*' -delete
  chmod +x '${REMOTE_ROOT}/parakeet_emotion_ft/'*.sh \
    '${REMOTE_ROOT}/parakeet_emotion_ft/'*.sbatch \
    '${REMOTE_ROOT}/parakeet_emotion_ft/'*.py
  ls -la '${REMOTE_ROOT}/parakeet_emotion_ft'
"

echo "staged to ${REMOTE_ROOT} on ${POD}"
