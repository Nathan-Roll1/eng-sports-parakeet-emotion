#!/usr/bin/env bash
set -euo pipefail

USER_NAME="${USER_NAME:-${USER:-}}"

command -v kubectl >/dev/null || {
  echo "missing kubectl on PATH" >&2
  exit 1
}

kubectl auth whoami
kubectl config current-context

POD="$(kubectl get pod -n slurm -l "stanford/user=${USER_NAME}" -o jsonpath='{.items[0].metadata.name}')"
if [[ -z "${POD}" ]]; then
  echo "no login pod found for stanford/user=${USER_NAME}" >&2
  exit 1
fi

echo "login pod: ${POD}"
kubectl exec -n slurm "${POD}" -c login -- runuser -l "${USER_NAME}" -c 'whoami; pwd; command -v sbatch; command -v squeue; python3 --version'
