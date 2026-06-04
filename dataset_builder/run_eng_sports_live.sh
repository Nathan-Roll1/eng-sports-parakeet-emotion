#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p runs/eng_sports_live

exec env PYTHONPATH="$ROOT${PYTHONPATH:+:${PYTHONPATH}}" python3 -m dataset_builder.eng_sports_collector \
  --allow-unverified-rights \
  --bucket "${GCS_BUCKET:-eng_sports}" \
  --duration "${SEGMENT_SECONDS:-120}" \
  --parallel "${PARALLEL_RECORDERS:-17}" \
  --continuous \
  --delete-after-upload \
  --retry-pending \
  --work-dir runs/eng_sports_live \
  --source-id talksport_uk \
  --source-id bbc_5live_uk \
  --source-id abc_grandstand_au \
  --source-id fox_sports_radio_la_us \
  --source-id cbs_sports_1053_us \
  --source-id sportsnet_590_ca \
  --source-id talksport2_uk \
  --source-id kzsu_stanford_us \
  --source-id fox_sports_1260_us \
  --source-id cbs_sports_1500_hawaii_us \
  --source-id espn_1017_team_us \
  --source-id knbr_1050_us \
  --source-id wtka_1050_us \
  --source-id abc_aleague_au \
  --source-id tsn_1050_toronto_ca \
  --source-id sen_1116_au \
  --source-id sports_byline_us
