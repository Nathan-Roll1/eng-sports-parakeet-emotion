# English Sports Parakeet Emotion IU Pipeline

Code for collecting English sports-radio audio, preparing intonation-unit datasets, annotating emotion labels, and training the public Parakeet IU boundary/emotion head:

https://huggingface.co/NathanRoll/parakeet-tdt-0.6b-v3-eng-sports-emotion-iu-boundary-head

This repository intentionally contains code only. It does not include raw audio, parquet datasets, annotation databases, cloud credentials, Hugging Face tokens, Slurm logs, or model weights.

## AI Assistance

AI coding assistants were used during this project to help research data sources, draft and refactor pipeline code, debug local and Slurm runs, prepare dataset-conversion scripts, and summarize training/evaluation results. Human operators reviewed commands before running them, controlled credentials and infrastructure access, made dataset/release decisions, and verified that this public repository excludes secrets, raw data, annotation databases, logs, and model artifacts.

## Contents

- `dataset_builder/`: live sports-radio source list, 24 kHz 24-bit PCM capture code, GCS upload support, Parakeet + PSST intonation-unit processing, and Hugging Face dataset packaging.
- `annotation/`: FastAPI/Railway IU emotion annotation UI with persistent SQLite storage.
- `scripts/`: Parakeet retranscription and final two-column emotion-token dataset preparation.
- `parakeet_emotion_ft/`: Parakeet TDT emotion-token experiments, emotion-head baselines, concat-IU experiments, and the final encoder-side IU boundary/emotion head trainer.

## Data And Rights

The collector records public internet radio streams into decoded WAV artifacts. Most internet streams are MP3/AAC/HLS, so decoded 24-bit WAV output is not proof of native 24-bit source fidelity. Stream rights are not verified here. Confirm rights before archival, redistribution, training, or commercial use.

## Secrets

Do not commit secrets. Required credentials are read from environment variables or platform-specific auth stores:

- `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` for Hugging Face.
- `GOOGLE_APPLICATION_CREDENTIALS` or `gcloud auth` for GCS.
- `IU_ANNOTATOR_ACCESS_KEY` for the annotation UI, if access control is desired.

## Typical Flow

1. Capture sports-radio segments:

```bash
python -m dataset_builder.eng_sports_collector \
  --allow-unverified-rights \
  --bucket eng_sports \
  --duration 300 \
  --parallel 2 \
  --continuous \
  --delete-after-upload \
  --work-dir runs/eng_sports_live
```

2. Process captured manifests into Parakeet + PSST intonation-unit datasets:

```bash
./dataset_builder/run_eng_sports_prosody.sh
```

3. Run the annotation UI:

```bash
export IU_HF_REPO_ID=NathanRoll/eng-sports-radio-psst-iu
export IU_ANNOTATOR_DATA_DIR=/data/iu_annotator
uvicorn annotation.app:app --host 0.0.0.0 --port 8000
```

4. Retranscribe IUs and merge manual emotion labels:

```bash
python scripts/eng_sports_retranscribe_ius.py \
  --input runs/eng_sports_processing/hf_iu_sampled_dataset/data/train-00000.parquet \
  --out-dir runs/eng_sports_processing/hf_iu_parakeet_labeled_dataset
```

5. Prepare mixture data and train the final IU boundary/emotion head:

```bash
cd parakeet_emotion_ft
python prepare_emotion_mixture_dataset.py --output-dir data/emotion_mixture_train
python train_iu_boundary_emotion_head.py \
  --model-dir runs/parakeet-emotion/mixture_train \
  --dataset-parquet data/emotion_mixture_train \
  --output-dir runs/parakeet-emotion/iu_boundary_head
```

On Slurm, use `parakeet_emotion_ft/run_iu_boundary_head.sbatch` after setting paths for your cluster.

## Final Model Contract

The final model package is an auxiliary encoder-side head:

1. Run the base Parakeet ASR model.
2. Use `boundary_head` to detect IU boundary frames.
3. Use `emotion_head` to classify each segment.
4. Insert `<|emotion|>` tokens at IU boundaries.

This avoids relying on the TDT decoder to emit rare special tokens directly.
