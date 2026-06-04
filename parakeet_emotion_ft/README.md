# Parakeet Emotion Token Fine-Tune

Fine-tunes `nvidia/parakeet-tdt-0.6b-v3` on the current intonation-unit dataset:

- Dataset: `NathanRoll/eng-sports-radio-psst-iu`
- Full train source: set with `FULL_DATASET_PARQUET` or `--dataset-parquet`
- Shape: `audio`, `text`; the full parquet also carries metadata such as `emotion`, `source_id`, and `iu_id`
- Text contract: one leading emotion token followed by transcript, e.g. `<|joy|> What a finish.`
- Tokens: `<|neutral|>`, `<|low_quality|>`, `<|joy|>`, `<|surprise|>`, `<|sadness|>`, `<|anger|>`, `<|disgust|>`, `<|fear|>`

The default run uses Hugging Face Transformers from source because Parakeet TDT support is newer than the local installed release. It adds `<blank>` at the configured Parakeet blank id, appends the eight emotion/quality tokens, resizes the TDT decoder embedding and joint token/duration head without disturbing duration logits, and updates generation suppression ids.

## Cluster Usage

From your own login pod:

```bash
cd parakeet_emotion_ft
sbatch run_parakeet_ft.sbatch env
sbatch run_parakeet_ft.sbatch smoke
sbatch run_parakeet_ft.sbatch train
```

The `train` mode defaults to one H100, bf16, encoder frozen, decoder/joint/projector trainable, deterministic 70/15/15 emotion-stratified split over the full private parquet, and private Hub push to:

```text
NathanRoll/parakeet-tdt-0.6b-v3-eng-sports-emotion-iu
```

Required on the cluster:

```bash
export HF_TOKEN="<your-hugging-face-token>"
```

Optional:

```bash
export WANDB_API_KEY="<your-wandb-token>"
export WANDB_PROJECT=parakeet-sports-emotion
```

If you use the same Kubernetes login-pod staging pattern, set `USER_NAME` and `REMOTE_ROOT` first:

```bash
export USER_NAME="$USER"
export REMOTE_ROOT="/home/$USER/eng-sports-parakeet-emotion"
./parakeet_emotion_ft/cluster_preflight.sh
./parakeet_emotion_ft/stage_to_cluster.sh
```

## Local Validation

This laptop has `kubectl`, `om`, and a kubeconfig for `amp-internal`. The local Python files pass syntax checks.
