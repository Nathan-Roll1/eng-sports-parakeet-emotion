# English Sports Radio Collection

This collector records live English sports-radio streams into closed WAV
segments at 24 kHz, mono, 24-bit PCM and uploads each segment plus a JSON
manifest to `gs://eng_sports`.

Important: most internet radio streams are MP3/AAC/HLS. The 24-bit WAV files
are decoded capture artifacts; they do not prove native 24-bit source fidelity.
The bundled streams are public, technically reachable sources, but their audio
rights are unverified. Confirm licensing before long-term archival, model
training, redistribution, or commercial use.

## Commands

Probe/select sources:

```bash
python3 -m dataset_builder.eng_sports_collector --sources dataset_builder/eng_sports_sources.json
```

Run a bounded smoke capture without upload:

```bash
python3 -m dataset_builder.eng_sports_collector \
  --allow-unverified-rights \
  --skip-upload \
  --duration 15 \
  --parallel 2 \
  --max-segments 2 \
  --work-dir runs/eng_sports_smoke
```

Run continuous collection to GCS:

```bash
python3 -m dataset_builder.eng_sports_collector \
  --allow-unverified-rights \
  --bucket "${GCS_BUCKET:-eng_sports}" \
  --duration 300 \
  --parallel 2 \
  --continuous \
  --delete-after-upload \
  --work-dir runs/eng_sports_live
```

The current live collector is running in a detached `screen` session:

```bash
screen -ls
tail -f runs/eng_sports_live/screen.log
```

Stop it cleanly:

```bash
screen -S eng_sports_live -X quit
```

Check collector output:

```bash
find runs/eng_sports_live -type f | tail
gcloud storage ls --long 'gs://eng_sports/**'
```
