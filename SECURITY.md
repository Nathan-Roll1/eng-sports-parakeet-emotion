# Security

This repository should contain source code only.

Before pushing changes, scan for credentials and generated data:

```bash
rg -n --hidden -S "(gho_|hf_|sk-|BEGIN .*PRIVATE KEY|DATABASE_URL|API_KEY|SECRET|TOKEN)" .
find . -type f \( -name "*.wav" -o -name "*.flac" -o -name "*.parquet" -o -name "*.pt" -o -name ".env*" \)
```

Use environment variables or local auth stores for credentials. Do not commit tokens, service-account JSON, annotation databases, raw audio, processed datasets, logs, or model weights.
