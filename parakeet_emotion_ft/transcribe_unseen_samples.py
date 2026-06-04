#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from train_parakeet_emotion_tokens import (
    canonicalize_emotion_text,
    compute_eval_metrics,
    decode_audio_array,
    decode_sequences,
    leading_emotion,
    load_and_split_dataset,
    normalize_asr_text,
    strip_leading_emotion,
    tensor_batch_to_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="runs/parakeet-emotion/train")
    parser.add_argument("--dataset-parquet", default="data/full_iu_parakeet_labeled_dataset.parquet")
    parser.add_argument("--output-dir", default="runs/parakeet-emotion/unseen_transcription")
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--samples-per-label", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=611)
    parser.add_argument("--max-duration-seconds", type=float, default=20.0)
    parser.add_argument("--min-duration-seconds", type=float, default=0.35)
    parser.add_argument("--dataset-id", default="NathanRoll/eng-sports-radio-psst-iu")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--emotion-column", default="emotion")
    parser.add_argument("--emotion-token-position", choices=["prefix", "suffix", "both"], default="prefix")
    parser.add_argument("--emotion-token-repeat", type=int, default=1)
    return parser.parse_args()


def sample_indices_by_label(dataset, samples_per_label: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    by_label: dict[str, list[int]] = {}
    for idx, text in enumerate(dataset["text"]):
        canonical_text = canonicalize_emotion_text(text)
        by_label.setdefault(leading_emotion(canonical_text) or "<|unknown|>", []).append(idx)

    selected: list[int] = []
    for label in sorted(by_label):
        indices = by_label[label]
        rng.shuffle(indices)
        selected.extend(indices[:samples_per_label])
    rng.shuffle(selected)
    return selected


def word_error_rate(ref: str, pred: str) -> float:
    ref_words = normalize_asr_text(ref).split()
    pred_words = normalize_asr_text(pred).split()
    if not ref_words:
        return 0.0 if not pred_words else 1.0
    prev = list(range(len(pred_words) + 1))
    for i, ref_word in enumerate(ref_words, start=1):
        cur = [i]
        for j, pred_word in enumerate(pred_words, start=1):
            cur.append(min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + (ref_word != pred_word),
            ))
        prev = cur
    return prev[-1] / len(ref_words)


def quality_bucket(wer: float) -> str:
    if wer <= 0.15:
        return "good"
    if wer <= 0.30:
        return "usable"
    return "needs_review"


def transcribe_rows(model, processor, rows: list[dict[str, Any]], batch_size: int) -> list[str]:
    import torch

    model.eval()
    predictions: list[str] = []
    sampling_rate = processor.feature_extractor.sampling_rate
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            batch = processor(
                audio=[decode_audio_array(row["audio"], sampling_rate) for row in batch_rows],
                sampling_rate=sampling_rate,
                padding=True,
                return_tensors="pt",
            )
            batch = tensor_batch_to_device(batch, model)
            generated = model.generate(**batch)
            predictions.extend(decode_sequences(processor, generated))
    return predictions


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForTDT, AutoProcessor
    import torch

    processor = AutoProcessor.from_pretrained(args.model_dir, token=True)
    model = AutoModelForTDT.from_pretrained(args.model_dir, token=True)
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = model.to(device="cuda", dtype=dtype)

    split_args = argparse.Namespace(
        dataset_id=args.dataset_id,
        dataset_parquet=args.dataset_parquet,
        text_column=args.text_column,
        emotion_column=args.emotion_column,
        min_duration_seconds=args.min_duration_seconds,
        max_duration_seconds=args.max_duration_seconds,
        train_frac=0.70,
        val_frac=0.15,
        test_frac=0.15,
        seed=153,
        oversample_minority=False,
        max_train_samples=0,
        oversample_neutral_fraction=0.35,
        oversample_max_repeat=3,
        emotion_token_position=args.emotion_token_position,
        emotion_token_repeat=args.emotion_token_repeat,
    )
    splits, split_info = load_and_split_dataset(split_args, processor.feature_extractor.sampling_rate)
    dataset = splits[args.split]
    selected = sample_indices_by_label(dataset, args.samples_per_label, args.seed)
    rows = [dataset[int(idx)] for idx in selected]
    predictions = transcribe_rows(model, processor, rows, args.batch_size)
    references = [canonicalize_emotion_text(row["text"]) for row in rows]
    predictions = [canonicalize_emotion_text(pred) for pred in predictions]

    examples = []
    for idx, row, ref, pred in zip(selected, rows, references, predictions):
        wer = word_error_rate(ref, pred)
        examples.append({
            "split": args.split,
            "split_index": int(idx),
            "source_id": row.get("source_id"),
            "iu_id": row.get("iu_id"),
            "reference": ref,
            "prediction": pred,
            "reference_text": strip_leading_emotion(ref),
            "prediction_text": strip_leading_emotion(pred),
            "reference_label": leading_emotion(ref),
            "prediction_label": leading_emotion(pred),
            "wer_without_emotion": wer,
            "quality_bucket": quality_bucket(wer),
        })

    metrics = compute_eval_metrics(references, predictions)
    bucket_counts: dict[str, int] = {}
    for example in examples:
        bucket_counts[example["quality_bucket"]] = bucket_counts.get(example["quality_bucket"], 0) + 1
    summary = {
        "model_dir": args.model_dir,
        "dataset_parquet": args.dataset_parquet,
        "split": args.split,
        "num_examples": len(examples),
        "samples_per_label": args.samples_per_label,
        "bucket_counts": bucket_counts,
        "metrics": {k: v for k, v in metrics.items() if k != "per_class"},
        "split_counts": split_info["counts"],
    }

    (output_dir / "unseen_transcriptions.json").write_text(
        json.dumps({"summary": summary, "examples": examples}, indent=2) + "\n"
    )
    report_lines = [
        "# Unseen Parakeet Transcription Spot Check",
        "",
        json.dumps(summary, indent=2),
        "",
    ]
    for i, example in enumerate(examples, start=1):
        report_lines.extend([
            f"## Example {i}: {example['reference_label']} / {example['quality_bucket']}",
            f"WER without emotion: {example['wer_without_emotion']:.3f}",
            f"Source: {example.get('source_id') or ''}",
            f"Reference: {example['reference_text']}",
            f"Prediction: {example['prediction_text']}",
            "",
        ])
    (output_dir / "unseen_transcriptions.md").write_text("\n".join(report_lines))
    print(json.dumps({"summary": summary, "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
