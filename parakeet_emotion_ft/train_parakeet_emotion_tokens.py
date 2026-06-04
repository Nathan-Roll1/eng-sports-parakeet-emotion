#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import re
import string
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"
DATASET_ID = "NathanRoll/eng-sports-radio-psst-iu"
DEFAULT_HUB_MODEL_ID = "NathanRoll/parakeet-tdt-0.6b-v3-eng-sports-emotion-iu"
EMOTION_TOKENS = [
    "<|neutral|>",
    "<|low_quality|>",
    "<|joy|>",
    "<|surprise|>",
    "<|sadness|>",
    "<|anger|>",
    "<|disgust|>",
    "<|fear|>",
]
EMOTION_PATTERN = re.compile(r"^\s*(<\|(?:neutral|low_quality|joy|surprise|sadness|anger|disgust|fear)\|>)\s*")
EMOTION_ANY_PATTERN = re.compile(r"<\|(?:neutral|low_quality|joy|surprise|sadness|anger|disgust|fear)\|>")
FULL_DATASET_PARQUET = "data/full_iu_parakeet_labeled_dataset.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["env", "train", "eval"], default="train")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--dataset-parquet", default=None)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--emotion-column", default="emotion")
    parser.add_argument("--output-dir", default="runs/parakeet-emotion/train")
    parser.add_argument("--hub-model-id", default=DEFAULT_HUB_MODEL_ID)
    parser.add_argument("--push-to-hub", dest="push_to_hub", action="store_true")
    parser.add_argument("--no-push-to-hub", dest="push_to_hub", action="store_false")
    parser.set_defaults(push_to_hub=False)
    parser.add_argument("--hub-private", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=153)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=512)
    parser.add_argument("--max-duration-seconds", type=float, default=20.0)
    parser.add_argument("--min-duration-seconds", type=float, default=0.35)
    parser.add_argument("--oversample-minority", action="store_true", default=True)
    parser.add_argument("--no-oversample-minority", dest="oversample_minority", action="store_false")
    parser.add_argument("--oversample-max-repeat", type=int, default=3)
    parser.add_argument("--oversample-neutral-fraction", type=float, default=0.35)
    parser.add_argument("--emotion-token-position", choices=["prefix", "suffix", "both"], default="prefix")
    parser.add_argument("--emotion-token-repeat", type=int, default=1)
    parser.add_argument("--freeze-mode", choices=["decoder_joint", "full", "encoder_last4"], default="decoder_joint")
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--required-token-present-rate", type=float, default=0.0)
    return parser.parse_args()


def require_runtime_imports() -> dict[str, Any]:
    import torch
    from datasets import Audio, Dataset, DatasetDict, load_dataset
    from transformers import (
        AutoModelForTDT,
        AutoProcessor,
        EarlyStoppingCallback,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    return {
        "torch": torch,
        "Audio": Audio,
        "Dataset": Dataset,
        "DatasetDict": DatasetDict,
        "load_dataset": load_dataset,
        "AutoModelForTDT": AutoModelForTDT,
        "AutoProcessor": AutoProcessor,
        "EarlyStoppingCallback": EarlyStoppingCallback,
        "Trainer": Trainer,
        "TrainerCallback": TrainerCallback,
        "TrainingArguments": TrainingArguments,
    }


def leading_emotion(text: str) -> str | None:
    match = EMOTION_PATTERN.match(text or "")
    return match.group(1) if match else None


def any_emotion(text: str) -> str | None:
    match = EMOTION_ANY_PATTERN.search(text or "")
    return match.group(0) if match else None


def emotion_tokens_in_text(text: str) -> list[str]:
    return EMOTION_ANY_PATTERN.findall(text or "")


def primary_emotion(text: str) -> str | None:
    return leading_emotion(text) or any_emotion(text)


def strip_leading_emotion(text: str) -> str:
    return EMOTION_PATTERN.sub("", text or "").strip()


def strip_all_emotion_tokens(text: str) -> str:
    return re.sub(r"\s+", " ", EMOTION_ANY_PATTERN.sub(" ", text or "")).strip()


def canonicalize_emotion_text(text: str) -> str:
    token = any_emotion(text)
    body = strip_all_emotion_tokens(text)
    return f"{token} {body}".strip() if token else body


def format_emotion_target(text: str, position: str, repeat: int = 1) -> str:
    token = primary_emotion(text)
    if token not in EMOTION_TOKENS:
        return text
    body = strip_all_emotion_tokens(text)
    repeated = " ".join([token] * max(1, int(repeat)))
    if position == "prefix":
        return f"{repeated} {body}".strip()
    if position == "suffix":
        return f"{body} {repeated}".strip()
    if position == "both":
        return f"{token} {body} {repeated}".strip()
    raise ValueError(f"unknown emotion token position: {position}")


def token_from_emotion_label(label: str | None) -> str | None:
    normalized = (label or "").strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"bad_quality", "bad", "poor_quality"}:
        normalized = "low_quality"
    token = f"<|{normalized}|>"
    return token if token in EMOTION_TOKENS else None


def normalize_asr_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = strip_all_emotion_tokens(text)
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def duration_seconds(audio: dict[str, Any]) -> float:
    array = audio.get("array")
    rate = audio.get("sampling_rate") or 16000
    if array is not None:
        return float(len(array)) / float(rate)
    try:
        import soundfile as sf

        if audio.get("bytes") is not None:
            info = sf.info(io.BytesIO(audio["bytes"]))
        elif audio.get("path"):
            info = sf.info(audio["path"])
        else:
            return 0.0
        return float(info.frames) / float(info.samplerate)
    except Exception:
        return 0.0


def row_duration_seconds(row: dict[str, Any]) -> float:
    value = row.get("duration_seconds")
    if value is not None:
        try:
            return float(value)
        except (TypeError, ValueError):
            pass
    return duration_seconds(row["audio"])


def decode_audio_array(audio: dict[str, Any], target_sampling_rate: int):
    import numpy as np
    import soundfile as sf

    array = audio.get("array")
    source_sampling_rate = int(audio.get("sampling_rate") or target_sampling_rate)
    if array is not None:
        samples = np.asarray(array, dtype=np.float32)
    elif audio.get("bytes") is not None:
        samples, source_sampling_rate = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=False)
    elif audio.get("path"):
        samples, source_sampling_rate = sf.read(audio["path"], dtype="float32", always_2d=False)
    else:
        raise ValueError("audio row has no array, bytes, or path")

    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if int(source_sampling_rate) != int(target_sampling_rate):
        import librosa

        samples = librosa.resample(samples, orig_sr=source_sampling_rate, target_sr=target_sampling_rate)
    return np.asarray(samples, dtype=np.float32)


def load_and_split_dataset(args: argparse.Namespace, sampling_rate: int):
    rt = require_runtime_imports()
    source_info: dict[str, Any] = {"dataset_id": args.dataset_id}
    if args.dataset_parquet:
        base, parquet_info = load_parquet_dataset(args, rt)
        source_info.update(parquet_info)
    else:
        ds = rt["load_dataset"](args.dataset_id, token=True)
        base = ds["train"] if "train" in ds else next(iter(ds.values()))
    base = base.cast_column("audio", rt["Audio"](sampling_rate=sampling_rate, decode=False))

    def keep(row: dict[str, Any]) -> bool:
        text = (row.get("text") or "").strip()
        if primary_emotion(text) not in EMOTION_TOKENS:
            return False
        if not strip_all_emotion_tokens(text):
            return False
        dur = row_duration_seconds(row)
        return args.min_duration_seconds <= dur < args.max_duration_seconds

    if source_info.get("duration_prefiltered"):
        keep_indices = [
            idx for idx, text in enumerate(base["text"])
            if primary_emotion(text) in EMOTION_TOKENS and strip_all_emotion_tokens(text)
        ]
    else:
        keep_indices = [
            idx for idx in range(len(base))
            if keep(base[idx])
        ]
    base = base.select(keep_indices)
    labels = [primary_emotion(t) for t in base["text"]]
    train_idx, val_idx, test_idx = stratified_indices(labels, args)

    split_info = {
        "source": source_info,
        "seed": args.seed,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
        "emotion_token_position": args.emotion_token_position,
        "emotion_token_repeat": args.emotion_token_repeat,
        "counts": {
            "all": label_counts(labels),
            "train": label_counts([labels[i] for i in train_idx]),
            "validation": label_counts([labels[i] for i in val_idx]),
            "test": label_counts([labels[i] for i in test_idx]),
        },
        "indices": {"train": train_idx, "validation": val_idx, "test": test_idx},
    }

    train_ds = base.select(train_idx)
    val_ds = base.select(val_idx)
    test_ds = base.select(test_idx)

    if args.emotion_token_position != "prefix" or args.emotion_token_repeat != 1:
        def rewrite_target(row: dict[str, Any]) -> dict[str, str]:
            return {"text": format_emotion_target(
                row["text"],
                args.emotion_token_position,
                args.emotion_token_repeat,
            )}

        train_ds = train_ds.map(rewrite_target, desc=f"Moving emotion token to {args.emotion_token_position}")
        val_ds = val_ds.map(rewrite_target, desc=f"Moving emotion token to {args.emotion_token_position}")
        test_ds = test_ds.map(rewrite_target, desc=f"Moving emotion token to {args.emotion_token_position}")

    if args.oversample_minority:
        train_ds = oversample_train_dataset(train_ds, args)

    if args.max_train_samples and args.max_train_samples > 0:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))

    return rt["DatasetDict"]({"train": train_ds, "validation": val_ds, "test": test_ds}), split_info


def load_parquet_dataset(args: argparse.Namespace, rt: dict[str, Any]):
    from collections import Counter

    import pyarrow.parquet as pq

    path = Path(args.dataset_parquet).expanduser()
    files = parquet_files_from_path(path)
    first_columns = pq.ParquetFile(files[0]).schema_arrow.names
    if args.emotion_column not in first_columns:
        return load_tokenized_parquet_dataset(files, args, rt)

    selected: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    label_counts_seen: Counter[str] = Counter()
    input_rows = 0
    raw_columns: list[str] = first_columns

    for file_path in files:
        parquet_file = pq.ParquetFile(file_path)
        for batch in parquet_file.iter_batches(batch_size=512):
            source_rows = batch.to_pylist()
            input_rows += len(source_rows)
            for row in source_rows:
                audio = row.get("audio")
                if not isinstance(audio, dict) or not (audio.get("bytes") or audio.get("path")):
                    skipped["missing_audio"] += 1
                    continue
                raw_text = str(row.get(args.text_column) or "").strip()
                token = token_from_emotion_label(row.get(args.emotion_column))
                if token is None:
                    token = primary_emotion(raw_text)
                if token is None:
                    skipped["missing_or_unknown_emotion"] += 1
                    continue
                text = strip_all_emotion_tokens(raw_text).strip()
                if not text:
                    skipped["empty_text"] += 1
                    continue
                selected.append({
                    "audio": audio,
                    "text": f"{token} {text}",
                    "duration_seconds": row.get("duration_seconds"),
                    "source_id": row.get("source_id"),
                    "iu_id": row.get("iu_id"),
                })
                label_counts_seen[token] += 1

    info = {
        "dataset_parquet": str(path),
        "parquet_files": [str(file_path) for file_path in files],
        "input_rows": input_rows,
        "converted_rows": len(selected),
        "raw_columns": raw_columns,
        "skipped_before_duration_filter": dict(sorted(skipped.items())),
        "converted_label_counts_before_duration_filter": dict(sorted(label_counts_seen.items())),
    }
    return rt["Dataset"].from_list(selected), info


def parquet_files_from_path(path: Path) -> list[Path]:
    if path.is_dir():
        files = sorted(path.glob("*.parquet"))
        if not files:
            files = sorted((path / "data").glob("*.parquet"))
    elif any(char in str(path) for char in "*?[]"):
        import glob

        files = [Path(item) for item in sorted(glob.glob(str(path)))]
    else:
        files = [path]
    if not files:
        raise FileNotFoundError(f"no parquet files found at {path}")
    return files


def load_tokenized_parquet_dataset(files: list[Path], args: argparse.Namespace, rt: dict[str, Any]):
    from collections import Counter

    data_files = [str(file_path) for file_path in files]
    ds = rt["load_dataset"]("parquet", data_files=data_files, split="train")
    selected_indices: list[int] = []
    skipped: Counter[str] = Counter()
    label_counts_seen: Counter[str] = Counter()

    for idx, raw_text in enumerate(ds[args.text_column]):
        text = str(raw_text or "").strip()
        token = primary_emotion(text)
        if token is None:
            skipped["missing_or_unknown_emotion"] += 1
            continue
        if not strip_all_emotion_tokens(text):
            skipped["empty_text"] += 1
            continue
        selected_indices.append(idx)
        label_counts_seen[token] += 1

    if len(selected_indices) != len(ds):
        ds = ds.select(selected_indices)

    info = {
        "dataset_parquet": str(files[0].parent if len(files) > 1 else files[0]),
        "parquet_files": data_files,
        "input_rows": len(selected_indices) + sum(skipped.values()),
        "converted_rows": len(selected_indices),
        "raw_columns": ds.column_names,
        "duration_prefiltered": True,
        "skipped_before_duration_filter": dict(sorted(skipped.items())),
        "converted_label_counts_before_duration_filter": dict(sorted(label_counts_seen.items())),
    }
    return ds, info


def label_counts(labels: list[str | None]) -> dict[str, int]:
    return {tok: sum(1 for item in labels if item == tok) for tok in EMOTION_TOKENS}


def stratified_indices(labels: list[str | None], args: argparse.Namespace) -> tuple[list[int], list[int], list[int]]:
    if not math.isclose(args.train_frac + args.val_frac + args.test_frac, 1.0, abs_tol=1e-6):
        raise ValueError("train/val/test fractions must sum to 1.0")
    rng = random.Random(args.seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    for token in EMOTION_TOKENS:
        group = [i for i, label in enumerate(labels) if label == token]
        rng.shuffle(group)
        n = len(group)
        n_train = int(round(n * args.train_frac))
        n_val = int(round(n * args.val_frac))
        if n_train + n_val > n:
            n_val = max(0, n - n_train)
        train_idx.extend(group[:n_train])
        val_idx.extend(group[n_train : n_train + n_val])
        test_idx.extend(group[n_train + n_val :])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def oversample_train_dataset(train_ds, args: argparse.Namespace):
    labels = [primary_emotion(t) for t in train_ds["text"]]
    counts = label_counts(labels)
    neutral = max(1, counts["<|neutral|>"])
    target = int(neutral * args.oversample_neutral_fraction)
    rng = random.Random(args.seed + 17)
    selected: list[int] = list(range(len(train_ds)))
    for token in EMOTION_TOKENS:
        if token == "<|neutral|>":
            continue
        group = [i for i, label in enumerate(labels) if label == token]
        if not group:
            continue
        desired = min(target, len(group) * args.oversample_max_repeat)
        extra = max(0, desired - len(group))
        selected.extend(rng.choice(group) for _ in range(extra))
    rng.shuffle(selected)
    return train_ds.select(selected)


def add_special_tokens_without_replacing(tokenizer, tokens: list[str]) -> int:
    missing = []
    unk = getattr(tokenizer, "unk_token_id", None)
    for token in tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or (unk is not None and token_id == unk and token != tokenizer.unk_token):
            missing.append(token)
    if not missing:
        return 0
    try:
        return tokenizer.add_special_tokens(
            {"additional_special_tokens": missing},
            replace_extra_special_tokens=False,
        )
    except TypeError:
        try:
            return tokenizer.add_special_tokens(
                {"additional_special_tokens": missing},
                replace_additional_special_tokens=False,
            )
        except TypeError:
            return tokenizer.add_special_tokens({"additional_special_tokens": missing})


def setup_processor_and_model(args: argparse.Namespace):
    rt = require_runtime_imports()
    torch = rt["torch"]
    processor = rt["AutoProcessor"].from_pretrained(args.model_id, token=True)

    model_kwargs: dict[str, Any] = {"token": True}
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        model_kwargs["dtype"] = torch.bfloat16
    try:
        model = rt["AutoModelForTDT"].from_pretrained(args.model_id, **model_kwargs)
    except TypeError:
        if "dtype" in model_kwargs:
            model_kwargs["torch_dtype"] = model_kwargs.pop("dtype")
        model = rt["AutoModelForTDT"].from_pretrained(args.model_id, **model_kwargs)

    tokenizer = processor.tokenizer
    old_model_vocab_size = int(model.config.vocab_size)
    blank_token = getattr(processor, "blank_token", None) or "<blank>"

    pre_add_piece_ids = {
        tok: tokenizer(emotion_token_seed_text(tok), add_special_tokens=False).input_ids
        for tok in EMOTION_TOKENS
    }

    blank_id_before = tokenizer.convert_tokens_to_ids(blank_token)
    if blank_id_before in (None, getattr(tokenizer, "unk_token_id", None)):
        if len(tokenizer) != int(model.config.blank_token_id):
            raise RuntimeError(
                f"Cannot add {blank_token} at configured blank id {model.config.blank_token_id}; "
                f"tokenizer length is {len(tokenizer)}."
            )
        add_special_tokens_without_replacing(tokenizer, [blank_token])

    blank_id = tokenizer.convert_tokens_to_ids(blank_token)
    if blank_id != int(model.config.blank_token_id):
        raise RuntimeError(f"blank token id mismatch: tokenizer={blank_id}, model={model.config.blank_token_id}")
    processor.blank_token = blank_token
    processor.blank_token_id = blank_id

    add_special_tokens_without_replacing(tokenizer, EMOTION_TOKENS)
    resize_parakeet_tdt_vocab(model, processor, old_model_vocab_size, pre_add_piece_ids)
    apply_freeze_mode(model, args.freeze_mode)
    validate_token_setup(model, processor)
    return processor, model


def emotion_token_seed_text(token: str) -> str:
    return token.removeprefix("<|").removesuffix("|>")


def resize_parakeet_tdt_vocab(model, processor, old_vocab_size: int, piece_ids: dict[str, list[int]]) -> None:
    import torch

    tokenizer = processor.tokenizer
    new_vocab_size = len(tokenizer)
    num_durations = len(model.config.durations)

    if new_vocab_size == old_vocab_size:
        sync_generation_config(model, processor)
        return
    if new_vocab_size < old_vocab_size:
        raise RuntimeError(f"new tokenizer vocab {new_vocab_size} is smaller than model vocab {old_vocab_size}")

    old_embedding = model.decoder.embedding
    old_head = model.joint.head
    hidden = int(model.config.decoder_hidden_size)
    device = old_embedding.weight.device
    dtype = old_embedding.weight.dtype

    new_embedding = torch.nn.Embedding(new_vocab_size, hidden, device=device, dtype=dtype)
    new_head = torch.nn.Linear(hidden, new_vocab_size + num_durations, device=device, dtype=dtype)

    with torch.no_grad():
        new_embedding.weight[:old_vocab_size].copy_(old_embedding.weight[:old_vocab_size])
        new_head.weight[:old_vocab_size].copy_(old_head.weight[:old_vocab_size])
        new_head.bias[:old_vocab_size].copy_(old_head.bias[:old_vocab_size])

        old_duration_slice = slice(old_vocab_size, old_vocab_size + num_durations)
        new_duration_slice = slice(new_vocab_size, new_vocab_size + num_durations)
        new_head.weight[new_duration_slice].copy_(old_head.weight[old_duration_slice])
        new_head.bias[new_duration_slice].copy_(old_head.bias[old_duration_slice])

        old_mean_embedding = old_embedding.weight[:old_vocab_size].mean(dim=0)
        old_mean_head = old_head.weight[:old_vocab_size].mean(dim=0)
        old_mean_bias = old_head.bias[:old_vocab_size].mean()

        for token in EMOTION_TOKENS:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id < old_vocab_size:
                continue
            valid_piece_ids = [
                idx for idx in piece_ids.get(token, [])
                if 0 <= int(idx) < old_vocab_size and int(idx) != tokenizer.unk_token_id
            ]
            if valid_piece_ids:
                ids = torch.tensor(valid_piece_ids, device=device, dtype=torch.long)
                new_embedding.weight[token_id].copy_(old_embedding.weight.index_select(0, ids).mean(dim=0))
                new_head.weight[token_id].copy_(old_head.weight.index_select(0, ids).mean(dim=0))
                new_head.bias[token_id].copy_(old_head.bias.index_select(0, ids).mean())
            else:
                new_embedding.weight[token_id].copy_(old_mean_embedding)
                new_head.weight[token_id].copy_(old_mean_head)
                new_head.bias[token_id].copy_(old_mean_bias)

        start = old_vocab_size
        end = new_vocab_size
        for row in range(start, end):
            token = tokenizer.convert_ids_to_tokens(row)
            if token in EMOTION_TOKENS:
                continue
            torch.nn.init.normal_(new_embedding.weight[row], mean=0.0, std=model.config.initializer_range)
            torch.nn.init.normal_(new_head.weight[row], mean=0.0, std=model.config.initializer_range)
            new_head.bias[row].zero_()

    model.decoder.embedding = new_embedding
    model.joint.head = new_head
    model.config.vocab_size = new_vocab_size
    model.joint.vocab_size = new_vocab_size
    sync_generation_config(model, processor)


def sync_generation_config(model, processor) -> None:
    tokenizer = processor.tokenizer
    blank_id = processor.blank_token_id
    model.config.blank_token_id = blank_id
    model.config.pad_token_id = tokenizer.pad_token_id
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.decoder_start_token_id = blank_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.suppress_tokens = list(
            range(int(model.config.vocab_size), int(model.config.vocab_size) + len(model.config.durations))
        )


def validate_token_setup(model, processor) -> None:
    tokenizer = processor.tokenizer
    assert processor.blank_token_id == model.config.blank_token_id
    assert model.decoder.embedding.num_embeddings == model.config.vocab_size
    assert model.joint.head.out_features == model.config.vocab_size + len(model.config.durations)
    expected_suppressed = list(range(model.config.vocab_size, model.config.vocab_size + len(model.config.durations)))
    if getattr(model, "generation_config", None) is not None:
        assert list(model.generation_config.suppress_tokens) == expected_suppressed
    for token in EMOTION_TOKENS:
        ids = tokenizer(token, add_special_tokens=False).input_ids
        if len(ids) != 1:
            raise RuntimeError(f"{token} is not atomic after tokenizer setup: {ids}")
        if ids[0] == processor.blank_token_id:
            raise RuntimeError(f"{token} collided with blank token id {processor.blank_token_id}")


def apply_freeze_mode(model, freeze_mode: str) -> None:
    if freeze_mode == "full":
        for param in model.parameters():
            param.requires_grad = True
        return
    if freeze_mode == "decoder_joint":
        for param in model.parameters():
            param.requires_grad = False
        for module_name in ["encoder_projector", "decoder", "joint"]:
            module = getattr(model, module_name)
            for param in module.parameters():
                param.requires_grad = True
        return
    if freeze_mode == "encoder_last4":
        for param in model.parameters():
            param.requires_grad = False
        for module_name in ["encoder_projector", "decoder", "joint"]:
            module = getattr(model, module_name)
            for param in module.parameters():
                param.requires_grad = True
        layers = getattr(getattr(model, "encoder", None), "layers", [])
        for layer in list(layers)[-4:]:
            for param in layer.parameters():
                param.requires_grad = True
        return
    raise ValueError(f"unknown freeze mode: {freeze_mode}")


def trainable_parameter_summary(model) -> dict[str, float | int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "trainable_fraction": trainable / total if total else 0.0,
    }


@dataclass
class ParakeetDataCollator:
    processor: Any
    sampling_rate: int

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        audio = [decode_audio_array(row["audio"], self.sampling_rate) for row in rows]
        text = [row["text"] for row in rows]
        batch = self.processor(
            audio=audio,
            text=text,
            sampling_rate=self.sampling_rate,
            padding=True,
            return_tensors="pt",
        )
        for key, value in list(batch.items()):
            if isinstance(value, torch.Tensor) and not value.is_floating_point() and value.dtype != torch.bool:
                batch[key] = value.long()
        return batch


def tensor_batch_to_device(batch: dict[str, Any], model) -> dict[str, Any]:
    import torch

    device = next(model.parameters()).device
    dtype = next((p.dtype for p in model.parameters() if p.is_floating_point()), torch.float32)
    out = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            if value.is_floating_point():
                out[key] = value.to(device=device, dtype=dtype)
            else:
                out[key] = value.to(device=device)
        else:
            out[key] = value
    return out


def decode_sequences(processor, sequences) -> list[str]:
    import torch

    if hasattr(sequences, "sequences"):
        sequences = sequences.sequences
    if isinstance(sequences, torch.Tensor):
        sequences = sequences.detach().cpu().tolist()
    skip_ids = {processor.tokenizer.pad_token_id, processor.blank_token_id}
    decoded = []
    for seq in sequences:
        filtered = [int(token_id) for token_id in seq if int(token_id) not in skip_ids]
        decoded.append(processor.tokenizer.decode(filtered, skip_special_tokens=False, group_tokens=True).strip())
    return decoded


def compute_eval_metrics(refs: list[str], preds: list[str]) -> dict[str, Any]:
    refs_canonical = [canonicalize_emotion_text(x) for x in refs]
    preds_canonical = [canonicalize_emotion_text(x) for x in preds]
    refs_norm = [normalize_asr_text(x) for x in refs_canonical]
    preds_norm = [normalize_asr_text(x) for x in preds_canonical]
    ref_emotions = [leading_emotion(x) for x in refs_canonical]
    pred_emotions = [leading_emotion(x) for x in preds_canonical]
    ref_boundary_tokens = [emotion_tokens_in_text(x) for x in refs]
    pred_boundary_tokens = [emotion_tokens_in_text(x) for x in preds]
    valid_pred = [p in EMOTION_TOKENS for p in pred_emotions]
    per_class = {}
    f1_values = []
    weighted_f1_num = 0.0
    support_total = 0
    for token in EMOTION_TOKENS:
        tp = sum(p == token and r == token for p, r in zip(pred_emotions, ref_emotions))
        fp = sum(p == token and r != token for p, r in zip(pred_emotions, ref_emotions))
        fn = sum(p != token and r == token for p, r in zip(pred_emotions, ref_emotions))
        support = sum(r == token for r in ref_emotions)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[token] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        f1_values.append(f1)
        weighted_f1_num += f1 * support
        support_total += support
    majority = max(label_counts(ref_emotions).values()) / len(ref_emotions) if ref_emotions else 0.0
    ref_boundary_total = sum(len(tokens) for tokens in ref_boundary_tokens)
    pred_boundary_total = sum(len(tokens) for tokens in pred_boundary_tokens)
    boundary_lcs_total = sum(
        token_lcs(ref_tokens, pred_tokens)
        for ref_tokens, pred_tokens in zip(ref_boundary_tokens, pred_boundary_tokens)
    )
    boundary_exact = sum(
        ref_tokens == pred_tokens
        for ref_tokens, pred_tokens in zip(ref_boundary_tokens, pred_boundary_tokens)
    )
    boundary_present = sum(bool(tokens) for tokens in pred_boundary_tokens)
    return {
        "wer_without_emotion": corpus_wer(refs_norm, preds_norm) if refs_norm else None,
        "cer_without_emotion": corpus_cer(refs_norm, preds_norm) if refs_norm else None,
        "leading_emotion_accuracy": (
            sum(p == r for p, r in zip(pred_emotions, ref_emotions)) / len(ref_emotions)
            if ref_emotions
            else None
        ),
        "leading_token_present_rate": sum(valid_pred) / len(valid_pred) if valid_pred else None,
        "invalid_token_rate": 1.0 - (sum(valid_pred) / len(valid_pred)) if valid_pred else None,
        "boundary_token_present_rate": boundary_present / len(pred_boundary_tokens) if pred_boundary_tokens else None,
        "boundary_token_recall": boundary_lcs_total / ref_boundary_total if ref_boundary_total else None,
        "boundary_token_precision": boundary_lcs_total / pred_boundary_total if pred_boundary_total else None,
        "boundary_token_sequence_accuracy": boundary_exact / len(ref_boundary_tokens) if ref_boundary_tokens else None,
        "reference_boundary_tokens": ref_boundary_total,
        "predicted_boundary_tokens": pred_boundary_total,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else None,
        "weighted_f1": weighted_f1_num / support_total if support_total else None,
        "majority_class_accuracy": majority,
        "per_class": per_class,
    }


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, item_a in enumerate(a, start=1):
        cur = [i]
        for j, item_b in enumerate(b, start=1):
            cur.append(min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + (item_a != item_b),
            ))
        prev = cur
    return prev[-1]


def token_lcs(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for item_a in a:
        cur = [0]
        for j, item_b in enumerate(b, start=1):
            cur.append(prev[j - 1] + 1 if item_a == item_b else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def corpus_wer(refs: list[str], preds: list[str]) -> float:
    try:
        from jiwer import wer

        return float(wer(refs, preds))
    except Exception:
        edits = 0
        total = 0
        for ref, pred in zip(refs, preds):
            ref_words = ref.split()
            pred_words = pred.split()
            edits += edit_distance(ref_words, pred_words)
            total += len(ref_words)
        return edits / total if total else 0.0


def corpus_cer(refs: list[str], preds: list[str]) -> float:
    try:
        from jiwer import cer

        return float(cer(refs, preds))
    except Exception:
        edits = 0
        total = 0
        for ref, pred in zip(refs, preds):
            edits += edit_distance(list(ref), list(pred))
            total += len(ref)
        return edits / total if total else 0.0


def run_generation_eval(model, processor, dataset, max_samples: int, batch_size: int, output_path: Path | None) -> dict[str, Any]:
    import torch

    model.eval()
    n = min(max_samples, len(dataset)) if max_samples and max_samples > 0 else len(dataset)
    refs: list[str] = []
    preds: list[str] = []
    examples: list[dict[str, str]] = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            rows = [dataset[i] for i in range(start, stop)]
            batch = processor(
                audio=[
                    decode_audio_array(row["audio"], processor.feature_extractor.sampling_rate)
                    for row in rows
                ],
                sampling_rate=processor.feature_extractor.sampling_rate,
                padding=True,
                return_tensors="pt",
            )
            batch = tensor_batch_to_device(batch, model)
            generated = model.generate(**batch)
            decoded = decode_sequences(processor, generated)
            refs.extend(row["text"] for row in rows)
            preds.extend(decoded)
            for ref, pred in zip([row["text"] for row in rows], decoded):
                if len(examples) < 50:
                    examples.append({
                        "reference": ref,
                        "prediction": pred,
                        "reference_canonical": canonicalize_emotion_text(ref),
                        "prediction_canonical": canonicalize_emotion_text(pred),
                    })
    metrics = compute_eval_metrics(refs, preds)
    metrics["num_eval_samples"] = n
    metrics["examples"] = examples
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2))
    return metrics


def make_eval_callback(processor, eval_dataset, args: argparse.Namespace):
    rt = require_runtime_imports()
    TrainerCallback = rt["TrainerCallback"]

    class GenerationEvalCallback(TrainerCallback):
        def __init__(self):
            self.trainer = None

        def on_evaluate(self, args_, state, control, model=None, **kwargs):
            if model is None:
                return control
            metrics = run_generation_eval(
                model=model,
                processor=processor,
                dataset=eval_dataset,
                max_samples=args.max_eval_samples,
                batch_size=args.per_device_eval_batch_size,
                output_path=None,
            )
            flat = {f"eval_gen_{k}": v for k, v in metrics.items() if isinstance(v, (int, float)) or v is None}
            if self.trainer is not None:
                self.trainer.log(flat)
            return control

    return GenerationEvalCallback()


def run_env(args: argparse.Namespace) -> None:
    rt = require_runtime_imports()
    import transformers
    try:
        from huggingface_hub import get_token
    except Exception:
        get_token = lambda: None

    torch = rt["torch"]
    print(json.dumps({
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "hf_token_present": bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or get_token()),
    }, indent=2))
    processor, model = setup_processor_and_model(args)
    print(json.dumps({
        "processor_sampling_rate": processor.feature_extractor.sampling_rate,
        "tokenizer_length": len(processor.tokenizer),
        "blank_token_id": processor.blank_token_id,
        "emotion_token_ids": {
            tok: processor.tokenizer(tok, add_special_tokens=False).input_ids for tok in EMOTION_TOKENS
        },
        "config_vocab_size": model.config.vocab_size,
        "generation_suppress_tokens": model.generation_config.suppress_tokens,
        "parameters": trainable_parameter_summary(model),
    }, indent=2))


def run_train(args: argparse.Namespace) -> None:
    rt = require_runtime_imports()
    torch = rt["torch"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processor, model = setup_processor_and_model(args)
    sampling_rate = processor.feature_extractor.sampling_rate
    splits, split_info = load_and_split_dataset(args, sampling_rate)
    (output_dir / "split_manifest.json").write_text(json.dumps(split_info, indent=2))
    (output_dir / "trainable_parameters.json").write_text(json.dumps(trainable_parameter_summary(model), indent=2))

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    callbacks = []
    eval_callback = make_eval_callback(processor, splits["validation"], args)
    callbacks.append(eval_callback)
    if args.early_stopping_patience > 0:
        callbacks.append(rt["EarlyStoppingCallback"](early_stopping_patience=args.early_stopping_patience))

    report_to = ["tensorboard"]
    if os.environ.get("WANDB_API_KEY"):
        report_to.append("wandb")

    train_args = rt["TrainingArguments"](
        output_dir=str(output_dir),
        hub_model_id=args.hub_model_id,
        push_to_hub=args.push_to_hub,
        hub_private_repo=args.hub_private,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        max_grad_norm=1.0,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        tf32=True,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
        label_names=["labels"],
        report_to=report_to,
    )

    trainer = rt["Trainer"](
        model=model,
        args=train_args,
        train_dataset=splits["train"],
        eval_dataset=splits["validation"],
        data_collator=ParakeetDataCollator(processor, sampling_rate),
        processing_class=processor,
        callbacks=callbacks,
    )
    eval_callback.trainer = trainer

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))

    val_metrics = run_generation_eval(
        model=trainer.model,
        processor=processor,
        dataset=splits["validation"],
        max_samples=args.max_eval_samples,
        batch_size=args.per_device_eval_batch_size,
        output_path=output_dir / "validation_generation_eval.json",
    )
    test_metrics = run_generation_eval(
        model=trainer.model,
        processor=processor,
        dataset=splits["test"],
        max_samples=0,
        batch_size=args.per_device_eval_batch_size,
        output_path=output_dir / "test_generation_eval.json",
    )
    print(json.dumps({"validation": val_metrics, "test": test_metrics}, indent=2))

    token_present_rate = (
        val_metrics.get("boundary_token_present_rate")
        if val_metrics.get("boundary_token_present_rate") is not None
        else val_metrics.get("leading_token_present_rate")
    ) or 0.0
    if args.required_token_present_rate and token_present_rate < args.required_token_present_rate:
        raise RuntimeError(
            "emotion token probe failed: "
            f"validation token present rate {token_present_rate:.4f} "
            f"< required {args.required_token_present_rate:.4f}"
        )

    if args.push_to_hub:
        trainer.push_to_hub(
            commit_message="Fine-tune Parakeet TDT v3 on emotion-token ASR data"
        )


def run_eval(args: argparse.Namespace) -> None:
    rt = require_runtime_imports()
    output_dir = Path(args.output_dir)
    processor = rt["AutoProcessor"].from_pretrained(str(output_dir), token=True)
    model = rt["AutoModelForTDT"].from_pretrained(str(output_dir), token=True)
    if rt["torch"].cuda.is_available():
        model = model.to("cuda")
    sampling_rate = processor.feature_extractor.sampling_rate
    splits, _ = load_and_split_dataset(args, sampling_rate)
    metrics = run_generation_eval(
        model=model,
        processor=processor,
        dataset=splits["test"],
        max_samples=0,
        batch_size=args.per_device_eval_batch_size,
        output_path=output_dir / "test_generation_eval.json",
    )
    print(json.dumps(metrics, indent=2))


def main() -> None:
    args = parse_args()
    if args.mode == "env":
        run_env(args)
    elif args.mode == "train":
        run_train(args)
    elif args.mode == "eval":
        run_eval(args)
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
