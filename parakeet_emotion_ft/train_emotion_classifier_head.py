#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

try:
    from train_parakeet_emotion_tokens import (
        EMOTION_TOKENS,
        canonicalize_emotion_text,
        decode_audio_array,
        leading_emotion,
        load_and_split_dataset,
        tensor_batch_to_device,
    )
except ModuleNotFoundError:
    from .train_parakeet_emotion_tokens import (
        EMOTION_TOKENS,
        canonicalize_emotion_text,
        decode_audio_array,
        leading_emotion,
        load_and_split_dataset,
        tensor_batch_to_device,
    )


DEFAULT_ASR_MODEL_DIR = "runs/parakeet-emotion/mixture_train"
DEFAULT_DATASET_PARQUET = "data/emotion_mixture_train"
DEFAULT_OUTPUT_DIR = "runs/parakeet-emotion/emotion_head"
DEFAULT_HUB_MODEL_ID = "NathanRoll/parakeet-tdt-0.6b-v3-eng-sports-emotion-iu-mixture-emotion-head"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=DEFAULT_ASR_MODEL_DIR)
    parser.add_argument("--dataset-parquet", default=DEFAULT_DATASET_PARQUET)
    parser.add_argument("--dataset-id", default="NathanRoll/eng-sports-radio-psst-iu")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--emotion-column", default="emotion")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--hub-model-id", default=DEFAULT_HUB_MODEL_ID)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-private", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=153)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--max-duration-seconds", type=float, default=20.0)
    parser.add_argument("--min-duration-seconds", type=float, default=0.35)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--oversample-minority", action="store_true", default=True)
    parser.add_argument("--no-oversample-minority", dest="oversample_minority", action="store_false")
    parser.add_argument("--oversample-max-repeat", type=int, default=3)
    parser.add_argument("--oversample-neutral-fraction", type=float, default=0.55)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--log-steps", type=int, default=20)
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--required-val-accuracy", type=float, default=0.0)
    parser.add_argument("--required-val-macro-f1", type=float, default=0.0)
    parser.add_argument("--transcribe-examples", type=int, default=24)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EmotionClassifierHead(nn.Module):
    def __init__(self, encoder_dim: int, hidden_dim: int, num_labels: int, dropout: float):
        super().__init__()
        pooled_dim = encoder_dim * 3
        self.net = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, pooled_features: torch.Tensor) -> torch.Tensor:
        return self.net(pooled_features)


def split_args_from(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        dataset_id=args.dataset_id,
        dataset_parquet=args.dataset_parquet,
        text_column=args.text_column,
        emotion_column=args.emotion_column,
        min_duration_seconds=args.min_duration_seconds,
        max_duration_seconds=args.max_duration_seconds,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
        oversample_minority=args.oversample_minority,
        max_train_samples=args.max_train_samples,
        oversample_neutral_fraction=args.oversample_neutral_fraction,
        oversample_max_repeat=args.oversample_max_repeat,
        emotion_token_position="prefix",
        emotion_token_repeat=1,
    )


def label_id_from_text(text: str, label_to_id: dict[str, int]) -> int:
    token = leading_emotion(canonicalize_emotion_text(text))
    if token not in label_to_id:
        raise ValueError(f"missing emotion token in text: {text!r}")
    return label_to_id[token]


def make_collate_fn(processor, label_to_id: dict[str, int]):
    sampling_rate = processor.feature_extractor.sampling_rate

    def collate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        batch = processor(
            audio=[decode_audio_array(row["audio"], sampling_rate) for row in rows],
            sampling_rate=sampling_rate,
            padding=True,
            return_tensors="pt",
        )
        labels = torch.tensor([label_id_from_text(row["text"], label_to_id) for row in rows], dtype=torch.long)
        batch["emotion_labels"] = labels
        batch["texts"] = [canonicalize_emotion_text(row["text"]) for row in rows]
        return batch

    return collate


def masked_pool(pooler_output: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    x = pooler_output.float()
    if attention_mask is None:
        mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
    else:
        mask = attention_mask.to(device=x.device, dtype=torch.bool)
    lengths = mask.sum(dim=1).clamp_min(1).float()
    mask_f = mask.unsqueeze(-1).float()

    mean = (x * mask_f).sum(dim=1) / lengths.unsqueeze(-1)
    centered = (x - mean.unsqueeze(1)) * mask_f
    std = torch.sqrt((centered.square().sum(dim=1) / lengths.unsqueeze(-1)).clamp_min(1e-6))
    max_values = x.masked_fill(~mask.unsqueeze(-1), torch.finfo(x.dtype).min).max(dim=1).values
    return torch.cat([mean, std, max_values], dim=-1)


def encoder_features(model, batch: dict[str, Any]) -> torch.Tensor:
    kwargs = {
        "input_features": batch["input_features"],
        "output_attention_mask": True,
    }
    if "attention_mask" in batch:
        kwargs["attention_mask"] = batch["attention_mask"]
    outputs = model.get_audio_features(**kwargs)
    return masked_pool(outputs.pooler_output, getattr(outputs, "attention_mask", None))


def class_weights(dataset, label_to_id: dict[str, int], power: float, device: torch.device) -> torch.Tensor:
    counts = torch.zeros(len(label_to_id), dtype=torch.float32)
    for text in dataset["text"]:
        counts[label_id_from_text(text, label_to_id)] += 1
    counts = counts.clamp_min(1)
    weights = (counts.sum() / counts).pow(power)
    weights = weights / weights.mean()
    return weights.to(device)


def prf_metrics(labels: list[int], preds: list[int], id_to_label: list[str]) -> dict[str, Any]:
    correct = sum(int(p == y) for p, y in zip(preds, labels))
    total = len(labels)
    per_class = {}
    macro_f1 = 0.0
    weighted_f1 = 0.0
    for idx, label in enumerate(id_to_label):
        tp = sum(p == idx and y == idx for p, y in zip(preds, labels))
        fp = sum(p == idx and y != idx for p, y in zip(preds, labels))
        fn = sum(p != idx and y == idx for p, y in zip(preds, labels))
        support = sum(y == idx for y in labels)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        macro_f1 += f1
        weighted_f1 += f1 * support
    macro_f1 /= len(id_to_label)
    weighted_f1 = weighted_f1 / total if total else 0.0
    return {
        "accuracy": correct / total if total else 0.0,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "num_eval_samples": total,
        "per_class": per_class,
    }


def evaluate(model, head, dataloader, device: torch.device, id_to_label: list[str], max_batches: int = 0) -> dict[str, Any]:
    model.eval()
    head.eval()
    all_labels: list[int] = []
    all_preds: list[int] = []
    examples: list[dict[str, str]] = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break
            texts = batch.pop("texts")
            labels = batch.pop("emotion_labels").to(device)
            batch = tensor_batch_to_device(batch, model)
            features = encoder_features(model, batch)
            logits = head(features)
            preds = logits.argmax(dim=-1)
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            for text, label_id, pred_id in zip(texts, labels.cpu().tolist(), preds.cpu().tolist()):
                if len(examples) < 50:
                    examples.append({
                        "reference": text,
                        "reference_label": id_to_label[label_id],
                        "prediction_label": id_to_label[pred_id],
                    })
    metrics = prf_metrics(all_labels, all_preds, id_to_label)
    metrics["examples"] = examples
    return metrics


def transcribe_examples(model, processor, head, dataset, device: torch.device, id_to_label: list[str], n: int) -> list[dict[str, str]]:
    if n <= 0:
        return []
    model.eval()
    head.eval()
    rows = [dataset[i] for i in range(min(n, len(dataset)))]
    collate = make_collate_fn(processor, {label: idx for idx, label in enumerate(id_to_label)})
    batch = collate(rows)
    texts = batch.pop("texts")
    labels = batch.pop("emotion_labels").to(device)
    batch = tensor_batch_to_device(batch, model)
    with torch.no_grad():
        features = encoder_features(model, batch)
        logits = head(features)
        pred_ids = logits.argmax(dim=-1).cpu().tolist()
        generated = model.generate(**batch)
    try:
        from train_parakeet_emotion_tokens import decode_sequences, strip_all_emotion_tokens
    except ModuleNotFoundError:
        from .train_parakeet_emotion_tokens import decode_sequences, strip_all_emotion_tokens

    transcripts = [strip_all_emotion_tokens(text) for text in decode_sequences(processor, generated)]
    return [
        {
            "reference": ref,
            "reference_label": id_to_label[int(label_id)],
            "prediction": f"{id_to_label[int(pred_id)]} {transcript}".strip(),
            "prediction_label": id_to_label[int(pred_id)],
            "transcript": transcript,
        }
        for ref, label_id, pred_id, transcript in zip(texts, labels.cpu().tolist(), pred_ids, transcripts)
    ]


def save_artifacts(
    output_dir: Path,
    head: EmotionClassifierHead,
    args: argparse.Namespace,
    id_to_label: list[str],
    encoder_dim: int,
    metrics: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": head.state_dict(),
            "id_to_label": id_to_label,
            "label_to_id": {label: idx for idx, label in enumerate(id_to_label)},
            "encoder_dim": encoder_dim,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "base_asr_model": args.model_dir,
        },
        output_dir / "emotion_head.pt",
    )
    config = {
        "base_asr_model": args.model_dir,
        "labels": id_to_label,
        "encoder_dim": encoder_dim,
        "pooled_feature_dim": encoder_dim * 3,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "pooling": "mean+std+max over Parakeet encoder frames",
        "format_contract": "prediction text is '<|emotion|> transcript'",
    }
    (output_dir / "emotion_head_config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (output_dir / "README.md").write_text(
        "# Parakeet Emotion Classifier Head\n\n"
        "This repo stores an auxiliary utterance-level emotion classifier for Parakeet ASR.\n"
        "Use the ASR model to transcribe, then prepend the classifier's predicted emotion token.\n\n"
        f"Base ASR model: `{args.model_dir}`\n\n"
        "Labels:\n\n"
        + "\n".join(f"- `{label}`" for label in id_to_label)
        + "\n"
    )


def push_to_hub(output_dir: Path, args: argparse.Namespace) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.hub_model_id, repo_type="model", private=args.hub_private, exist_ok=True)
    api.upload_folder(
        repo_id=args.hub_model_id,
        repo_type="model",
        folder_path=str(output_dir),
        commit_message="Train Parakeet emotion classifier head",
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForTDT, AutoProcessor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    processor = AutoProcessor.from_pretrained(args.model_dir, token=True)
    model = AutoModelForTDT.from_pretrained(args.model_dir, token=True)
    if device.type == "cuda":
        model = model.to(device=device, dtype=dtype)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    splits, split_info = load_and_split_dataset(split_args_from(args), processor.feature_extractor.sampling_rate)
    (output_dir / "split_manifest.json").write_text(json.dumps(split_info, indent=2) + "\n")

    id_to_label = EMOTION_TOKENS
    label_to_id = {label: idx for idx, label in enumerate(id_to_label)}
    collate = make_collate_fn(processor, label_to_id)
    train_loader = DataLoader(
        splits["train"],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        splits["validation"],
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        splits["test"],
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
        collate_fn=collate,
    )

    first_batch = next(iter(train_loader))
    labels = first_batch.pop("emotion_labels").to(device)
    first_batch.pop("texts")
    first_batch = tensor_batch_to_device(first_batch, model)
    with torch.no_grad():
        first_features = encoder_features(model, first_batch)
    encoder_dim = int(first_features.shape[-1] // 3)
    head = EmotionClassifierHead(
        encoder_dim=encoder_dim,
        hidden_dim=args.hidden_dim,
        num_labels=len(id_to_label),
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights(splits["train"], label_to_id, args.class_weight_power, device))

    best_val = {"macro_f1": -1.0}
    best_state = None
    step = 0
    train_iter = iter(train_loader)
    history = []
    while step < args.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        batch.pop("texts")
        labels = batch.pop("emotion_labels").to(device)
        batch = tensor_batch_to_device(batch, model)
        head.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            features = encoder_features(model, batch)
        logits = head(features)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        step += 1

        if step % args.log_steps == 0:
            print(json.dumps({"step": step, "train_loss": float(loss.detach().cpu())}))
        if step % args.eval_steps == 0 or step == args.max_steps:
            val_metrics = evaluate(model, head, val_loader, device, id_to_label)
            row = {
                "step": step,
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_weighted_f1": val_metrics["weighted_f1"],
            }
            print(json.dumps(row))
            history.append(row)
            if val_metrics["macro_f1"] > best_val.get("macro_f1", -1):
                best_val = val_metrics
                best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}

    if best_state is not None:
        head.load_state_dict(best_state)
    val_metrics = evaluate(model, head, val_loader, device, id_to_label)
    test_metrics = evaluate(model, head, test_loader, device, id_to_label)
    examples = transcribe_examples(
        model,
        processor,
        head,
        splits["test"],
        device,
        id_to_label,
        args.transcribe_examples,
    )
    metrics = {
        "validation": val_metrics,
        "test": test_metrics,
        "combined_examples": examples,
        "history": history,
        "split_counts": split_info["counts"],
    }
    save_artifacts(output_dir, head, args, id_to_label, encoder_dim, metrics)
    print(json.dumps({
        "validation": {k: v for k, v in val_metrics.items() if k not in {"per_class", "examples"}},
        "test": {k: v for k, v in test_metrics.items() if k not in {"per_class", "examples"}},
        "output_dir": str(output_dir),
    }, indent=2))

    if args.required_val_accuracy and val_metrics["accuracy"] < args.required_val_accuracy:
        raise RuntimeError(
            f"emotion head validation accuracy {val_metrics['accuracy']:.4f} "
            f"< required {args.required_val_accuracy:.4f}"
        )
    if args.required_val_macro_f1 and val_metrics["macro_f1"] < args.required_val_macro_f1:
        raise RuntimeError(
            f"emotion head validation macro_f1 {val_metrics['macro_f1']:.4f} "
            f"< required {args.required_val_macro_f1:.4f}"
        )
    if args.push_to_hub:
        push_to_hub(output_dir, args)


if __name__ == "__main__":
    main()
