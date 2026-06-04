#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    from train_parakeet_emotion_tokens import (
        EMOTION_TOKENS,
        decode_audio_array,
        load_and_split_dataset,
        primary_emotion,
        row_duration_seconds,
        strip_all_emotion_tokens,
        tensor_batch_to_device,
    )
except ModuleNotFoundError:
    from .train_parakeet_emotion_tokens import (
        EMOTION_TOKENS,
        decode_audio_array,
        load_and_split_dataset,
        primary_emotion,
        row_duration_seconds,
        strip_all_emotion_tokens,
        tensor_batch_to_device,
    )


DEFAULT_ASR_MODEL_DIR = "runs/parakeet-emotion/mixture_train"
DEFAULT_DATASET_PARQUET = "data/emotion_mixture_train"
DEFAULT_OUTPUT_DIR = "runs/parakeet-emotion/iu_boundary_emotion_head"
DEFAULT_HUB_MODEL_ID = "NathanRoll/parakeet-tdt-0.6b-v3-eng-sports-emotion-iu-boundary-head"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=DEFAULT_ASR_MODEL_DIR)
    parser.add_argument("--base-hub-model-id", default="NathanRoll/parakeet-tdt-0.6b-v3-eng-sports-emotion-iu-mixture")
    parser.add_argument("--dataset-parquet", default=DEFAULT_DATASET_PARQUET)
    parser.add_argument("--dataset-id", default="NathanRoll/eng-sports-radio-psst-iu")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--emotion-column", default="emotion")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--hub-model-id", default=DEFAULT_HUB_MODEL_ID)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-private", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=641)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--min-duration-seconds", type=float, default=0.35)
    parser.add_argument("--max-duration-seconds", type=float, default=20.0)
    parser.add_argument("--oversample-minority", action="store_true", default=True)
    parser.add_argument("--no-oversample-minority", dest="oversample_minority", action="store_false")
    parser.add_argument("--oversample-neutral-fraction", type=float, default=0.55)
    parser.add_argument("--oversample-max-repeat", type=int, default=4)
    parser.add_argument("--min-ius", type=int, default=2)
    parser.add_argument("--max-ius", type=int, default=6)
    parser.add_argument("--target-duration-seconds", type=float, default=28.0)
    parser.add_argument("--max-concat-duration-seconds", type=float, default=55.0)
    parser.add_argument("--silence-ms", type=float, default=80.0)
    parser.add_argument("--boundary-window-frames", type=int, default=1)
    parser.add_argument("--boundary-threshold", type=float, default=0.45)
    parser.add_argument(
        "--boundary-thresholds",
        default="0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.70",
    )
    parser.add_argument("--boundary-match-tolerance-frames", type=int, default=5)
    parser.add_argument("--boundary-pos-weight", type=float, default=30.0)
    parser.add_argument("--segment-loss-weight", type=float, default=1.0)
    parser.add_argument("--boundary-loss-weight", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--max-steps", type=int, default=2500)
    parser.add_argument("--eval-steps", type=int, default=250)
    parser.add_argument("--log-steps", type=int, default=25)
    parser.add_argument("--max-train-groups", type=int, default=0)
    parser.add_argument("--max-eval-groups", type=int, default=0)
    parser.add_argument("--required-boundary-f1", type=float, default=0.0)
    parser.add_argument("--required-segment-macro-f1", type=float, default=0.0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
        max_train_samples=0,
        oversample_neutral_fraction=args.oversample_neutral_fraction,
        oversample_max_repeat=args.oversample_max_repeat,
        emotion_token_position="prefix",
        emotion_token_repeat=1,
    )


class BoundaryHead(nn.Module):
    def __init__(self, encoder_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.net(frames.float()).squeeze(-1)


class SegmentEmotionHead(nn.Module):
    def __init__(self, encoder_dim: int, hidden_dim: int, num_labels: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(encoder_dim * 3),
            nn.Linear(encoder_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, pooled_segments: torch.Tensor) -> torch.Tensor:
        return self.net(pooled_segments.float())


@dataclass
class GroupItem:
    indices: list[int]
    durations: list[float]


class GroupedIUDataset(Dataset):
    def __init__(
        self,
        hf_dataset,
        *,
        seed: int,
        min_ius: int,
        max_ius: int,
        target_duration_seconds: float,
        max_concat_duration_seconds: float,
        silence_seconds: float,
        max_groups: int = 0,
    ):
        self.dataset = hf_dataset
        self.seed = seed
        self.silence_seconds = silence_seconds
        self.durations = [row_duration_seconds(self.dataset[i]) for i in range(len(self.dataset))]
        self.groups = self._make_groups(
            min_ius=min_ius,
            max_ius=max_ius,
            target_duration_seconds=target_duration_seconds,
            max_concat_duration_seconds=max_concat_duration_seconds,
        )
        if max_groups > 0:
            self.groups = self.groups[:max_groups]

    def _make_groups(
        self,
        *,
        min_ius: int,
        max_ius: int,
        target_duration_seconds: float,
        max_concat_duration_seconds: float,
    ) -> list[GroupItem]:
        rng = random.Random(self.seed)
        indices = list(range(len(self.dataset)))
        rng.shuffle(indices)
        groups: list[GroupItem] = []
        idx = 0
        while idx < len(indices):
            want = rng.randint(min_ius, max_ius)
            group: list[int] = []
            durations: list[float] = []
            total = 0.0
            while idx < len(indices) and len(group) < want:
                item = indices[idx]
                dur = float(self.durations[item])
                added = dur + (self.silence_seconds if group else 0.0)
                if group and len(group) >= min_ius and total + added > max_concat_duration_seconds:
                    break
                group.append(item)
                durations.append(dur)
                total += added
                idx += 1
                if len(group) >= min_ius and total >= target_duration_seconds:
                    break
            if len(group) >= min_ius:
                groups.append(GroupItem(group, durations))
            else:
                idx += max(1, len(group))
        return groups

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        group = self.groups[idx]
        return {
            "rows": [self.dataset[i] for i in group.indices],
            "durations": group.durations,
        }


def token_id_from_text(text: str, label_to_id: dict[str, int]) -> int:
    token = primary_emotion(text)
    if token not in label_to_id:
        raise ValueError(f"missing emotion token in text: {text!r}")
    return label_to_id[token]


def make_collate_fn(processor, label_to_id: dict[str, int], silence_seconds: float):
    sampling_rate = int(processor.feature_extractor.sampling_rate)
    silence_samples = int(round(sampling_rate * silence_seconds))

    def collate(items: list[dict[str, Any]]) -> dict[str, Any]:
        import numpy as np

        audios = []
        event_fractions: list[list[float]] = []
        event_label_ids: list[list[int]] = []
        segment_fractions: list[list[tuple[float, float]]] = []
        segment_label_ids: list[list[int]] = []
        reference_texts: list[str] = []
        total_samples: list[int] = []
        for item in items:
            parts = []
            cursor = 0
            boundaries = []
            segment_bounds = []
            label_ids = []
            target_parts = []
            rows = item["rows"]
            for seg_idx, row in enumerate(rows):
                if seg_idx:
                    silence = np.zeros(silence_samples, dtype=np.float32)
                    parts.append(silence)
                    cursor += len(silence)
                start = cursor
                samples = decode_audio_array(row["audio"], sampling_rate)
                parts.append(samples)
                cursor += len(samples)
                end = cursor
                text = str(row["text"])
                label_id = token_id_from_text(text, label_to_id)
                boundaries.append(end)
                segment_bounds.append((start, end))
                label_ids.append(label_id)
                target_parts.extend([strip_all_emotion_tokens(text), EMOTION_TOKENS[label_id]])
            audio = np.concatenate(parts).astype(np.float32)
            denominator = max(1, len(audio) - 1)
            audios.append(audio)
            total_samples.append(len(audio))
            event_fractions.append([min(1.0, max(0.0, b / denominator)) for b in boundaries])
            segment_fractions.append([
                (min(1.0, max(0.0, s / denominator)), min(1.0, max(0.0, e / denominator)))
                for s, e in segment_bounds
            ])
            event_label_ids.append(label_ids)
            segment_label_ids.append(label_ids)
            reference_texts.append(" ".join(target_parts))
        batch = processor(
            audio=audios,
            sampling_rate=sampling_rate,
            padding=True,
            return_tensors="pt",
        )
        batch["event_fractions"] = event_fractions
        batch["event_label_ids"] = event_label_ids
        batch["segment_fractions"] = segment_fractions
        batch["segment_label_ids"] = segment_label_ids
        batch["reference_texts"] = reference_texts
        batch["total_samples"] = total_samples
        return batch

    return collate


def audio_features(model, batch: dict[str, Any]):
    kwargs = {
        "input_features": batch["input_features"],
        "output_attention_mask": True,
    }
    if "attention_mask" in batch:
        kwargs["attention_mask"] = batch["attention_mask"]
    return model.get_audio_features(**kwargs)


def valid_frame_lengths(attention_mask: torch.Tensor | None, batch_size: int, frame_count: int, device) -> torch.Tensor:
    if attention_mask is None:
        return torch.full((batch_size,), frame_count, dtype=torch.long, device=device)
    return attention_mask.to(device=device, dtype=torch.bool).sum(dim=1).clamp_min(1)


def build_boundary_targets(
    *,
    event_fractions: list[list[float]],
    lengths: torch.Tensor,
    frame_count: int,
    window: int,
    device,
) -> tuple[torch.Tensor, torch.Tensor, list[list[int]]]:
    targets = torch.zeros((len(event_fractions), frame_count), dtype=torch.float32, device=device)
    valid = torch.zeros_like(targets, dtype=torch.bool)
    ref_indices: list[list[int]] = []
    for batch_idx, fractions in enumerate(event_fractions):
        length = int(lengths[batch_idx].item())
        valid[batch_idx, :length] = True
        indices = []
        for fraction in fractions:
            frame = min(length - 1, max(0, int(round(fraction * max(0, length - 1)))))
            indices.append(frame)
            lo = max(0, frame - window)
            hi = min(length, frame + window + 1)
            targets[batch_idx, lo:hi] = 1.0
        ref_indices.append(indices)
    return targets, valid, ref_indices


def pool_segments(
    frames: torch.Tensor,
    lengths: torch.Tensor,
    segment_fractions: list[list[tuple[float, float]]],
    segment_label_ids: list[list[int]],
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, int]]]:
    pooled = []
    labels = []
    refs = []
    for batch_idx, bounds in enumerate(segment_fractions):
        length = int(lengths[batch_idx].item())
        for segment_idx, (start_fraction, end_fraction) in enumerate(bounds):
            start = min(length - 1, max(0, int(round(start_fraction * max(0, length - 1)))))
            end = min(length, max(start + 1, int(round(end_fraction * max(0, length - 1))) + 1))
            x = frames[batch_idx, start:end].float()
            mean = x.mean(dim=0)
            std = x.std(dim=0, unbiased=False)
            max_values = x.max(dim=0).values
            pooled.append(torch.cat([mean, std, max_values], dim=-1))
            labels.append(int(segment_label_ids[batch_idx][segment_idx]))
            refs.append({"batch_idx": batch_idx, "segment_idx": segment_idx, "start": start, "end": end})
    if not pooled:
        raise RuntimeError("empty segment batch")
    return torch.stack(pooled), torch.tensor(labels, dtype=torch.long, device=frames.device), refs


def boundary_loss(logits: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor, pos_weight: float) -> torch.Tensor:
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    weights = torch.ones_like(loss)
    weights = torch.where(targets > 0.5, weights * pos_weight, weights)
    loss = loss * weights
    return loss[valid].mean()


def prf(labels: list[int], preds: list[int], id_to_label: list[str]) -> dict[str, Any]:
    per_class = {}
    total = len(labels)
    correct = sum(int(p == y) for p, y in zip(preds, labels))
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
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        macro_f1 += f1
        weighted_f1 += f1 * support
    macro_f1 /= len(id_to_label)
    weighted_f1 = weighted_f1 / total if total else 0.0
    return {
        "accuracy": correct / total if total else 0.0,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "num_segments": total,
    }


def segment_class_weights(dataset, label_to_id: dict[str, int], power: float, device) -> torch.Tensor | None:
    if power <= 0:
        return None
    labels = [primary_emotion(text) for text in dataset["text"]]
    counts = Counter(label for label in labels if label in label_to_id)
    total = sum(counts.values())
    if total == 0:
        return None
    raw = []
    for label in EMOTION_TOKENS:
        count = max(1, counts.get(label, 0))
        raw.append((total / (len(label_to_id) * count)) ** power)
    weights = torch.tensor(raw, dtype=torch.float32, device=device)
    return weights / weights.mean().clamp_min(1e-6)


def boundary_thresholds(args: argparse.Namespace) -> list[float]:
    values = {float(args.boundary_threshold)}
    for part in str(args.boundary_thresholds).split(","):
        part = part.strip()
        if part:
            values.add(float(part))
    return sorted(values)


def predicted_boundary_events(probabilities: torch.Tensor, length: int, threshold: float) -> list[int]:
    active = (probabilities[:length] >= threshold).detach().cpu().tolist()
    probs = probabilities[:length].detach().cpu().tolist()
    events = []
    start = None
    for idx, is_active in enumerate(active + [False]):
        if is_active and start is None:
            start = idx
        elif not is_active and start is not None:
            stop = idx
            best = max(range(start, stop), key=lambda item: probs[item])
            events.append(best)
            start = None
    return events


def match_events(refs: list[int], preds: list[int], tolerance: int) -> tuple[int, list[tuple[int, int]]]:
    matched = []
    used = set()
    for ref in refs:
        candidates = [
            (abs(pred - ref), pred_idx, pred)
            for pred_idx, pred in enumerate(preds)
            if pred_idx not in used and abs(pred - ref) <= tolerance
        ]
        if not candidates:
            continue
        _dist, pred_idx, pred = min(candidates)
        used.add(pred_idx)
        matched.append((ref, pred))
    return len(matched), matched


def boundary_metrics_for_records(records: list[dict[str, Any]], threshold: float, tolerance: int) -> dict[str, Any]:
    ref_event_total = 0
    pred_event_total = 0
    matched_total = 0
    for record in records:
        refs = record["refs"]
        preds = predicted_boundary_events(record["probabilities"], record["length"], threshold)
        matched, _pairs = match_events(refs, preds, tolerance)
        ref_event_total += len(refs)
        pred_event_total += len(preds)
        matched_total += matched
    precision = matched_total / pred_event_total if pred_event_total else 0.0
    recall = matched_total / ref_event_total if ref_event_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "boundary_threshold": threshold,
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_f1": f1,
        "reference_boundaries": ref_event_total,
        "predicted_boundaries": pred_event_total,
        "matched_boundaries": matched_total,
    }


def evaluate(
    model,
    boundary_head,
    emotion_head,
    dataloader,
    args,
    device,
    id_to_label: list[str],
    fixed_boundary_threshold: float | None = None,
) -> dict[str, Any]:
    model.eval()
    boundary_head.eval()
    emotion_head.eval()
    boundary_records: list[dict[str, Any]] = []
    all_labels: list[int] = []
    all_preds: list[int] = []
    example_records: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in dataloader:
            reference_texts = batch.pop("reference_texts")
            event_fractions = batch.pop("event_fractions")
            _event_label_ids = batch.pop("event_label_ids")
            segment_fractions = batch.pop("segment_fractions")
            segment_label_ids = batch.pop("segment_label_ids")
            batch.pop("total_samples", None)
            batch = tensor_batch_to_device(batch, model)
            outputs = audio_features(model, batch)
            frames = outputs.pooler_output
            lengths = valid_frame_lengths(getattr(outputs, "attention_mask", None), frames.shape[0], frames.shape[1], device)
            targets, valid, ref_indices = build_boundary_targets(
                event_fractions=event_fractions,
                lengths=lengths,
                frame_count=frames.shape[1],
                window=0,
                device=device,
            )
            boundary_probs = torch.sigmoid(boundary_head(frames))
            segment_features, segment_labels, segment_refs = pool_segments(
                frames, lengths, segment_fractions, segment_label_ids
            )
            segment_logits = emotion_head(segment_features)
            segment_preds = segment_logits.argmax(dim=-1)
            all_labels.extend(segment_labels.detach().cpu().tolist())
            all_preds.extend(segment_preds.detach().cpu().tolist())
            for batch_idx in range(frames.shape[0]):
                length = int(lengths[batch_idx].item())
                refs = ref_indices[batch_idx]
                record = {
                    "probabilities": boundary_probs[batch_idx].detach().cpu(),
                    "length": length,
                    "refs": refs,
                }
                boundary_records.append(record)
                if len(example_records) < 50:
                    record = dict(record)
                    record["reference"] = reference_texts[batch_idx]
                    record["reference_segment_labels"] = [
                        id_to_label[int(segment_labels[idx].detach().cpu())]
                        for idx, ref in enumerate(segment_refs)
                        if ref["batch_idx"] == batch_idx
                    ]
                    record["predicted_segment_labels"] = [
                        id_to_label[int(segment_preds[idx].detach().cpu())]
                        for idx, ref in enumerate(segment_refs)
                        if ref["batch_idx"] == batch_idx
                    ]
                    example_records.append(record)
    thresholds = [fixed_boundary_threshold] if fixed_boundary_threshold is not None else boundary_thresholds(args)
    threshold_metrics = [
        boundary_metrics_for_records(boundary_records, threshold, args.boundary_match_tolerance_frames)
        for threshold in thresholds
    ]
    best_boundary = max(
        threshold_metrics,
        key=lambda item: (item["boundary_f1"], item["boundary_recall"], item["boundary_precision"]),
    )
    examples = []
    for record in example_records:
        preds = predicted_boundary_events(
            record["probabilities"],
            record["length"],
            best_boundary["boundary_threshold"],
        )
        examples.append({
            "reference": record["reference"],
            "reference_boundary_frames": record["refs"],
            "predicted_boundary_frames": preds,
            "reference_segment_labels": record["reference_segment_labels"],
            "predicted_segment_labels": record["predicted_segment_labels"],
        })
    segment_metrics = prf(all_labels, all_preds, id_to_label)
    return {
        **best_boundary,
        "boundary_threshold_sweep": threshold_metrics,
        "segment_accuracy": segment_metrics["accuracy"],
        "segment_macro_f1": segment_metrics["macro_f1"],
        "segment_weighted_f1": segment_metrics["weighted_f1"],
        "num_segments": segment_metrics["num_segments"],
        "segment_per_class": segment_metrics["per_class"],
        "examples": examples,
    }


def run_step(model, boundary_head, emotion_head, batch, args, device, class_weights) -> tuple[torch.Tensor, dict[str, float]]:
    event_fractions = batch.pop("event_fractions")
    batch.pop("event_label_ids")
    segment_fractions = batch.pop("segment_fractions")
    segment_label_ids = batch.pop("segment_label_ids")
    batch.pop("reference_texts")
    batch.pop("total_samples", None)
    batch = tensor_batch_to_device(batch, model)
    with torch.no_grad():
        outputs = audio_features(model, batch)
        frames = outputs.pooler_output
        lengths = valid_frame_lengths(getattr(outputs, "attention_mask", None), frames.shape[0], frames.shape[1], device)
    boundary_logits = boundary_head(frames)
    targets, valid, _ = build_boundary_targets(
        event_fractions=event_fractions,
        lengths=lengths,
        frame_count=frames.shape[1],
        window=args.boundary_window_frames,
        device=device,
    )
    b_loss = boundary_loss(boundary_logits, targets, valid, args.boundary_pos_weight)
    segment_features, segment_labels, _refs = pool_segments(frames, lengths, segment_fractions, segment_label_ids)
    segment_logits = emotion_head(segment_features)
    s_loss = torch.nn.functional.cross_entropy(segment_logits, segment_labels, weight=class_weights)
    loss = args.boundary_loss_weight * b_loss + args.segment_loss_weight * s_loss
    return loss, {"boundary_loss": float(b_loss.detach().cpu()), "segment_loss": float(s_loss.detach().cpu())}


def save_artifacts(output_dir: Path, boundary_head, emotion_head, args, encoder_dim: int, metrics: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "boundary_head_state_dict": boundary_head.state_dict(),
            "emotion_head_state_dict": emotion_head.state_dict(),
            "labels": EMOTION_TOKENS,
            "encoder_dim": encoder_dim,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "boundary_threshold": args.boundary_threshold,
            "boundary_match_tolerance_frames": args.boundary_match_tolerance_frames,
            "base_asr_model": args.base_hub_model_id or args.model_dir,
        },
        output_dir / "iu_boundary_emotion_head.pt",
    )
    (output_dir / "iu_boundary_emotion_config.json").write_text(json.dumps({
        "base_asr_model": args.base_hub_model_id or args.model_dir,
        "labels": EMOTION_TOKENS,
        "encoder_dim": encoder_dim,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "boundary_threshold": args.boundary_threshold,
        "format_contract": "Run Parakeet ASR, detect IU boundary frames with boundary_head, classify each detected segment with emotion_head, then insert '<|emotion|>' at each IU boundary.",
    }, indent=2) + "\n")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (output_dir / "README.md").write_text(
        "# Parakeet IU Boundary Emotion Head\n\n"
        "This model package augments Parakeet ASR with explicit encoder-side IU boundary detection and "
        "segment-level emotion classification. It avoids relying on the TDT text decoder to emit rare "
        "emotion special tokens.\n\n"
        f"Base ASR model: `{args.base_hub_model_id or args.model_dir}`\n\n"
        "Output contract: transcribe with the base ASR model, segment encoder frames with `boundary_head`, "
        "classify each segment with `emotion_head`, and insert the predicted emotion token at each IU boundary.\n"
    )


def push_to_hub(output_dir: Path, args: argparse.Namespace) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.hub_model_id, repo_type="model", private=args.hub_private, exist_ok=True)
    api.upload_folder(
        repo_id=args.hub_model_id,
        repo_type="model",
        folder_path=str(output_dir),
        commit_message="Train Parakeet IU boundary emotion head",
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
    label_to_id = {label: idx for idx, label in enumerate(EMOTION_TOKENS)}
    silence_seconds = args.silence_ms / 1000.0
    train_groups = GroupedIUDataset(
        splits["train"],
        seed=args.seed,
        min_ius=args.min_ius,
        max_ius=args.max_ius,
        target_duration_seconds=args.target_duration_seconds,
        max_concat_duration_seconds=args.max_concat_duration_seconds,
        silence_seconds=silence_seconds,
        max_groups=args.max_train_groups,
    )
    val_groups = GroupedIUDataset(
        splits["validation"],
        seed=args.seed + 1,
        min_ius=args.min_ius,
        max_ius=args.max_ius,
        target_duration_seconds=args.target_duration_seconds,
        max_concat_duration_seconds=args.max_concat_duration_seconds,
        silence_seconds=silence_seconds,
        max_groups=args.max_eval_groups,
    )
    test_groups = GroupedIUDataset(
        splits["test"],
        seed=args.seed + 2,
        min_ius=args.min_ius,
        max_ius=args.max_ius,
        target_duration_seconds=args.target_duration_seconds,
        max_concat_duration_seconds=args.max_concat_duration_seconds,
        silence_seconds=silence_seconds,
        max_groups=args.max_eval_groups,
    )
    group_summary = {
        "train_groups": len(train_groups),
        "validation_groups": len(val_groups),
        "test_groups": len(test_groups),
        "train_iu_count": len(splits["train"]),
        "validation_iu_count": len(splits["validation"]),
        "test_iu_count": len(splits["test"]),
    }
    (output_dir / "group_manifest.json").write_text(json.dumps(group_summary, indent=2) + "\n")
    print(json.dumps(group_summary), flush=True)

    collate = make_collate_fn(processor, label_to_id, silence_seconds)
    train_loader = DataLoader(
        train_groups,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_groups,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        test_groups,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
        collate_fn=collate,
    )

    first_batch = next(iter(train_loader))
    for key in ["event_fractions", "event_label_ids", "segment_fractions", "segment_label_ids", "reference_texts", "total_samples"]:
        first_batch.pop(key, None)
    first_batch = tensor_batch_to_device(first_batch, model)
    with torch.no_grad():
        first_outputs = audio_features(model, first_batch)
    encoder_dim = int(first_outputs.pooler_output.shape[-1])
    boundary_head = BoundaryHead(encoder_dim, args.hidden_dim, args.dropout).to(device)
    emotion_head = SegmentEmotionHead(encoder_dim, args.hidden_dim, len(EMOTION_TOKENS), args.dropout).to(device)
    optimizer = torch.optim.AdamW(
        list(boundary_head.parameters()) + list(emotion_head.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    class_weights = segment_class_weights(splits["train"], label_to_id, args.class_weight_power, device)

    best_metric = -1.0
    best_state = None
    history = []
    train_iter = iter(train_loader)
    for step in range(1, args.max_steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        boundary_head.train()
        emotion_head.train()
        optimizer.zero_grad(set_to_none=True)
        loss, parts = run_step(model, boundary_head, emotion_head, batch, args, device, class_weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(boundary_head.parameters()) + list(emotion_head.parameters()), 1.0)
        optimizer.step()
        if step % args.log_steps == 0:
            print(json.dumps({"step": step, "loss": float(loss.detach().cpu()), **parts}), flush=True)
        if step % args.eval_steps == 0 or step == args.max_steps:
            val_metrics = evaluate(model, boundary_head, emotion_head, val_loader, args, device, EMOTION_TOKENS)
            row = {
                "step": step,
                "boundary_f1": val_metrics["boundary_f1"],
                "boundary_precision": val_metrics["boundary_precision"],
                "boundary_recall": val_metrics["boundary_recall"],
                "segment_accuracy": val_metrics["segment_accuracy"],
                "segment_macro_f1": val_metrics["segment_macro_f1"],
            }
            print(json.dumps(row), flush=True)
            history.append(row)
            score = val_metrics["boundary_f1"] + val_metrics["segment_macro_f1"]
            if score > best_metric:
                best_metric = score
                best_state = {
                    "boundary": {k: v.detach().cpu().clone() for k, v in boundary_head.state_dict().items()},
                    "emotion": {k: v.detach().cpu().clone() for k, v in emotion_head.state_dict().items()},
                }

    if best_state is not None:
        boundary_head.load_state_dict(best_state["boundary"])
        emotion_head.load_state_dict(best_state["emotion"])
    val_metrics = evaluate(model, boundary_head, emotion_head, val_loader, args, device, EMOTION_TOKENS)
    args.boundary_threshold = float(val_metrics["boundary_threshold"])
    test_metrics = evaluate(
        model,
        boundary_head,
        emotion_head,
        test_loader,
        args,
        device,
        EMOTION_TOKENS,
        fixed_boundary_threshold=args.boundary_threshold,
    )
    metrics = {
        "validation": val_metrics,
        "test": test_metrics,
        "history": history,
        "split_counts": split_info["counts"],
        "group_summary": group_summary,
    }
    save_artifacts(output_dir, boundary_head, emotion_head, args, encoder_dim, metrics)
    print(json.dumps({
        "validation": {k: v for k, v in val_metrics.items() if k not in {"segment_per_class", "examples"}},
        "test": {k: v for k, v in test_metrics.items() if k not in {"segment_per_class", "examples"}},
        "output_dir": str(output_dir),
    }, indent=2), flush=True)
    if args.required_boundary_f1 and val_metrics["boundary_f1"] < args.required_boundary_f1:
        raise RuntimeError(
            f"boundary_f1 {val_metrics['boundary_f1']:.4f} < required {args.required_boundary_f1:.4f}"
        )
    if args.required_segment_macro_f1 and val_metrics["segment_macro_f1"] < args.required_segment_macro_f1:
        raise RuntimeError(
            f"segment_macro_f1 {val_metrics['segment_macro_f1']:.4f} < required {args.required_segment_macro_f1:.4f}"
        )
    if args.push_to_hub:
        push_to_hub(output_dir, args)


if __name__ == "__main__":
    main()
