"""Read-only release guards; imports NumPy only when checking array contents."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import re


FOLDS = {
    "lls48": (
        (10, 15, 22, 27, 30, 39, 42, 46, 47),
        (4, 6, 7, 8, 18, 29, 32, 36, 37, 43),
        (2, 5, 13, 16, 19, 24, 28, 31, 45),
        (1, 3, 20, 23, 26, 35, 38, 40, 44, 48),
        (9, 11, 12, 14, 17, 21, 25, 33, 34, 41),
    ),
    "cholect45": (
        (79, 2, 51, 6, 25, 14, 66, 23, 50),
        (80, 32, 5, 15, 40, 47, 26, 48, 70),
        (31, 57, 36, 18, 52, 68, 10, 8, 73),
        (42, 29, 60, 27, 65, 75, 22, 49, 12),
        (78, 43, 62, 35, 74, 1, 56, 4, 13),
    ),
}
SCHEMAS = {"lls48": (195, 266), "cholect45": (100, 131)}
FRAME_COUNTS = {"lls48": 9560, "cholect45": 90489}
TEMPORAL_CFG = {"nhead": 4, "num_layers": 3, "dim_feedforward": 2048, "dropout": 0.1}


def expected_videos(dataset: str) -> list[set[str]]:
    return [{f"VID{number:02d}" for number in fold} for fold in FOLDS[dataset]]


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    dataset = config.get("dataset")
    if dataset not in SCHEMAS:
        raise ValueError(
            "Only the published cholect45 and lls48 configurations are supported"
        )
    fixed = {
        "schema_version": 1,
        "model_type": "mr2",
        "n_fold": 5,
        "triplet_size": SCHEMAS[dataset][0],
        "num_classes": SCHEMAS[dataset][1],
        "fusion": "learned",
        "use_spatial_norm": True,
        "ks_weights": True,
        "spatial_dropout": 0.1,
        "temporal_cfg": TEMPORAL_CFG,
        "optimizer": "AdamW",
        "weight_decay": 0.01,
        "special_lr_multiplier": 1.0,
        "scheduler": {"name": "StepLR", "step_size": 30, "gamma": 0.5},
        "seed": 42,
        "batch_size": 1,
        "ignore_null": True,
        "sd_ddof": 0,
    }
    for key, value in fixed.items():
        if config.get(key) != value:
            raise ValueError(
                f"{key} must match the canonical implementation: {value!r}"
            )
    if config.get("dim") not in (1024, 1536):
        raise ValueError("Feature dimension must be 1024 (SwinB) or 1536 (SwinL)")
    if config.get("ks") not in ([4, 5, 6], [4, 5, 6, 7]):
        raise ValueError("Unsupported published temporal scales")
    if config.get("epochs") != 30 or config.get("learning_rate") != 0.0003:
        raise ValueError(
            "Published temporal training uses 30 epochs and learning_rate=0.0003"
        )
    if not isinstance(config.get("num_workers"), int) or config["num_workers"] < 0:
        raise ValueError("num_workers must be a nonnegative integer")
    for key in ("model_name", "experiment_suffix"):
        value = config.get(key)
        if (
            not isinstance(value, str)
            or not value
            or any(c in value for c in "/\\:")
            or value in (".", "..")
        ):
            raise ValueError(f"Unsafe or missing {key}")
    if [set(x) for x in config.get("validation_videos", [])] != expected_videos(
        dataset
    ):
        raise ValueError(
            "Validation procedure memberships differ from the published five folds"
        )
    return config


def array_pair(feature_path: Path, label_path: Path, dim: int, num_classes: int) -> int:
    """Validate one pair without casting, modifying, or executing any model."""
    import numpy as np

    feature = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    label = np.load(label_path, mmap_mode="r", allow_pickle=False)
    if feature.ndim != 2 or feature.shape[1] != dim or feature.shape[0] < 1:
        raise ValueError(
            f"Feature shape must be (T>0, {dim}): {feature_path}: {feature.shape}"
        )
    if label.shape != (feature.shape[0], num_classes):
        raise ValueError(
            f"Label shape must be ({feature.shape[0]}, {num_classes}): {label_path}: {label.shape}"
        )
    if feature.dtype != np.dtype("float32") or not np.isfinite(feature).all():
        raise ValueError(f"Features must be finite float32 values: {feature_path}")
    if not np.isin(label, [0, 1]).all():
        raise ValueError(f"Labels must be binary: {label_path}")
    return int(feature.shape[0])


def _npy_names(path: Path) -> set[str]:
    if not path.is_dir():
        raise FileNotFoundError(f"Required directory not found: {path}")
    unexpected = [
        x.name for x in path.iterdir() if not x.is_file() or x.suffix != ".npy"
    ]
    if unexpected:
        raise ValueError(
            f"Only .npy files are allowed in {path}; unexpected: {unexpected[:5]}"
        )
    return {x.stem for x in path.iterdir()}


def preflight(
    config: dict, feature_root: Path, label_root: Path, *, inference_only: bool = False
) -> dict:
    """Check all published folds; array order semantics require an external manifest."""
    dataset = config["dataset"]
    folds = expected_videos(dataset)
    all_videos = set.union(*folds)
    labels = label_root / dataset / "triplet_label"
    if _npy_names(labels) != all_videos:
        raise ValueError(
            "Label file set must equal the published procedure set exactly"
        )
    features = feature_root / dataset / config["model_name"]
    if not features.is_dir() or {p.name for p in features.iterdir()} != {
        str(f) for f in range(5)
    }:
        raise ValueError(
            f"Feature experiment must contain exactly fold directories 0-4: {features}"
        )
    pairs = 0
    validation_frames = 0
    for fold, validation_videos in enumerate(folds):
        split_map = {"val": validation_videos}
        if not inference_only:
            split_map["train"] = all_videos - validation_videos
        for split, wanted in split_map.items():
            folder = features / str(fold) / split
            if _npy_names(folder) != wanted:
                raise ValueError(f"Wrong {split} video set in fold {fold}: {folder}")
            for video in sorted(wanted):
                count = array_pair(
                    folder / f"{video}.npy",
                    labels / f"{video}.npy",
                    config["dim"],
                    config["num_classes"],
                )
                if dataset == "lls48" and count % 5:
                    raise ValueError(
                        f"LLS48 frame count must be a multiple of five: {video}"
                    )
                if split == "val":
                    validation_frames += count
                pairs += 1
    if validation_frames != FRAME_COUNTS[dataset]:
        raise ValueError(
            f"Expected {FRAME_COUNTS[dataset]} validation frames, found {validation_frames}"
        )
    return {
        "status": "PASS_ARRAY_AND_FOLD_CHECKS",
        "dataset": dataset,
        "procedures": len(all_videos),
        "folds": 5,
        "validated_pairs": pairs,
        "validation_frames": validation_frames,
        "scope": "val only" if inference_only else "train and val",
        "limitation": "Shape and fold checks cannot certify feature/label row identity; use the release checksums and ordered frame manifest. LLS48 temporal inference output alone lacks clip identities for segmental evaluation.",
    }


def validate_checkpoint_fold(meta: dict, fold: int) -> None:
    if not isinstance(meta, dict):
        raise ValueError("Checkpoint metadata must be a dictionary")
    if type(fold) is not int or fold not in range(5):
        raise ValueError("Expected checkpoint fold must be an integer from 0 to 4")
    if type(meta.get("fold")) is not int or meta["fold"] != fold:
        raise ValueError(
            f"Checkpoint fold mismatch/missing: expected {fold}, got {meta.get('fold')!r}"
        )


def validate_checkpoint_features(meta: dict, manifest_sha256: str | None) -> None:
    """Bind newly keyed training weights to their input record before inference.

    Historical released model-only weights have no keyed-feature identity and
    remain supported without inventing such a provenance claim.
    """
    expected = meta.get("feature_manifest_sha256")
    if expected in (None, ""):
        return
    if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise ValueError("Checkpoint keyed-feature manifest SHA-256 is malformed")
    if manifest_sha256 != expected:
        raise ValueError(
            "Checkpoint requires its matching --feature-manifest; input SHA-256 is missing or differs"
        )


def validate_metadata(meta: dict, config: dict, *, fold: int | None = None) -> None:
    """Reject missing/guessed checkpoint metadata and mismatched configurations."""
    if not isinstance(meta, dict):
        raise ValueError("Checkpoint metadata must be a dictionary")
    if fold is not None:
        validate_checkpoint_fold(meta, fold)
    expected = {
        "model_type": config["model_type"],
        "embed_dim": config["dim"],
        "num_classes": config["num_classes"],
        "dataset": config["dataset"],
        "name": config["model_name"],
        "ks": config["ks"],
        "fusion": config["fusion"],
        "use_spatial_norm": config["use_spatial_norm"],
        "spatial_dropout": config["spatial_dropout"],
        "ks_weights": config["ks_weights"],
        "temporal_cfg": config["temporal_cfg"],
    }
    for key, value in expected.items():
        actual = meta.get(key)
        if key == "ks" and isinstance(actual, tuple):
            actual = list(actual)
        if actual != value:
            raise ValueError(
                f"Checkpoint metadata mismatch/missing {key}: expected {value!r}, got {actual!r}"
            )


def validate_prediction_csv(path: Path, dataset: str) -> dict:
    """Require probabilities, canonical triplet blocks, exact folds and unique row keys."""
    triplets, _ = SCHEMAS[dataset]
    wanted_folds = expected_videos(dataset)
    observed = [set() for _ in range(5)]
    seen_keys = set()
    count = 0
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)):
            raise ValueError("Duplicate CSV columns are not permitted")
        gt = [f"tri{i}" for i in range(triplets)]
        pred = [str(i) for i in range(triplets)]
        if not {"video", "fold", *gt, *pred}.issubset(fields):
            raise ValueError(
                "CSV must include video, fold, every triN GT and numeric prediction column"
            )
        for block in (gt, pred):
            start = fields.index(block[0])
            if fields[start : start + triplets] != block:
                raise ValueError(
                    "Canonical contiguous triplet column ordering is required"
                )
        keys = [
            k for k in ("image_id", "image_path", "frame", "frame_id") if k in fields
        ]
        if not keys:
            raise ValueError(
                "A frame identity column is required (image_id/image_path/frame/frame_id)"
            )
        for row in reader:
            count += 1
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed CSV row {count}")
            if row["fold"] not in {"0", "1", "2", "3", "4"}:
                raise ValueError(f"Invalid fold at row {count}")
            fold = int(row["fold"])
            video = row["video"]
            video = video[:-4] if video.endswith(".npy") else video
            if video not in wanted_folds[fold]:
                raise ValueError(
                    f"Wrong video/fold membership at row {count}: {video}, {fold}"
                )
            observed[fold].add(video)
            row_key = next((row[k] for k in keys if row[k].strip()), None)
            identity = (fold, video, row_key)
            if row_key is None or identity in seen_keys:
                raise ValueError(f"Missing/duplicate frame identity at row {count}")
            seen_keys.add(identity)
            for key in gt:
                if float(row[key]) not in (0.0, 1.0):
                    raise ValueError(f"Nonbinary GT at row {count}, {key}")
            for key in pred:
                value = float(row[key])
                if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                    raise ValueError(
                        f"Expected probabilities (no extra sigmoid), row {count}, {key}"
                    )
    if observed != wanted_folds:
        raise ValueError(
            "Prediction CSV does not cover every published procedure in all five folds"
        )
    if count != FRAME_COUNTS[dataset]:
        raise ValueError(
            f"Expected {FRAME_COUNTS[dataset]} prediction rows, found {count}"
        )
    return {"rows": count, "folds": 5, "status": "PASS_SCHEMA_PROBABILITY_FOLD_CHECKS"}
