"""Record procedure iteration without claiming to recover historical row order.

The original loader consumed the filesystem's enumeration order. A new training
run captures that order once, then reuses this explicit record for every epoch.
Replaying the record on another machine is independent of its directory order.
This records procedure order only, not the provenance/order of frames in arrays.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from curconmix_repro.validation import expected_videos


def validate_order(record: dict, dataset: str, model_name: str) -> dict:
    if not isinstance(record, dict):
        raise ValueError("Procedure-order manifest must be a dictionary")
    identity = {"schema_version": 1, "dataset": dataset, "model_name": model_name}
    for key, expected in identity.items():
        if record.get(key) != expected:
            raise ValueError(f"Procedure-order manifest mismatch: {key}")
    folds = record.get("folds")
    if not isinstance(folds, dict) or set(folds) != {str(i) for i in range(5)}:
        raise ValueError("Procedure-order manifest must contain exactly folds 0-4")
    expected_folds = expected_videos(dataset)
    all_videos = set.union(*expected_folds)
    for fold, validation in enumerate(expected_folds):
        splits = folds[str(fold)]
        if not isinstance(splits, dict) or set(splits) != {"train", "val"}:
            raise ValueError("Each procedure-order fold must contain train and val")
        for split, wanted in (("train", all_videos - validation), ("val", validation)):
            names = splits[split]
            if (
                not isinstance(names, list)
                or any(not isinstance(name, str) for name in names)
                or len(names) != len(wanted)
                or set(names) != wanted
            ):
                raise ValueError(f"Wrong/duplicate procedures in fold {fold} {split}")
    return record


def capture_order(config: dict, feature_root: Path) -> dict:
    """Snapshot observed directory order; never reorder or read array contents."""
    dataset, model_name = config["dataset"], config["model_name"]
    root = feature_root / dataset / model_name
    folds = {}
    for fold in range(5):
        splits = {}
        for split in ("train", "val"):
            folder = root / str(fold) / split
            names = os.listdir(folder)
            if any(
                not name.endswith(".npy") or not (folder / name).is_file()
                for name in names
            ):
                raise ValueError(f"Only procedure .npy files are allowed: {folder}")
            splits[split] = [name[:-4] for name in names]
        folds[str(fold)] = splits
    return validate_order(
        {
            "schema_version": 1,
            "dataset": dataset,
            "model_name": model_name,
            "order_policy": "captured_directory_enumeration_no_historical_claim",
            "scope": "procedure_iteration_only_not_frame_identity",
            "folds": folds,
        },
        dataset,
        model_name,
    )


def load_order(path: Path, dataset: str, model_name: str) -> dict:
    with path.open(encoding="utf-8") as stream:
        return validate_order(json.load(stream), dataset, model_name)


def order_sha256(record: dict) -> str:
    canonical = json.dumps(
        record, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def ordered_files(
    record: dict, dataset: str, model_name: str, fold: int, split: str, folder: Path
) -> list[str]:
    validate_order(record, dataset, model_name)
    if type(fold) is not int or fold not in range(5) or split not in ("train", "val"):
        raise ValueError("Expected one canonical fold and train/val split")
    filenames = [f"{name}.npy" for name in record["folds"][str(fold)][split]]
    actual = os.listdir(folder)
    if set(actual) != set(filenames) or any(
        not (folder / name).is_file() for name in filenames
    ):
        raise ValueError(f"Procedure-order manifest disagrees with files: {folder}")
    return filenames
