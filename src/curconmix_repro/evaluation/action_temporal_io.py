"""Strict prediction schema and temporal-unit alignment for action metrics."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd


class AlignmentError(ValueError):
    """Raised when identity or ordering cannot be proven without guessing."""


def natural_key(value: object) -> Tuple[object, ...]:
    """Match the natural sort used by ``video_label_extraction.py``."""

    return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(value)))


def normalize_video_id(value: object) -> str:
    normalized = str(value).strip()
    if normalized.lower().endswith(".npy"):
        normalized = normalized[:-4]
    return normalized


def temporal_image_ordinal(video: object, image_id: object) -> int:
    """Convert ``VIDxx.npy_{t+1}`` to the zero-based video ordinal ``t``."""

    normalized_video = normalize_video_id(video)
    text = str(image_id).strip()
    match = re.fullmatch(r"(.+?)(?:\.npy)?_(\d+)", text, flags=re.IGNORECASE)
    if match is None or normalize_video_id(match.group(1)) != normalized_video:
        raise AlignmentError(f"invalid temporal image_id {image_id!r} for video {video!r}")
    one_based = int(match.group(2))
    if one_based < 1:
        raise AlignmentError(f"temporal image ordinal must be one-based, got {one_based}")
    return one_based - 1


def triplet_gt_columns(number_of_triplets: int) -> List[str]:
    return [f"tri{index}" for index in range(number_of_triplets)]


def prediction_columns(number_of_triplets: int) -> List[str]:
    return [str(index) for index in range(number_of_triplets)]


def require_columns(frame: pd.DataFrame, columns: Sequence[str], source: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        preview = ", ".join(missing[:10])
        raise AlignmentError(f"{source} is missing {len(missing)} required columns: {preview}")


def load_prediction_csv(path: Path, number_of_triplets: int) -> pd.DataFrame:
    """Load only official triplet GT/probabilities plus strict alignment metadata."""

    path = Path(path)
    header = pd.read_csv(path, nrows=0)
    required = ["video", "image_id", "fold"] + triplet_gt_columns(number_of_triplets) + prediction_columns(number_of_triplets)
    require_columns(header, required, str(path))
    optional = [column for column in ("clip_name", "image_path", "frame") if column in header.columns]
    frame = pd.read_csv(path, usecols=required + optional)

    gt = frame[triplet_gt_columns(number_of_triplets)].to_numpy()
    probabilities = frame[prediction_columns(number_of_triplets)].to_numpy(dtype=np.float64)
    if not np.all(np.isin(gt, (0, 1))):
        raise AlignmentError(f"{path} contains non-binary official triplet ground truth")
    if not np.all(np.isfinite(probabilities)):
        raise AlignmentError(f"{path} contains non-finite triplet predictions")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise AlignmentError(f"{path} predictions are not sigmoid probabilities in [0, 1]")
    folds = sorted(pd.unique(frame["fold"]).tolist())
    if folds != [0, 1, 2, 3, 4]:
        raise AlignmentError(f"{path} fold set is {folds}, expected [0, 1, 2, 3, 4]")
    return frame


def _master_with_video_ordinals(master: pd.DataFrame, number_of_triplets: int) -> pd.DataFrame:
    required = ["video", "image_path", "fold"] + triplet_gt_columns(number_of_triplets)
    require_columns(master, required, "master dataframe")
    result = master.copy()
    for column in ("video", "image_path"):
        if result[column].isna().any() or (result[column].astype(str).str.strip() == "").any():
            raise AlignmentError(f"master dataframe has missing {column}")
    result["_video_norm"] = result["video"].map(normalize_video_id)
    result["_natural_key"] = result["image_path"].map(natural_key)
    result = result.sort_values(["_video_norm", "_natural_key"], kind="mergesort").reset_index(drop=True)
    result["_video_ordinal"] = result.groupby("_video_norm", sort=False).cumcount()
    if result.duplicated(["_video_norm", "_video_ordinal"]).any():
        raise AlignmentError("master video ordinals are not unique")
    if result["image_path"].duplicated().any():
        raise AlignmentError("master image_path values are not unique")
    return result.drop(columns=["_natural_key"])


def rehydrate_temporal_metadata(
    temporal: pd.DataFrame,
    master: pd.DataFrame,
    number_of_triplets: int,
    dataset: str,
) -> pd.DataFrame:
    """Attach master metadata after proving one-to-one row and GT identity."""

    gt_columns = triplet_gt_columns(number_of_triplets)
    require_columns(temporal, ["video", "image_id", "fold"] + gt_columns, "temporal prediction")
    temporal_work = temporal.copy()
    temporal_work["_video_norm"] = temporal_work["video"].map(normalize_video_id)
    temporal_work["_video_ordinal"] = [
        temporal_image_ordinal(video, image_id)
        for video, image_id in zip(temporal_work["video"], temporal_work["image_id"])
    ]
    if temporal_work.duplicated(["_video_norm", "_video_ordinal"]).any():
        raise AlignmentError("temporal prediction has duplicate video ordinals")
    for video, group in temporal_work.groupby("_video_norm", sort=False):
        observed = sorted(group["_video_ordinal"].astype(int).tolist())
        expected = list(range(len(group)))
        if observed != expected:
            raise AlignmentError(f"{video} temporal ordinals are missing, duplicate, or nonconsecutive")

    master_work = _master_with_video_ordinals(master, number_of_triplets)
    master_columns = [
        "_video_norm",
        "_video_ordinal",
        "video",
        "image_path",
        "fold",
    ] + gt_columns
    for optional in ("clip_name", "image_id", "frame"):
        if optional in master_work.columns and optional not in master_columns:
            master_columns.append(optional)
    renamed = {
        column: f"_master_{column}"
        for column in master_columns
        if column not in {"_video_norm", "_video_ordinal"}
    }
    joined = temporal_work.merge(
        master_work[master_columns].rename(columns=renamed),
        on=["_video_norm", "_video_ordinal"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if len(joined) != len(temporal_work) or not (joined["_merge"] == "both").all():
        missing = int((joined["_merge"] != "both").sum())
        raise AlignmentError(f"{dataset} master mapping is not one-to-one; unmatched rows={missing}")
    if len(master_work) != len(joined):
        raise AlignmentError(f"{dataset} row count differs: prediction={len(joined)}, master={len(master_work)}")

    if not np.array_equal(
        joined[gt_columns].to_numpy(),
        joined[[f"_master_{column}" for column in gt_columns]].to_numpy(),
    ):
        mismatched_rows = int(
            np.any(
                joined[gt_columns].to_numpy()
                != joined[[f"_master_{column}" for column in gt_columns]].to_numpy(),
                axis=1,
            ).sum()
        )
        raise AlignmentError(f"{dataset} triplet GT mismatch in {mismatched_rows} rows")
    if not np.array_equal(joined["fold"].to_numpy(), joined["_master_fold"].to_numpy()):
        raise AlignmentError(f"{dataset} fold mapping differs from master")

    joined["video"] = joined["_master_video"].map(normalize_video_id)
    joined["image_path"] = joined["_master_image_path"]
    if "_master_image_id" in joined:
        joined["master_image_id"] = joined["_master_image_id"]
    if "_master_frame" in joined:
        joined["frame"] = joined["_master_frame"]
    if "_master_clip_name" in joined:
        joined["clip_name"] = joined["_master_clip_name"]

    cleanup = [column for column in joined.columns if column.startswith("_master_")]
    return joined.drop(columns=cleanup + ["_merge"])


def validate_temporal_units(
    frame: pd.DataFrame,
    unit_columns: Sequence[str],
    order_column: str,
    expected_unit_length: int | None = None,
) -> None:
    """Fail closed on missing, duplicate, unsorted, or nonconsecutive metadata."""

    required = list(unit_columns) + [order_column]
    require_columns(frame, required, "temporal metadata")
    if frame[required].isna().any().any():
        raise AlignmentError("temporal metadata contains missing values")
    if frame.duplicated(required).any():
        raise AlignmentError("temporal metadata contains duplicate frame keys")

    numeric_order = pd.to_numeric(frame[order_column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(numeric_order).all() or not np.equal(numeric_order, np.floor(numeric_order)).all():
        raise AlignmentError("temporal frame order must contain finite integers")

    row_keys = [
        tuple(row) if len(unit_columns) > 1 else row[0]
        for row in frame[list(unit_columns)].itertuples(index=False, name=None)
    ]
    closed_units = set()
    sentinel = object()
    previous = sentinel
    for key in row_keys:
        if key != previous:
            if key in closed_units:
                raise AlignmentError(f"temporal unit {key!r} appears in multiple noncontiguous blocks")
            if previous is not sentinel:
                closed_units.add(previous)
            previous = key

    for unit_key, group in frame.groupby(list(unit_columns), sort=False, dropna=False):
        order_values = pd.to_numeric(group[order_column], errors="raise").to_numpy(dtype=float)
        if not np.equal(order_values, np.floor(order_values)).all():
            raise AlignmentError(f"unit {unit_key!r} has fractional frame metadata")
        order = order_values.astype(int).tolist()
        if order != sorted(order):
            raise AlignmentError(f"unit {unit_key!r} has unsorted frame metadata")
        if order != list(range(order[0], order[0] + len(order))):
            raise AlignmentError(f"unit {unit_key!r} has missing/nonconsecutive frames")
        if expected_unit_length is not None and len(group) != expected_unit_length:
            raise AlignmentError(
                f"unit {unit_key!r} has {len(group)} frames, expected {expected_unit_length}"
            )
