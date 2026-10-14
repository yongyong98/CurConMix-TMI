"""Build safe-loadable training checkpoints without changing metric semantics."""

from __future__ import annotations

import math
from numbers import Integral, Real


def _plain_metadata(value):
    """Convert NumPy numeric scalars to primitives; reject opaque Python objects."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("Checkpoint metadata must contain finite numbers")
        return result
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("Checkpoint metadata keys must be strings")
        return {key: _plain_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_metadata(item) for item in value]
    raise ValueError(f"Unsupported checkpoint metadata type: {type(value).__name__}")


def training_checkpoint(
    *,
    epoch,
    model_state,
    optimizer_state,
    scheduler_state,
    best_map,
    arguments,
    metadata,
    fold,
):
    """Keep tensor/optimizer state intact; serialize scalar metadata safely.

    This changes the storage representation only. The caller retains the original
    metric for checkpoint selection and reporting. No model is executed here.
    """
    if type(fold) is not int or fold not in range(5):
        raise ValueError("Checkpoint fold must be an integer from 0 through 4")
    if type(epoch) is not int or epoch < 1:
        raise ValueError("Checkpoint epoch must be a positive integer")
    score = float(best_map)
    if not math.isfinite(score):
        raise ValueError("Best validation AP must be finite")
    meta = _plain_metadata(metadata)
    if not isinstance(meta, dict):
        raise ValueError("Checkpoint metadata must be a dictionary")
    if "fold" in meta and (type(meta["fold"]) is not int or meta["fold"] != fold):
        raise ValueError("Conflicting checkpoint fold metadata")
    meta["fold"] = fold
    return {
        "epoch": epoch,
        "model": model_state,
        "optimizer": optimizer_state,
        "scheduler": scheduler_state,
        "best_mAP": score,
        "args": _plain_metadata(arguments),
        "meta": meta,
    }
