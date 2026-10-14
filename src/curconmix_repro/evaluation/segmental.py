"""Class-wise multi-label segmental F1 for surgical action components.

The implementation follows the conventional chronological greedy matching
used by segmental F1, applied independently to every temporal unit and class.
Intervals are half-open ``[start, end)`` and simultaneous classes are allowed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True, order=True)
class Event:
    """One maximal contiguous positive run using an exclusive end index."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError(f"invalid event interval [{self.start}, {self.end})")

    @property
    def duration(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class Counts:
    """Raw event counts; these are summed before pooled-micro scoring."""

    tp: int = 0
    fp: int = 0
    fn: int = 0

    def __post_init__(self) -> None:
        if min(self.tp, self.fp, self.fn) < 0:
            raise ValueError("event counts must be non-negative")

    def __add__(self, other: "Counts") -> "Counts":
        if not isinstance(other, Counts):
            return NotImplemented
        return Counts(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn)


@dataclass(frozen=True)
class Score:
    counts: Counts
    precision: float
    recall: float
    f1: float


COMPONENT_TO_BANK_COLUMN: Mapping[str, int] = {
    "ivt": 0,
    "i": 1,
    "v": 2,
    "t": 3,
    "iv": 4,
    "it": 5,
}


def extract_events(binary_stream: Sequence[object]) -> Tuple[Event, ...]:
    """Extract maximal positive runs from a one-dimensional binary stream."""

    values = np.asarray(binary_stream)
    if values.ndim != 1:
        raise ValueError(f"binary_stream must be 1-D, got shape {values.shape}")
    if values.size == 0:
        return ()
    if not np.all(np.isfinite(values.astype(float, copy=False))):
        raise ValueError("binary_stream contains a non-finite value")
    if not np.all(np.isin(values, (0, 1, False, True))):
        raise ValueError("binary_stream must contain only binary values")

    active = values.astype(bool, copy=False)
    padded = np.pad(active.astype(np.int8), (1, 1), constant_values=0)
    transitions = np.diff(padded)
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    return tuple(Event(int(start), int(end)) for start, end in zip(starts, ends))


def temporal_iou(first: Event, second: Event) -> float:
    """Return temporal intersection-over-union for half-open intervals."""

    intersection = max(0, min(first.end, second.end) - max(first.start, second.start))
    union = first.duration + second.duration - intersection
    return float(intersection / union) if union else 0.0


def match_events(
    ground_truth: Iterable[Event],
    predictions: Iterable[Event],
    iou_threshold: float,
) -> Counts:
    """Chronologically match predictions to the best GT event.

    Every prediction selects its maximum-IoU event from *all* GT events. An IoU
    tie is resolved in favor of the earliest GT event. The prediction is a true
    positive only when that selected event is still unmatched and its IoU is at
    least the threshold; a previously matched selection is a false positive,
    with no fallback to a lower-IoU unmatched event.
    """

    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be within [0, 1]")
    gt = sorted(tuple(ground_truth), key=lambda event: (event.start, event.end))
    pred = sorted(tuple(predictions), key=lambda event: (event.start, event.end))
    matched = set()
    true_positives = 0

    for predicted in pred:
        if not gt:
            continue
        best_index = min(
            range(len(gt)),
            key=lambda index: (
                -temporal_iou(gt[index], predicted),
                gt[index].start,
                gt[index].end,
                index,
            ),
        )
        best_iou = temporal_iou(gt[best_index], predicted)
        if best_iou >= iou_threshold and best_index not in matched:
            true_positives += 1
            matched.add(best_index)

    return Counts(
        tp=true_positives,
        fp=len(pred) - true_positives,
        fn=len(gt) - true_positives,
    )


def metrics_from_counts(counts: Counts) -> Score:
    """Convert raw counts to metrics; an all-empty score is undefined/NaN."""

    precision_denominator = counts.tp + counts.fp
    recall_denominator = counts.tp + counts.fn
    precision = counts.tp / precision_denominator if precision_denominator else np.nan
    recall = counts.tp / recall_denominator if recall_denominator else np.nan
    f1_denominator = 2 * counts.tp + counts.fp + counts.fn
    f1 = 2 * counts.tp / f1_denominator if f1_denominator else np.nan
    return Score(counts, float(precision), float(recall), float(f1))


def mapping_ids(mapping_bank: np.ndarray, component: str) -> np.ndarray:
    """Return sorted original component IDs used by the official mapping."""

    if component not in COMPONENT_TO_BANK_COLUMN:
        raise ValueError(f"unsupported component: {component}")
    bank = np.asarray(mapping_bank)
    if bank.ndim != 2 or bank.shape[1] < 6:
        raise ValueError(f"mapping_bank must have shape [N, >=6], got {bank.shape}")
    return np.unique(bank[:, COMPONENT_TO_BANK_COLUMN[component]]).astype(int)


def decompose_triplets(
    triplet_values: np.ndarray,
    mapping_bank: np.ndarray,
    component: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply the official max/OR triplet-to-component decomposition."""

    values = np.asarray(triplet_values)
    bank = np.asarray(mapping_bank)
    if values.ndim != 2:
        raise ValueError(f"triplet_values must be [T, N], got {values.shape}")
    if bank.ndim != 2 or bank.shape[0] != values.shape[1] or bank.shape[1] < 6:
        raise ValueError(
            f"mapping_bank shape {bank.shape} is incompatible with values {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("triplet_values contain non-finite values")
    if component == "ivt":
        return values.copy(), mapping_ids(bank, component)

    column = COMPONENT_TO_BANK_COLUMN.get(component)
    if column is None:
        raise ValueError(f"unsupported component: {component}")
    ids = mapping_ids(bank, component)
    output = np.empty((values.shape[0], len(ids)), dtype=values.dtype)
    for dense_index, mapping_id in enumerate(ids):
        source_indices = np.flatnonzero(bank[:, column] == mapping_id)
        output[:, dense_index] = np.max(values[:, source_indices], axis=1)
    return output, ids


def evaluate_multilabel_unit(
    ground_truth: np.ndarray,
    probabilities: np.ndarray,
    iou_thresholds: Sequence[float] = (0.10, 0.25, 0.50),
    probability_threshold: float = 0.5,
    class_mask: Optional[Sequence[bool]] = None,
) -> Dict[float, Dict[str, object]]:
    """Evaluate one temporal unit independently for every retained class."""

    gt = np.asarray(ground_truth)
    prob = np.asarray(probabilities)
    if gt.ndim != 2 or prob.ndim != 2 or gt.shape != prob.shape:
        raise ValueError(f"ground_truth/probabilities must share [T,C] shape, got {gt.shape}/{prob.shape}")
    if not np.all(np.isin(gt, (0, 1, False, True))):
        raise ValueError("ground_truth must be binary")
    if not np.all(np.isfinite(prob)):
        raise ValueError("probabilities contain non-finite values")
    if np.any((prob < 0.0) | (prob > 1.0)):
        raise ValueError("probabilities must be within [0, 1]")
    if not 0.0 <= probability_threshold <= 1.0:
        raise ValueError("probability_threshold must be within [0, 1]")

    if class_mask is None:
        mask = np.ones(gt.shape[1], dtype=bool)
    else:
        mask = np.asarray(class_mask, dtype=bool)
        if mask.shape != (gt.shape[1],):
            raise ValueError(f"class_mask must have shape ({gt.shape[1]},), got {mask.shape}")

    predicted = prob >= probability_threshold
    results: Dict[float, Dict[str, object]] = {}
    for iou_threshold in iou_thresholds:
        pooled = Counts()
        per_class: Dict[int, Dict[str, object]] = {}
        for class_index in np.flatnonzero(mask):
            gt_events = extract_events(gt[:, class_index])
            pred_events = extract_events(predicted[:, class_index])
            counts = match_events(gt_events, pred_events, float(iou_threshold))
            pooled = pooled + counts
            per_class[int(class_index)] = {
                "counts": counts,
                "score": metrics_from_counts(counts),
                "gt_events": gt_events,
                "pred_events": pred_events,
            }
        results[float(iou_threshold)] = {
            "counts": pooled,
            "score": metrics_from_counts(pooled),
            "per_class": per_class,
        }
    return results


def headline_ivt_class_mask(dataset: str, number_of_triplets: int) -> np.ndarray:
    """Return the manuscript-consistent IVT mask excluding only null triplets."""

    normalized = dataset.strip().lower()
    expected, null_count = {
        "cholect45": (100, 6),
        "cholec45": (100, 6),
        "lls48": (195, 16),
    }.get(normalized, (None, None))
    if expected is None:
        raise ValueError(f"unsupported dataset: {dataset}")
    if number_of_triplets != expected:
        raise ValueError(f"{dataset} requires {expected} triplets, got {number_of_triplets}")
    mask = np.ones(number_of_triplets, dtype=bool)
    mask[-null_count:] = False
    return mask


def count_events_across_units(
    unit_streams: Sequence[np.ndarray],
    class_index: int = 0,
) -> int:
    """Count events with a mandatory reset at each video/clip boundary."""

    total = 0
    for stream in unit_streams:
        values = np.asarray(stream)
        if values.ndim == 2:
            values = values[:, class_index]
        total += len(extract_events(values))
    return total
