"""Explicit, procedure-disjoint splits for new training runs (stdlib only).

The published outer folds are retained, but never used for checkpoint selection.
An investigator must supply the inner split before running any training stage.
This module deliberately does not choose validation procedures automatically.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from .validation import expected_videos


PROTOCOL = "nested_procedure_validation_v1"
PARTITIONS = ("train", "inner_validation", "outer_test")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key in training split: {key}")
        result[key] = value
    return result


def validate_training_split(
    record, dataset, expected_outer_folds=None, procedure_ids=None
):
    if not isinstance(record, dict):
        raise ValueError("Training split must be a JSON object")
    if type(record.get("schema_version")) is not int:
        raise ValueError("Training schema_version must be the integer 1")
    required = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "dataset": dataset,
        "selection_metric": "AP_IVT",
    }
    for key, value in required.items():
        if record.get(key) != value:
            raise ValueError(f"Training split mismatch: {key} must be {value!r}")
    if not re.fullmatch(r"[0-9a-f]{64}", str(record.get("annotation_sha256", ""))):
        raise ValueError("Training split requires the actual annotation_sha256")
    outer = expected_outer_folds
    if outer is None:
        outer = expected_videos(dataset)
    if isinstance(outer, dict):
        if set(outer) != {str(i) for i in range(5)}:
            raise ValueError("Expected outer folds must be exactly 0-4")
        outer = [outer[str(i)] for i in range(5)]
    if len(outer) != 5:
        raise ValueError("Exactly five outer folds are required")
    outer = [set(items) for items in outer]
    universe = set.union(*outer)
    if any(not items for items in outer) or sum(map(len, outer)) != len(universe):
        raise ValueError("Outer folds must be nonempty and mutually disjoint")
    if any(
        not isinstance(x, str) or not re.fullmatch(r"VID[0-9]+", x) for x in universe
    ):
        raise ValueError("Expected procedure IDs must use VID<number> identifiers")
    if procedure_ids is not None and set(procedure_ids) != universe:
        raise ValueError(
            "Annotation procedure IDs differ from the canonical outer folds"
        )
    folds = record.get("folds")
    if not isinstance(folds, dict) or set(folds) != {str(i) for i in range(5)}:
        raise ValueError("Training split must contain exactly folds 0-4")
    groups = record.get("patient_groups")
    if groups is not None:
        if not isinstance(groups, dict) or set(groups) != universe:
            raise ValueError("patient_groups must cover all procedures exactly")
        if any(
            not isinstance(v, str) or not v.strip() or v != v.strip()
            for v in groups.values()
        ):
            raise ValueError("patient_groups must contain nonempty anonymized IDs")
    for fold in range(5):
        partitions = folds[str(fold)]
        if not isinstance(partitions, dict) or set(partitions) != set(PARTITIONS):
            raise ValueError("Each fold needs train, inner_validation and outer_test")
        sets = {}
        for role in PARTITIONS:
            names = partitions[role]
            if (
                not isinstance(names, list)
                or not names
                or any(not isinstance(name, str) for name in names)
                or len(set(names)) != len(names)
            ):
                raise ValueError(
                    f"Empty, invalid or duplicate procedures: fold {fold} {role}"
                )
            sets[role] = set(names)
        if set.union(*sets.values()) != universe:
            raise ValueError(f"Missing or unknown procedures in fold {fold}")
        if sum(map(len, sets.values())) != len(universe):
            raise ValueError(
                f"Overlapping training/validation/test procedures in fold {fold}"
            )
        if sets["outer_test"] != outer[fold]:
            raise ValueError(
                f"Outer-test membership differs from published fold {fold}"
            )
        if groups is not None:
            group_sets = [{groups[v] for v in sets[role]} for role in PARTITIONS]
            if sum(map(len, group_sets)) != len(set.union(*group_sets)):
                raise ValueError(f"Patient overlap across partitions in fold {fold}")
    return record


def load_training_split(path, dataset, expected_outer_folds=None, procedure_ids=None):
    if path is None or str(path).strip() in ("", "None"):
        raise ValueError(
            "New training requires an explicit independent --training-split"
        )
    record = load_json(path)
    return validate_training_split(record, dataset, expected_outer_folds, procedure_ids)


def load_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream, object_pairs_hook=_unique_object)


def split_digest(record):
    data = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def fold_partition(record, fold):
    if type(fold) is not int or fold not in range(5):
        raise ValueError("Fold must be an integer from 0 through 4")
    validate_training_split(record, record.get("dataset"))
    return {name: list(record["folds"][str(fold)][name]) for name in PARTITIONS}


def checkpoint_provenance(
    record, fold, *, stage, selection_partition="inner_validation"
):
    if selection_partition not in ("inner_validation", "fixed_training_schedule"):
        raise ValueError(
            "Checkpoint selection cannot use outer-test or training-set scores"
        )
    partitions = fold_partition(record, fold)
    return {
        "training_protocol": PROTOCOL,
        "training_split_sha256": split_digest(record),
        "annotation_sha256": record["annotation_sha256"],
        "fold": fold,
        "stage": stage,
        "selection_partition": selection_partition,
        "selection_metric": (
            "AP_IVT" if selection_partition == "inner_validation" else None
        ),
        **{f"{key}_procedures": names for key, names in partitions.items()},
    }
