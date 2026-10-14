"""Validate new-run feature lineage without loading a model or outer-test arrays."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from curconmix_repro.training_split import (
    PROTOCOL,
    fold_partition,
    load_json,
    split_digest,
    validate_training_split,
)
from curconmix_repro.validation import SCHEMAS, array_pair


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def label_columns(dataset):
    sizes = (195, 17, 16, 38) if dataset == "lls48" else (100, 6, 10, 15)
    return [
        f"{prefix}{i}"
        for prefix, size in zip(("tri", "inst", "v", "t"), sizes)
        for i in range(size)
    ]


def annotation_key(index, row):
    return {
        "row_index": index,
        **{
            key: row[key]
            for key in ("video", "image_id", "image_path", "clip_name")
            if key in row
        },
    }


def training_order(config, split):
    validate_training_split(split, config["dataset"])
    return {
        "schema_version": 2,
        "dataset": config["dataset"],
        "model_name": config["model_name"],
        "training_split_sha256": split_digest(split),
        "order_policy": "explicit_inner_split_order",
        "folds": {
            str(fold): {
                role: fold_partition(split, fold)[role]
                for role in ("train", "inner_validation")
            }
            for fold in range(5)
        },
    }


def load_training_order(path, config, split):
    record = load_json(path)
    if record != training_order(config, split):
        raise ValueError("Training procedure order must match the explicit inner split")
    return record


def validate_encoder(encoder, split, fold, config=None):
    """Bind a newly trained student checkpoint to its actual selection protocol."""
    checkpoint = Path(encoder["checkpoint_path"])
    sidecar = Path(encoder["metadata_sidecar_path"])
    record = load_json(sidecar)
    actual = file_sha256(checkpoint)
    if actual != encoder.get("sha256") or actual != record.get("checkpoint_sha256"):
        raise ValueError("Encoder checkpoint/sidecar SHA-256 mismatch")
    if encoder.get("metadata_sidecar_sha256") != file_sha256(sidecar):
        raise ValueError("Encoder sidecar SHA-256 mismatch")
    if (
        checkpoint.stat().st_size != encoder.get("size_bytes")
        or record.get("size_bytes") != checkpoint.stat().st_size
    ):
        raise ValueError("Encoder checkpoint size mismatch")
    if type(record.get("schema_version")) is not int or record["schema_version"] != 1:
        raise ValueError("Unsupported encoder provenance schema")
    meta = record.get("meta", {})
    if type(meta.get("fold")) is not int:
        raise ValueError("Encoder fold must be an integer")
    expected = {
        "training_protocol": PROTOCOL,
        "training_split_sha256": split_digest(split),
        "annotation_sha256": split["annotation_sha256"],
        "dataset": split["dataset"],
        "fold": fold,
        "stage": "student",
        "selection_partition": "inner_validation",
        "selection_metric": "AP_IVT",
    }
    expected.update(
        {f"{role}_procedures": ids for role, ids in fold_partition(split, fold).items()}
    )
    if any(meta.get(key) != value for key, value in expected.items()):
        raise ValueError(
            "Encoder was not trained/selected with this inner-validation split"
        )
    if config is not None:
        # New spatial training is currently B224 only. Logical feature-directory
        # names must not relabel B224 arrays as a B384/L224 experiment.
        model_names = {
            "lls48": "swin_bas_266_LLS48_SwinB_Curriculum_Feature_Mixup_e7_Student",
            "cholect45": "swin_bas_131_FeatureMixupBatch_CurriculumSupcon_Student_mixup_alpha0_4_e3",
        }
        dataset = split["dataset"]
        if (
            config.get("dataset") != dataset
            or config.get("model_name") != model_names.get(dataset)
            or config.get("dim") != 1024
        ):
            raise ValueError(
                "New-protocol encoder features support packaged B224 configurations only; B384/L224 are not implemented"
            )
        architecture = {
            "model_class": "TripletModel",
            "model_name": "swin_base_patch4_window7_224",
            "input_resolution": 224,
            "feature_dim": 1024,
            "num_classes": SCHEMAS[dataset][1],
            "triplet_size": SCHEMAS[dataset][0],
        }
        if any(meta.get(key) != value for key, value in architecture.items()) or any(
            type(meta.get(key)) is not int
            for key in (
                "input_resolution",
                "feature_dim",
                "num_classes",
                "triplet_size",
            )
        ):
            raise ValueError(
                "Encoder architecture metadata does not match the B224 feature configuration"
            )
    return meta


def _checked_path(root, record, expected_path):
    root = Path(root).resolve()
    if record.get("path") != expected_path:
        raise ValueError("Training array path/partition identity mismatch")
    path = (root / expected_path).resolve()
    if root not in path.parents:
        raise ValueError("Training array escapes its declared root")
    if path.stat().st_size != record.get("size_bytes") or file_sha256(
        path
    ) != record.get("sha256"):
        raise ValueError(f"Training array size/SHA-256 mismatch: {expected_path}")
    return path


def validate_training_features(
    path,
    config,
    feature_root,
    label_root,
    split,
    *,
    fold=None,
    include_outer_test=False,
):
    """Check only fit/selection arrays for requested folds; never open outer-test arrays.

    This verifies new encoder provenance and annotation-aligned rows. It is not
    an inference test or a claim to reproduce previously reported paper scores.
    """
    import numpy as np

    dataset = config["dataset"]
    validate_training_split(split, dataset)
    if fold is not None and (type(fold) is not int or fold not in range(5)):
        raise ValueError("Fold must be 0 through 4")
    path = Path(path)
    raw = path.read_bytes()
    manifest = load_json(path)
    if type(manifest.get("schema_version")) is not int:
        raise ValueError("Training feature schema_version must be integer 2")
    identity = {
        "schema_version": 2,
        "kind": "inner_validation_training_features",
        "dataset": dataset,
        "model_name": config["model_name"],
        "annotation_sha256": split["annotation_sha256"],
        "training_split_sha256": split_digest(split),
    }
    if any(manifest.get(key) != value for key, value in identity.items()):
        raise ValueError("Training feature manifest identity/split mismatch")
    if set(manifest.get("folds", {})) != {str(i) for i in range(5)}:
        raise ValueError("Training features require exactly five fold records")
    annotation = Path(manifest["annotation_path"])
    if file_sha256(annotation) != split["annotation_sha256"]:
        raise ValueError("Training annotation SHA-256 mismatch")
    columns = label_columns(dataset)
    by_video = {}
    with annotation.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if len(reader.fieldnames or []) != len(set(reader.fieldnames or [])):
            raise ValueError("Duplicate annotation column names are not allowed")
        if not {"video", "image_id", *columns}.issubset(reader.fieldnames or []):
            raise ValueError(
                "Training annotation lacks original keys or canonical label columns"
            )
        for index, row in enumerate(reader):
            by_video.setdefault(row["video"], []).append((index, row))
    validate_training_split(split, dataset, procedure_ids=by_video)
    checked = 0
    for k in range(5) if fold is None else [fold]:
        entry = manifest["folds"][str(k)]
        validate_encoder(entry["encoder"], split, k, config=config)
        partition = fold_partition(split, k)
        roles = (
            ("train", "inner_validation", "outer_test")
            if include_outer_test
            else ("train", "inner_validation")
        )
        for role in roles:
            if set(entry.get(role, {})) != set(partition[role]):
                raise ValueError(
                    f"Training feature membership mismatch: fold {k} {role}"
                )
            for video in partition[role]:
                item = entry[role][video]
                storage = "val" if role == "outer_test" else "train"
                feature = _checked_path(
                    feature_root,
                    item["feature"],
                    f"{dataset}/{config['model_name']}/{k}/{storage}/{video}.npy",
                )
                label = _checked_path(
                    label_root, item["label"], f"{dataset}/triplet_label/{video}.npy"
                )
                count = array_pair(feature, label, config["dim"], SCHEMAS[dataset][1])
                source_rows = by_video[video]
                keys = [annotation_key(index, row) for index, row in source_rows]
                if count != len(keys) or item.get("frame_keys") != keys:
                    raise ValueError(
                        "Training features differ from original annotation row order"
                    )
                ids = [row["image_id"] for _, row in source_rows]
                if len(set(ids)) != len(ids):
                    raise ValueError("Duplicate frame keys within a training procedure")
                truth = np.asarray(
                    [[int(row[column]) for column in columns] for _, row in source_rows]
                )
                if not np.array_equal(np.load(label, allow_pickle=False), truth):
                    raise ValueError(
                        "Training labels differ from the recorded annotation"
                    )
                checked += 1
    if (
        file_sha256(annotation) != split["annotation_sha256"]
        or path.read_bytes() != raw
    ):
        raise ValueError("Training input manifest/annotation changed during validation")
    return {
        "status": "PASS_INNER_VALIDATION_FEATURE_LINEAGE",
        "training_split_sha256": split_digest(split),
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "checked_fit_selection_pairs": checked,
        "outer_test_arrays_opened": bool(include_outer_test),
        "numerical_paper_reproduction": False,
    }


def restore_training_prediction_keys(frame, manifest, label_root, *, fold):
    """Restore held-out LLS48 keys using the existing full-GT alignment guard.

    Caller must validate this version-2 manifest including outer-test arrays first.
    """
    if manifest["dataset"] != "lls48":
        raise ValueError("New-run key restoration currently supports LLS48 only")
    from curconmix_release.feature_contract import restore_prediction_keys

    procedures = {
        video: {"frames": item["frame_keys"], "labels": item["label"]}
        for video, item in manifest["folds"][str(fold)]["outer_test"].items()
    }
    return restore_prediction_keys(
        frame, {"procedures": procedures}, label_root, fold=fold
    )


def guard_training_loaders(split, fold, train_loader, validation_loader):
    """Refuse a direct training call using a held-out test loader."""
    partition = fold_partition(split, fold)
    for role, loader in (
        ("train", train_loader),
        ("inner_validation", validation_loader),
    ):
        data = loader.dataset
        if getattr(data, "selection_role", None) != role or getattr(
            data, "training_split_sha256", None
        ) != split_digest(split):
            raise ValueError(
                "Training loader is missing matching inner-split provenance"
            )
        if getattr(data, "video_names", None) != [f"{v}.npy" for v in partition[role]]:
            raise ValueError("Training loader contains wrong/outer-test procedures")
