"""Plan or explicitly export LLS48 B224 features from new split-matched students.

The default is a model-free, write-free plan. --preflight hashes existing inputs
without importing a model runtime. Only --execute opts into image decoding and
model forward passes. This is the public 195-triplet schema, not the separate
unresolved AIS 193-class experiment and not historical numerical reproduction.
"""

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import shutil
import sys
import tempfile

from curconmix_repro.temporal.training_inputs import (
    annotation_key,
    validate_training_features,
)
from curconmix_repro.training_split import (
    fold_partition,
    load_training_split,
    split_digest,
)

from .feature_contract import (
    LIMITATION,
    RUNTIME_VERSIONS,
    SOURCE_HASH_POLICY,
    config_identity,
    preprocessing_spec,
    source_code_hash,
)
from .feature_export import (
    _extract_rows,
    _new_destination,
    _publish_without_replace,
    _runtime,
    _save_array,
    _validate_config,
)
from .lls48 import (
    LABEL_COLUMNS,
    MASTER_RELATIVE,
    inspect_annotations,
    json_bytes,
    sha256_file,
    source_file,
    verify_dataset,
)
from .spatial_checkpoint import _safe_load, _strict_model_load
from .spatial_protocol import sidecar_path, validate_meta, verify_sidecar


DEFAULT_PATTERN = "fold{fold}_swin_bas_266_CurConMix_B224_Student.pth"
ARCHITECTURE = {
    "dataset": "lls48",
    "model_class": "TripletModel",
    "model_name": "swin_base_patch4_window7_224",
    "input_resolution": 224,
    "feature_dim": 1024,
    "num_classes": 266,
    "triplet_size": 195,
}
SOURCE_FILES = (
    "legacy/spatial/lls48/models.py",
    "legacy/spatial/lls48/augmentation.py",
    "src/curconmix_release/training_feature_export.py",
    "src/curconmix_release/feature_export.py",
    "src/curconmix_release/spatial_protocol.py",
    "src/curconmix_repro/training_split.py",
    "src/curconmix_repro/temporal/training_inputs.py",
)


def _checkpoint_paths(root, pattern):
    if (
        not isinstance(pattern, str)
        or pattern.count("{fold}") != 1
        or any(character in pattern for character in ("/", "\\", ":"))
        or "{" in pattern.replace("{fold}", "")
        or "}" in pattern.replace("{fold}", "")
        or not pattern.endswith(".pth")
    ):
        raise ValueError("Checkpoint pattern must be a .pth basename with one {fold}")
    root = Path(root).resolve()
    paths = [root / pattern.format(fold=fold) for fold in range(5)]
    if any(path.resolve().parent != root or path.is_symlink() for path in paths):
        raise ValueError("Checkpoint files must remain directly within checkpoint-root")
    return paths


def plan_export(
    config,
    training_split,
    dataset_root,
    checkpoint_root,
    output,
    *,
    batch_size=16,
    device="cpu",
    checkpoint_pattern=DEFAULT_PATTERN,
    include_outer_test=False,
):
    if config.get("dataset") != "lls48":
        raise ValueError("New training feature export currently supports LLS48 only")
    _validate_config(config, "lls48-spatial-b224")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Batch size must be a positive integer")
    if device not in ("cpu", "cuda"):
        raise ValueError("Device must be cpu or cuda")
    if training_split is None or not str(training_split).strip():
        raise ValueError("An explicit --training-split is required")
    if type(include_outer_test) is not bool:
        raise ValueError("include_outer_test must be boolean")
    checkpoints = _checkpoint_paths(checkpoint_root, checkpoint_pattern)
    output = Path(output).absolute()
    return {
        "mode": "plan_only",
        "dataset": "lls48",
        "triplet_size": 195,
        "spatial_backbone": ARCHITECTURE["model_name"],
        "model_name": config["model_name"],
        "training_split": str(Path(training_split).absolute()),
        "dataset_root": str(Path(dataset_root).absolute()),
        "checkpoints": [str(path) for path in checkpoints],
        "new_output_root": str(output),
        "feature_root": str(output / "features"),
        "label_root": str(output / "labels"),
        "feature_manifest": str(output / "training_feature_manifest.json"),
        "folds": list(range(5)),
        "export_partitions": ["train", "inner_validation"]
        + (["outer_test"] if include_outer_test else []),
        "selection_partition": "inner_validation",
        "outer_test_arrays_used_for_selection": False,
        "include_outer_test": include_outer_test,
        "batch_size": batch_size,
        "device": device,
        "preprocessing": preprocessing_spec(224),
        "model_execution": False,
        "numerical_paper_reproduction": False,
        "limitation": LIMITATION,
    }


def _checked_metadata(info, record, fold):
    meta = info.get("meta")
    validate_meta(meta, record, fold, "student")
    if any(meta.get(key) != value for key, value in ARCHITECTURE.items()):
        raise ValueError("Student architecture metadata must match LLS48 SwinB224")
    return meta


def _inspect_inputs(config, training_split, dataset_root, checkpoint_root, pattern):
    split_file_hash = sha256_file(training_split)
    record = load_training_split(training_split, "lls48")
    rows, _, summary = inspect_annotations(dataset_root)
    annotation = source_file(dataset_root, MASTER_RELATIVE)
    if summary["annotation_sha256"] != record["annotation_sha256"]:
        raise ValueError(
            "Training split does not match the canonical annotation SHA-256"
        )
    record = load_training_split(
        training_split, "lls48", procedure_ids={row["video"] for row in rows}
    )
    grouped = {}
    seen = set()
    for index, row in enumerate(rows):
        key = (row["video"], row["image_id"])
        if key in seen:
            raise ValueError("Duplicate frame key within a procedure")
        seen.add(key)
        source_file(dataset_root, f"data/{row['image_path']}")
        grouped.setdefault(row["video"], []).append((index, row))
    checkpoints = []
    for fold, checkpoint in enumerate(_checkpoint_paths(checkpoint_root, pattern)):
        sidecar = sidecar_path(checkpoint)
        if not sidecar.is_file():
            raise ValueError(
                "Missing training provenance; legacy weights are not accepted"
            )
        if (
            sidecar.is_symlink()
            or sidecar.resolve().parent != checkpoint.parent.resolve()
        ):
            raise ValueError("Checkpoint provenance must remain within checkpoint-root")
        sidecar_hash = sha256_file(sidecar)
        info = verify_sidecar(checkpoint, record, fold, "student")
        _checked_metadata(info, record, fold)
        if sha256_file(sidecar) != sidecar_hash:
            raise ValueError("Checkpoint provenance changed during inspection")
        entry = {
            "checkpoint_path": str(checkpoint.resolve()),
            "sha256": info["checkpoint_sha256"],
            "size_bytes": info["size_bytes"],
            "metadata_sidecar_path": str(sidecar.resolve()),
            "metadata_sidecar_sha256": sidecar_hash,
        }
        checkpoints.append((checkpoint, info, entry))
    if sha256_file(training_split) != split_file_hash:
        raise ValueError("Training split changed during inspection")
    if sha256_file(annotation) != record["annotation_sha256"]:
        raise ValueError("Annotation changed during inspection")
    return record, grouped, annotation, summary, checkpoints


def preflight_export(
    config,
    training_split,
    dataset_root,
    checkpoint_root,
    output,
    *,
    batch_size=16,
    device="cpu",
    checkpoint_pattern=DEFAULT_PATTERN,
    include_outer_test=False,
):
    plan = plan_export(
        config,
        training_split,
        dataset_root,
        checkpoint_root,
        output,
        batch_size=batch_size,
        device=device,
        checkpoint_pattern=checkpoint_pattern,
        include_outer_test=include_outer_test,
    )
    _new_destination(output, [dataset_root, checkpoint_root])
    record, _, _, summary, checkpoints = _inspect_inputs(
        config, training_split, dataset_root, checkpoint_root, checkpoint_pattern
    )
    return {
        **plan,
        "mode": "preflight_only",
        "status": "PASS_MODEL_FREE_INPUT_LINEAGE",
        "training_split_sha256": split_digest(record),
        "annotation_summary": summary,
        "verified_checkpoints": [entry for _, _, entry in checkpoints],
        "model_execution": False,
        "image_decoding": False,
        "new_files_written": False,
    }


def _load_student(path, info, record, fold, project_root):
    """Execution-only: restricted loading, exact metadata, no backbone download."""
    payload = _safe_load(path)
    if not isinstance(payload, dict) or "model" not in payload or "meta" not in payload:
        raise ValueError("Expected a newly trained model/meta checkpoint payload")
    if payload["meta"] != info["meta"]:
        raise ValueError("Embedded checkpoint metadata differs from its sidecar")
    _checked_metadata(info, record, fold)
    return _strict_model_load(
        {**ARCHITECTURE, "timm_version": RUNTIME_VERSIONS["timm"]},
        payload["model"],
        project_root,
    )


def execute_export(
    config,
    training_split,
    dataset_root,
    checkpoint_root,
    output,
    *,
    project_root=".",
    batch_size=16,
    device="cpu",
    checkpoint_pattern=DEFAULT_PATTERN,
    include_outer_test=False,
):
    """Explicit forward execution, never training, with atomic new-only output."""
    plan_export(
        config,
        training_split,
        dataset_root,
        checkpoint_root,
        output,
        batch_size=batch_size,
        device=device,
        checkpoint_pattern=checkpoint_pattern,
        include_outer_test=include_outer_test,
    )
    dataset_root = Path(dataset_root).resolve()
    output = _new_destination(output, [dataset_root, checkpoint_root])
    record, grouped, annotation, _, checkpoints = _inspect_inputs(
        config, training_split, dataset_root, checkpoint_root, checkpoint_pattern
    )
    split_file_hash = sha256_file(training_split)
    dataset_summary = verify_dataset(dataset_root)
    source_hashes = {
        name: source_code_hash(Path(project_root) / name) for name in SOURCE_FILES
    }
    frame_count = sum(len(rows) for rows in grouped.values())
    estimate = frame_count * (5 * 1024 + 266) * 4 + 64 * 1024**2
    if shutil.disk_usage(output.parent).free < estimate + 1024**3:
        raise OSError("Insufficient free space for export plus a 1 GiB reserve")
    torch, np, cv2, transform, versions = _runtime(project_root, 224)
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; no silent substitution")
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        manifest = {
            "schema_version": 2,
            "kind": "inner_validation_training_features",
            "dataset": "lls48",
            "model_name": config["model_name"],
            "config_identity_sha256": config_identity(config),
            "annotation_path": str(annotation.resolve()),
            "annotation_sha256": record["annotation_sha256"],
            "training_split_sha256": split_digest(record),
            "training_split_file_sha256": split_file_hash,
            "training_split": record,
            "label_columns": LABEL_COLUMNS,
            "dim": 1024,
            "num_classes": 266,
            "preprocessing": preprocessing_spec(224),
            "runtime": versions,
            "device": device,
            "batch_size": batch_size,
            "include_outer_test": include_outer_test,
            "outer_test_arrays_used_for_selection": False,
            "source_code_sha256": source_hashes,
            "source_code_hash_policy": SOURCE_HASH_POLICY,
            "dataset_verification": dataset_summary,
            "reproduction_status": "NEW_PROTOCOL_NOT_NUMERICALLY_QUALIFIED",
            "limitation": LIMITATION,
            "folds": {},
        }
        labels = {}
        for video, entries in grouped.items():
            values = np.asarray(
                [[int(row[column]) for column in LABEL_COLUMNS] for _, row in entries],
                dtype=np.float32,
            )
            labels[video] = _save_array(
                stage / "labels", f"lls48/triplet_label/{video}.npy", values
            )
        for fold, (path, info, encoder) in enumerate(checkpoints):
            with redirect_stdout(sys.stderr):
                model = _load_student(path, info, record, fold, project_root)
            model.to(device)
            model.eval()
            partitions = fold_partition(record, fold)
            fold_entry = {"encoder": encoder, "train": {}, "inner_validation": {}}
            roles = ["train", "inner_validation"]
            if include_outer_test:
                fold_entry["outer_test"] = {}
                roles.append("outer_test")
            for role in roles:
                for video in partitions[role]:
                    entries = grouped[video]
                    image_rows = [
                        {
                            **row,
                            "image_path": source_file(
                                dataset_root, f"data/{row['image_path']}"
                            ),
                        }
                        for _, row in entries
                    ]
                    values = _extract_rows(
                        model,
                        image_rows,
                        transform,
                        batch_size,
                        device,
                        (torch, np, cv2),
                    )
                    disk_role = "val" if role == "outer_test" else "train"
                    relative = (
                        f"lls48/{config['model_name']}/{fold}/{disk_role}/{video}.npy"
                    )
                    fold_entry[role][video] = {
                        "feature": _save_array(stage / "features", relative, values),
                        "label": labels[video],
                        "frame_keys": [
                            annotation_key(index, row) for index, row in entries
                        ],
                    }
            manifest["folds"][str(fold)] = fold_entry
            del model
        if verify_dataset(dataset_root) != dataset_summary:
            raise ValueError("Dataset changed during feature export")
        if sha256_file(training_split) != split_file_hash:
            raise ValueError("Training split changed during feature export")
        for path, _, encoder in checkpoints:
            if (
                sha256_file(path) != encoder["sha256"]
                or sha256_file(encoder["metadata_sidecar_path"])
                != encoder["metadata_sidecar_sha256"]
            ):
                raise ValueError(
                    "Checkpoint or provenance changed during feature export"
                )
        if any(
            source_code_hash(Path(project_root) / name) != digest
            for name, digest in source_hashes.items()
        ):
            raise ValueError("Exporter source changed during execution")
        manifest_path = stage / "training_feature_manifest.json"
        with manifest_path.open("xb") as handle:
            handle.write(json_bytes(manifest))
        validate_training_features(
            manifest_path,
            config,
            stage / "features",
            stage / "labels",
            record,
            include_outer_test=include_outer_test,
        )
        _publish_without_replace(stage, output)
    except Exception as exc:
        raise RuntimeError(
            f"Training feature export failed; staging preserved at {stage}: {exc}"
        ) from exc
    return {
        "status": "EXPORTED_NEW_PROTOCOL_FEATURES",
        "output": str(output),
        "feature_manifest": str(output / "training_feature_manifest.json"),
        "include_outer_test": include_outer_test,
        "training_executed": False,
        "numerical_paper_reproduction": False,
        "limitation": LIMITATION,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--training-split", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--checkpoint-pattern", default=DEFAULT_PATTERN)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--include-outer-test",
        action="store_true",
        help="Also export frozen-encoder outer-test features for later final inference; never used for selection",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--preflight",
        action="store_true",
        help="Read-only file and lineage checks; no ML imports",
    )
    modes.add_argument(
        "--execute",
        action="store_true",
        help="Explicitly perform image decoding and encoder forward passes",
    )
    args = parser.parse_args(argv)
    try:
        from curconmix_repro.validation import load_config

        config = load_config(args.config)
        arguments = (
            config,
            args.training_split,
            args.dataset_root,
            args.checkpoint_root,
            args.output,
        )
        options = {
            "batch_size": args.batch_size,
            "device": args.device,
            "checkpoint_pattern": args.checkpoint_pattern,
            "include_outer_test": args.include_outer_test,
        }
        if args.execute:
            result = execute_export(
                *arguments, project_root=args.project_root, **options
            )
        elif args.preflight:
            result = preflight_export(*arguments, **options)
        else:
            result = plan_export(*arguments, **options)
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, ImportError, KeyError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
