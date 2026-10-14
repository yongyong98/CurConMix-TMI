"""Plan or explicitly execute a provenance-tracked LLS48 spatial feature export.

Without --execute this module does not import torch, open images, load weights,
create files or download anything. Execution is an opt-in model forward pass.
New exports are not established reproductions of the historical MRTT features.
"""

import argparse
import ctypes
from contextlib import redirect_stdout
import errno
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace

from .dataset import LLS48Index
from .feature_contract import (
    CHECKPOINT_HASH_KEYS,
    LIMITATION,
    MODEL_NAMES,
    RUNTIME_VERSIONS,
    SOURCE_FILES,
    SOURCE_HASH_POLICY,
    config_identity,
    frame_key,
    preprocessing_spec,
    source_code_hash,
    validate_feature_manifest,
)
from .lls48 import (
    CANONICAL_FOLDS,
    LABEL_COLUMNS,
    MASTER_RELATIVE,
    MASTER_SHA256,
    json_bytes,
    sha256_file,
    source_file,
    verify_dataset,
)
from .spatial_checkpoint import VARIANTS, _validated_file, load_spatial_checkpoint


def _validate_config(config, asset):
    if asset not in MODEL_NAMES:
        raise ValueError("Only the verified LLS48 B224/B384 students are supported")
    expected = {
        "dataset": "lls48",
        "model_name": MODEL_NAMES[asset],
        "dim": 1024,
        "num_classes": 266,
        "triplet_size": 195,
        "n_fold": 5,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("The spatial asset and temporal configuration do not match")
    if config.get("validation_videos") != list(CANONICAL_FOLDS.values()):
        raise ValueError("Configuration must retain the canonical five-fold split")


def plan_export(
    config, asset, dataset_root, checkpoint_root, output, batch_size, device
):
    _validate_config(config, asset)
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Batch size must be a positive integer")
    if device not in ("cpu", "cuda"):
        raise ValueError("Device must be cpu or cuda")
    return {
        "mode": "plan_only",
        "spatial_asset": asset,
        "dataset_root": str(Path(dataset_root).absolute()),
        "checkpoint_root": str(Path(checkpoint_root).absolute()),
        "new_output_root": str(Path(output).absolute()),
        "feature_root": str(Path(output).absolute() / "features"),
        "label_root": str(Path(output).absolute() / "labels"),
        "feature_manifest": str(Path(output).absolute() / "feature_manifest.json"),
        "model_name": config["model_name"],
        "folds": list(range(5)),
        "splits": ["train", "val"],
        "batch_size": batch_size,
        "device": device,
        "preprocessing": preprocessing_spec(VARIANTS[asset]["input_resolution"]),
        "requires_published_matching_spatial_weights": True,
        "model_execution": False,
        "limitation": LIMITATION,
    }


def _reject_links(path):
    path = Path(path).absolute()
    for candidate in [path, *path.parents]:
        if candidate.exists() and (
            candidate.is_symlink()
            or getattr(candidate.stat(), "st_file_attributes", 0) & 0x400
        ):
            raise ValueError(
                "Output paths must not traverse symlinks or reparse points"
            )


def _new_destination(output, protected):
    if sys.platform != "win32" and not sys.platform.startswith("linux"):
        raise OSError(errno.ENOTSUP, "Atomic no-replace export supports Windows/Linux")
    if sys.platform.startswith("linux") and not hasattr(ctypes.CDLL(None), "renameat2"):
        raise OSError("Atomic no-replace export requires Linux renameat2")
    _reject_links(output)
    output = Path(output).resolve(strict=False)
    if output.exists():
        raise FileExistsError(
            "Output must be a new directory; existing files are preserved"
        )
    for source in protected:
        resolved = Path(source).resolve()
        if (
            output == resolved
            or output.is_relative_to(resolved)
            or resolved.is_relative_to(output)
        ):
            raise ValueError("Output must not overlap dataset or checkpoint sources")
    if not output.parent.is_dir():
        raise FileNotFoundError("Create the output parent directory before execution")
    return output


def _publish_without_replace(stage, output):
    """Atomic directory rename that refuses an existing destination."""
    if sys.platform == "win32":
        os.rename(stage, output)  # Windows MoveFile refuses any existing destination.
    elif sys.platform.startswith("linux"):
        library = ctypes.CDLL(None, use_errno=True)
        rename = getattr(library, "renameat2", None)
        if rename is None:
            raise OSError("Atomic no-replace rename requires Linux renameat2")
        result = rename(-100, os.fsencode(stage), -100, os.fsencode(output), 1)
        if result:
            number = ctypes.get_errno()
            raise OSError(number, os.strerror(number), str(output))
    else:
        raise OSError(errno.ENOTSUP, "Atomic no-replace export supports Windows/Linux")


def _save_array(root, relative, array):
    import numpy as np

    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.save(handle, array, allow_pickle=False)
    return {
        "path": relative,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }


def _runtime(project_root, resolution):
    import albumentations
    import cv2
    import numpy as np
    import torch
    import timm

    versions = {
        "torch": torch.__version__,
        "numpy": np.__version__,
        "albumentations": albumentations.__version__,
        "opencv": cv2.__version__,
        "timm": timm.__version__,
    }
    if any(
        versions[key].split("+")[0] != value for key, value in RUNTIME_VERSIONS.items()
    ):
        raise ValueError("Use requirements-spatial.txt; preprocessing runtime mismatch")
    path = Path(project_root) / "legacy/spatial/lls48/augmentation.py"
    spec = importlib.util.spec_from_file_location(
        "curconmix_lls48_export_augmentation", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    transform = module.get_transforms(
        data="valid", CFG=SimpleNamespace(width=resolution, height=resolution)
    )
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch, np, cv2, transform, versions


def _extract_rows(model, rows, transform, batch_size, device, runtime):
    """Execution-only path: historical pooled backbone features, never logits."""
    torch, np, cv2 = runtime
    pool = type(model).forward.__globals__["pool_backbone_feature"]
    batches = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            images = []
            for row in rows[start : start + batch_size]:
                image = cv2.imread(str(row["image_path"]), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"Image decoding failed: {row['image_path'].name}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images.append(transform(image=image)["image"])
            image_batch = torch.stack(images).to(device=device, dtype=torch.float32)
            features = pool(model.model(image_batch))
            values = features.detach().cpu().numpy()
            if values.shape != (len(images), 1024) or not np.isfinite(values).all():
                raise ValueError(
                    "Expected finite pooled 1024-dimensional student features"
                )
            batches.append(values.astype(np.float32, copy=False))
    return np.concatenate(batches, axis=0)


def execute_export(
    config,
    asset,
    dataset_root,
    checkpoint_root,
    output,
    *,
    spatial_manifest="manifests/spatial_assets.json",
    project_root=".",
    batch_size=16,
    device="cpu",
):
    """Explicit opt-in export. Never modify an input or an existing output."""
    plan_export(
        config, asset, dataset_root, checkpoint_root, output, batch_size, device
    )
    dataset_root = Path(dataset_root).resolve()
    output = _new_destination(output, [dataset_root, checkpoint_root])
    # Gates run before importing torch, decoding images or constructing a model.
    checkpoint_files = [
        _validated_file(asset, checkpoint_root, fold, spatial_manifest)
        for fold in range(5)
    ]
    dataset_summary = verify_dataset(dataset_root)
    index = LLS48Index(dataset_root)
    grouped = {}
    for row in index:
        grouped.setdefault(row["video"], []).append(row)
    specification = VARIANTS[asset]
    estimate = len(index) * (5 * 1024 + 266) * 4 + 64 * 1024**2
    if shutil.disk_usage(output.parent).free < estimate + 1024**3:
        raise OSError("Insufficient free space for export plus a 1 GiB reserve")
    source_hashes = {
        name: source_code_hash(Path(project_root) / name) for name in SOURCE_FILES
    }
    spatial_manifest_hash = sha256_file(spatial_manifest)
    torch, np, cv2, transform, versions = _runtime(
        project_root, specification["input_resolution"]
    )
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "CUDA requested but unavailable; no silent device substitution"
        )
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        manifest = {
            "schema_version": 1,
            "dataset": "lls48",
            "spatial_asset": asset,
            "model_name": config["model_name"],
            "config_identity_sha256": config_identity(config),
            "config": config,
            "annotation_sha256": MASTER_SHA256,
            "label_columns": LABEL_COLUMNS,
            "dim": 1024,
            "num_classes": 266,
            "event_extraction_scope": "independent_five_frame_clip",
            "reproduction_status": "NOT_NUMERICALLY_QUALIFIED",
            "limitation": LIMITATION,
            "preprocessing": preprocessing_spec(specification["input_resolution"]),
            "runtime": versions,
            "device": device,
            "batch_size": batch_size,
            "dataset_verification": dataset_summary,
            "spatial_manifest_sha256": spatial_manifest_hash,
            "source_code_sha256": source_hashes,
            "source_code_hash_policy": SOURCE_HASH_POLICY,
            "procedures": {},
            "folds": {},
        }
        annotation = source_file(dataset_root, MASTER_RELATIVE)
        copied_annotation = stage / MASTER_RELATIVE
        copied_annotation.parent.mkdir(parents=True)
        with annotation.open("rb") as source, copied_annotation.open("xb") as target:
            shutil.copyfileobj(source, target)
        manifest["annotation"] = {
            "path": MASTER_RELATIVE,
            "sha256": sha256_file(copied_annotation),
            "size_bytes": copied_annotation.stat().st_size,
        }
        for video, rows in grouped.items():
            keys = []
            for row in rows:
                relative = (
                    row["image_path"].relative_to(dataset_root / "data").as_posix()
                )
                keys.append(
                    frame_key(row["row_index"], {**row, "image_path": relative})
                )
            labels = np.asarray([row["labels"] for row in rows], dtype=np.float32)
            manifest["procedures"][video] = {
                "frames": keys,
                "labels": _save_array(
                    stage, f"labels/lls48/triplet_label/{video}.npy", labels
                ),
            }
        for fold, (checkpoint_path, checkpoint_record) in enumerate(checkpoint_files):
            print(
                f"Extracting {asset}, fold {fold}; sparse clip identities preserved",
                file=sys.stderr,
            )
            with redirect_stdout(sys.stderr):
                model, _ = load_spatial_checkpoint(
                    asset, checkpoint_root, fold, spatial_manifest, project_root
                )
            model.to(device)
            model.eval()
            entry = {
                "checkpoint": {
                    "asset": asset,
                    "fold": fold,
                    **{key: checkpoint_record[key] for key in CHECKPOINT_HASH_KEYS},
                },
                "files": [],
            }
            for video, rows in grouped.items():
                split = "val" if video in CANONICAL_FOLDS[str(fold)] else "train"
                values = _extract_rows(
                    model, rows, transform, batch_size, device, (torch, np, cv2)
                )
                relative = (
                    f"features/lls48/{config['model_name']}/{fold}/{split}/{video}.npy"
                )
                entry["files"].append(
                    {
                        "video": video,
                        "split": split,
                        **_save_array(stage, relative, values),
                    }
                )
            if sha256_file(checkpoint_path) != checkpoint_record["sha256"]:
                raise ValueError("Source checkpoint changed during feature extraction")
            manifest["folds"][str(fold)] = entry
            del model
        # Recheck all immutable source images/metadata after execution as well.
        if verify_dataset(dataset_root) != dataset_summary:
            raise ValueError("Dataset source changed during export")
        if any(sha256_file(path) != item["sha256"] for path, item in checkpoint_files):
            raise ValueError("Spatial checkpoint changed during export")
        if sha256_file(spatial_manifest) != spatial_manifest_hash or any(
            source_code_hash(Path(project_root) / name) != digest
            for name, digest in source_hashes.items()
        ):
            raise ValueError(
                "Spatial manifest or exporter source changed during export"
            )
        manifest_path = stage / "feature_manifest.json"
        with manifest_path.open("xb") as handle:
            handle.write(json_bytes(manifest))
        validate_feature_manifest(
            manifest_path,
            config,
            stage / "features",
            stage / "labels",
            spatial_manifest_path=spatial_manifest,
            project_root=project_root,
        )
        _publish_without_replace(stage, output)
    except Exception as exc:
        raise RuntimeError(
            f"Export failed; unpublished staging preserved at {stage}: {exc}"
        ) from exc
    return {
        "status": "EXPORTED_WITH_ALIGNMENT_CHECKS",
        "output": str(output),
        "feature_manifest": str(output / "feature_manifest.json"),
        "reproduction_status": "NOT_NUMERICALLY_QUALIFIED",
        "limitation": LIMITATION,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--asset", choices=MODEL_NAMES, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--spatial-manifest", type=Path, default=Path("manifests/spatial_assets.json")
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Opt in to image decoding and spatial model forward passes",
    )
    args = parser.parse_args(argv)
    try:
        from curconmix_repro.validation import load_config

        config = load_config(args.config)
        arguments = (
            config,
            args.asset,
            args.dataset_root,
            args.checkpoint_root,
            args.output,
        )
        if args.execute:
            result = execute_export(
                *arguments,
                spatial_manifest=args.spatial_manifest,
                project_root=args.project_root,
                batch_size=args.batch_size,
                device=args.device,
            )
        else:
            result = plan_export(*arguments, args.batch_size, args.device)
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, ImportError, KeyError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
