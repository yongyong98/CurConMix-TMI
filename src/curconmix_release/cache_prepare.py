"""Explicit source-derived B224 contrastive-cache preparation, not paper reproduction.

Default/--check-only never imports a model. --execute performs new feature
extraction (not training) and requires permission for timm pretrained downloads.
These are ImageNet-backbone contrastive caches, NOT final-student MRTT features.
"""

import argparse
import ctypes
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path
import pickle
import shutil
import sys
import tempfile
from types import SimpleNamespace

from .feature_contract import SOURCE_HASH_POLICY, source_code_hash
from .spatial_training import (
    DATASETS,
    MODEL_NAME,
    cache_filenames,
    image_path,
    inspect_inputs,
    repository_root,
    row_identity,
    row_order_sha256,
    sha256_file,
)

SEED = 142
FEATURE_DIM = 1024
RUNTIME = {
    "torch": "2.0.1",
    "torchvision": "0.15.2",
    "timm": "0.5.4",
    "numpy": "1.22.4",
    "scikit-learn": "1.5.0",
}


def available_memory():
    """Best-effort host available RAM, without importing a model/runtime."""
    if os.name == "nt":

        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong)
                for name in (
                    "total",
                    "available",
                    "page_total",
                    "page_available",
                    "virtual_total",
                    "virtual_available",
                    "extended",
                )
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return status.available
    elif Path("/proc/meminfo").is_file():
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    return None


def estimate_resources(rows, training_split=None):
    from .spatial_protocol import select_rows

    folds = {}
    for fold in range(5):
        n = (
            len(select_rows(rows, training_split, fold))
            if training_split
            else sum(row["fold"] != fold for row in rows)
        )
        folds[str(fold)] = {
            "training_frames": n,
            "feature_shape": [n, FEATURE_DIM],
            "feature_bytes_float32": n * FEATURE_DIM * 4,
            "similarity_shape": [n, n],
            "similarity_bytes_float32": n * n * 4,
        }
    total = sum(
        v["feature_bytes_float32"] + v["similarity_bytes_float32"]
        for v in folds.values()
    )
    peak = (
        max(
            (
                2 * v["similarity_bytes_float32"] + 3 * v["feature_bytes_float32"]
                for v in folds.values()
            ),
            default=0,
        )
        + 1024**3
    )
    return {
        "folds": folds,
        "payload_bytes_estimate": total,
        "required_free_disk_bytes": int(total * 1.10) + 3 * 1024**3,
        "conservative_host_ram_bytes": peak,
        "note": "Dense NxN cosine matrices; no subsampling or approximation. Estimates include headroom, not a measured peak.",
    }


def check_plan(args):
    from .spatial_protocol import load_split
    from curconmix_repro.training_split import split_digest

    rows, report = inspect_inputs(args.dataset, args.data_root, args.annotation_csv)
    training_split = None
    try:
        training_split = load_split(
            getattr(args, "training_split", None), args.dataset, report["annotation"]
        )
        report["training_split"] = training_split
        report["training_split_sha256"] = split_digest(training_split)
    except (ValueError, OSError) as error:
        report["errors"].append(str(error))
    for part in (args.output_dir.absolute(), *args.output_dir.absolute().parents):
        if part.is_symlink() or (
            part.exists() and getattr(part.lstat(), "st_file_attributes", 0) & 0x400
        ):
            report["errors"].append(
                "Output directory and its ancestors must not be symlinks or reparse points."
            )
            break
    output = args.output_dir.resolve()
    if output.exists() or output.is_symlink():
        report["errors"].append(
            "--output-dir must not exist; partial or previous caches are never overwritten."
        )
    if output.is_relative_to(args.data_root.resolve()):
        report["errors"].append(
            "Place new caches outside the source dataset directory."
        )
    ancestor = output.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    resources = estimate_resources(rows, training_split)
    resources["free_disk_bytes"] = shutil.disk_usage(ancestor).free
    resources["available_host_ram_bytes"] = available_memory()
    if resources["free_disk_bytes"] < resources["required_free_disk_bytes"]:
        report["errors"].append(
            "Insufficient free disk for all five dense feature/similarity pairs and reserve."
        )
    if (
        resources["available_host_ram_bytes"] is not None
        and resources["available_host_ram_bytes"]
        < resources["conservative_host_ram_bytes"]
    ):
        report["errors"].append(
            "Insufficient available host RAM for the dense cosine matrix estimate."
        )
    report["resources"] = resources
    report["status"] = (
        "BLOCKED"
        if report["errors"] or report["missing_files"]
        else "INPUT_FILES_PRESENT"
    )
    report["model_executed"] = False
    report["qualification"] = (
        "SOURCE_DERIVED_NEW_CACHE_NOT_HISTORICAL_NUMERIC_REPRODUCTION"
    )
    return rows, report


class ImageDataset:
    """Pickleable on Windows workers; no torch/PIL imports until explicit use."""

    def __init__(self, rows, root, transform):
        self.rows, self.root, self.transform = rows, root, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        from PIL import Image

        data = image_path(self.root, self.rows[index]["image_path"]).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        with Image.open(io.BytesIO(data)) as opened:
            image = self.transform(opened.convert("RGB"))
        return image, digest


def source_hashes(dataset):
    root = repository_root()
    paths = [
        root / "legacy" / "spatial" / dataset / name
        for name in (
            "models.py",
            "preprocess.py",
            "helper.py",
            "config.yaml",
            "pretrain.yaml",
        )
    ]
    paths.extend([Path(__file__), Path(__file__).with_name("spatial_training.py")])
    return {path.relative_to(root).as_posix(): source_code_hash(path) for path in paths}


def check_runtime():
    versions = {name: importlib.metadata.version(name) for name in RUNTIME}
    for name, required in RUNTIME.items():
        if versions[name].split("+")[0] != required:
            raise ValueError(
                f"Cache recipe requires {name}=={required}; found {versions[name]}. Use requirements-spatial.txt."
            )
    versions["pillow"] = importlib.metadata.version("pillow")
    versions["scipy"] = importlib.metadata.version("scipy")
    versions["python"] = sys.version
    return versions


def backbone_fingerprint(model):
    digest = hashlib.sha256()
    for key, tensor in model.model.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([key, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def publish_staged(staging, output, names):
    """Each file appears atomically; hard-link publication can never overwrite."""
    for name in names:
        os.link(staging / name, output / name)


def probe_hardlinks(output):
    """Fail before downloading a model if atomic no-overwrite publication is unsupported."""
    with tempfile.NamedTemporaryFile(
        prefix=".hardlink-probe-", dir=output, delete=False
    ) as handle:
        source = Path(handle.name)
    target = source.with_name(source.name + ".link")
    try:
        os.link(source, target)
    except OSError as error:
        raise ValueError(
            "Cache destination must support atomic hard-link publication (for example NTFS/ext4)."
        ) from error
    finally:
        if target.exists():
            target.unlink()
        source.unlink()


def execute(args, rows, report):
    """Only the explicit CLI execution branch can call this new-computation path."""
    if not args.allow_pretrained_download:
        raise ValueError(
            "Execution requires --allow-pretrained-download (timm may use its cache or download ImageNet weights)."
        )
    if report["status"] != "INPUT_FILES_PRESENT":
        raise ValueError("Input preflight must pass before model construction.")
    versions = check_runtime()
    # Recheck disk/RAM/files immediately before any model construction/download.
    current_rows, current = check_plan(args)
    if (
        current["status"] != "INPUT_FILES_PRESENT"
        or current["annotation_sha256"] != report["annotation_sha256"]
        or current["training_split_sha256"] != report["training_split_sha256"]
    ):
        raise ValueError("Inputs/resources changed since preflight.")
    if row_order_sha256(current_rows) != row_order_sha256(rows):
        raise ValueError("Input row order changed since preflight.")
    output = args.output_dir.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    incomplete = output / "INCOMPLETE.json"
    with incomplete.open("x", encoding="utf-8") as handle:
        json.dump(
            {
                "status": "incomplete",
                "note": "Never train from this directory until cache_provenance.json is present and this marker is removed.",
            },
            handle,
        )
    probe_hardlinks(output)
    original_sources = source_hashes(args.dataset)

    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from sklearn.metrics.pairwise import cosine_similarity

    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("The source-derived cache runtime supports CPU or CUDA only.")
    if device.type == "cuda" and (
        not torch.cuda.is_available()
        or (device.index is not None and device.index >= torch.cuda.device_count())
    ):
        raise ValueError(
            "Requested CUDA device is unavailable; no pretrained model was downloaded or constructed."
        )

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model_path = repository_root() / "legacy" / "spatial" / args.dataset / "models.py"
    spec = importlib.util.spec_from_file_location("_cache_source_models", model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = SimpleNamespace(target_size=sum(DATASETS[args.dataset]["counts"]))
    model = module.supcon_Model(cfg, MODEL_NAME, pretrained=True).to(args.device).eval()
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    provenance = {
        "schema_version": 1,
        "complete": True,
        "dataset": args.dataset,
        "qualification": report["qualification"],
        "model_name": MODEL_NAME,
        "model_tag": args.model_tag,
        "seed": SEED,
        "feature_dimension": FEATURE_DIM,
        "annotation_sha256": report["annotation_sha256"],
        "training_split_sha256": report["training_split_sha256"],
        "training_split": report["training_split"],
        "source_sha256": original_sources,
        "source_hash_policy": SOURCE_HASH_POLICY,
        "pretrained_backbone_tensor_sha256": backbone_fingerprint(model),
        "runtime": versions,
        "device": args.device,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "preprocessing": [
            "PIL RGB",
            "torchvision Resize((224,224)), default bilinear/interpolation behavior of recorded runtime",
            "ToTensor",
            "ImageNet mean/std",
        ],
        "algorithm": "Source-derived cache recipe: eval-mode timm pretrained supcon_Model second output, explicit train-partition frames only (excluding inner validation and outer test), sklearn cosine_similarity, dense float32 pickle arrays.",
        "not_final_student_mrtt_features": True,
        "folds": {},
    }
    image_digests = {}
    with tempfile.TemporaryDirectory(prefix=".staging-", dir=output) as temp:
        staging = Path(temp)
        all_names = []
        for fold in range(5):
            from .spatial_protocol import select_rows

            selected = select_rows(rows, report["training_split"], fold)
            loader = DataLoader(
                ImageDataset(selected, args.data_root, transform),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=True,
            )
            parts, digests = [], []
            with torch.no_grad():
                for images, hashes in loader:
                    _, features = model(images.to(args.device))
                    parts.append(features.cpu().numpy())
                    digests.extend(hashes)
            features = np.concatenate(parts, axis=0)
            del parts
            if (
                features.shape != (len(selected), FEATURE_DIM)
                or features.dtype != np.float32
                or not np.isfinite(features).all()
            ):
                raise ValueError(
                    "Unexpected backbone feature shape, dtype, or nonfinite values."
                )
            for row, digest in zip(selected, digests):
                previous = image_digests.setdefault(row["image_path"], digest)
                if previous != digest:
                    raise ValueError(
                        f"Source image changed between folds: {row['image_path']}"
                    )
            similarity = cosine_similarity(features)
            if (
                similarity.shape != (len(selected), len(selected))
                or similarity.dtype != np.float32
            ):
                raise ValueError("Unexpected dense cosine matrix shape/dtype.")
            names = cache_filenames(args.model_tag, fold)
            files = {}
            for name, value in zip(names, (features, similarity)):
                path = staging / name
                with path.open("xb") as handle:
                    pickle.dump(value, handle, protocol=pickle.DEFAULT_PROTOCOL)
                    handle.flush()
                    os.fsync(handle.fileno())
                files[name] = {
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            provenance["folds"][str(fold)] = {
                "training_frames": len(selected),
                "row_order_sha256": row_order_sha256(selected),
                "feature_shape": list(features.shape),
                "similarity_shape": list(similarity.shape),
                "dtype": "float32",
                "files": files,
                "rows": [
                    {"key": row_identity(row), "image_sha256": digest}
                    for row, digest in zip(selected, digests)
                ],
            }
            all_names.extend(names)
            del features, similarity
        if (
            sha256_file(report["annotation"]) != report["annotation_sha256"]
            or source_hashes(args.dataset) != original_sources
        ):
            raise ValueError(
                "Annotation or recipe source changed during cache extraction."
            )
        from .spatial_protocol import load_split
        from curconmix_repro.training_split import split_digest

        if (
            split_digest(
                load_split(args.training_split, args.dataset, report["annotation"])
            )
            != report["training_split_sha256"]
        ):
            raise ValueError("Training split changed during cache extraction.")
        # Rehash every source image once after extraction to bind the completed cache.
        for relative, digest in image_digests.items():
            if sha256_file(image_path(args.data_root, relative)) != digest:
                raise ValueError(f"Source image changed during extraction: {relative}")
        manifest = staging / "cache_provenance.json"
        with manifest.open("x", encoding="utf-8") as handle:
            json.dump(provenance, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        publish_staged(staging, output, all_names + [manifest.name])
    incomplete.unlink()  # Only our exact marker in the newly reserved output directory.
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--annotation-csv")
    parser.add_argument(
        "--training-split",
        type=Path,
        help="Required explicit nested procedure split JSON for cache preparation.",
    )
    parser.add_argument("--model-tag", default="SwinB224")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--allow-pretrained-download", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        parser.error("--batch-size must be positive and --workers nonnegative.")
    try:
        names = [
            name for fold in range(5) for name in cache_filenames(args.model_tag, fold)
        ]
        print(
            json.dumps(
                {
                    "dataset": args.dataset,
                    "model_name": MODEL_NAME,
                    "seed": SEED,
                    "new_cache_directory": str(args.output_dir.resolve()),
                    "files": names + ["cache_provenance.json"],
                    "recipe": "New source-derived B224 contrastive caches; not historical numeric reproduction or final-student MRTT features.",
                },
                indent=2,
            )
        )
        if not (args.execute or args.check_only):
            print(
                "Preview only. No model, image, or checkpoint was loaded; no network request was made."
            )
            return
        rows, report = check_plan(args)
        print(json.dumps(report, indent=2))
        if report["status"] == "BLOCKED":
            parser.exit(2, "Input/resource checks failed. No model was run.\n")
        if args.check_only:
            print(
                "Input files checked without model execution. Runtime/CUDA and numerical reproduction remain unqualified."
            )
            return
        execute(args, rows, report)
        print(
            "Completed new source-derived contrastive caches. This is not a paper-metric reproduction result."
        )
    except (ValueError, OSError, importlib.metadata.PackageNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
