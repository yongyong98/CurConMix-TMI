"""Maintainer-only model-only export of explicitly qualified spatial students.

Reads the original model+preds files with a narrowly restricted NumPy allowlist.
No training, inference, forward pass, feature generation or network access occurs.
Use a dataset-specific qualified runtime for strict architecture loading.
"""

import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
LICENSE = "CC-BY-NC-4.0"
VARIANTS = {
    "b224": ("swin_base_patch4_window7_224", 224),
    "b384": ("swin_base_patch4_window12_384", 384),
}
DATASETS = {
    "lls48": {
        "registry_prefix": "lls48",
        "display_name": "LLS48",
        "variants": VARIANTS,
        "feature_dim": 1024,
        "num_classes": 266,
        "triplet_size": 195,
        "class_counts": {"triplet": 195, "instrument": 17, "verb": 16, "target": 38},
        "mapping": "ivtmetrics_lls/maps.txt",
        "mapping_sha256": "927957a5942d62e18168cdf761013c08b4f361a537ad6cf20ebe8a7c7cdfe758",
        "timm_version": "0.5.4",
    },
    "cholect45": {
        "registry_prefix": "cholec45",
        "display_name": "CholecT45",
        "variants": {"l224": ("swin_large_patch4_window7_224.ms_in22k", 224)},
        "feature_dim": 1536,
        "num_classes": 131,
        "triplet_size": 100,
        "class_counts": {"triplet": 100, "instrument": 6, "verb": 10, "target": 15},
        "mapping": "ivtmetrics_cholec/maps.txt",
        "mapping_sha256": "076a862728566ccd641e2f21bcc5f18e9e020c74a9bf54d8cbdb51639c14e67e",
        "timm_version": "1.0.20",
    },
}
NUMPY_GLOBALS = {"numpy.core.multiarray._reconstruct", "numpy.ndarray", "numpy.dtype"}
STRICT_CHECK = """
import importlib.util, json, sys
from pathlib import Path
from types import SimpleNamespace
import torch, timm
path, root = Path(sys.argv[1]), Path(sys.argv[2])
obj = torch.load(path, map_location='cpu', weights_only=True)
if timm.__version__ != obj['meta']['timm_version']:
    raise RuntimeError('Strict qualification requires the specified timm version')
spec = importlib.util.spec_from_file_location('qualified_spatial_models', root/'legacy/spatial'/obj['meta']['dataset']/'models.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
model = module.TripletModel(SimpleNamespace(target_size=obj['meta']['num_classes']), obj['meta']['model_name'], pretrained=False)
model.load_state_dict(obj['model'], strict=True)
print('STRICT_MODEL_LOAD_PASS')
"""


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paths_alias(left, right):
    """Include case aliases, resolved links, and existing hard links."""
    left, right = Path(left), Path(right)
    if str(left.resolve()).casefold() == str(right.resolve()).casefold():
        return True
    try:
        return left.samefile(right)
    except FileNotFoundError:
        return False


def validate_output_paths(args, audit, registry, sources):
    """Reject report/source collisions before any directory or file is written."""
    from curconmix_release.assets import _assert_plain_path

    reports = (args.private_report, args.public_manifest)
    for path in (*reports, args.output):
        _assert_plain_path(path)
    if paths_alias(*reports):
        raise ValueError("Private report and public manifest must not alias")
    protected = [args.audit, Path(audit["registry"]), Path(__file__)]
    protected.extend(
        ROOT / "src/curconmix_repro/temporal" / spec["mapping"]
        for spec in DATASETS.values()
    )
    for model in audit["models"]:
        if model.get("canonical_prediction"):
            protected.append(Path(model["canonical_prediction"]))
        protected.extend(Path(item["path"]) for item in model["candidates"])
    protected.extend(
        Path(row["absolute_prediction_path"])
        for row in registry.values()
        if row.get("absolute_prediction_path")
    )
    protected.extend(
        Path(item["path"])
        for item in audit.get("model_save_evidence", {}).values()
        if isinstance(item, dict) and item.get("path")
    )
    destinations = [args.output]
    for source in sources:
        name = f"{args.dataset}-spatial-{source['variant']}-fold{source['fold']}.pth"
        destinations.extend((args.output / name, args.output / (name + ".part")))
    for report in reports:
        if any(paths_alias(report, path) for path in (*protected, *destinations)):
            raise ValueError("Report output aliases a protected source or export path")
    for destination in destinations:
        if any(paths_alias(destination, path) for path in protected):
            raise ValueError("Export destination aliases a protected source path")


def select_sources(audit, dataset="lls48"):
    if dataset not in DATASETS:
        raise ValueError("Dataset is not qualified for spatial checkpoint export")
    spec = DATASETS[dataset]
    selected = []
    for variant in spec["variants"]:
        matches = [
            m
            for m in audit["models"]
            if m["model_id"] == f"{spec['registry_prefix']}_curconmix_{variant}"
        ]
        if len(matches) != 1:
            raise ValueError("Expected one canonical spatial-model audit entry")
        model = matches[0]
        if model.get("verified_embedded_pairing_folds") != list(range(5)):
            raise ValueError(
                "All five spatial folds must have exact embedded-prediction pairing"
            )
        for fold in range(5):
            candidates = [
                c
                for c in model["candidates"]
                if c.get("fold_from_filename") == fold
                and c.get("identity_status") == "VERIFIED_EMBEDDED_PREDICTION_PAIRING"
            ]
            if (
                not candidates
                or len({c["model_state_fingerprint"] for c in candidates}) != 1
            ):
                raise ValueError("Missing or conflicting checkpoint tensor identities")
            chosen = candidates[0]
            if (
                chosen.get("canonical_prediction_comparison", {}).get(
                    "exact_at_stored_dtype"
                )
                is not True
            ):
                raise ValueError("Exact prediction audit gate is missing")
            selected.append(
                {"variant": variant, "fold": fold, "candidate": chosen, "model": model}
            )
    return selected


def load_original(path):
    import numpy as np
    import torch

    found = set(torch.serialization.get_unsafe_globals_in_checkpoint(path))
    if not found.issubset(NUMPY_GLOBALS):
        raise ValueError(f"Unexpected original-checkpoint globals: {sorted(found)}")
    scalar_module = getattr(np, "_core", np.core).multiarray
    safe = [
        (scalar_module._reconstruct, "numpy.core.multiarray._reconstruct"),
        np.ndarray,
        np.dtype,
        type(np.dtype("float32")),
        type(np.dtype("float64")),
    ]
    with torch.serialization.safe_globals(safe):
        obj = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(obj, dict) or set(obj) != {"model", "preds"}:
        raise ValueError(
            "Only audited student model+preds checkpoint schema is accepted"
        )
    if not isinstance(obj["model"], dict) or not obj["model"]:
        raise ValueError("Missing named model tensors")
    if not all(
        isinstance(k, str) and isinstance(v, torch.Tensor)
        for k, v in obj["model"].items()
    ):
        raise ValueError("Model state must contain named tensors only")
    return obj


def validate_prediction_pairing(predictions, canonical):
    import numpy as np

    if not isinstance(predictions, np.ndarray) or predictions.dtype != np.dtype(
        "float32"
    ):
        raise ValueError("Original student predictions must be float32 arrays")
    if predictions.shape != canonical.shape or not np.array_equal(
        predictions, canonical.astype("float32")
    ):
        raise ValueError(
            "Original embedded predictions do not exactly match canonical fold probabilities"
        )


def metadata(variant, fold, mapping_hash, dataset="lls48"):
    if dataset not in DATASETS:
        raise ValueError("Dataset is not qualified for spatial checkpoint export")
    spec = DATASETS[dataset]
    if variant not in spec["variants"] or type(fold) is not int or fold not in range(5):
        raise ValueError("Dataset/variant/fold combination is not qualified")
    if not re.fullmatch(r"[0-9a-f]{64}", mapping_hash):
        raise ValueError("Invalid mapping SHA-256")
    model_name, resolution = spec["variants"][variant]
    return {
        "schema_version": 1,
        "dataset": dataset,
        "variant": variant,
        "fold": fold,
        "model_class": "TripletModel",
        "model_name": model_name,
        "input_resolution": resolution,
        "feature_dim": spec["feature_dim"],
        "num_classes": spec["num_classes"],
        "triplet_size": spec["triplet_size"],
        "class_counts": dict(spec["class_counts"]),
        "mapping_sha256": mapping_hash,
        "license": LICENSE,
        "timm_version": spec["timm_version"],
    }


def verify_tensors(left, right):
    import torch

    if list(left) != list(right):
        raise ValueError("Exported tensor keys/order differ")
    for key in left:
        if left[key].shape != right[key].shape or left[key].dtype != right[key].dtype:
            raise ValueError(f"Exported tensor shape/dtype differs: {key}")
        if not torch.equal(left[key], right[key]):
            raise ValueError(f"Exported tensor values differ: {key}")


def strict_load(path, python):
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [str(python), "-c", STRICT_CHECK, str(path), str(ROOT)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    if result.returncode:
        raise RuntimeError(
            f"Strict architecture qualification failed ({result.returncode}):\n"
            f"{result.stdout}\n{result.stderr}"
        )
    if "STRICT_MODEL_LOAD_PASS" not in result.stdout:
        raise ValueError("Strict architecture qualification did not report success")
    return result.stdout, result.stderr


def export(args):
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    dataset = args.dataset
    spec = DATASETS[dataset]
    sources = select_sources(audit, dataset)
    registry_path = Path(audit["registry"])
    if sha256(registry_path) != audit["registry_sha256"]:
        raise ValueError("Canonical registry changed since audit")
    with registry_path.open(encoding="utf-8-sig", newline="") as handle:
        registry = {row["model_id"]: row for row in csv.DictReader(handle)}
    validate_output_paths(args, audit, registry, sources)

    import pandas as pd
    import torch

    mapping_hash = sha256(ROOT / "src/curconmix_repro/temporal" / spec["mapping"])
    if mapping_hash != spec["mapping_sha256"]:
        raise ValueError("Public class mapping differs from the qualified mapping")
    public = {"schema_version": 1, "license": LICENSE, "artifacts": []}
    private = {
        "schema_version": 1,
        "audit": str(args.audit.resolve()),
        "audit_sha256": sha256(args.audit),
        "training_inference_or_forward_run": False,
        "exports": [],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    frames = {}
    for source in sources:
        variant, fold = source["variant"], source["fold"]
        model, candidate = source["model"], source["candidate"]
        row = registry[model["model_id"]]
        if any(
            row.get(k) != "VERIFIED"
            for k in ("ap_verification_status", "row_gt_verification_status")
        ):
            raise ValueError("Canonical AP/GT identity gates are not verified")
        if variant not in frames:
            pred_path = Path(row["absolute_prediction_path"])
            pred_hash = sha256(pred_path)
            if (
                pred_hash != model["canonical_prediction_sha256"]
                or pred_hash != row["prediction_sha256"]
            ):
                raise ValueError(
                    "Canonical prediction hash differs from verified audit/registry"
                )
            frames[variant] = pd.read_csv(pred_path)
        frame = frames[variant]
        canonical = frame.loc[
            frame["fold"] == fold, [str(i) for i in range(spec["num_classes"])]
        ].to_numpy()
        original_path = Path(candidate["path"])
        original_hash = sha256(original_path)
        if original_hash != candidate["sha256"]:
            raise ValueError("Original checkpoint differs from audited file")
        print(f"Checking and exporting {variant} fold {fold}", flush=True)
        obj = load_original(original_path)
        validate_prediction_pairing(obj["preds"], canonical)
        state = obj["model"]
        if list(state["head.weight"].shape) != [
            spec["num_classes"],
            spec["feature_dim"],
        ] or list(state["head.bias"].shape) != [spec["num_classes"]]:
            raise ValueError("Checkpoint classifier dimensions are not canonical")
        tensor_bytes = sum(t.numel() * t.element_size() for t in state.values())
        if (
            shutil.disk_usage(args.output).free
            < tensor_bytes + 6 * 1024**3 + 1024**2
        ):
            raise OSError("Insufficient disk space to retain the 6 GiB reserve")
        name = f"{dataset}-spatial-{variant}-fold{fold}.pth"
        destination, temporary = args.output / name, args.output / (name + ".part")
        if destination.exists() or temporary.exists():
            raise FileExistsError("Refusing to overwrite existing checkpoint export")
        meta = metadata(variant, fold, mapping_hash, dataset)
        released = {
            "model": state,
            "meta": meta,
            "release": {
                "schema_version": 1,
                "kind": "spatial_model_only",
                "dataset": spec["display_name"],
                "variant": variant,
                "fold": fold,
                "license": LICENSE,
                "source_sha256": original_hash,
                "canonical_prediction_sha256": model["canonical_prediction_sha256"],
                "embedded_prediction_pairing": "EXACT_FLOAT32",
            },
        }
        try:
            torch.save(released, temporary)
            checked = torch.load(temporary, map_location="cpu", weights_only=True)
            if (
                set(checked) != {"model", "meta", "release"}
                or checked["meta"] != meta
                or checked["release"] != released["release"]
            ):
                raise ValueError("Model-only export schema/metadata changed")
            verify_tensors(state, checked["model"])
            stdout, stderr = strict_load(temporary, args.strict_python)
            if sha256(original_path) != original_hash:
                raise ValueError("Original checkpoint changed during export")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        record = {
            "id": f"{dataset}-spatial-{variant}-fold{fold}",
            "filename": name,
            "size_bytes": destination.stat().st_size,
            "sha256": sha256(destination),
            "source_sha256": original_hash,
            "canonical_prediction_sha256": model["canonical_prediction_sha256"],
            "mapping_sha256": mapping_hash,
            "variant": variant,
            "fold": fold,
            "dataset": dataset,
            "license": LICENSE,
            "metadata": meta,
            "tensor_equality": "PASS",
            "strict_model_load": "PASS",
            "source_preserved": True,
            "prediction_pairing": "EXACT_FLOAT32",
            "install_path": f"{dataset}/{variant}/fold{fold}.pth",
            "model_tensor_count": len(state),
            "model_tensor_bytes": tensor_bytes,
        }
        public["artifacts"].append(record)
        private["exports"].append(
            {
                **record,
                "source_path": str(original_path),
                "output_path": str(destination.resolve()),
                "strict_load_stdout": stdout,
                "strict_load_stderr": stderr,
            }
        )
        for path, data in (
            (args.private_report, private),
            (args.public_manifest, public),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        print(f"PASS: {name} ({record['size_bytes']} bytes)", flush=True)
        del obj, state, released, checked
        gc.collect()
    return public


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="lls48")
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--private-report", type=Path, required=True)
    parser.add_argument("--public-manifest", type=Path, required=True)
    parser.add_argument("--strict-python", type=Path, required=True)
    export(parser.parse_args())


if __name__ == "__main__":
    main()
