"""Load or inspect verified public spatial student weights without inference.

Run from a source checkout with the historical spatial environment installed.
The loader verifies the published manifest and SHA-256 before restricted PyTorch
loading. It constructs the existing TripletModel with pretrained=False and loads
strictly. It does not download a backbone, read images, extract features, or run
the spatial model or MRTT. Successful loading is not end-to-end reproduction.
The required timm version is a qualified loading runtime, not an independently
established historical training version.
"""

import argparse
from contextlib import redirect_stdout
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import re
import sys
from types import SimpleNamespace

from .assets import load_manifest, validate_downloadable, verify_file


VARIANTS = {
    "lls48-spatial-b224": {
        "dataset": "lls48",
        "release_dataset": "LLS48",
        "variant": "b224",
        "public_variant": "swinb224",
        "model_name": "swin_base_patch4_window7_224",
        "input_resolution": 224,
        "feature_dim": 1024,
        "num_classes": 266,
        "triplet_size": 195,
        "class_counts": {"triplet": 195, "instrument": 17, "verb": 16, "target": 38},
        "metric_package": "ivtmetrics_lls",
        "timm_version": "0.5.4",
    },
    "lls48-spatial-b384": {
        "dataset": "lls48",
        "release_dataset": "LLS48",
        "variant": "b384",
        "public_variant": "swinb384",
        "model_name": "swin_base_patch4_window12_384",
        "input_resolution": 384,
        "feature_dim": 1024,
        "num_classes": 266,
        "triplet_size": 195,
        "class_counts": {"triplet": 195, "instrument": 17, "verb": 16, "target": 38},
        "metric_package": "ivtmetrics_lls",
        "timm_version": "0.5.4",
    },
    "cholect45-spatial-l224": {
        "dataset": "cholect45",
        "release_dataset": "CholecT45",
        "variant": "l224",
        "public_variant": "swinl224",
        "model_name": "swin_large_patch4_window7_224.ms_in22k",
        "input_resolution": 224,
        "feature_dim": 1536,
        "num_classes": 131,
        "triplet_size": 100,
        "class_counts": {"triplet": 100, "instrument": 6, "verb": 10, "target": 15},
        "metric_package": "ivtmetrics_cholec",
        "timm_version": "1.0.20",
    },
}
LICENSE = "CC-BY-NC-4.0"


def _safe_load(path):
    import torch

    # Never fall back to unrestricted pickle loading or custom safe globals.
    return torch.load(path, map_location="cpu", weights_only=True)


def _historical_model_class(project_root, dataset="lls48"):
    if dataset not in ("lls48", "cholect45"):
        raise ValueError("Unsupported historical spatial dataset")
    path = Path(project_root) / "legacy/spatial" / dataset / "models.py"
    if not path.is_file():
        raise FileNotFoundError(
            "Historical spatial model source is missing. Run from the repository "
            "root or pass --project-root pointing to the CurConMix-TMI checkout."
        )
    spec = importlib.util.spec_from_file_location(
        f"curconmix_{dataset}_spatial_models", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TripletModel


def _strict_model_load(meta, state, project_root):
    import torch
    import timm

    # This is the qualified loading version, not an inferred training version.
    expected_version = meta["timm_version"]
    if getattr(timm, "__version__", None) != expected_version:
        raise ValueError(
            f"This released spatial architecture requires timm=={expected_version}. "
            "Use the matching separate spatial environment; different model aliases "
            "are not silently substituted."
        )
    if not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("Released model state must contain tensors only")
    model_class = _historical_model_class(project_root, meta["dataset"])
    # The historical constructor prints feature width. Keep CLI JSON on stdout.
    with redirect_stdout(sys.stderr):
        model = model_class(
            SimpleNamespace(target_size=meta["num_classes"]),
            meta["model_name"],
            pretrained=False,
        )
    model.cpu()
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _validated_file(asset_id, root, fold, manifest_path):
    if asset_id not in VARIANTS:
        raise ValueError(
            "Only released LLS48 B224/B384 and CholecT45 L224 spatial students are supported"
        )
    if type(fold) is not int or fold not in range(5):
        raise ValueError("Fold must be an integer from 0 through 4")
    specification = VARIANTS[asset_id]
    dataset, variant = specification["dataset"], specification["variant"]
    manifest = load_manifest(manifest_path)
    selected = [asset for asset in manifest["assets"] if asset["id"] == asset_id]
    if len(selected) != 1:
        raise ValueError(f"Asset is missing from the release manifest: {asset_id}")
    asset = selected[0]
    validate_downloadable(asset)
    expected = {
        "kind": "spatial_checkpoint_bundle",
        "dataset": dataset,
        "variant": specification["public_variant"],
        "architecture": "TripletModel",
        "folds": list(range(5)),
        "license": LICENSE,
    }
    if any(asset.get(key) != value for key, value in expected.items()):
        raise ValueError("Manifest spatial checkpoint identity mismatch")
    files = asset.get("files", [])
    if len(files) != 5 or any(type(item.get("fold")) is not int for item in files):
        raise ValueError(
            "Manifest must contain one plain checkpoint for each integer fold"
        )
    if sorted(item["fold"] for item in files) != list(range(5)):
        raise ValueError("Manifest must contain exactly folds 0 through 4")
    for item in files:
        number = item["fold"]
        if (
            item.get("format") != "file"
            or item.get("filename") != f"{asset_id}-fold{number}.pth"
            or item.get("install_path") != f"{dataset}/{variant}/fold{number}.pth"
        ):
            raise ValueError(
                "Manifest checkpoint filename/install path does not match its fold"
            )
    item = next(item for item in files if item["fold"] == fold)
    path = Path(root) / item["install_path"]
    verify_file(item, path)
    return path, item


def _validate_payload(checkpoint, asset_id, fold, manifest_item):
    if not isinstance(checkpoint, dict) or set(checkpoint) != {
        "model",
        "meta",
        "release",
    }:
        raise ValueError("Expected the released model-only checkpoint schema")
    state, meta, release = (
        checkpoint["model"],
        checkpoint["meta"],
        checkpoint["release"],
    )
    if not isinstance(meta, dict) or not isinstance(release, dict):
        raise ValueError("Checkpoint metadata must be dictionaries")
    specification = VARIANTS[asset_id]
    expected_meta = {
        "schema_version": 1,
        "fold": fold,
        "model_class": "TripletModel",
        "license": LICENSE,
        **{
            key: specification[key]
            for key in (
                "dataset",
                "variant",
                "model_name",
                "input_resolution",
                "feature_dim",
                "num_classes",
                "triplet_size",
                "class_counts",
                "timm_version",
            )
        },
    }
    if any(meta.get(key) != value for key, value in expected_meta.items()):
        raise ValueError("Checkpoint spatial architecture/fold metadata mismatch")
    if type(meta["fold"]) is not int:
        raise ValueError("Checkpoint fold must be an integer")
    mapping = (
        Path(__file__).resolve().parents[1]
        / "curconmix_repro/temporal"
        / specification["metric_package"]
        / "maps.txt"
    )
    if meta.get("mapping_sha256") != hashlib.sha256(mapping.read_bytes()).hexdigest():
        raise ValueError("Checkpoint class mapping does not match the public evaluator")
    expected_release = {
        "schema_version": 1,
        "kind": "spatial_model_only",
        "dataset": specification["release_dataset"],
        "variant": specification["variant"],
        "fold": fold,
        "license": LICENSE,
        "embedded_prediction_pairing": "EXACT_FLOAT32",
    }
    if any(release.get(key) != value for key, value in expected_release.items()):
        raise ValueError("Checkpoint release identity mismatch")
    if type(release["fold"]) is not int:
        raise ValueError("Release fold must be an integer")
    for key in ("source_sha256", "canonical_prediction_sha256"):
        if not isinstance(release.get(key), str) or not re.fullmatch(
            r"[0-9a-f]{64}", release[key]
        ):
            raise ValueError(
                "Checkpoint provenance requires source and prediction SHA-256 values"
            )
    provenance = {
        "source_sha256": release["source_sha256"],
        "canonical_prediction_sha256": release["canonical_prediction_sha256"],
        "mapping_sha256": meta["mapping_sha256"],
    }
    for key, value in provenance.items():
        if manifest_item.get(key) != value:
            raise ValueError(f"Checkpoint provenance does not match manifest: {key}")
    if (
        not isinstance(state, dict)
        or not state
        or not all(isinstance(key, str) for key in state)
    ):
        raise ValueError("Expected a nonempty named model state")
    return state, meta, release


def _load_and_inspect(asset_id, root, fold, manifest_path, project_root):
    path, manifest_item = _validated_file(asset_id, root, fold, manifest_path)
    # All publication, file-identity and hash gates precede any PyTorch import.
    state, meta, release = _validate_payload(
        _safe_load(path), asset_id, fold, manifest_item
    )
    model = _strict_model_load(meta, state, project_root)
    summary = {
        "dataset": meta["dataset"],
        "variant": VARIANTS[asset_id]["public_variant"],
        "fold": fold,
        "model_name": meta["model_name"],
        "input_resolution": meta["input_resolution"],
        "num_classes": meta["num_classes"],
        "feature_dim": meta["feature_dim"],
        "qualified_timm_version": meta["timm_version"],
        "tensor_count": len(state),
        "sha256_status": "PASS",
        "strict_loading_status": "PASS",
        "embedded_prediction_pairing": release["embedded_prediction_pairing"],
        "feature_extraction_run": False,
        "inference_run": False,
    }
    return model, meta, summary


def load_spatial_checkpoint(
    asset_id,
    root,
    fold=0,
    manifest_path="manifests/spatial_assets.json",
    project_root=".",
):
    """Return ``(model, metadata)`` on CPU in evaluation mode; never run forward.

    Only immutable public model-only exports registered in a published manifest
    are accepted. Supplying an image preprocessing/feature-export pipeline and
    qualifying its ordering and numerical outputs remain separate responsibilities.
    """
    model, meta, _ = _load_and_inspect(
        asset_id, root, fold, manifest_path, project_root
    )
    return model, meta


def inspect_checkpoint(
    asset_id,
    root,
    fold=0,
    manifest_path="manifests/spatial_assets.json",
    project_root=".",
):
    """Verify file identity and strict architecture compatibility, without forward."""
    _, _, summary = _load_and_inspect(asset_id, root, fold, manifest_path, project_root)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", required=True, choices=VARIANTS)
    parser.add_argument(
        "--root", type=Path, default=Path("artifacts/spatial_checkpoints")
    )
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    parser.add_argument(
        "--manifest", type=Path, default=Path("manifests/spatial_assets.json")
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    try:
        print(
            json.dumps(
                inspect_checkpoint(
                    args.asset, args.root, args.fold, args.manifest, args.project_root
                ),
                indent=2,
            )
        )
        return 0
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        RuntimeError,
        ImportError,
        pickle.UnpicklingError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
