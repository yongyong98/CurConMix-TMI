"""Inspect a published LLS48 MRTT checkpoint locally, without inference.

Verify the release checksum before restricted loading, then require exact
architecture/fold metadata and strict model state loading. No features or images
are needed. This command does not validate numerical prediction reproducibility.
"""

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys

from .assets import load_manifest, validate_downloadable, verify_file
from curconmix_repro.validation import load_config, validate_metadata


VARIANTS = {
    "lls48-mrtt-b224": ("b224", "swinb224", "lls48_base224_plus.json"),
    "lls48-mrtt-b384": ("b384", "swinb384", "lls48_base384_plus.json"),
    "lls48-mrtt-l224": ("l224", "swinl224", "lls48_large224_plus.json"),
}
LICENSE = "CC-BY-NC-4.0"


def _safe_load(path):
    import torch

    # No fallback to unrestricted pickle loading and no custom global allowlist.
    return torch.load(path, map_location="cpu", weights_only=True)


def _strict_model_load(meta, state):
    import torch
    from curconmix_repro.temporal.temporal_model2 import MultiResTemporalModel2

    if not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("Released model state must contain tensors only")
    model = MultiResTemporalModel2(
        embed_dim=meta["embed_dim"],
        num_classes=meta["num_classes"],
        ks=tuple(meta["ks"]),
        fusion=meta["fusion"],
        use_spatial_norm=meta["use_spatial_norm"],
        spatial_dropout=meta["spatial_dropout"],
        ks_weights=meta["ks_weights"],
        temporal_cfg=meta["temporal_cfg"],
    )
    model.load_state_dict(state, strict=True)
    # Deliberately no model.forward(), feature/image load, optimizer or scheduler.


def inspect_checkpoint(asset_id, root, fold, manifest_path, config_root):
    if asset_id not in VARIANTS:
        raise ValueError(
            "Only the three published LLS48 MRTT configurations are supported"
        )
    if type(fold) is not int or fold not in range(5):
        raise ValueError("Fold must be an integer from 0 through 4")
    variant, public_variant, config_name = VARIANTS[asset_id]
    manifest = load_manifest(manifest_path)
    selected = [asset for asset in manifest["assets"] if asset["id"] == asset_id]
    if len(selected) != 1:
        raise ValueError(f"Asset is missing from the release manifest: {asset_id}")
    asset = selected[0]
    validate_downloadable(asset)
    expected_asset = {
        "kind": "checkpoint_bundle",
        "dataset": "lls48",
        "variant": public_variant,
        "temporal_model": "mr2",
        "folds": list(range(5)),
        "license": LICENSE,
    }
    if any(asset.get(key) != expected for key, expected in expected_asset.items()):
        raise ValueError("Manifest checkpoint identity/configuration mismatch")
    files = asset.get("files", [])
    if len(files) != 5 or sorted(item.get("fold", -1) for item in files) != list(
        range(5)
    ):
        raise ValueError("Manifest must contain exactly one checkpoint for every fold")
    for item in files:
        number = item["fold"]
        if type(number) is not int or item.get("format") != "file":
            raise ValueError(
                "Checkpoint entries must be plain files with integer fold IDs"
            )
        if (
            item.get("filename") != f"{asset_id}-fold{number}.pth"
            or item.get("install_path")
            != f"lls48/{variant}/best_model_fold{number}.pth"
        ):
            raise ValueError(
                "Manifest checkpoint filename/install path does not match its fold"
            )
    item = next(item for item in files if item["fold"] == fold)
    path = Path(root) / item["install_path"]
    verify_file(item, path)
    # Hash/size/identity gates have all passed before any PyTorch import/loading.
    config = load_config(Path(config_root) / config_name)
    checkpoint = _safe_load(path)
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
    validate_metadata(meta, config)
    if (
        type(meta.get("fold")) is not int
        or meta["fold"] != fold
        or meta.get("license") != LICENSE
    ):
        raise ValueError("Checkpoint fold/license metadata mismatch")
    mapping = (
        Path(__file__).resolve().parents[1]
        / "curconmix_repro/temporal/ivtmetrics_lls/maps.txt"
    )
    if meta.get("mapping_sha256") != hashlib.sha256(mapping.read_bytes()).hexdigest():
        raise ValueError("Checkpoint class mapping does not match the public evaluator")
    if any(
        release.get(key) != value
        for key, value in {
            "schema_version": 1,
            "dataset": "LLS48",
            "variant": variant,
            "fold": fold,
            "license": LICENSE,
        }.items()
    ):
        raise ValueError("Checkpoint release identity mismatch")
    if (
        not isinstance(state, dict)
        or not state
        or not all(isinstance(key, str) for key in state)
    ):
        raise ValueError("Expected a nonempty named model state")
    _strict_model_load(meta, state)
    return {
        "dataset": "lls48",
        "variant": public_variant,
        "fold": fold,
        "num_classes": meta["num_classes"],
        "embedding_dim": meta["embed_dim"],
        "tensor_count": len(state),
        "sha256_status": "PASS",
        "strict_loading_status": "PASS",
        "inference_run": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", required=True, choices=VARIANTS)
    parser.add_argument("--root", type=Path, default=Path("artifacts/checkpoints"))
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    parser.add_argument("--manifest", type=Path, default=Path("manifests/assets.json"))
    parser.add_argument("--config-root", type=Path, default=Path("configs/temporal"))
    args = parser.parse_args(argv)
    try:
        result = inspect_checkpoint(
            args.asset, args.root, args.fold, args.manifest, args.config_root
        )
        print(json.dumps(result, indent=2))
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
