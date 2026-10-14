"""Export model-only LLS48 MRTT release weights without training or inference.

Maintainer utility: the registry and private report are local-only provenance.
The public output has no optimizer, scheduler, training arguments or source paths.
Requires PyTorch >=2.6 only for reading the original NumPy-scalar metadata safely.
Released weights themselves contain tensors and pure Python metadata only.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

VARIANTS = {
    "b224": "lls48_base224_plus.json",
    "b384": "lls48_base384_plus.json",
    "l224": "lls48_large224_plus.json",
}
META_FIELDS = (
    "model_type",
    "embed_dim",
    "num_classes",
    "ks",
    "fusion",
    "use_spatial_norm",
    "spatial_dropout",
    "ks_weights",
    "temporal_cfg",
    "dataset",
    "name",
)
ALLOWED_NUMPY_GLOBALS = {"numpy.core.multiarray.scalar", "numpy.dtype"}
LICENSE_ID = "CC-BY-NC-4.0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def canonical_sources(registry: Path) -> list[dict]:
    """Do not select a checkpoint by partial filename, score or recency."""
    with registry.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    records = []
    for variant in VARIANTS:
        model_id = f"lls48_curconmix_plus_{variant}"
        selected = [row for row in rows if row["model_id"] == model_id]
        if len(selected) != 1:
            raise ValueError(f"Expected one canonical registry row for {model_id}")
        row = selected[0]
        for field in ("ap_verification_status", "row_gt_verification_status"):
            if row.get(field) != "VERIFIED":
                raise ValueError(f"Registry gate failed: {model_id}: {field}")
        if row.get("manuscript_eligibility") != "YES":
            raise ValueError(f"Registry model is not manuscript eligible: {model_id}")
        if row.get("dataset") != "LLS48" or row.get("temporal_module") != "MRTT":
            raise ValueError(f"Wrong dataset/module: {model_id}")
        paths = [Path(item) for item in row["absolute_checkpoint_paths"].split("|")]
        if len(paths) != 5 or len(set(paths)) != 5:
            raise ValueError(f"Exactly five unique checkpoints required: {model_id}")
        by_fold = {}
        for path in paths:
            match = re.fullmatch(r"best_model_fold([0-4])\.pth", path.name)
            if not match or not path.is_file():
                raise ValueError(f"Missing/noncanonical fold checkpoint: {path}")
            fold = int(match.group(1))
            if fold in by_fold:
                raise ValueError(f"Duplicate fold: {model_id}: {fold}")
            by_fold[fold] = path
        if set(by_fold) != set(range(5)):
            raise ValueError(f"Missing fold: {model_id}")
        for fold, path in sorted(by_fold.items()):
            records.append(
                {
                    "variant": variant,
                    "model_id": model_id,
                    "fold": fold,
                    "source": path,
                    "source_config": Path(row["absolute_config_path"]),
                }
            )
    return records


def restricted_original_load(path: Path):
    import numpy as np
    import torch

    if not hasattr(torch.serialization, "get_unsafe_globals_in_checkpoint"):
        raise RuntimeError("Original checkpoint export requires PyTorch >=2.6")
    globals_found = set(torch.serialization.get_unsafe_globals_in_checkpoint(path))
    if not globals_found.issubset(ALLOWED_NUMPY_GLOBALS):
        raise ValueError(f"Unexpected checkpoint globals: {sorted(globals_found)}")
    # Original best_mAP is a NumPy float64; no arbitrary/custom objects are allowed.
    scalar = getattr(np, "_core", np.core).multiarray.scalar
    allowlist = [
        (scalar, "numpy.core.multiarray.scalar"),
        np.dtype,
        type(np.dtype("float64")),
    ]
    with torch.serialization.safe_globals(allowlist):
        result = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(result, dict) or not {"model", "meta"}.issubset(result):
        raise ValueError("Expected original model+meta checkpoint schema")
    state = result["model"]
    if not isinstance(state, dict) or not state:
        raise ValueError("Empty or non-dictionary model state")
    if not all(
        isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()
    ):
        raise ValueError("Model state must contain named tensors only")
    return result


def sanitized_metadata(
    meta: dict, public_config: dict, fold: int, mapping_hash: str
) -> dict:
    from curconmix_repro.validation import validate_metadata

    if fold not in range(5):
        raise ValueError("Fold must be 0 through 4")
    if not re.fullmatch(r"[0-9a-f]{64}", mapping_hash):
        raise ValueError("Invalid mapping SHA-256")
    if any(field not in meta for field in META_FIELDS):
        raise ValueError("Complete architecture metadata is required")
    selected = {field: meta[field] for field in META_FIELDS}
    validate_metadata(selected, public_config)
    # Round-trip rejects custom values; strict known config values reject private paths.
    selected = json.loads(json.dumps(selected, allow_nan=False))
    selected.update(
        {"fold": fold, "license": LICENSE_ID, "mapping_sha256": mapping_hash}
    )
    return selected


def validate_original_config(meta: dict, config: dict) -> None:
    """Legacy JSON is partial; check every present architecture field, never fill gaps."""
    aliases = {"embed_dim": "dim", "name": "model_name"}
    required = {
        "model_type",
        "dim",
        "dataset",
        "model_name",
        "ks",
        "fusion",
        "use_spatial_norm",
        "ks_weights",
    }
    if not required.issubset(config):
        raise ValueError("Canonical original configuration is incomplete")
    for field in META_FIELDS:
        config_key = aliases.get(field, field)
        if config_key in config and config[config_key] != meta[field]:
            raise ValueError(f"Original configuration disagrees with metadata: {field}")


def verify_state_equal(original: dict, exported: dict) -> None:
    import torch

    if list(original) != list(exported):
        raise ValueError("State key order/names changed")
    for key, before in original.items():
        after = exported[key]
        if before.shape != after.shape or before.dtype != after.dtype:
            raise ValueError(f"Tensor shape/dtype changed: {key}")
        if not torch.equal(before, after):
            raise ValueError(f"Tensor value changed: {key}")


def strict_model_load(meta: dict, state: dict) -> None:
    from curconmix_repro.temporal.temporal_model2 import MultiResTemporalModel2

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
    # Intentionally no forward pass, optimizer or scheduler.
    del model


def export_all(args) -> dict:
    import torch

    records = canonical_sources(args.registry)
    mapping = ROOT / "src/curconmix_repro/temporal/ivtmetrics_lls/maps.txt"
    mapping_hash = sha256(mapping)
    report = {
        "schema_version": 1,
        "registry_path": str(args.registry.resolve()),
        "registry_sha256": sha256(args.registry),
        "mapping_sha256": mapping_hash,
        "runtime_torch": torch.__version__,
        "training_or_inference_run": False,
        "safe_load": "weights_only=True with NumPy float64 scalar/dtype allowlist only",
        "exports": [],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    args.private_report.parent.mkdir(parents=True, exist_ok=True)
    public = {"schema_version": 1, "license": LICENSE_ID, "artifacts": []}
    for record in records:
        variant, fold, source = record["variant"], record["fold"], record["source"]
        filename = f"lls48-mrtt-{variant}-fold{fold}.pth"
        destination = args.output / filename
        temporary = destination.with_suffix(".pth.part")
        if not args.inspect_only and (destination.exists() or temporary.exists()):
            raise FileExistsError(
                f"Refusing to overwrite release weight: {destination}"
            )
        print(f"Inspecting {variant} fold {fold}", flush=True)
        source_hash = sha256(source)
        original = restricted_original_load(source)
        config = read_json(ROOT / "configs/temporal" / VARIANTS[variant])
        source_config = read_json(record["source_config"])
        validate_original_config(original["meta"], source_config)
        sidecar = read_json(source.with_name(f"experiment_fold{fold}.json"))
        if sidecar != original["meta"]:
            raise ValueError("Embedded metadata differs from canonical sidecar")
        meta = sanitized_metadata(original["meta"], config, fold, mapping_hash)
        model_bytes = sum(
            t.numel() * t.element_size() for t in original["model"].values()
        )
        if args.inspect_only:
            strict_model_load(meta, original["model"])
            report["exports"].append(
                {
                    "variant": variant,
                    "fold": fold,
                    "source_path": str(source),
                    "source_sha256": source_hash,
                    "source_size_bytes": source.stat().st_size,
                    "model_tensor_bytes": model_bytes,
                    "model_tensor_count": len(original["model"]),
                    "metadata": meta,
                    "strict_model_load": "PASS",
                    "sidecar_match": "PASS",
                    "fold_identity_basis": "canonical registry exact fold filename; original has no fold ID",
                    "source_args_folds": original.get("args", {}).get("folds"),
                    "embedded_source_fold": original["meta"].get("fold"),
                }
            )
            args.private_report.write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
            print(
                f"Inspected {variant} fold {fold}: {model_bytes} model bytes; strict load PASS",
                flush=True,
            )
            del original
            gc.collect()
            continue
        if model_bytes >= 2 * 1024**3:
            raise ValueError("Model-only checkpoint exceeds release asset size limit")
        reserve = int(args.reserve_gib * 1024**3)
        if shutil.disk_usage(args.output).free < model_bytes + reserve + 1024**2:
            raise OSError("Insufficient free disk for checkpoint plus reserved space")
        strict_model_load(meta, original["model"])
        exported = {
            "model": original["model"],
            "meta": meta,
            "release": {
                "schema_version": 1,
                "kind": "mrt_tensors_only",
                "dataset": "LLS48",
                "variant": variant,
                "fold": fold,
                "license": LICENSE_ID,
                "source_sha256": source_hash,
                "fold_identity_basis": "canonical_registry_and_source_filename",
            },
        }
        torch.save(exported, temporary)
        checked = torch.load(temporary, map_location="cpu", weights_only=True)
        if set(checked) != {"model", "meta", "release"}:
            raise ValueError("Unexpected model-only export fields")
        if checked["meta"] != meta or checked["release"] != exported["release"]:
            raise ValueError("Export metadata changed")
        verify_state_equal(original["model"], checked["model"])
        if sha256(source) != source_hash:
            raise ValueError("Original checkpoint changed during export")
        temporary.replace(destination)
        result = {
            "id": f"lls48-mrtt-{variant}-fold{fold}",
            "variant": variant,
            "fold": fold,
            "filename": filename,
            "size_bytes": destination.stat().st_size,
            "sha256": sha256(destination),
            "source_sha256": source_hash,
            "mapping_sha256": mapping_hash,
            "license": LICENSE_ID,
            "target_path": f"artifacts/checkpoints/lls48/{variant}/best_model_fold{fold}.pth",
            "config": f"configs/temporal/{VARIANTS[variant]}",
            "model_tensor_count": len(original["model"]),
            "model_tensor_bytes": model_bytes,
            "tensor_equality": "PASS",
            "strict_model_load": "PASS",
            "architecture_metadata": "PASS",
            "source_preserved": True,
        }
        public["artifacts"].append(result)
        report["exports"].append(
            {
                **result,
                "source_path": str(source),
                "output_path": str(destination.resolve()),
            }
        )
        args.private_report.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"Exported {filename}: {result['size_bytes']} bytes; all checks PASS",
            flush=True,
        )
        del original, exported, checked
        gc.collect()
    if not args.inspect_only:
        args.public_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.public_manifest.write_text(
            json.dumps(public, indent=2) + "\n", encoding="utf-8"
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--private-report", type=Path, required=True)
    parser.add_argument("--public-manifest", type=Path, required=True)
    parser.add_argument("--reserve-gib", type=float, default=6.0)
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Read/validate originals; do not export weights",
    )
    args = parser.parse_args()
    if args.reserve_gib < 6:
        parser.error("At least 6 GiB must remain reserved")
    export_all(args)


if __name__ == "__main__":
    main()
