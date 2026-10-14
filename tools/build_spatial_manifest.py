"""Prepare fail-closed metadata for fifteen qualified spatial students.

This local-only builder reads completed export QA and rehashes every checkpoint.
It never loads a model, uploads files, or publishes a release. --published is an
operator assertion that remote release verification has already succeeded.
Defaults are private and preserve existing files. Explicit output paths may
replace their previous generated outputs, but never the input manifest/weights.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from build_release_manifest import (  # noqa: E402
    checksum_text,
    local_file,
    public_paths_only,
    read_json,
    save_json,
    save_text,
)

REPOSITORY = "yongyong98/CurConMix-TMI"
TAG = "spatial-encoders-v1.0.0"
LICENSE = "CC-BY-NC-4.0"
VARIANTS = {
    "lls48": {"b224": "swinb224", "b384": "swinb384"},
    "cholect45": {"l224": "swinl224"},
}
CLASS_COUNTS = {
    "lls48": {"triplet": 195, "instrument": 17, "verb": 16, "target": 38},
    "cholect45": {"triplet": 100, "instrument": 6, "verb": 10, "target": 15},
}
CHECKSUM_FILENAME = "spatial_sha256sums.txt"


def release_url(filename, published):
    return (
        f"https://github.com/{REPOSITORY}/releases/download/{TAG}/{filename}"
        if published
        else None
    )


def _hash_field(record, field):
    value = record.get(field)
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"Missing or invalid {field}")
    return value


def prepare_dataset(dataset, checkpoint_dir, checkpoint_manifest, published):
    """Require every allowed fold for one dataset; never broaden its variant list."""
    source = read_json(checkpoint_manifest)
    if source.get("schema_version") != 1 or source.get("license") != LICENSE:
        raise ValueError("Spatial export schema/license mismatch")
    records = source.get("artifacts")
    variants = VARIANTS[dataset]
    expected = {(variant, fold) for variant in variants for fold in range(5)}
    if not isinstance(records, list) or len(records) != len(expected):
        raise ValueError(f"Complete unique {dataset} variant/fold exports are required")
    if any(
        not isinstance(record, dict) or type(record.get("fold")) is not int
        for record in records
    ):
        raise ValueError("Each spatial export requires an integer fold")
    identities = {(record.get("variant"), record["fold"]) for record in records}
    if identities != expected:
        raise ValueError(f"Complete unique {dataset} variant/fold exports are required")

    grouped = {variant: [] for variant in variants}
    plan_files = []
    prediction_hashes = {variant: set() for variant in variants}
    mapping_hashes = set()
    for record in sorted(records, key=lambda item: (item["variant"], item["fold"])):
        variant, fold = record["variant"], record["fold"]
        artifact_id = f"{dataset}-spatial-{variant}-fold{fold}"
        filename = f"{artifact_id}.pth"
        install_path = f"{dataset}/{variant}/fold{fold}.pth"
        if (
            record.get("dataset") != dataset
            or record.get("license") != LICENSE
            or record.get("id") != artifact_id
            or record.get("filename") != filename
            or record.get("install_path") != install_path
        ):
            raise ValueError(
                "Spatial artifact dataset/license/installation identity mismatch"
            )
        if (
            record.get("tensor_equality") != "PASS"
            or record.get("strict_model_load") != "PASS"
            or record.get("prediction_pairing") != "EXACT_FLOAT32"
            or record.get("source_preserved") is not True
        ):
            raise ValueError("Spatial checkpoint export QA did not pass")
        for field in ("model_tensor_count", "model_tensor_bytes"):
            if type(record.get(field)) is not int or record[field] <= 0:
                raise ValueError(f"Missing or invalid {field}")
        provenance = {
            field: _hash_field(record, field)
            for field in (
                "source_sha256",
                "canonical_prediction_sha256",
                "mapping_sha256",
            )
        }
        metadata = record.get("metadata", {})
        expected_metadata = {
            "schema_version": 1,
            "dataset": dataset,
            "variant": variant,
            "fold": fold,
            "model_class": "TripletModel",
            "num_classes": sum(CLASS_COUNTS[dataset].values()),
            "triplet_size": CLASS_COUNTS[dataset]["triplet"],
            "class_counts": CLASS_COUNTS[dataset],
            "mapping_sha256": provenance["mapping_sha256"],
            "license": LICENSE,
        }
        if not isinstance(metadata, dict) or any(
            metadata.get(key) != value for key, value in expected_metadata.items()
        ):
            raise ValueError("Spatial checkpoint architecture metadata mismatch")
        prediction_hashes[variant].add(provenance["canonical_prediction_sha256"])
        mapping_hashes.add(provenance["mapping_sha256"])
        # Hashes are always recomputed from export bytes, not trusted from QA alone.
        plan_files.append(local_file(checkpoint_dir, record, verify_hashes=True))
        grouped[variant].append(
            {
                "id": artifact_id,
                "filename": filename,
                "format": "file",
                "install_path": install_path,
                "fold": fold,
                "url": release_url(filename, published),
                "sha256": record["sha256"],
                "size_bytes": record["size_bytes"],
                **provenance,
            }
        )
    if len(mapping_hashes) != 1 or any(
        len(values) != 1 for values in prediction_hashes.values()
    ):
        raise ValueError(
            "Fold exports must share their variant prediction and dataset mapping"
        )

    assets = []
    for variant, public_variant in variants.items():
        assets.append(
            {
                "id": f"{dataset}-spatial-{variant}",
                "kind": "spatial_checkpoint_bundle",
                "dataset": dataset,
                "variant": public_variant,
                "architecture": "TripletModel",
                "folds": list(range(5)),
                "status": "published" if published else "unpublished",
                "available": published,
                "license": LICENSE,
                "license_status": "approved",
                "release_tag": TAG,
                "files": grouped[variant],
                "validation": {
                    "tensor_equality": "PASS",
                    "strict_model_load": "PASS",
                    "prediction_pairing": "EXACT_FLOAT32",
                    "source_preserved": True,
                    "export_sha256": "recomputed",
                },
                "release_note": (
                    "Five spatial student checkpoints with unchanged model tensors, strict "
                    "architecture loading, and exact float32 pairing to stored canonical "
                    "predictions. Export preparation performs no new inference. Matching "
                    "feature extraction and MRTT inputs require separate qualification."
                ),
            }
        )
    return assets, plan_files


def build(
    checkpoint_dir,
    checkpoint_manifest,
    cholect45_checkpoint_dir,
    cholect45_checkpoint_manifest,
    published=False,
):
    """Return public metadata and a private fifteen-weight plan, without writes."""
    assets = []
    plan = {"repository": REPOSITORY, "tag": TAG, "files": []}
    sources = (
        ("lls48", checkpoint_dir, checkpoint_manifest),
        ("cholect45", cholect45_checkpoint_dir, cholect45_checkpoint_manifest),
    )
    if any(not directory or not manifest for _, directory, manifest in sources):
        raise ValueError(
            "Both complete LLS48 and CholecT45 spatial exports are required"
        )
    for dataset, directory, manifest in sources:
        dataset_assets, files = prepare_dataset(dataset, directory, manifest, published)
        assets.extend(dataset_assets)
        plan["files"].extend(files)
    checksums = checksum_text(plan).encode("utf-8")
    result = {
        "schema_version": 2,
        "assets": assets,
        "checksums": {
            "filename": CHECKSUM_FILENAME,
            "url": release_url(CHECKSUM_FILENAME, published),
            "sha256": hashlib.sha256(checksums).hexdigest(),
            "size_bytes": len(checksums),
            "scope": "Fifteen spatial checkpoint files only; excludes this checksum file.",
        },
    }
    public_paths_only(result)
    return result, plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--checkpoint-manifest", required=True, type=Path)
    parser.add_argument("--cholect45-checkpoint-dir", required=True, type=Path)
    parser.add_argument("--cholect45-checkpoint-manifest", required=True, type=Path)
    parser.add_argument(
        "--output", type=Path, help="Explicit public JSON target may replace it"
    )
    parser.add_argument(
        "--upload-plan",
        "--private-upload-plan",
        dest="upload_plan",
        type=Path,
        help="Private local-path plan; explicit target may replace it",
    )
    parser.add_argument(
        "--checksums-output",
        type=Path,
        help=f"Public checksum text; basename must be {CHECKSUM_FILENAME}",
    )
    parser.add_argument(
        "--published",
        action="store_true",
        help="Operator confirms all sixteen remote release assets are already public and verified",
    )
    args = parser.parse_args(argv)
    output = args.output or ROOT / ".release-local/pending_spatial_assets.json"
    upload_plan = args.upload_plan or ROOT / ".release-local/spatial_upload_plan.json"
    checksums_output = (
        args.checksums_output or ROOT / ".release-local" / CHECKSUM_FILENAME
    )
    outputs = (
        (output, args.output),
        (upload_plan, args.upload_plan),
        (checksums_output, args.checksums_output),
    )
    if checksums_output.name != CHECKSUM_FILENAME:
        raise ValueError(f"Checksum basename must be {CHECKSUM_FILENAME}")
    resolved = [str(path.resolve()).casefold() for path, _ in outputs]
    if len(set(resolved)) != 3:
        raise ValueError(
            "Public manifest, private upload plan and checksums must be separate"
        )
    protected = {
        str(args.checkpoint_manifest.resolve()).casefold(),
        str(args.cholect45_checkpoint_manifest.resolve()).casefold(),
    }
    for dataset, directory in (
        ("lls48", args.checkpoint_dir),
        ("cholect45", args.cholect45_checkpoint_dir),
    ):
        protected.update(
            str(
                (directory / f"{dataset}-spatial-{variant}-fold{fold}.pth").resolve()
            ).casefold()
            for variant in VARIANTS[dataset]
            for fold in range(5)
        )
    if protected.intersection(resolved):
        raise ValueError("Outputs must not overwrite input metadata or weights")
    for path, explicit in outputs:
        if path.exists() and explicit is None:
            raise FileExistsError(f"Existing default output preserved: {path}")

    result, plan = build(
        args.checkpoint_dir,
        args.checkpoint_manifest,
        args.cholect45_checkpoint_dir,
        args.cholect45_checkpoint_manifest,
        args.published,
    )
    checksums = checksum_text(plan)
    checksum_record = result["checksums"]
    plan["files"].append(
        {
            "path": str(checksums_output.resolve()),
            "sha256": checksum_record["sha256"],
            "size_bytes": checksum_record["size_bytes"],
        }
    )
    save_text(checksums_output, checksums, replace=args.checksums_output is not None)
    save_json(output, result, replace=args.output is not None)
    save_json(upload_plan, plan, replace=args.upload_plan is not None)
    print(
        json.dumps(
            {
                "manifest": str(output),
                "upload_plan": str(upload_plan),
                "checksums": str(checksums_output),
                "upload_files": len(plan["files"]),
                "published_metadata": args.published,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
