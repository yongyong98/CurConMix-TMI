"""Derive public metadata and a private upload plan from checked release exports.

No network, Git, upload, or publication occurs here. By default, all prepared
assets remain unpublished with null URLs. --published is an explicit operator
assertion that the corresponding assets have already been publicly released.
Local sizes and source QA are always checked. The uploader rehashes every file;
--verify-file-hashes additionally rehashes them while preparing these manifests.
"""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from curconmix_release.assets import _assert_plain_path, _relative_path  # noqa: E402

REPOSITORY = "yongyong98/CurConMix-TMI"
TAG = "lls48-v1.0.0"
LICENSE = "CC-BY-NC-4.0"
VARIANTS = ("b224", "b384", "l224")
EXPECTED_IDS = {"lls48-dataset"} | {
    f"{dataset}-mrtt-{variant}"
    for dataset in ("lls48", "cholect45")
    for variant in VARIANTS
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def hash_size(record, allow_empty=False):
    if not re.fullmatch(r"[0-9a-f]{64}", record.get("sha256", "")):
        raise ValueError("Missing or invalid SHA-256")
    size = record.get("size_bytes")
    if type(size) is not int or size < (0 if allow_empty else 1):
        raise ValueError("Missing or invalid size")


def local_file(directory, record, verify_hashes):
    name = _relative_path(record["filename"])
    if len(name.parts) != 1:
        raise ValueError("Release filenames must be safe basenames")
    hash_size(record)
    if record["size_bytes"] >= 2 * 1024**3:
        raise ValueError("Release files must be below 2 GiB")
    path = Path(directory) / name
    _assert_plain_path(path)
    if not path.is_file() or path.stat().st_size != record["size_bytes"]:
        raise ValueError(f"Local size mismatch or missing file: {name}")
    if verify_hashes and digest(path) != record["sha256"]:
        raise ValueError(f"Local SHA-256 mismatch: {name}")
    return {
        "path": str(path.resolve()),
        "size_bytes": record["size_bytes"],
        "sha256": record["sha256"],
    }


def public_file(record, published):
    return {
        "filename": record["filename"],
        "url": (
            f"https://github.com/{REPOSITORY}/releases/download/{TAG}/{record['filename']}"
            if published
            else None
        ),
        "sha256": record["sha256"],
        "size_bytes": record["size_bytes"],
    }


def public_paths_only(value):
    if isinstance(value, dict):
        for child in value.values():
            public_paths_only(child)
    elif isinstance(value, list):
        for child in value:
            public_paths_only(child)
    elif isinstance(value, str):
        if re.search(
            r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]|\\\\|/(?:Users|home|mnt)/", value
        ):
            raise ValueError("Private absolute path found in public metadata")


def prepare_dataset(directory, published, verify_hashes):
    source = read_json(Path(directory) / "release_files.json")
    qa = source.get("validation", {})
    if (
        any(
            qa.get(key) != "passed"
            for key in ("zip_crc", "membership", "per_member_sha256")
        )
        or qa.get("source_annotation_unchanged") is not True
    ):
        raise ValueError("Dataset source QA did not pass")
    summary = source.get("summary", {})
    if any(
        summary.get(key) != count
        for key, count in (("frames", 9560), ("clips", 1912), ("procedures", 48))
    ):
        raise ValueError(
            "Dataset is not the canonical 9560-frame/1912-clip/48-procedure selection"
        )
    files, plan, seen = [], [], set()
    image_count = 0
    for record in source["files"]:
        if record.get("format") != "zip" or record.get("install_path") != ".":
            raise ValueError("Dataset parts must be root-relative ZIPs")
        plan.append(local_file(directory, record, verify_hashes))
        result = public_file(record, published)
        members = []
        for member in record["members"]:
            path = _relative_path(member["path"]).as_posix()
            if path.casefold() in seen:
                raise ValueError("Duplicate dataset member across parts")
            seen.add(path.casefold())
            hash_size(member, allow_empty=True)
            members.append(
                {
                    "path": path,
                    "size_bytes": member["size_bytes"],
                    "sha256": member["sha256"],
                }
            )
            image_count += path.startswith("data/")
        if (
            not members
            or sum(m["size_bytes"] for m in members) != record["expanded_size_bytes"]
        ):
            raise ValueError("Dataset expanded-size/member list mismatch")
        result.update(
            {
                "format": "zip",
                "install_path": ".",
                "expanded_size_bytes": record["expanded_size_bytes"],
                "member_count": len(members),
                "members": members,
            }
        )
        files.append(result)
    required = {
        "dataframe/lls48_tri195.csv",
        "splits/folds.json",
        "manifests/frames.json",
        "manifests/checksums.json",
        "data_and_model_license.md",
    }
    if image_count != 9560 or not required.issubset(seen):
        raise ValueError("Dataset image/member inventory is incomplete")
    annotation = next(
        m
        for f in files
        for m in f["members"]
        if m["path"] == "dataframe/LLS48_TRI195.csv"
    )
    if annotation["sha256"] != summary["annotation_sha256"]:
        raise ValueError("Dataset annotation checksum mismatch")
    return files, plan, summary


def prepare_checkpoints(directory, manifest, published, verify_hashes):
    source = read_json(manifest)
    if source.get("license") != LICENSE:
        raise ValueError("Checkpoint license mismatch")
    records = source.get("artifacts", [])
    identities = [(r.get("variant"), r.get("fold")) for r in records]
    expected = {(variant, fold) for variant in VARIANTS for fold in range(5)}
    if len(records) != 15 or set(identities) != expected:
        raise ValueError(
            "Exactly 15 unique checkpoint variant/fold exports are required"
        )
    grouped = {variant: [] for variant in VARIANTS}
    plan = []
    for record in sorted(records, key=lambda r: (r["variant"], r["fold"])):
        if (
            any(
                record.get(key) != "PASS"
                for key in (
                    "tensor_equality",
                    "strict_model_load",
                    "architecture_metadata",
                )
            )
            or record.get("source_preserved") is not True
        ):
            raise ValueError("Checkpoint source QA did not pass")
        if record.get("license") != LICENSE:
            raise ValueError("Checkpoint artifact license mismatch")
        variant, fold = record["variant"], record["fold"]
        expected_name = f"lls48-mrtt-{variant}-fold{fold}.pth"
        install = f"lls48/{variant}/best_model_fold{fold}.pth"
        if (
            record["filename"] != expected_name
            or record["target_path"] != f"artifacts/checkpoints/{install}"
        ):
            raise ValueError("Checkpoint filename or installation identity mismatch")
        plan.append(local_file(directory, record, verify_hashes))
        result = public_file(record, published)
        result.update(
            {
                "id": record["id"],
                "format": "file",
                "install_path": install,
                "fold": fold,
            }
        )
        grouped[variant].append(result)
    return grouped, plan


def build(
    base_manifest,
    dataset_dir=None,
    checkpoint_dir=None,
    checkpoint_manifest=None,
    published=False,
    verify_file_hashes=False,
):
    if bool(checkpoint_dir) != bool(checkpoint_manifest):
        raise ValueError("Provide checkpoint directory and manifest together")
    if not dataset_dir and not checkpoint_dir:
        raise ValueError("Provide at least one checked dataset or checkpoint export")
    base = read_json(base_manifest)
    result = copy.deepcopy(base)
    assets = result.get("assets", [])
    ids = [a["id"] for a in assets]
    if len(ids) != 7 or set(ids) != EXPECTED_IDS:
        raise ValueError(
            "Base manifest must contain exactly the seven canonical asset IDs"
        )
    result["schema_version"] = 2
    indexed = {a["id"]: a for a in assets}
    for asset in assets:
        if asset.get("license") != LICENSE or asset.get("license_status") != "approved":
            raise ValueError("Base release license must already be approved")
        asset.update(
            {
                "status": "unpublished",
                "available": False,
                "url": None,
                "sha256": None,
                "size_bytes": None,
                "files": [],
            }
        )
    plan = {"repository": REPOSITORY, "tag": TAG, "files": []}
    prepared = []
    if dataset_dir:
        files, entries, summary = prepare_dataset(
            dataset_dir, published, verify_file_hashes
        )
        asset = indexed["lls48-dataset"]
        asset.update(
            {
                "files": files,
                "annotation_sha256": summary["annotation_sha256"],
                "release_tag": TAG,
            }
        )
        asset[
            "release_note"
        ] = "Original approved LLS48 images and annotations; 1,912 independent five-frame clips from 48 procedures. No features or checkpoints are included."
        prepared.append(asset)
        plan["files"].extend(entries)
    if checkpoint_dir:
        grouped, entries = prepare_checkpoints(
            checkpoint_dir, checkpoint_manifest, published, verify_file_hashes
        )
        for variant, files in grouped.items():
            asset = indexed[f"lls48-mrtt-{variant}"]
            asset.update({"files": files, "release_tag": TAG})
            asset[
                "release_note"
            ] = "Five MRTT checkpoints exported with unchanged model tensors and verified strict architecture loading. Matching spatial encoder is not included; aligned input features are required. No end-to-end inference reproduction is claimed."
            prepared.append(asset)
        plan["files"].extend(entries)
    for asset in prepared:
        asset.update(
            {
                "status": "published" if published else "unpublished",
                "available": published,
            }
        )
    names = [Path(item["path"]).name.casefold() for item in plan["files"]]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate release asset basenames")
    public_paths_only(result)
    return result, plan


def checksum_text(plan):
    """Return standard SHA256SUMS lines containing basenames, never local paths."""
    entries = {}
    seen = set()
    for record in plan["files"]:
        hash_size(record)
        filename = re.split(r"[\\/]", str(record["path"]))[-1]
        _relative_path(filename)
        if filename.casefold() in seen:
            raise ValueError("Duplicate checksum filename")
        seen.add(filename.casefold())
        entries[filename] = record["sha256"]
    if not entries:
        raise ValueError("Checksum list must not be empty")
    return "".join(f"{entries[name]}  {name}\n" for name in sorted(entries))


def save_text(path, content, replace=False):
    path = Path(path)
    _assert_plain_path(path)
    if path.exists() and not replace:
        raise FileExistsError(f"Existing default output preserved: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=".manifest-",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_json(path, payload, replace=False):
    save_text(path, json.dumps(payload, indent=2) + "\n", replace=replace)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-manifest", type=Path, default=ROOT / "manifests/assets.json"
    )
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--checkpoint-manifest", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="Explicit target may replace existing JSON; default preserves existing .release-local/pending_assets.json",
    )
    parser.add_argument(
        "--upload-plan",
        type=Path,
        help="Private plan; explicit target may replace existing JSON",
    )
    parser.add_argument(
        "--published",
        action="store_true",
        help="Operator confirms these files are already public at the release URLs",
    )
    parser.add_argument(
        "--checksums-output",
        type=Path,
        help="Optional public SHA256SUMS text; explicit target may replace an existing file",
    )
    parser.add_argument("--verify-file-hashes", action="store_true")
    args = parser.parse_args(argv)
    result, plan = build(
        args.base_manifest,
        args.dataset_dir,
        args.checkpoint_dir,
        args.checkpoint_manifest,
        args.published,
        args.verify_file_hashes,
    )
    output = args.output or ROOT / ".release-local/pending_assets.json"
    upload_plan = args.upload_plan or ROOT / ".release-local/upload_plan.json"
    output_paths = [output, upload_plan]
    if args.checksums_output is not None:
        output_paths.append(args.checksums_output)
    if len({path.resolve() for path in output_paths}) != len(output_paths):
        raise ValueError(
            "Public manifest, private upload plan and checksums must be separate"
        )
    checksums = checksum_text(plan) if args.checksums_output is not None else None
    for path, explicit in ((output, args.output), (upload_plan, args.upload_plan)):
        if path.exists() and explicit is None:
            raise FileExistsError(f"Existing default output preserved: {path}")
    save_json(output, result, replace=args.output is not None)
    save_json(upload_plan, plan, replace=args.upload_plan is not None)
    if args.checksums_output is not None:
        save_text(args.checksums_output, checksums, replace=True)
    print(
        json.dumps(
            {
                "manifest": str(output),
                "upload_plan": str(upload_plan),
                "upload_files": len(plan["files"]),
                "published_metadata": args.published,
                "checksums": str(args.checksums_output)
                if args.checksums_output
                else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
