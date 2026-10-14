"""Prepare and verify the immutable, annotation-selected LLS48 release.

This module never trains, infers, relabels, transforms images, or changes sources.
"""

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import zipfile


MASTER_RELATIVE = "dataframe/LLS48_TRI195.csv"
MASTER_SHA256 = "6835eb20fb1272088fdf263a9c417dbc7bd56e0a4e5c5580f513496ca947af50"
MAPPING_COUNTS = {"instrument": 17, "verb": 16, "target": 38, "triplet": 195}
METADATA_COLUMNS = [
    "video",
    "clip_name",
    "image_path",
    "image_id",
    "triplets_idx",
    "inst_idx",
    "verb_idx",
    "target_idx",
    "step",
    "task",
]
LABEL_COLUMNS = (
    [f"tri{i}" for i in range(195)]
    + [f"inst{i}" for i in range(17)]
    + [f"v{i}" for i in range(16)]
    + [f"t{i}" for i in range(38)]
)
MAX_SHARD_BYTES = 1024**3
MIN_FREE_AFTER_BYTES = 6 * 1024**3
CANONICAL_FOLDS = {
    "0": [
        "VID10",
        "VID15",
        "VID22",
        "VID27",
        "VID30",
        "VID39",
        "VID42",
        "VID46",
        "VID47",
    ],
    "1": [
        "VID04",
        "VID06",
        "VID07",
        "VID08",
        "VID18",
        "VID29",
        "VID32",
        "VID36",
        "VID37",
        "VID43",
    ],
    "2": [
        "VID02",
        "VID05",
        "VID13",
        "VID16",
        "VID19",
        "VID24",
        "VID28",
        "VID31",
        "VID45",
    ],
    "3": [
        "VID01",
        "VID03",
        "VID20",
        "VID23",
        "VID26",
        "VID35",
        "VID38",
        "VID40",
        "VID44",
        "VID48",
    ],
    "4": [
        "VID09",
        "VID11",
        "VID12",
        "VID14",
        "VID17",
        "VID21",
        "VID25",
        "VID33",
        "VID34",
        "VID41",
    ],
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_relative(value):
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError("Unsafe relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(p in ("", ".", "..") for p in value.split("/")):
        raise ValueError("Unsafe relative path")
    if any(p.endswith((".", " ")) for p in path.parts):
        raise ValueError("Unsafe relative path")
    return path


def source_file(root, relative):
    root = Path(root).resolve()
    path = root.joinpath(*safe_relative(relative).parts)
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError("Source resolves outside dataset root or is a symlink")
    if not path.is_file():
        raise FileNotFoundError(f"Required dataset file is missing: {relative}")
    return path


def read_canonical_folds(preprocess_path):
    """Read literal fold membership without importing the historical trainer."""
    tree = ast.parse(Path(preprocess_path).read_text(encoding="utf-8-sig"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "split_selector":
            for statement in node.body:
                if isinstance(statement, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "switcher"
                    for t in statement.targets
                ):
                    split = ast.literal_eval(statement.value)["LLS48"]
                    result = {
                        str(f): [f"VID{x:02d}" for x in ids] for f, ids in split.items()
                    }
                    videos = [v for values in result.values() for v in values]
                    if set(result) != set(map(str, range(5))) or len(videos) != 48:
                        raise ValueError("Unexpected LLS48 five-fold definition")
                    if len(set(videos)) != 48:
                        raise ValueError("A procedure appears in multiple folds")
                    if result != CANONICAL_FOLDS:
                        raise ValueError(
                            "Fold assignment differs from the released benchmark"
                        )
                    return result
    raise ValueError("Canonical LLS48 split was not found")


def inspect_annotations(root, require_canonical=True):
    master = source_file(root, MASTER_RELATIVE)
    digest = sha256_file(master)
    if require_canonical and digest != MASTER_SHA256:
        raise ValueError(
            "Master annotation SHA-256 does not match the canonical release"
        )
    with master.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != METADATA_COLUMNS + LABEL_COLUMNS:
            raise ValueError("Annotation column names or order changed")
        rows = list(reader)
    clips, paths, procedures = {}, set(), set()
    for index, row in enumerate(rows):
        if not re.fullmatch(r"VID\d{2}", row["video"]):
            raise ValueError("Unexpected procedure identifier")
        if not re.fullmatch(r"\d{2}_\d{3}", row["clip_name"]):
            raise ValueError("Unexpected clip identifier")
        image = safe_relative(row["image_path"])
        if len(image.parts) != 3 or image.parts[:2] != (row["video"], row["clip_name"]):
            raise ValueError("Image path does not match its procedure and clip")
        if not re.fullmatch(r"frame_\d{5}\.png", image.name):
            raise ValueError("Unexpected image filename")
        if row["image_path"] in paths:
            raise ValueError("Duplicate image path in annotation")
        if not all(row[c] in ("0", "1") for c in LABEL_COLUMNS):
            raise ValueError("Action labels must be binary")
        if not re.fullmatch(r"\d+_\d+", row["image_id"]):
            raise ValueError("Unexpected image identifier")
        if int(row["step"]) not in range(5) or int(row["task"]) not in range(18):
            raise ValueError("Unexpected hierarchical label index")
        paths.add(row["image_path"])
        procedures.add(row["video"])
        clips.setdefault((row["video"], row["clip_name"]), []).append(index)
    if any(len(indices) != 5 for indices in clips.values()):
        raise ValueError("Every independent clip must contain exactly five frames")
    for indices in clips.values():
        if indices != list(range(indices[0], indices[0] + 5)):
            raise ValueError("Clip rows must remain contiguous in the source order")
        frames = [
            int(PurePosixPath(rows[i]["image_path"]).stem.split("_")[1])
            for i in indices
        ]
        if frames != [1, 2, 3, 4, 5]:
            raise ValueError("Within-clip frame order must remain 1 through 5")
    counts = {"frames": len(rows), "clips": len(clips), "procedures": len(procedures)}
    if require_canonical and counts != {
        "frames": 9560,
        "clips": 1912,
        "procedures": 48,
    }:
        raise ValueError("Canonical dataset counts changed")
    return rows, clips, {**counts, "annotation_sha256": digest}


def validate_mappings(root, numeric_mapping):
    mappings = {}
    for kind, expected_count in MAPPING_COUNTS.items():
        path = source_file(root, f"dict/{kind}_class_mapping.txt")
        entries = {}
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            index, name = line.split(":", 1)
            if int(index) in entries:
                raise ValueError("Duplicate class mapping index")
            entries[int(index)] = name.strip()
        if set(entries) != set(range(expected_count)):
            raise ValueError(f"Unexpected {kind} class mapping")
        mappings[kind] = entries
    with Path(numeric_mapping).open(encoding="utf-8-sig", newline="") as handle:
        numeric_rows = [r for r in csv.reader(handle) if r and not r[0].startswith("#")]
    if len(numeric_rows) != 195 or [int(r[0]) for r in numeric_rows] != list(
        range(195)
    ):
        raise ValueError("Unexpected canonical numeric triplet mapping")
    for row in numeric_rows:
        index, instrument, verb, target = map(int, row[:4])
        label = ", ".join(
            [
                mappings["instrument"][instrument],
                mappings["verb"][verb],
                mappings["target"][target],
            ]
        )
        if mappings["triplet"][index] != label:
            raise ValueError(
                f"Triplet class text disagrees with canonical mapping: {index}"
            )
    return mappings


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def record(path, relative):
    return {
        "path": str(safe_relative(relative)),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def shard_members(members, maximum=MAX_SHARD_BYTES):
    """Keep ZIP_STORED payload plus conservative ZIP headers below the cap."""
    groups, current, used = [], [], 65536
    for member in members:
        cost = member["size_bytes"] + 2 * len(member["path"].encode("utf-8")) + 256
        if cost + 65536 > maximum:
            raise ValueError("One member exceeds the archive size limit")
        if current and used + cost > maximum:
            groups.append(current)
            current, used = [], 65536
        current.append(member)
        used += cost
    if current:
        groups.append(current)
    return groups


def write_archive(
    destination, members, root=None, payloads=None, maximum=MAX_SHARD_BYTES
):
    """Exclusive deterministic ZIP writer with membership, CRC and SHA checks."""
    destination = Path(destination)
    names = [str(safe_relative(item["path"])) for item in members]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate archive members")
    with zipfile.ZipFile(
        destination, "x", compression=zipfile.ZIP_STORED, allowZip64=True
    ) as archive:
        for item in members:
            info = zipfile.ZipInfo(item["path"], date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_STORED
            info.file_size = item["size_bytes"]
            with archive.open(info, "w") as output:
                if payloads is not None:
                    output.write(payloads[item["path"]])
                else:
                    with source_file(root, item["path"]).open("rb") as source:
                        shutil.copyfileobj(source, output, 4 * 1024**2)
    if destination.stat().st_size >= maximum:
        raise ValueError("Archive exceeds its size limit")
    with zipfile.ZipFile(destination) as archive:
        if archive.namelist() != names or archive.testzip() is not None:
            raise ValueError("Archive membership or CRC check failed")
        for item in members:
            digest = hashlib.sha256()
            with archive.open(item["path"]) as handle:
                for block in iter(lambda: handle.read(4 * 1024**2), b""):
                    digest.update(block)
            if digest.hexdigest() != item["sha256"]:
                raise ValueError("Archived file differs from its source checksum")
    return {
        "filename": destination.name,
        "url": None,
        "sha256": sha256_file(destination),
        "size_bytes": destination.stat().st_size,
        "format": "zip",
        "install_path": ".",
        "expanded_size_bytes": sum(item["size_bytes"] for item in members),
        "members": members,
    }


def prepare_dataset(
    root,
    destination,
    preprocess_path,
    numeric_mapping,
    dataset_card,
    license_path,
    maximum=MAX_SHARD_BYTES,
    minimum_free=MIN_FREE_AFTER_BYTES,
):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    if destination == root or destination.is_relative_to(root):
        raise ValueError("Release output must be outside the source dataset")
    if destination.exists():
        raise FileExistsError(
            "Release destination must be new; existing outputs are preserved"
        )
    rows, clips, summary = inspect_annotations(root)
    validate_mappings(root, numeric_mapping)
    folds = read_canonical_folds(preprocess_path)
    fold_by_video = {
        video: int(fold) for fold, videos in folds.items() for video in videos
    }
    if set(fold_by_video) != {row["video"] for row in rows}:
        raise ValueError(
            "Fold membership does not cover the exact annotation procedures"
        )
    image_paths = [f"data/{row['image_path']}" for row in rows]
    image_bytes = sum(source_file(root, path).stat().st_size for path in image_paths)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if (
        shutil.disk_usage(destination.parent).free
        < image_bytes + minimum_free + 64 * 1024**2
    ):
        raise OSError(
            "Insufficient disk space for archives and the reserved free-space budget"
        )
    print(
        f"Hashing {len(image_paths)} selected images ({image_bytes} bytes)", flush=True
    )
    images = [record(source_file(root, path), path) for path in image_paths]
    selected_metadata = [MASTER_RELATIVE] + [
        f"dict/{kind}_class_mapping.txt" for kind in MAPPING_COUNTS
    ]
    source_metadata = [
        record(source_file(root, path), path) for path in selected_metadata
    ]
    frame_manifest = {
        "schema_version": 1,
        "dataset": "LLS48",
        "summary": summary,
        "event_extraction_scope": "independent_five_frame_clip",
        "frames": [],
    }
    for index, row in enumerate(rows):
        frame_manifest["frames"].append(
            {
                "row_index": index,
                "video": row["video"],
                "clip_name": row["clip_name"],
                "image_id": row["image_id"],
                "image_path": row["image_path"],
                "fold": fold_by_video[row["video"]],
                "frame_in_clip": int(
                    PurePosixPath(row["image_path"]).stem.split("_")[1]
                ),
            }
        )
    generated = {
        "splits/folds.json": json_bytes(
            {
                "schema_version": 1,
                "dataset": "LLS48",
                "folds": folds,
                "unit": "procedure",
                "indexing": "zero_based",
            }
        ),
        "manifests/frames.json": json_bytes(frame_manifest),
        "manifests/checksums.json": json_bytes(
            {"schema_version": 1, "dataset": "LLS48", "files": source_metadata + images}
        ),
        "DATASET_CARD.md": Path(dataset_card).read_bytes(),
        "DATA_AND_MODEL_LICENSE.md": Path(license_path).read_bytes(),
        "third_party/licenses/CC-BY-NC-4.0.txt": (
            Path(license_path).parent / "third_party/licenses/CC-BY-NC-4.0.txt"
        ).read_bytes(),
    }
    payloads = {
        path: source_file(root, path).read_bytes() for path in selected_metadata
    }
    payloads.update(generated)
    metadata_members = [
        {
            "path": path,
            "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
        }
        for path, data in payloads.items()
    ]
    groups = shard_members(images, maximum)
    destination.mkdir()
    files = [
        write_archive(
            destination / "lls48-metadata.zip",
            metadata_members,
            payloads=payloads,
            maximum=maximum,
        )
    ]
    for index, group in enumerate(groups, 1):
        name = f"lls48-images-{index:03d}-of-{len(groups):03d}.zip"
        print(f"Preparing and verifying {name}", flush=True)
        files.append(
            write_archive(destination / name, group, root=root, maximum=maximum)
        )
    if sha256_file(root / MASTER_RELATIVE) != summary["annotation_sha256"]:
        raise ValueError("Source annotation changed during packaging")
    report = {
        "schema_version": 1,
        "asset_id": "lls48",
        "summary": summary,
        "selected_image_bytes": image_bytes,
        "files": files,
        "validation": {
            "zip_crc": "passed",
            "membership": "passed",
            "per_member_sha256": "passed",
            "source_annotation_unchanged": True,
        },
    }
    with (destination / "release_files.json").open(
        "x", encoding="utf-8", newline="\n"
    ) as handle:
        handle.write(json_bytes(report).decode("utf-8"))
    return report


def verify_dataset(root, require_canonical=True):
    root = Path(root).resolve()
    rows, clips, summary = inspect_annotations(
        root, require_canonical=require_canonical
    )
    manifest = json.loads(
        source_file(root, "manifests/checksums.json").read_text(encoding="utf-8")
    )
    if manifest.get("schema_version") != 1 or manifest.get("dataset") != "LLS48":
        raise ValueError("Unsupported checksum manifest")
    expected = {MASTER_RELATIVE} | {
        f"dict/{kind}_class_mapping.txt" for kind in MAPPING_COUNTS
    }
    expected.update(f"data/{row['image_path']}" for row in rows)
    members = manifest["files"]
    paths = [item["path"] for item in members]
    if len(paths) != len(set(paths)) or set(paths) != expected:
        raise ValueError(
            "Checksum manifest does not cover exactly the benchmark selection"
        )
    for item in members:
        path = source_file(root, item["path"])
        if (
            path.stat().st_size != item["size_bytes"]
            or sha256_file(path) != item["sha256"]
        ):
            raise ValueError(f"File checksum or size mismatch: {item['path']}")
    frame_manifest = json.loads(
        source_file(root, "manifests/frames.json").read_text(encoding="utf-8")
    )
    fold_manifest = json.loads(
        source_file(root, "splits/folds.json").read_text(encoding="utf-8")
    )
    folds = fold_manifest["folds"]
    if require_canonical and folds != CANONICAL_FOLDS:
        raise ValueError("Fold assignment differs from the released benchmark")
    fold_by_video = {
        video: int(fold) for fold, videos in folds.items() for video in videos
    }
    flattened = [video for videos in folds.values() for video in videos]
    if len(flattened) != len(set(flattened)) or set(fold_by_video) != {
        r["video"] for r in rows
    }:
        raise ValueError("Fold membership is incomplete or overlaps")
    if frame_manifest.get("event_extraction_scope") != "independent_five_frame_clip":
        raise ValueError("Clip-bounded temporal scope is missing")
    frames = frame_manifest["frames"]
    if len(frames) != len(rows):
        raise ValueError("Frame manifest length does not match the annotation")
    for index, (row, entry) in enumerate(zip(rows, frames)):
        expected_frame = int(PurePosixPath(row["image_path"]).stem.split("_")[1])
        if entry["row_index"] != index or entry["frame_in_clip"] != expected_frame:
            raise ValueError("Frame manifest order changed")
        if any(
            entry[key] != row[key]
            for key in ("video", "clip_name", "image_id", "image_path")
        ):
            raise ValueError("Frame manifest identifiers disagree with the annotation")
        if entry["fold"] != fold_by_video[row["video"]]:
            raise ValueError("Frame fold does not match its procedure")
    if require_canonical:
        validate_mappings(
            root,
            Path(__file__).resolve().parents[1]
            / "curconmix_repro/temporal/ivtmetrics_lls/maps.txt",
        )
    return {**summary, "verified_files": len(members), "status": "VERIFIED"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser(
        "inspect", help="Read-only annotation and selected-image audit"
    )
    inspect.add_argument("--root", type=Path, required=True)
    verify = commands.add_parser(
        "verify", help="Verify an installed release, including every image SHA-256"
    )
    verify.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "verify":
            result = verify_dataset(args.root)
        else:
            rows, _, result = inspect_annotations(args.root)
            result["selected_image_bytes"] = sum(
                source_file(args.root, f"data/{row['image_path']}").stat().st_size
                for row in rows
            )
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
