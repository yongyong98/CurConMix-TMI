"""Model-free required-file checks for the preserved B224 spatial launchers."""

import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys

STAGES = {
    "pretrain": ("main.py", "pretrain", None),
    "teacher": ("main.py", "teacher", None),
    "softlabels": ("generate.py", "teacher", False),
    "student": ("main.py", "student", None),
    "predict": ("generate.py", "student", True),
}
DATASETS = {
    "cholect45": {
        "annotation": "dataframes/CholecT50.csv",
        "counts": (100, 6, 10, 15),
        "split": "cholect45-crossval",
        "pretrain_epoch": 2,
        "softlabel_key": "image_id",
    },
    "lls48": {
        "annotation": "dataframe/LLS48_TRI195.csv",
        "counts": (195, 17, 16, 38),
        "split": "LLS48",
        "pretrain_epoch": 6,
        "softlabel_key": "image_path",
    },
}
MODEL_NAME = "swin_base_patch4_window7_224"


def repository_root():
    root = Path(__file__).resolve().parents[2]
    if not (root / "legacy" / "spatial").is_dir():
        raise ValueError("Run this source workflow from an editable repository clone.")
    return root


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def row_identity(row):
    return [str(row["video"]), str(row["image_path"]), str(row["image_id"])]


def row_order_sha256(rows):
    payload = json.dumps(
        [row_identity(row) for row in rows], separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def fold_map(dataset):
    """Read the original literal split without importing pandas or a trainer."""
    path = repository_root() / "legacy" / "spatial" / dataset / "preprocess.py"
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "split_selector":
            for statement in node.body:
                if isinstance(statement, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == "switcher"
                    for target in statement.targets
                ):
                    mapping = ast.literal_eval(statement.value)[
                        DATASETS[dataset]["split"]
                    ]
                    offset = 1 if dataset == "cholect45" else 0
                    return {
                        f"VID{v:02d}": f - offset
                        for f, videos in mapping.items()
                        for v in videos
                    }
    raise ValueError("Could not read the historical fold split.")


def annotation_path(dataset, data_root, annotation_csv=None):
    return Path(data_root).resolve() / (
        annotation_csv or DATASETS[dataset]["annotation"]
    )


def image_path(data_root, relative):
    relative = str(relative).replace("\\", "/")
    if not relative or ":" in relative or relative.startswith("/"):
        raise ValueError(f"Image path must be relative to data/: {relative!r}")
    if any(part in ("", ".", "..") for part in relative.split("/")):
        raise ValueError(f"Unsafe image path: {relative!r}")
    root = (Path(data_root).resolve() / "data").resolve()
    result = root.joinpath(*relative.split("/"))
    if not result.resolve().is_relative_to(root):
        raise ValueError(f"Image resolves outside the data directory: {relative!r}")
    return result


def inspect_inputs(dataset, data_root, annotation_csv=None):
    """Validate selected image paths, fold coverage, and the contiguous label block."""
    annotation = annotation_path(dataset, data_root, annotation_csv)
    details = {"annotation": str(annotation), "dataset": dataset}
    if not annotation.is_file():
        return [], {**details, "errors": [], "missing_files": [str(annotation)]}
    columns = [
        f"{prefix}{i}"
        for prefix, n in zip(("tri", "inst", "v", "t"), DATASETS[dataset]["counts"])
        for i in range(n)
    ]
    mapping = fold_map(dataset)
    rows, errors, missing, seen, clips = [], [], [], set(), {}
    digest = sha256_file(annotation)
    with annotation.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        required = ["video", "image_path", "image_id", "tri0"] + (
            ["clip_name"] if dataset == "lls48" else []
        )
        if len(headers) != len(set(headers)) or any(k not in headers for k in required):
            return [], {
                **details,
                "errors": ["Missing or duplicate annotation columns."],
                "missing_files": [],
            }
        start = headers.index("tri0")
        if headers[start : start + len(columns)] != columns:
            return [], {
                **details,
                "errors": [
                    "The contiguous triplet/instrument/verb/target label block changed."
                ],
                "missing_files": [],
            }
        for line, row in enumerate(reader, 2):
            if row["video"] not in mapping:
                continue  # Preserve the historical canonical split filtering.
            try:
                if row["image_path"] in seen or not row["image_id"]:
                    raise ValueError("duplicate image_path or empty image_id")
                if any(float(row[name]) not in (0.0, 1.0) for name in columns):
                    raise ValueError("action labels must be binary")
                path = image_path(data_root, row["image_path"])
                if not path.is_file():
                    missing.append(str(path))
            except (ValueError, TypeError, KeyError) as error:
                errors.append(f"Annotation line {line}: {error}")
                continue
            seen.add(row["image_path"])
            row["fold"] = mapping[row["video"]]
            rows.append(row)
            if dataset == "lls48":
                clips.setdefault((row["video"], row["clip_name"]), []).append(row)
    absent = sorted(set(mapping) - {row["video"] for row in rows})
    if absent:
        errors.append("Missing canonical procedures: " + ", ".join(absent))
    for (video, clip), values in clips.items():
        if [Path(r["image_path"]).name for r in values] != [
            f"frame_{i:05d}.png" for i in range(1, 6)
        ]:
            errors.append(
                f"{video}/{clip}: expected five ordered independent clip frames."
            )
    if sha256_file(annotation) != digest:
        errors.append("Annotation changed while it was checked.")
    return rows, {
        **details,
        "annotation_sha256": digest,
        "selected_frames": len(rows),
        "selected_procedures": len({r["video"] for r in rows}),
        "fold_frames": {str(f): sum(r["fold"] == f for r in rows) for f in range(5)},
        "row_order_sha256": row_order_sha256(rows),
        "errors": errors,
        "missing_files": missing,
    }


def cache_filenames(model_tag, fold):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", model_tag) or model_tag in (".", ".."):
        raise ValueError("--model-tag must be a safe filename prefix.")
    return (
        f"{model_tag}_fold{fold}_trainset_features_threshold_10.pkl",
        f"{model_tag}_fold_{fold}_similarity_matrix.pkl",
    )


def required_artifacts(dataset, args):
    root = Path(args.output_dir).resolve()
    count = sum(DATASETS[dataset]["counts"])
    paths = []
    for fold in range(5):
        if args.stage == "pretrain":
            if not args.feature_dir:
                raise ValueError("--feature-dir is required for pretraining.")
            paths.extend(
                Path(args.feature_dir).resolve() / n
                for n in cache_filenames(args.model_tag, fold)
            )
        elif args.stage == "teacher":
            epoch = DATASETS[dataset]["pretrain_epoch"]
            paths.append(
                root
                / "checkpoints"
                / f"fold{fold}_swin_bas_CurConMix_B224_Pretrain_{epoch}.pth"
            )
        elif args.stage == "student":
            paths.append(
                root
                / "softlabels"
                / f"sl_f{fold}_swin_bas_{count}_CurConMix_B224_Teacher.csv"
            )
        else:
            stage = "Teacher" if args.stage == "softlabels" else "Student"
            paths.append(
                root
                / "checkpoints"
                / f"fold{fold}_swin_bas_{count}_CurConMix_B224_{stage}.pth"
            )
    return paths


def check_softlabels(path, rows, dataset, fold, training_split=None):
    count, key = sum(DATASETS[dataset]["counts"]), DATASETS[dataset]["softlabel_key"]
    from .spatial_protocol import select_rows

    if training_split is None:
        raise ValueError("Soft-label checking requires an explicit training split.")
    expected = {r[key] for r in select_rows(rows, training_split, fold)}
    seen, probabilities = set(), [str(i) for i in range(count)]
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        if (
            len(headers) != len(set(headers))
            or key not in headers
            or "0" not in headers
        ):
            raise ValueError(f"Missing {key!r} or prediction columns.")
        if headers[headers.index("0") : headers.index("0") + count] != probabilities:
            raise ValueError(
                "Soft-label probability columns are not contiguous/in order."
            )
        for row in reader:
            if row[key] in seen:
                raise ValueError(f"Duplicate soft-label key: {row[key]}")
            seen.add(row[key])
            for name in probabilities:
                value = float(row[name])
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f"Invalid probability for {row[key]}/{name}")
    if seen != expected:
        raise ValueError(
            f"Training-row mismatch: missing={len(expected-seen)}, extra={len(seen-expected)}"
        )


def preflight(dataset, args):
    from .spatial_protocol import load_split, select_rows, verify_sidecar
    from curconmix_repro.training_split import split_digest

    rows, report = inspect_inputs(dataset, args.data_root, args.annotation_csv)
    training_split = None
    if args.stage != "predict" or getattr(args, "training_split", None):
        try:
            training_split = load_split(
                getattr(args, "training_split", None), dataset, report["annotation"]
            )
            report["training_split_sha256"] = split_digest(training_split)
        except (ValueError, OSError) as error:
            report["errors"].append(str(error))
    paths = required_artifacts(dataset, args)
    report["required_artifacts"] = [str(p) for p in paths]
    report["warnings"] = [
        "File preflight is not model/runtime validation or numerical paper reproduction.",
        "Legacy checkpoints and pickle caches must be trusted; no payload is loaded here.",
    ]
    for path in paths:
        if not path.is_file():
            report["missing_files"].append(str(path))
        elif path.stat().st_size == 0:
            report["errors"].append(f"Empty required file: {path}")
    if args.stage in ("teacher", "softlabels", "predict") and training_split:
        prior_stage = {
            "teacher": "pretrain",
            "softlabels": "teacher",
            "predict": "student",
        }[args.stage]
        for fold, path in enumerate(paths):
            if path.is_file():
                try:
                    verify_sidecar(path, training_split, fold, prior_stage)
                except (ValueError, OSError, KeyError, TypeError) as error:
                    report["errors"].append(f"{path}: {error}")
    if args.stage == "student" and rows and training_split:
        for fold, path in enumerate(paths):
            if path.is_file():
                try:
                    check_softlabels(path, rows, dataset, fold, training_split)
                    info = verify_sidecar(
                        path,
                        training_split,
                        fold,
                        "softlabels",
                        select_rows(rows, training_split, fold),
                    )
                    if not info.get("teacher_checkpoint_sha256"):
                        raise ValueError(
                            "Soft-label provenance lacks teacher checkpoint identity."
                        )
                except (ValueError, TypeError, KeyError, OSError) as error:
                    report["errors"].append(f"{path}: {error}")
    if args.stage == "pretrain":
        manifest = Path(args.feature_dir).resolve() / "cache_provenance.json"
        if (Path(args.feature_dir).resolve() / "INCOMPLETE.json").exists():
            report["errors"].append(
                "Cache preparation is incomplete. Use a complete cache directory."
            )
        if not manifest.is_file():
            report["errors"].append(
                "No cache provenance sidecar: new training requires split-matched caches."
            )
        elif not report["missing_files"] and not report["errors"]:
            try:
                provenance = json.loads(manifest.read_text(encoding="utf-8"))
                if not isinstance(provenance, dict):
                    raise ValueError("Cache provenance must be a JSON object.")
                if (
                    provenance.get("complete") is not True
                    or provenance.get("schema_version") != 1
                ):
                    raise ValueError("Cache provenance is not marked complete.")
                if set(provenance["folds"]) != {str(f) for f in range(5)}:
                    raise ValueError("Cache provenance must contain exactly folds 0-4.")
                if (
                    provenance["dataset"] != dataset
                    or provenance["annotation_sha256"] != report["annotation_sha256"]
                    or provenance.get("training_split_sha256")
                    != split_digest(training_split)
                ):
                    raise ValueError("Cache annotation/dataset identity differs.")
                if (
                    provenance["model_name"] != MODEL_NAME
                    or provenance["model_tag"] != args.model_tag
                ):
                    raise ValueError("Cache backbone or prefix differs.")
                images = {}
                for fold in range(5):
                    entry = provenance["folds"][str(fold)]
                    if set(entry["files"]) != set(
                        cache_filenames(args.model_tag, fold)
                    ):
                        raise ValueError(f"Unexpected cache filenames for fold {fold}.")
                    expected_rows = select_rows(rows, training_split, fold)
                    if entry["row_order_sha256"] != row_order_sha256(expected_rows):
                        raise ValueError(
                            f"Cache training row order differs for fold {fold}."
                        )
                    if [value["key"] for value in entry["rows"]] != [
                        row_identity(r) for r in expected_rows
                    ]:
                        raise ValueError(f"Cache image keys differ for fold {fold}.")
                    for value in entry["rows"]:
                        relative = value["key"][1]
                        digest = value["image_sha256"]
                        if images.setdefault(relative, digest) != digest:
                            raise ValueError(
                                f"Cache image digests differ between folds: {relative}"
                            )
                    for path in paths[fold * 2 : fold * 2 + 2]:
                        identity = entry["files"][path.name]
                        if (
                            path.stat().st_size != identity["size_bytes"]
                            or sha256_file(path) != identity["sha256"]
                        ):
                            raise ValueError(f"Cache bytes differ: {path.name}")
                for relative, digest in images.items():
                    if sha256_file(image_path(args.data_root, relative)) != digest:
                        raise ValueError(
                            f"Source image differs from prepared cache: {relative}"
                        )
            except (ValueError, KeyError, TypeError) as error:
                report["errors"].append(f"Cache provenance check failed: {error}")
    report["status"] = (
        "BLOCKED"
        if report["errors"] or report["missing_files"]
        else "REQUIRED_FILES_PRESENT"
    )
    report["model_executed"] = False
    return report


def build_command(args):
    script, config, inference = STAGES[args.stage]
    command = [
        sys.executable,
        script,
        "--config-name",
        config,
        "parent_path=" + json.dumps(str(args.data_root.resolve())),
        "output_dir=" + json.dumps(str(args.output_dir.resolve())),
        "device=" + args.device,
        "nworkers=" + str(args.workers),
    ]
    if args.annotation_csv:
        command.append("path_csv=" + json.dumps(args.annotation_csv))
    if getattr(args, "training_split", None):
        command.append(
            "training_split=" + json.dumps(str(Path(args.training_split).resolve()))
        )
    if args.stage == "pretrain":
        if not args.feature_dir:
            raise ValueError("--feature-dir is required for pretraining.")
        cache_filenames(args.model_tag, 0)
        command.extend(
            [
                "feature_dir=" + json.dumps(str(args.feature_dir.resolve())),
                "model_tag=" + json.dumps(args.model_tag),
            ]
        )
    if inference is not None:
        command.append("inference=" + str(inference).lower())
    return command


def main(dataset, directory=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--annotation-csv", help="Relative to --data-root, or an absolute path."
    )
    parser.add_argument("--feature-dir", type=Path)
    parser.add_argument(
        "--training-split",
        type=Path,
        help="Explicit procedure-disjoint nested training split JSON; required except for legacy prediction.",
    )
    parser.add_argument("--model-tag", default="SwinB224")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=4)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--execute",
        action="store_true",
        help="Check inputs, then explicitly run the stage.",
    )
    mode.add_argument(
        "--check-only",
        action="store_true",
        help="Check files without importing a model or running a stage.",
    )
    args = parser.parse_args()
    if args.workers < 0:
        parser.error("--workers must be nonnegative.")
    try:
        command = build_command(args)
        required = required_artifacts(dataset, args)
        directory = directory or repository_root() / "legacy" / "spatial" / dataset
        print(
            json.dumps(
                {
                    "working_directory": str(directory),
                    "command": command,
                    "annotation": str(
                        annotation_path(dataset, args.data_root, args.annotation_csv)
                    ),
                    "required_artifacts": [str(p) for p in required],
                },
                indent=2,
            )
        )
        if not (args.execute or args.check_only):
            print(
                "Preview only. No dataset, checkpoint, model, or training process was loaded."
            )
            return
        report = preflight(dataset, args)
        print(json.dumps(report, indent=2))
        if report["status"] == "BLOCKED":
            parser.exit(2, "Required-file checks failed. No model or stage was run.\n")
        if args.check_only:
            print("Required-file checks passed. No model or stage was run.")
            return
        subprocess.run(command, cwd=directory, check=True)
    except (ValueError, OSError) as error:
        parser.error(str(error))
