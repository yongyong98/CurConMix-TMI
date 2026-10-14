"""Portable command entrypoints. Help/dry-run import only the Python standard library."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from .validation import (
    load_config,
    preflight,
    validate_checkpoint_features,
    validate_metadata,
    validate_prediction_csv,
)
from .temporal.procedure_order import load_order, order_sha256
from .training_split import PROTOCOL, load_json, load_training_split, split_digest
from .temporal.training_inputs import (
    load_training_order,
    training_order,
    validate_training_features,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="CurConMix+ research release. No command trains by default."
    )
    commands = result.add_subparsers(dest="command")
    for name in ("preflight", "train-temporal", "infer-temporal"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument(
            "--feature-root",
            type=Path,
            required=True,
            help="Parent of <dataset>/<model_name>/<fold>/{train,val}/VIDxx.npy",
        )
        command.add_argument(
            "--label-root",
            type=Path,
            required=True,
            help="Parent of <dataset>/triplet_label/VIDxx.npy",
        )
        command.add_argument(
            "--feature-manifest",
            type=Path,
            help="Optional keyed LLS48 feature export manifest; verifies hashes and original row identities before any model load",
        )
        command.add_argument(
            "--training-split",
            type=Path,
            help="Explicit independent train/inner-validation/outer-test JSON for new training",
        )
        command.add_argument(
            "--training-feature-manifest",
            type=Path,
            help="Version-2 features from student encoders selected on the matching inner validation set",
        )
        command.add_argument(
            "--dry-run",
            action="store_true",
            help="Print plan only; no data/dependency requirement and no files written",
        )
        if name == "preflight":
            command.add_argument(
                "--inference-only",
                action="store_true",
                help="Validate val features only, still requiring all five folds",
            )
        else:
            command.add_argument(
                "--output",
                type=Path,
                required=True,
                help="New run directory (training) or new prediction CSV (inference)",
            )
            command.add_argument("--device", default="cuda:0")
            command.add_argument(
                "--procedure-order",
                type=Path,
                help="Replay a recorded order; new training uses the explicit inner-split order",
            )
        if name == "infer-temporal":
            command.add_argument(
                "--checkpoint-dir",
                type=Path,
                required=True,
                help="One experiment directory containing best_model_fold0.pth through best_model_fold4.pth",
            )
    evaluation = commands.add_parser(
        "evaluate-ap",
        help="Evaluate stored probabilities without any model or new sigmoid",
    )
    evaluation.add_argument("--dataset", required=True, choices=["cholect45", "lls48"])
    evaluation.add_argument("--predictions", required=True, type=Path)
    evaluation.add_argument(
        "--output",
        required=True,
        type=Path,
        help="New JSON result; never overwrites an existing file",
    )
    evaluation.add_argument("--dry-run", action="store_true")
    return result


def _roots(args) -> dict:
    return {
        "CURCONMIX_FEATURE_ROOT": str(args.feature_root.resolve()),
        "CURCONMIX_LABEL_ROOT": str(args.label_root.resolve()),
    }


def train_command(config: dict, args) -> list[str]:
    output = args.output.resolve()
    command = [sys.executable, "-m", "curconmix_repro.temporal.all_main"]
    options = {
        "name": config["model_name"],
        "dataset": config["dataset"],
        "dim": config["dim"],
        "model_type": config["model_type"],
        "save_root": output / "checkpoints",
        "results_dir": output / "results",
        "results_filename": config["experiment_suffix"],
        "procedure_order": output / "procedure_order.json",
        "training_split": output / "training_split.json",
        "training_feature_manifest": (
            str(args.training_feature_manifest.resolve())
            if getattr(args, "training_feature_manifest", None)
            else "REQUIRED_TRAINING_FEATURE_MANIFEST"
        ),
        "ks": ",".join(str(x) for x in config["ks"]),
        "fusion": config["fusion"],
        "spatial_dropout": config["spatial_dropout"],
        "epochs": config["epochs"],
        "learning_rate": config["learning_rate"],
        "folds": config["n_fold"],
        "batch_size": config["batch_size"],
        "num_workers": config["num_workers"],
        "device": args.device,
        "wandb_mode": "disabled",
    }
    for key, value in options.items():
        command.extend([f"--{key}", str(value)])
    if getattr(args, "feature_manifest_sha256", None):
        command.extend(["--feature_manifest_sha256", args.feature_manifest_sha256])
    if getattr(args, "training_feature_manifest_sha256", None):
        command.extend(
            [
                "--training_feature_manifest_sha256",
                args.training_feature_manifest_sha256,
            ]
        )
    command.extend(["--use_spatial_norm", "--ks_weights"])
    return command


def _new_output(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite output: {path}")


def _json_save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _json_finite(value):
    """Represent canonical undefined AP as JSON null, without altering aggregation."""
    import math

    if isinstance(value, dict):
        return {k: _json_finite(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_finite(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _validated_feature_manifest(config: dict, args):
    """Validate explicit keyed input before model work; keep provenance path-free."""
    if args.feature_manifest is None:
        return None, None
    from curconmix_release.feature_contract import validate_feature_manifest

    path = args.feature_manifest
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = validate_feature_manifest(
        path, config, args.feature_root, args.label_root
    )
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError("Feature manifest changed during validation")
    provenance = {
        "schema_version": 1,
        "dataset": config["dataset"],
        "model_name": config["model_name"],
        "manifest_filename": path.name,
        "manifest_sha256": digest,
        "validation": "PASS_KEYED_FEATURE_MANIFEST",
        "historical_numerical_reproduction": False,
    }
    return manifest, provenance


def evaluate(args) -> None:
    _new_output(args.output)
    audit = validate_prediction_csv(args.predictions, args.dataset)
    from .evaluation.common import evaluate_prediction_csv_detailed
    from .validation import SCHEMAS

    fold, mean, sd = evaluate_prediction_csv_detailed(
        csv_path=args.predictions,
        dataset=args.dataset,
        triplet_size=SCHEMAS[args.dataset][0],
        n_fold=5,
        ignore_null=True,
    )
    _json_save(
        args.output,
        _json_finite(
            {
                "dataset": args.dataset,
                "input": str(args.predictions.resolve()),
                "metric": "canonical video/procedure-wise AP",
                "unit": "0-1",
                "ignore_null": True,
                "fold_sd_ddof": 0,
                "validation": audit,
                "fold_ap": fold,
                "mean_ap": mean,
                "sd_ap": sd,
                "note": "AP only; no F1, inference, thresholding, or additional sigmoid. Undefined AP is null.",
            }
        ),
    )
    print(f"AP result written to {args.output}")


def infer(config: dict, args) -> None:
    """Reuse canonical forward/CSV code; enforce all metadata and strict state loading."""
    _new_output(args.output)
    checkpoint_paths = [
        args.checkpoint_dir / f"best_model_fold{fold}.pth" for fold in range(5)
    ]
    for path in checkpoint_paths:
        if not path.is_file():
            raise FileNotFoundError(f"All five checkpoints are required: {path}")
    feature_manifest, feature_provenance = _validated_feature_manifest(config, args)
    new_feature_audit = None
    new_feature_manifest = None
    new_split = None
    if getattr(args, "training_feature_manifest", None) is not None:
        if args.feature_manifest is not None:
            raise ValueError(
                "Use either published-feature or new-training-feature provenance, not both"
            )
        new_split = load_training_split(args.training_split, config["dataset"])
        new_feature_audit = validate_training_features(
            args.training_feature_manifest,
            config,
            args.feature_root,
            args.label_root,
            new_split,
            include_outer_test=True,
        )
        new_feature_manifest = load_json(args.training_feature_manifest)
    feature_digest = (
        feature_provenance["manifest_sha256"]
        if feature_provenance is not None
        else None
    )
    audit = preflight(config, args.feature_root, args.label_root, inference_only=True)
    if feature_provenance is not None:
        audit["keyed_feature_provenance"] = feature_provenance
    print(json.dumps(audit, indent=2))
    os.environ.update(_roots(args))
    import pandas as pd
    import torch
    from .temporal.infer_temporal import (
        DATASET_CONFIGS,
        VideoDataset,
        _build_model_from_meta,
        _infer_fold_df_like_format,
        _load_ckpt,
        _load_meta,
    )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable; explicitly select --device cpu if desired"
        )
    metadata = [
        _load_meta(str(path), expected_fold=fold)
        for fold, path in enumerate(checkpoint_paths)
    ]
    for fold, meta in enumerate(metadata):
        validate_metadata(meta, config, fold=fold)
        validate_checkpoint_features(meta, feature_digest)
        if meta.get("training_protocol") is not None or new_feature_audit is not None:
            if (
                new_feature_audit is None
                or meta.get("training_protocol") != PROTOCOL
                or meta.get("stage") != "mrtt"
                or meta.get("selection_partition") != "inner_validation"
                or meta.get("selection_metric") != "AP_IVT"
                or meta.get("training_split_sha256") != split_digest(new_split)
                or meta.get("training_feature_manifest_sha256")
                != new_feature_audit["manifest_sha256"]
            ):
                raise ValueError(
                    "New training checkpoints require matching --training-split and --training-feature-manifest, including outer-test features"
                )
    inference_order = args.procedure_order
    if args.procedure_order is not None:
        if new_split is not None:
            order = load_training_order(args.procedure_order, config, new_split)
            inference_order = None  # v2 records train/inner order, not inference order
        else:
            order = load_order(
                args.procedure_order, config["dataset"], config["model_name"]
            )
        digest = order_sha256(order)
        print(
            json.dumps(
                {
                    "inference_procedure_order_sha256": digest,
                    "checkpoint_training_order_matches": [
                        meta.get("procedure_order_sha256") == digest
                        for meta in metadata
                    ],
                    "scope": "procedure iteration only; does not establish within-procedure frame identity",
                },
                indent=2,
            )
        )
    model = _build_model_from_meta(metadata[0]).to(device)
    frames = []
    for fold, path in enumerate(checkpoint_paths):
        checkpoint = _load_ckpt(str(path))
        if not isinstance(checkpoint, dict) or "meta" not in checkpoint:
            raise ValueError(
                f"An embedded, explicit checkpoint fold is required: {path}"
            )
        validate_metadata(checkpoint["meta"], config, fold=fold)
        validate_checkpoint_features(checkpoint["meta"], feature_digest)
        if checkpoint["meta"] != metadata[fold]:
            raise ValueError("Checkpoint metadata changed after preflight")
        state = (
            checkpoint["model"]
            if isinstance(checkpoint, dict) and "model" in checkpoint
            else checkpoint
        )
        model.load_state_dict(state, strict=True)
        data = VideoDataset(
            config["dataset"],
            config["model_name"],
            "val",
            fold,
            procedure_order=inference_order,
        )
        frame = _infer_fold_df_like_format(
            model,
            data,
            device,
            False,
            fold,
            DATASET_CONFIGS[config["dataset"]],
            config["dataset"],
            "val",
        )
        if feature_manifest is not None:
            from curconmix_release.feature_contract import restore_prediction_keys

            frame = restore_prediction_keys(
                frame, feature_manifest, args.label_root, fold=fold
            )
        elif new_feature_manifest is not None and config["dataset"] == "lls48":
            from .temporal.training_inputs import restore_training_prediction_keys

            frame = restore_training_prediction_keys(
                frame, new_feature_manifest, args.label_root, fold=fold
            )
        frames.append(frame)
    output = pd.concat(frames, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False, encoding="utf-8-sig", mode="x")
    print(f"Predictions written to {args.output}")
    if config["dataset"] == "lls48":
        if feature_manifest is None and new_feature_manifest is None:
            print(
                "LLS48: output row IDs are sequential, not original clip IDs. Join the approved ordered frame manifest before clip-bounded segmental F1; never infer clip continuity across sampling gaps."
            )
        else:
            print(
                "LLS48: original row/clip keys were restored after full-GT verification. Segmental F1 is not computed here and must remain clip-bounded; no continuity across sampling gaps is implied."
            )


def main(argv=None) -> int:
    command_parser = parser()
    args = command_parser.parse_args(argv)
    if not args.command:
        command_parser.print_help()
        return 0
    try:
        if args.command == "evaluate-ap":
            if args.dry_run:
                print(
                    json.dumps(
                        {
                            "command": args.command,
                            "dataset": args.dataset,
                            "predictions": str(args.predictions),
                            "output": str(args.output),
                            "executes_model": False,
                        },
                        indent=2,
                    )
                )
            else:
                evaluate(args)
            return 0
        config = load_config(args.config)
        if args.dry_run:
            plan = {
                "command": args.command,
                "dataset": config["dataset"],
                "environment": _roots(args),
                "reads_data": False,
                "writes_files": False,
                "feature_manifest": (
                    str(args.feature_manifest) if args.feature_manifest else None
                ),
            }
            if args.command == "train-temporal":
                plan["argv"] = train_command(config, args)
            elif args.command == "infer-temporal":
                plan.update(
                    checkpoint_dir=str(args.checkpoint_dir),
                    output=str(args.output),
                    required_folds=[0, 1, 2, 3, 4],
                )
            print(json.dumps(plan, indent=2))
            return 0
        if args.command == "preflight":
            if args.training_feature_manifest is not None:
                split = load_training_split(args.training_split, config["dataset"])
                audit = validate_training_features(
                    args.training_feature_manifest,
                    config,
                    args.feature_root,
                    args.label_root,
                    split,
                    include_outer_test=args.inference_only,
                )
            else:
                _, feature_provenance = _validated_feature_manifest(config, args)
                audit = preflight(
                    config,
                    args.feature_root,
                    args.label_root,
                    inference_only=args.inference_only,
                )
                if feature_provenance is not None:
                    audit["keyed_feature_provenance"] = feature_provenance
            print(json.dumps(audit, indent=2))
        elif args.command == "train-temporal":
            _new_output(args.output)
            split = load_training_split(args.training_split, config["dataset"])
            if args.training_feature_manifest is None:
                raise ValueError(
                    "New temporal training requires --training-feature-manifest from newly trained inner-validation student encoders"
                )
            if args.feature_manifest is not None:
                raise ValueError(
                    "Published-encoder feature manifests are for inference, not new inner-validation training"
                )
            audit = validate_training_features(
                args.training_feature_manifest,
                config,
                args.feature_root,
                args.label_root,
                split,
            )
            print(json.dumps(audit, indent=2))
            environment = os.environ.copy()
            environment.update(_roots(args))
            environment["WANDB_MODE"] = "disabled"
            procedure_order = (
                load_training_order(args.procedure_order, config, split)
                if args.procedure_order is not None
                else training_order(config, split)
            )
            args.output.mkdir(parents=True, exist_ok=False)
            _json_save(args.output / "release_config.json", config)
            _json_save(args.output / "procedure_order.json", procedure_order)
            _json_save(args.output / "training_split.json", split)
            _json_save(args.output / "training_input_audit.json", audit)
            print(f"Recorded procedure-order SHA-256: {order_sha256(procedure_order)}")
            args.training_feature_manifest_sha256 = audit["manifest_sha256"]
            subprocess.run(train_command(config, args), env=environment, check=True)
            checkpoint_dir = args.output / "checkpoints" / config["experiment_suffix"]
            for fold in range(5):
                expected = checkpoint_dir / f"best_model_fold{fold}.pth"
                if not expected.is_file():
                    raise RuntimeError(
                        f"Training did not produce every fold checkpoint: missing {expected}"
                    )
        elif args.command == "infer-temporal":
            infer(config, args)
    except (
        ValueError,
        OSError,
        ImportError,
        RuntimeError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
