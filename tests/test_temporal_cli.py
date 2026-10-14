"""CPU-only synthetic release checks; no model is trained or inferred."""

from __future__ import annotations

import contextlib
import copy
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from curconmix_repro.cli import main, parser, train_command
from curconmix_repro.validation import (
    array_pair,
    expected_videos,
    load_config,
    preflight,
    validate_metadata,
    validate_prediction_csv,
)


class TemporalCLI(unittest.TestCase):
    def setUp(self):
        self.path = ROOT / "configs/temporal/lls48_base224_plus.json"
        self.config = load_config(self.path)

    def test_six_configs(self):
        paths = list((ROOT / "configs/temporal").glob("*.json"))
        self.assertEqual(len(paths), 6)
        for path in paths:
            with self.subTest(path=path):
                load_config(path)
                text = path.read_text(encoding="utf-8")
                self.assertNotIn("../../", text)
                self.assertNotIn("C:/", text)
                self.assertNotIn("reference_summary_csv", text)

    def test_help_and_all_dry_runs_are_stdlib_only(self):
        # A fresh interpreter fails if any heavy dependency is imported.
        options = [
            [],
            ["--help"],
            [
                "evaluate-ap",
                "--dataset",
                "lls48",
                "--predictions",
                "missing.csv",
                "--output",
                "missing.json",
                "--dry-run",
            ],
        ]
        roots = [
            "--config",
            str(self.path),
            "--feature-root",
            "missing_features",
            "--label-root",
            "missing_labels",
            "--dry-run",
        ]
        options.extend(
            [
                ["preflight", *roots],
                ["train-temporal", *roots, "--output", "missing_run"],
                [
                    "infer-temporal",
                    *roots,
                    "--checkpoint-dir",
                    "missing_checkpoints",
                    "--output",
                    "missing.csv",
                ],
            ]
        )
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(ROOT / "src")
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        code = """
import builtins, json, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'numpy', 'pandas', 'sklearn', 'tqdm', 'wandb'}:
        raise RuntimeError('heavy import forbidden: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from curconmix_repro.cli import main
raise SystemExit(main(json.loads(sys.argv[1])))
"""
        with tempfile.TemporaryDirectory() as directory:
            for option in options:
                with self.subTest(option=option):
                    completed = subprocess.run(
                        [sys.executable, "-B", "-c", code, json.dumps(option)],
                        cwd=directory,
                        env=environment,
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_train_plan_forwards_original_defaults(self):
        args = parser().parse_args(
            [
                "train-temporal",
                "--config",
                str(self.path),
                "--feature-root",
                "features",
                "--label-root",
                "labels",
                "--output",
                "run",
                "--dry-run",
            ]
        )
        command = train_command(self.config, args)
        for key, value in (
            ("--epochs", "30"),
            ("--learning_rate", "0.0003"),
            ("--batch_size", "1"),
            ("--folds", "5"),
            ("--wandb_mode", "disabled"),
        ):
            self.assertEqual(command[command.index(key) + 1], value)
        self.assertIn("--ks_weights", command)
        self.assertIn("--use_spatial_norm", command)

    def test_incorrect_config_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            bad = copy.deepcopy(self.config)
            bad["batch_size"] = 2
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "batch_size"):
                load_config(path)

    def test_metadata_cannot_be_guessed(self):
        with self.assertRaisesRegex(ValueError, "model_type"):
            validate_metadata({}, self.config)
        meta = {
            key: self.config[key]
            for key in (
                "model_type",
                "num_classes",
                "dataset",
                "ks",
                "fusion",
                "use_spatial_norm",
                "spatial_dropout",
                "ks_weights",
                "temporal_cfg",
            )
        }
        meta.update(embed_dim=self.config["dim"], name=self.config["model_name"])
        validate_metadata(meta, self.config)
        meta["dataset"] = "cholect45"
        with self.assertRaisesRegex(ValueError, "dataset"):
            validate_metadata(meta, self.config)

    def test_array_shape_dtype_and_binary_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            feature, label = Path(directory) / "x.npy", Path(directory) / "y.npy"
            np.save(feature, np.zeros((5, 2), dtype=np.float32))
            np.save(label, np.zeros((5, 3), dtype=np.int64))
            self.assertEqual(array_pair(feature, label, 2, 3), 5)
            np.save(label, np.zeros((4, 3), dtype=np.int64))
            with self.assertRaisesRegex(ValueError, "Label shape"):
                array_pair(feature, label, 2, 3)
            np.save(label, np.full((5, 3), 0.5))
            with self.assertRaisesRegex(ValueError, "binary"):
                array_pair(feature, label, 2, 3)
            np.save(label, np.zeros((5, 3)))
            np.save(feature, np.zeros((5, 2), dtype=np.float64))
            with self.assertRaisesRegex(ValueError, "float32"):
                array_pair(feature, label, 2, 3)

    def test_preflight_exact_folds_and_video_sets(self):
        # Small synthetic dimensions/row counts, preserving all canonical memberships.
        config = dict(self.config, dim=2, num_classes=3)
        folds = expected_videos("lls48")
        videos = set.union(*folds)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = root / "labels/lls48/triplet_label"
            labels.mkdir(parents=True)
            for video in videos:
                np.save(labels / f"{video}.npy", np.zeros((5, 3), dtype=np.int64))
            for fold, val in enumerate(folds):
                for split, members in (("val", val), ("train", videos - val)):
                    folder = (
                        root
                        / "features/lls48"
                        / config["model_name"]
                        / str(fold)
                        / split
                    )
                    folder.mkdir(parents=True)
                    for video in members:
                        np.save(
                            folder / f"{video}.npy", np.zeros((5, 2), dtype=np.float32)
                        )
            with patch("curconmix_repro.validation.FRAME_COUNTS", {"lls48": 240}):
                result = preflight(config, root / "features", root / "labels")
                self.assertEqual(result["validated_pairs"], 240)
                bad = root / "features/lls48" / config["model_name"] / "0/val/VID01.npy"
                np.save(bad, np.zeros((5, 2), dtype=np.float32))
                with self.assertRaisesRegex(ValueError, "video set"):
                    preflight(config, root / "features", root / "labels")

    def _prediction(self, path, *, probability=0.7, duplicate=False, wrong_fold=False):
        fields = ["video", "fold", "image_id", "tri0", "tri1", "0", "1"]
        rows = []
        for fold, videos in enumerate(expected_videos("lls48")):
            for video in sorted(videos):
                rows.append([video, fold, video + "_1", 1, 0, probability, 0.1])
        if duplicate:
            rows[1] = rows[0]
        if wrong_fold:
            rows[0][1] = 4
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(fields)
            writer.writerows(rows)

    def test_ap_probability_identity_fold_guards(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "curconmix_repro.validation.SCHEMAS", {"lls48": (2, 3)}
        ), patch("curconmix_repro.validation.FRAME_COUNTS", {"lls48": 48}):
            path = Path(directory) / "predictions.csv"
            self._prediction(path)
            self.assertEqual(validate_prediction_csv(path, "lls48")["rows"], 48)
            for kwargs, message in (
                ({"probability": 2.0}, "probabilities"),
                ({"duplicate": True}, "duplicate"),
                ({"wrong_fold": True}, "membership"),
            ):
                self._prediction(path, **kwargs)
                with self.subTest(kwargs=kwargs), self.assertRaisesRegex(
                    ValueError, message
                ):
                    validate_prediction_csv(path, "lls48")

    def test_missing_data_returns_failure_without_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            with contextlib.redirect_stderr(io.StringIO()):
                status = main(
                    [
                        "train-temporal",
                        "--config",
                        str(self.path),
                        "--feature-root",
                        str(Path(directory) / "missing"),
                        "--label-root",
                        str(Path(directory) / "missing"),
                        "--output",
                        str(output),
                    ]
                )
            self.assertEqual(status, 1)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
