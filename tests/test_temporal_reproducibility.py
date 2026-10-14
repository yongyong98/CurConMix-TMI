"""No-forward checks for new-run order, fold identity, and safe serialization."""

from __future__ import annotations

import ast
import copy
import contextlib
import hashlib
import importlib.util
import io
import json
from collections import OrderedDict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from curconmix_repro.cli import (
    _validated_feature_manifest,
    infer,
    main,
    parser,
    train_command,
)
from curconmix_repro.temporal.procedure_order import (
    capture_order,
    load_order,
    ordered_files,
    order_sha256,
    validate_order,
)
from curconmix_repro.temporal.training_checkpoint import training_checkpoint
from curconmix_repro.validation import (
    expected_videos,
    load_config,
    validate_checkpoint_features,
    validate_metadata,
)


ROOT = Path(__file__).resolve().parents[1]
TORCH_PRESENT = importlib.util.find_spec("torch") is not None


class TemporalReproducibility(unittest.TestCase):
    def setUp(self):
        self.config_path = ROOT / "configs/temporal/lls48_base224_plus.json"
        self.config = load_config(self.config_path)
        self.metadata = {
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
        self.metadata.update(
            embed_dim=self.config["dim"], name=self.config["model_name"]
        )

    def record(self):
        folds = expected_videos("lls48")
        videos = set.union(*folds)
        return {
            "schema_version": 1,
            "dataset": "lls48",
            "model_name": self.config["model_name"],
            "order_policy": "captured_directory_enumeration_no_historical_claim",
            "scope": "procedure_iteration_only_not_frame_identity",
            "folds": {
                str(fold): {
                    "train": sorted(videos - val, reverse=True),
                    "val": sorted(val, reverse=True),
                }
                for fold, val in enumerate(folds)
            },
        }

    def payload(self, **overrides):
        options = {
            "epoch": 2,
            "model_state": OrderedDict(),
            "optimizer_state": {"state": {}, "param_groups": []},
            "scheduler_state": {"last_epoch": 2, "_last_lr": [0.0003]},
            "best_map": np.float64(0.5123456789),
            "arguments": {"seed": np.int64(42), "learning_rate": np.float64(0.0003)},
            "metadata": self.metadata,
            "fold": 3,
        }
        options.update(overrides)
        return training_checkpoint(**options)

    def test_numpy_score_is_serialized_as_float_without_changing_value(self):
        score = np.float64(0.5123456789)
        payload = self.payload(best_map=score)
        self.assertIs(type(payload["best_mAP"]), float)
        self.assertEqual(payload["best_mAP"], score)
        self.assertIs(type(payload["args"]["seed"]), int)
        self.assertEqual(payload["meta"]["fold"], 3)
        self.assertNotIn("fold", self.metadata)
        validate_metadata(payload["meta"], self.config, fold=3)

    def test_state_objects_are_preserved_not_rebuilt_or_executed(self):
        states = [OrderedDict(), {"state": {}}, {"last_epoch": 2}]
        result = self.payload(
            model_state=states[0], optimizer_state=states[1], scheduler_state=states[2]
        )
        for key, value in zip(("model", "optimizer", "scheduler"), states):
            self.assertIs(result[key], value)

    def test_checkpoint_rejects_opaque_metadata_and_nonfinite_score(self):
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "finite"
            ):
                self.payload(best_map=value)
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            self.payload(arguments={"opaque": object()})
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            self.payload(metadata={**self.metadata, "fold": 4})

    def test_public_fold_guard_rejects_missing_swapped_or_noninteger_fold(self):
        for invalid in (None, 4, "3", True, 3.0, np.int64(3)):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                ValueError, "fold"
            ):
                validate_metadata(
                    {**self.metadata, "fold": invalid}, self.config, fold=3
                )
        validate_metadata({**self.metadata, "fold": 3}, self.config, fold=3)
        # Historical export/audit callers without a requested fold remain supported.
        validate_metadata(self.metadata, self.config)

    def test_keyed_training_checkpoint_requires_matching_manifest(self):
        digest = "a" * 64
        meta = {**self.metadata, "feature_manifest_sha256": digest}
        validate_checkpoint_features(meta, digest)
        for provided in (None, "", "b" * 64):
            with self.subTest(provided=provided), self.assertRaisesRegex(
                ValueError, "matching"
            ):
                validate_checkpoint_features(meta, provided)
        for invalid in (42, ["a" * 64], "not-a-digest"):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                ValueError, "malformed"
            ):
                validate_checkpoint_features(
                    {"feature_manifest_sha256": invalid}, digest
                )
        # Published historical weights do not acquire a fabricated feature identity.
        validate_checkpoint_features(self.metadata, None)
        validate_checkpoint_features(self.metadata, digest)

    def test_legacy_inference_module_entry_stays_disabled_without_model_imports(self):
        path = ROOT / "src/curconmix_repro/temporal/infer_temporal.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        guard = tree.body[-1]
        self.assertIsInstance(guard, ast.If)
        self.assertEqual(ast.unparse(guard.test), "__name__ == '__main__'")
        self.assertEqual(len(guard.body), 1)
        self.assertIsInstance(guard.body[0], ast.Raise)
        module = ast.fix_missing_locations(ast.Module(body=[guard], type_ignores=[]))
        # Execute only the terminal guard, never torch imports or the old batch main.
        with self.assertRaisesRegex(SystemExit, "curconmix_repro.cli infer-temporal"):
            exec(compile(module, str(path), "exec"), {"__name__": "__main__"})

    def test_replay_record_preserves_order_and_stable_digest(self):
        record = self.record()
        reordered_keys = dict(reversed(list(record.items())))
        self.assertEqual(order_sha256(record), order_sha256(reordered_keys))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "procedure_order.json"
            path.write_text(json.dumps(record), encoding="utf-8")
            loaded = load_order(path, self.config["dataset"], self.config["model_name"])
            folder = root / "val"
            folder.mkdir()
            wanted = [f"{video}.npy" for video in record["folds"]["0"]["val"]]
            for filename in reversed(wanted):
                (folder / filename).touch()
            with patch(
                "curconmix_repro.temporal.procedure_order.os.listdir",
                return_value=list(reversed(wanted)),
            ):
                found = ordered_files(
                    loaded, "lls48", self.config["model_name"], 0, "val", folder
                )
            self.assertEqual(found, wanted)

    def test_capture_records_filesystem_order_without_sorting(self):
        record = self.record()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected_listing = {}
            for fold, splits in record["folds"].items():
                for split, videos in splits.items():
                    folder = root / "lls48" / self.config["model_name"] / fold / split
                    folder.mkdir(parents=True)
                    filenames = [f"{video}.npy" for video in videos]
                    expected_listing[str(folder)] = filenames
                    for filename in filenames:
                        (folder / filename).touch()
            with patch(
                "curconmix_repro.temporal.procedure_order.os.listdir",
                side_effect=lambda path: expected_listing[str(path)],
            ):
                captured = capture_order(self.config, root)
            self.assertEqual(captured, record)

    def test_order_rejects_duplicate_wrong_split_and_wrong_identity(self):
        mutations = (
            lambda value: value.update(dataset="cholect45"),
            lambda value: value["folds"]["0"]["val"].__setitem__(0, "../VID10"),
            lambda value: value["folds"]["0"]["val"].__setitem__(
                0, value["folds"]["0"]["val"][1]
            ),
            lambda value: value["folds"].pop("4"),
        )
        for change in mutations:
            value = copy.deepcopy(self.record())
            change(value)
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_order(value, "lls48", self.config["model_name"])

    def test_training_command_consumes_new_run_record_not_original_input(self):
        args = parser().parse_args(
            [
                "train-temporal",
                "--config",
                str(self.config_path),
                "--feature-root",
                "features",
                "--label-root",
                "labels",
                "--output",
                "new_run",
                "--procedure-order",
                "previous_run/procedure_order.json",
                "--dry-run",
            ]
        )
        command = train_command(self.config, args)
        self.assertEqual(
            command[command.index("--procedure_order") + 1],
            str(Path("new_run/procedure_order.json").resolve()),
        )

    def arguments(self, command, root):
        options = [
            command,
            "--config",
            str(self.config_path),
            "--feature-root",
            str(root / "features"),
            "--label-root",
            str(root / "labels"),
            "--feature-manifest",
            str(root / "feature_manifest.json"),
        ]
        if command != "preflight":
            options += ["--output", str(root / "output"), "--device", "cpu"]
        if command == "infer-temporal":
            options += ["--checkpoint-dir", str(root / "checkpoints")]
        return options

    def test_feature_manifest_dry_runs_never_read_manifest_or_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for command in ("preflight", "train-temporal", "infer-temporal"):
                output = io.StringIO()
                with self.subTest(command=command), contextlib.redirect_stdout(output):
                    status = main(self.arguments(command, root) + ["--dry-run"])
                self.assertEqual(status, 0)
                plan = json.loads(output.getvalue())
                self.assertEqual(
                    plan["feature_manifest"], str(root / "feature_manifest.json")
                )
                self.assertFalse(plan["reads_data"])
            self.assertEqual(list(root.iterdir()), [])

    def test_invalid_keyed_manifest_fails_before_array_or_training_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "feature_manifest.json").write_text("{}", encoding="utf-8")
            with (
                patch(
                    "curconmix_release.feature_contract.validate_feature_manifest",
                    side_effect=ValueError("bad keyed manifest"),
                ),
                patch("curconmix_repro.cli.preflight") as array_check,
                patch("curconmix_repro.cli.subprocess.run") as execution,
                contextlib.redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(main(self.arguments("train-temporal", root)), 1)
            array_check.assert_not_called()
            execution.assert_not_called()
            self.assertFalse((root / "output").exists())

    def test_invalid_keyed_manifest_fails_before_any_inference_model_import(self):
        import builtins

        original_import = builtins.__import__

        def no_torch(name, *args, **kwargs):
            if name.split(".")[0] == "torch":
                raise AssertionError(
                    "Model imports must not precede feature validation"
                )
            return original_import(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "feature_manifest.json").write_text("{}", encoding="utf-8")
            (root / "checkpoints").mkdir()
            for fold in range(5):
                (root / "checkpoints" / f"best_model_fold{fold}.pth").touch()
            args = parser().parse_args(self.arguments("infer-temporal", root))
            with (
                patch(
                    "curconmix_release.feature_contract.validate_feature_manifest",
                    side_effect=ValueError("bad keyed manifest"),
                ),
                patch("builtins.__import__", side_effect=no_torch),
                self.assertRaisesRegex(ValueError, "bad keyed manifest"),
            ):
                infer(self.config, args)
            self.assertFalse((root / "output").exists())

    def test_feature_provenance_is_hash_bound_without_absolute_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "feature_manifest.json"
            path.write_text("{}", encoding="utf-8")
            args = parser().parse_args(self.arguments("train-temporal", root))
            with (
                patch(
                    "curconmix_release.feature_contract.validate_feature_manifest",
                    return_value={"verified": True},
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                manifest, provenance = _validated_feature_manifest(self.config, args)
            self.assertEqual(manifest, {"verified": True})
            self.assertEqual(
                provenance["manifest_sha256"],
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            self.assertEqual(provenance["manifest_filename"], path.name)
            self.assertNotIn(str(root), json.dumps(provenance))
            self.assertFalse(provenance["historical_numerical_reproduction"])

    @unittest.skipUnless(
        TORCH_PRESENT, "Optional loader contract check; no model execution"
    )
    def test_inference_loader_rejects_wrong_or_missing_embedded_fold(self):
        from curconmix_repro.temporal.infer_temporal import _load_meta

        for checkpoint in (
            {"model": {}},
            {"model": {}, "meta": self.metadata},
            {"model": {}, "meta": {**self.metadata, "fold": 1}},
        ):
            with (
                self.subTest(checkpoint=checkpoint),
                patch(
                    "curconmix_repro.temporal.infer_temporal._load_ckpt",
                    return_value=checkpoint,
                ),
                self.assertRaisesRegex(ValueError, "fold"),
            ):
                _load_meta("not-read.pth", expected_fold=0)
        released_style = {
            "model": {},
            "meta": {**self.metadata, "fold": 0},
            "release": {"schema_version": 1},
        }
        with patch(
            "curconmix_repro.temporal.infer_temporal._load_ckpt",
            return_value=released_style,
        ):
            self.assertEqual(
                _load_meta("not-read.pth", expected_fold=0), released_style["meta"]
            )

    @unittest.skipUnless(
        TORCH_PRESENT, "Optional PyTorch-only serialization test; no model execution"
    )
    def test_entire_checkpoint_payload_roundtrips_with_weights_only(self):
        import torch

        state = OrderedDict({"weight": torch.tensor([[1.0, 2.0]])})
        state._metadata = OrderedDict({"": {"version": 1}})
        optimizer = {
            "state": {
                0: {
                    "step": torch.tensor(2.0),
                    "exp_avg": torch.zeros(1, 2),
                    "exp_avg_sq": torch.ones(1, 2),
                }
            },
            "param_groups": [
                {
                    "params": [0],
                    "lr": 0.0003,
                    "betas": (0.9, 0.999),
                    "eps": 1e-8,
                    "weight_decay": 0.01,
                }
            ],
        }
        scheduler = {
            "step_size": 30,
            "gamma": 0.5,
            "base_lrs": [0.0003],
            "last_epoch": 2,
            "_step_count": 3,
            "_last_lr": [0.0003],
        }
        payload = self.payload(
            model_state=state, optimizer_state=optimizer, scheduler_state=scheduler
        )
        data = io.BytesIO()
        torch.save(payload, data)
        data.seek(0)
        loaded = torch.load(data, map_location="cpu", weights_only=True)
        self.assertEqual(set(loaded), set(payload))
        self.assertIs(type(loaded["best_mAP"]), float)
        self.assertEqual(loaded["meta"], payload["meta"])
        self.assertEqual(loaded["args"], payload["args"])
        self.assertEqual(loaded["scheduler"], scheduler)
        self.assertEqual(loaded["model"]._metadata, state._metadata)
        self.assertTrue(torch.equal(loaded["model"]["weight"], state["weight"]))
        for key in optimizer["state"][0]:
            self.assertTrue(
                torch.equal(
                    loaded["optimizer"]["state"][0][key], optimizer["state"][0][key]
                )
            )


if __name__ == "__main__":
    unittest.main()
