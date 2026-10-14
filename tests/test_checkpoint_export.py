"""Synthetic serialization tests only: never train or infer a model."""

import copy
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_checkpoints", ROOT / "tools/prepare_checkpoints.py"
)
EXPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORT)


class ExportMetadata(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(
            (ROOT / "configs/temporal/lls48_base224_plus.json").read_text()
        )
        self.meta = {
            "model_type": "mr2",
            "embed_dim": self.config["dim"],
            "num_classes": self.config["num_classes"],
            "dataset": "lls48",
            "name": self.config["model_name"],
            **{
                key: self.config[key]
                for key in (
                    "ks",
                    "fusion",
                    "use_spatial_norm",
                    "spatial_dropout",
                    "ks_weights",
                    "temporal_cfg",
                )
            },
        }

    def test_private_metadata_is_excluded(self):
        self.meta["private_training_path"] = "LOCAL_ONLY"
        result = EXPORT.sanitized_metadata(self.meta, self.config, 2, "a" * 64)
        self.assertNotIn("private_training_path", result)
        self.assertEqual(result["fold"], 2)
        self.assertEqual(result["license"], "CC-BY-NC-4.0")
        self.assertEqual(result["mapping_sha256"], "a" * 64)

    def test_missing_architecture_rejected(self):
        del self.meta["ks"]
        with self.assertRaises(ValueError):
            EXPORT.sanitized_metadata(self.meta, self.config, 0, "a" * 64)

    def test_wrong_architecture_rejected(self):
        self.meta["embed_dim"] = 1
        with self.assertRaises(ValueError):
            EXPORT.sanitized_metadata(self.meta, self.config, 0, "a" * 64)

    def test_fold_and_hash_are_required(self):
        for fold, digest in ((5, "a" * 64), (0, "not-a-sha")):
            with self.assertRaises(ValueError):
                EXPORT.sanitized_metadata(self.meta, self.config, fold, digest)

    def test_numpy_allowlist_is_narrow(self):
        self.assertEqual(
            EXPORT.ALLOWED_NUMPY_GLOBALS,
            {"numpy.core.multiarray.scalar", "numpy.dtype"},
        )

    def test_partial_original_config_must_agree(self):
        partial = {
            key: self.config[key]
            for key in (
                "model_type",
                "dim",
                "dataset",
                "model_name",
                "ks",
                "fusion",
                "use_spatial_norm",
                "ks_weights",
            )
        }
        EXPORT.validate_original_config(self.meta, partial)
        partial["ks"] = [1, 2, 3]
        with self.assertRaises(ValueError):
            EXPORT.validate_original_config(self.meta, partial)


class ExportRegistry(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = []
        for variant in EXPORT.VARIANTS:
            folder = self.root / variant
            folder.mkdir()
            paths = []
            for fold in range(5):
                path = folder / f"best_model_fold{fold}.pth"
                path.write_bytes(b"synthetic-file-not-a-checkpoint")
                paths.append(str(path))
            self.rows.append(
                {
                    "model_id": f"lls48_curconmix_plus_{variant}",
                    "ap_verification_status": "VERIFIED",
                    "row_gt_verification_status": "VERIFIED",
                    "manuscript_eligibility": "YES",
                    "dataset": "LLS48",
                    "temporal_module": "MRTT",
                    "absolute_checkpoint_paths": "|".join(paths),
                    "absolute_config_path": "unused.json",
                }
            )

    def registry(self):
        path = self.root / "registry.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        return path

    def test_exact_fifteen_sources(self):
        sources = EXPORT.canonical_sources(self.registry())
        self.assertEqual(len(sources), 15)
        self.assertEqual([row["fold"] for row in sources[:5]], list(range(5)))

    def test_unverified_source_rejected(self):
        self.rows[0]["ap_verification_status"] = "BLOCKED"
        with self.assertRaises(ValueError):
            EXPORT.canonical_sources(self.registry())

    def test_duplicate_fold_rejected(self):
        paths = self.rows[0]["absolute_checkpoint_paths"].split("|")
        paths[-1] = paths[0]
        self.rows[0]["absolute_checkpoint_paths"] = "|".join(paths)
        with self.assertRaises(ValueError):
            EXPORT.canonical_sources(self.registry())

    def test_duplicate_model_rejected(self):
        self.rows.append(dict(self.rows[0]))
        with self.assertRaises(ValueError):
            EXPORT.canonical_sources(self.registry())


class ExportTensors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("PyTorch is optional in lightweight CI")
        cls.torch = torch

    def test_exact_equality_and_rejections(self):
        state = {"weight": self.torch.tensor([1.0, 2.0])}
        EXPORT.verify_state_equal(state, copy.deepcopy(state))
        for changed in (
            {"weight": self.torch.tensor([1.0, 3.0])},
            {"other": self.torch.tensor([1.0, 2.0])},
            {"weight": self.torch.tensor([1.0, 2.0], dtype=self.torch.float64)},
        ):
            with self.assertRaises(ValueError):
                EXPORT.verify_state_equal(state, changed)

    def test_export_is_safe_loadable_without_numpy_globals(self):
        import io

        buffer = io.BytesIO()
        obj = {
            "model": {"x": self.torch.ones(1)},
            "meta": {"fold": 0, "license": "CC-BY-NC-4.0"},
            "release": {"schema_version": 1},
        }
        self.torch.save(obj, buffer)
        buffer.seek(0)
        actual = self.torch.load(buffer, weights_only=True, map_location="cpu")
        EXPORT.verify_state_equal(obj["model"], actual["model"])
        self.assertEqual(set(actual), {"model", "meta", "release"})
