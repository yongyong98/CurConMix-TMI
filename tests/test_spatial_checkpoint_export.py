"""Pure metadata and synthetic prediction pairing tests; no original assets."""

import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_spatial_checkpoints", ROOT / "tools/prepare_spatial_checkpoints.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SpatialExportTests(unittest.TestCase):
    def audit(self):
        return {
            "models": [
                {
                    "model_id": f"lls48_curconmix_{variant}",
                    "verified_embedded_pairing_folds": list(range(5)),
                    "candidates": [
                        {
                            "fold_from_filename": fold,
                            "identity_status": "VERIFIED_EMBEDDED_PREDICTION_PAIRING",
                            "model_state_fingerprint": "a" * 64,
                            "canonical_prediction_comparison": {
                                "exact_at_stored_dtype": True
                            },
                        }
                        for fold in range(5)
                    ],
                }
                for variant in ("b224", "b384")
            ]
        }

    def test_exact_ten_sources(self):
        self.assertEqual(len(MODULE.select_sources(self.audit())), 10)

    def test_only_cholec_large_five_sources_selected(self):
        audit = self.audit()
        large = copy.deepcopy(audit["models"][0])
        large["model_id"] = "cholec45_curconmix_l224"
        audit["models"].append(large)
        selected = MODULE.select_sources(audit, "cholect45")
        self.assertEqual(len(selected), 5)
        self.assertEqual({row["variant"] for row in selected}, {"l224"})

    def test_cholec_missing_or_incomplete_large_rejected(self):
        with self.assertRaises(ValueError):
            MODULE.select_sources(self.audit(), "cholect45")
        with self.assertRaises(ValueError):
            MODULE.select_sources(self.audit(), "unknown")

    def test_incomplete_folds_rejected(self):
        audit = self.audit()
        audit["models"][0]["verified_embedded_pairing_folds"] = [0, 1, 2, 3]
        with self.assertRaises(ValueError):
            MODULE.select_sources(audit)

    def test_conflicting_duplicate_rejected(self):
        audit = self.audit()
        candidate = copy.deepcopy(audit["models"][0]["candidates"][0])
        candidate["model_state_fingerprint"] = "b" * 64
        audit["models"][0]["candidates"].append(candidate)
        with self.assertRaises(ValueError):
            MODULE.select_sources(audit)

    def test_nonexact_prediction_gate_rejected(self):
        audit = self.audit()
        audit["models"][0]["candidates"][0]["canonical_prediction_comparison"][
            "exact_at_stored_dtype"
        ] = False
        with self.assertRaises(ValueError):
            MODULE.select_sources(audit)

    def test_metadata_contract(self):
        meta = MODULE.metadata("b384", 4, "a" * 64)
        self.assertEqual(meta["input_resolution"], 384)
        self.assertEqual(meta["feature_dim"], 1024)
        self.assertEqual(meta["num_classes"], 266)
        self.assertEqual(meta["timm_version"], "0.5.4")
        self.assertFalse({"preds", "args", "optimizer", "source_path"} & set(meta))

    def test_cholec_metadata_contract(self):
        meta = MODULE.metadata("l224", 0, "a" * 64, "cholect45")
        self.assertEqual(meta["dataset"], "cholect45")
        self.assertEqual(meta["model_name"], "swin_large_patch4_window7_224.ms_in22k")
        self.assertEqual(meta["timm_version"], "1.0.20")
        self.assertEqual(meta["feature_dim"], 1536)
        self.assertEqual(meta["num_classes"], 131)
        self.assertEqual(meta["triplet_size"], 100)
        self.assertEqual(
            meta["class_counts"],
            {"triplet": 100, "instrument": 6, "verb": 10, "target": 15},
        )
        with self.assertRaises(ValueError):
            MODULE.metadata("b224", 0, "a" * 64, "cholect45")
        with self.assertRaises(ValueError):
            MODULE.metadata("l224", 0, "a" * 64, "unknown")

    def test_mapping_digests_match_public_files(self):
        for dataset in MODULE.DATASETS.values():
            self.assertEqual(
                MODULE.sha256(
                    ROOT / "src/curconmix_repro/temporal" / dataset["mapping"]
                ),
                dataset["mapping_sha256"],
            )

    def test_unqualified_variant_fold_and_mapping_rejected(self):
        for variant, fold, digest in (
            ("l224", 0, "a" * 64),
            ("b224", True, "a" * 64),
            ("b224", 5, "a" * 64),
            ("b224", 0, "not-sha"),
        ):
            with self.subTest(variant=variant, fold=fold), self.assertRaises(
                ValueError
            ):
                MODULE.metadata(variant, fold, digest)

    def test_float32_exact_pairing(self):
        original = np.array([[0.3, 0.8]], dtype="float32")
        MODULE.validate_prediction_pairing(
            original, np.array([[0.3, 0.8]], dtype="float64")
        )
        with self.assertRaises(ValueError):
            MODULE.validate_prediction_pairing(original, np.array([[0.3, 0.8001]]))

    def test_wrong_dtype_or_shape_rejected(self):
        with self.assertRaises(ValueError):
            MODULE.validate_prediction_pairing(
                np.ones((2, 3), dtype="float64"), np.ones((2, 3))
            )
        with self.assertRaises(ValueError):
            MODULE.validate_prediction_pairing(
                np.ones((2, 3), dtype="float32"), np.ones((3, 2))
            )

    def test_numpy_globals_are_narrow(self):
        self.assertEqual(
            MODULE.NUMPY_GLOBALS,
            {"numpy.core.multiarray._reconstruct", "numpy.ndarray", "numpy.dtype"},
        )

    def test_strict_failure_preserves_actionable_error(self):
        result = SimpleNamespace(returncode=1, stdout="", stderr="shape mismatch")
        with patch.object(MODULE.subprocess, "run", return_value=result):
            with self.assertRaisesRegex(RuntimeError, "shape mismatch"):
                MODULE.strict_load(Path("synthetic.pth"), Path("python"))

    def test_strict_success_requires_explicit_pass(self):
        result = SimpleNamespace(returncode=0, stdout="", stderr="")
        with patch.object(MODULE.subprocess, "run", return_value=result):
            with self.assertRaisesRegex(ValueError, "did not report success"):
                MODULE.strict_load(Path("synthetic.pth"), Path("python"))

    def path_fixture(self, directory):
        root = Path(directory)
        original = root / "original.pth"
        original.write_bytes(b"original checkpoint must remain unchanged")
        registry = root / "registry.csv"
        registry.write_text("model_id,absolute_prediction_path\n", encoding="utf-8")
        audit = self.audit()
        audit.update(registry=str(registry), registry_sha256=MODULE.sha256(registry))
        for model in audit["models"]:
            model["canonical_prediction"] = str(root / "predictions.csv")
            for candidate in model["candidates"]:
                candidate["path"] = str(original)
        args = SimpleNamespace(
            dataset="lls48",
            audit=root / "audit.json",
            output=root / "not-created",
            private_report=root / "private.json",
            public_manifest=root / "public.json",
        )
        args.audit.write_text(json.dumps(audit), encoding="utf-8")
        return args, audit, original

    def test_export_rejects_report_overwriting_source_before_any_write(self):
        with tempfile.TemporaryDirectory() as directory:
            args, audit, original = self.path_fixture(directory)
            before = original.read_bytes()
            args.private_report = original
            with patch.object(MODULE, "load_original") as load:
                with self.assertRaisesRegex(ValueError, "protected source"):
                    MODULE.export(args)
                load.assert_not_called()
            self.assertEqual(original.read_bytes(), before)
            self.assertFalse(args.output.exists())
            self.assertFalse(args.public_manifest.exists())

    def test_all_protected_and_export_report_collisions_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args, audit, original = self.path_fixture(directory)
            sources = MODULE.select_sources(audit)
            registry_prediction = Path(directory) / "registry-prediction.csv"
            registry = {
                "example": {"absolute_prediction_path": str(registry_prediction)}
            }
            for target in (
                args.audit,
                Path(audit["registry"]),
                original,
                Path(audit["models"][0]["canonical_prediction"]),
                registry_prediction,
                args.output / "lls48-spatial-b224-fold0.pth",
                args.output / "lls48-spatial-b384-fold4.pth.part",
                args.output,
            ):
                for field in ("private_report", "public_manifest"):
                    modified = copy.copy(args)
                    setattr(modified, field, target)
                    with self.subTest(target=target, field=field):
                        with self.assertRaises(ValueError):
                            MODULE.validate_output_paths(
                                modified, audit, registry, sources
                            )
            self.assertFalse(args.output.exists())

    def test_report_case_alias_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args, audit, _ = self.path_fixture(directory)
            args.public_manifest = args.private_report.with_name("PRIVATE.JSON")
            with self.assertRaisesRegex(ValueError, "must not alias"):
                MODULE.validate_output_paths(
                    args, audit, {}, MODULE.select_sources(audit)
                )

    def test_report_hardlink_to_source_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args, audit, original = self.path_fixture(directory)
            os.link(original, args.private_report)
            before = original.read_bytes()
            with self.assertRaisesRegex(ValueError, "protected source"):
                MODULE.validate_output_paths(
                    args, audit, {}, MODULE.select_sources(audit)
                )
            self.assertEqual(original.read_bytes(), before)

    def test_report_symlink_to_source_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args, audit, original = self.path_fixture(directory)
            try:
                args.private_report.symlink_to(original)
            except OSError as exc:
                self.skipTest(f"Symlinks unavailable: {exc}")
            with self.assertRaisesRegex(ValueError, "Symlink|reparse"):
                MODULE.validate_output_paths(
                    args, audit, {}, MODULE.select_sources(audit)
                )
