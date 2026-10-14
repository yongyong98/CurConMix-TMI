"""Spatial consumer gates use synthetic files and mocks, not images or inference."""

import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from curconmix_release import spatial_checkpoint as spatial


ROOT = Path(__file__).resolve().parents[1]


class SpatialCheckpointInspectionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="spatial-inspect-synthetic-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.payload = b"synthetic checkpoint stand-in; no model or clinical data"
        self.asset_id = "lls48-spatial-b224"
        self.asset = {
            "id": self.asset_id,
            "kind": "spatial_checkpoint_bundle",
            "dataset": "lls48",
            "variant": "swinb224",
            "architecture": "TripletModel",
            "folds": list(range(5)),
            "license": "CC-BY-NC-4.0",
            "license_status": "approved",
            "status": "published",
            "available": True,
            "files": [
                {
                    "filename": f"{self.asset_id}-fold{fold}.pth",
                    "fold": fold,
                    "format": "file",
                    "install_path": f"lls48/b224/fold{fold}.pth",
                    "url": f"https://downloads.invalid/fold{fold}.pth",
                    "sha256": hashlib.sha256(self.payload).hexdigest(),
                    "size_bytes": len(self.payload),
                }
                for fold in range(5)
            ],
        }
        path = self.root / self.asset["files"][0]["install_path"]
        path.parent.mkdir(parents=True)
        path.write_bytes(self.payload)
        self.meta = {
            "schema_version": 1,
            "dataset": "lls48",
            "variant": "b224",
            "fold": 0,
            "model_class": "TripletModel",
            "model_name": "swin_base_patch4_window7_224",
            "input_resolution": 224,
            "feature_dim": 1024,
            "num_classes": 266,
            "triplet_size": 195,
            "class_counts": spatial.VARIANTS[self.asset_id]["class_counts"].copy(),
            "license": "CC-BY-NC-4.0",
            "timm_version": "0.5.4",
            "mapping_sha256": hashlib.sha256(
                (
                    ROOT / "src/curconmix_repro/temporal/ivtmetrics_lls/maps.txt"
                ).read_bytes()
            ).hexdigest(),
        }
        self.obj = {
            "model": {"fake_tensor": object()},
            "meta": self.meta,
            "release": {
                "schema_version": 1,
                "kind": "spatial_model_only",
                "dataset": "LLS48",
                "variant": "b224",
                "fold": 0,
                "license": "CC-BY-NC-4.0",
                "source_sha256": "1" * 64,
                "canonical_prediction_sha256": "2" * 64,
                "embedded_prediction_pairing": "EXACT_FLOAT32",
            },
        }
        for item in self.asset["files"]:
            item.update(
                source_sha256=self.obj["release"]["source_sha256"],
                canonical_prediction_sha256=self.obj["release"][
                    "canonical_prediction_sha256"
                ],
                mapping_sha256=self.meta["mapping_sha256"],
            )

    def manifest(self):
        path = self.root / "manifest.json"
        path.write_text(
            json.dumps({"schema_version": 2, "assets": [self.asset]}), encoding="utf-8"
        )
        return path

    def inspect(self):
        return spatial.inspect_checkpoint(
            self.asset_id, self.root, 0, self.manifest(), ROOT
        )

    def select_variant(self, asset_id):
        self.asset_id = asset_id
        spec = spatial.VARIANTS[asset_id]
        self.asset.update(
            id=asset_id, dataset=spec["dataset"], variant=spec["public_variant"]
        )
        for item in self.asset["files"]:
            fold = item["fold"]
            item["filename"] = f"{asset_id}-fold{fold}.pth"
            item["install_path"] = f"{spec['dataset']}/{spec['variant']}/fold{fold}.pth"
        path = self.root / self.asset["files"][0]["install_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.payload)
        for key in (
            "dataset",
            "variant",
            "model_name",
            "input_resolution",
            "feature_dim",
            "num_classes",
            "triplet_size",
            "class_counts",
            "timm_version",
        ):
            self.meta[key] = spec[key]
        self.meta["mapping_sha256"] = hashlib.sha256(
            (
                ROOT
                / "src/curconmix_repro/temporal"
                / spec["metric_package"]
                / "maps.txt"
            ).read_bytes()
        ).hexdigest()
        self.obj["release"].update(
            dataset=spec["release_dataset"], variant=spec["variant"]
        )
        for item in self.asset["files"]:
            item["mapping_sha256"] = self.meta["mapping_sha256"]

    def test_b384_and_cholec_l224_are_explicitly_supported(self):
        for asset_id, classes, features in (
            ("lls48-spatial-b384", 266, 1024),
            ("cholect45-spatial-l224", 131, 1536),
        ):
            with self.subTest(asset=asset_id):
                self.select_variant(asset_id)
                with patch.object(
                    spatial, "_safe_load", return_value=self.obj
                ), patch.object(spatial, "_strict_model_load"):
                    result = self.inspect()
                self.assertEqual(result["num_classes"], classes)
                self.assertEqual(result["feature_dim"], features)
                self.assertFalse(result["inference_run"])

    def test_cholec_rejects_lls_class_mapping(self):
        lls_mapping = self.meta["mapping_sha256"]
        self.select_variant("cholect45-spatial-l224")
        self.meta["mapping_sha256"] = lls_mapping
        with patch.object(spatial, "_safe_load", return_value=self.obj), patch.object(
            spatial, "_strict_model_load"
        ) as strict:
            with self.assertRaisesRegex(ValueError, "mapping"):
                self.inspect()
            strict.assert_not_called()

    def test_matching_file_returns_bounded_verification(self):
        with patch.object(
            spatial, "_safe_load", return_value=self.obj
        ) as loader, patch.object(spatial, "_strict_model_load") as strict:
            result = self.inspect()
        loader.assert_called_once()
        strict.assert_called_once_with(self.meta, self.obj["model"], ROOT)
        self.assertEqual(result["strict_loading_status"], "PASS")
        self.assertFalse(result["inference_run"])
        self.assertFalse(result["feature_extraction_run"])

    def test_unpublished_checkpoint_never_loads(self):
        self.asset["status"] = "unpublished"
        with patch.object(spatial, "_safe_load") as loader:
            with self.assertRaisesRegex(ValueError, "not published"):
                self.inspect()
            loader.assert_not_called()

    def test_wrong_checksum_never_loads(self):
        self.asset["files"][0]["sha256"] = "0" * 64
        with patch.object(spatial, "_safe_load") as loader:
            with self.assertRaisesRegex(ValueError, "mismatch"):
                self.inspect()
            loader.assert_not_called()

    def test_wrong_manifest_identity_never_loads(self):
        for change in (
            {"variant": "swinb384"},
            {"architecture": "Other"},
            {"dataset": "cholect45"},
        ):
            with self.subTest(change=change):
                original = copy.deepcopy(self.asset)
                self.asset.update(change)
                with patch.object(spatial, "_safe_load") as loader:
                    with self.assertRaisesRegex(ValueError, "identity"):
                        self.inspect()
                    loader.assert_not_called()
                self.asset = original

    def test_wrong_fold_layout_never_loads(self):
        for change in (
            {"fold": 1},
            {"fold": False},
            {"install_path": "lls48/b384/fold0.pth"},
            {"filename": "wrong.pth"},
            {"format": "zip"},
        ):
            with self.subTest(change=change):
                original = copy.deepcopy(self.asset["files"][0])
                self.asset["files"][0].update(change)
                with patch.object(spatial, "_safe_load") as loader:
                    with self.assertRaises(ValueError):
                        self.inspect()
                    loader.assert_not_called()
                self.asset["files"][0] = original

    def test_wrong_model_meta_never_builds_model(self):
        for change in (
            {"fold": 1},
            {"fold": False},
            {"feature_dim": 1536},
            {"input_resolution": 384},
            {"timm_version": "1.0.0"},
            {"mapping_sha256": "0" * 64},
            {"class_counts": {"triplet": 100}},
            {"license": "MIT"},
        ):
            with self.subTest(change=change):
                obj = {**self.obj, "meta": {**self.meta, **change}}
                with patch.object(
                    spatial, "_safe_load", return_value=obj
                ), patch.object(spatial, "_strict_model_load") as strict:
                    with self.assertRaises(ValueError):
                        self.inspect()
                    strict.assert_not_called()

    def test_non_model_payload_rejected(self):
        for extra in ("optimizer", "preds", "args"):
            with self.subTest(extra=extra), patch.object(
                spatial, "_safe_load", return_value={**self.obj, extra: {}}
            ):
                with self.assertRaisesRegex(ValueError, "model-only"):
                    self.inspect()

    def test_unpaired_release_rejected(self):
        for change in (
            {"embedded_prediction_pairing": "NOT_VERIFIED"},
            {"source_sha256": "unknown"},
            {"canonical_prediction_sha256": ""},
            {"fold": False},
        ):
            with self.subTest(change=change):
                obj = {**self.obj, "release": {**self.obj["release"], **change}}
                with patch.object(
                    spatial, "_safe_load", return_value=obj
                ), patch.object(spatial, "_strict_model_load") as strict:
                    with self.assertRaises(ValueError):
                        self.inspect()
                    strict.assert_not_called()

    def test_restricted_load_has_no_fallback(self):
        with patch.object(
            spatial, "_safe_load", side_effect=ValueError("unsafe global")
        ) as loader, patch.object(spatial, "_strict_model_load") as strict:
            with self.assertRaisesRegex(ValueError, "unsafe global"):
                self.inspect()
            loader.assert_called_once()
            strict.assert_not_called()

    def test_valid_but_different_payload_provenance_is_rejected(self):
        for key in ("source_sha256", "canonical_prediction_sha256"):
            with self.subTest(key=key):
                obj = {
                    **self.obj,
                    "release": {**self.obj["release"], key: "3" * 64},
                }
                with patch.object(
                    spatial, "_safe_load", return_value=obj
                ), patch.object(spatial, "_strict_model_load") as strict:
                    with self.assertRaisesRegex(ValueError, f"manifest: {key}"):
                        self.inspect()
                    strict.assert_not_called()

    def test_manifest_provenance_must_match_selected_checkpoint(self):
        for key in (
            "source_sha256",
            "canonical_prediction_sha256",
            "mapping_sha256",
        ):
            for value in ("3" * 64, None):
                with self.subTest(key=key, value=value):
                    original = self.asset["files"][0].copy()
                    if value is None:
                        del self.asset["files"][0][key]
                    else:
                        self.asset["files"][0][key] = value
                    with patch.object(
                        spatial, "_safe_load", return_value=self.obj
                    ), patch.object(spatial, "_strict_model_load") as strict:
                        with self.assertRaisesRegex(ValueError, f"manifest: {key}"):
                            self.inspect()
                        strict.assert_not_called()
                    self.asset["files"][0] = original

    def test_loader_requests_cpu_weights_only(self):
        loader = Mock(return_value=self.obj)
        path = self.root / "synthetic.pth"
        with patch.dict("sys.modules", {"torch": SimpleNamespace(load=loader)}):
            self.assertIs(spatial._safe_load(path), self.obj)
        loader.assert_called_once_with(path, map_location="cpu", weights_only=True)

    def test_strict_constructor_never_requests_pretraining_or_forward(self):
        class FakeTensor:
            pass

        state = {"tensor": FakeTensor()}
        model = Mock()
        factory = Mock(return_value=model)
        modules = {
            "torch": SimpleNamespace(Tensor=FakeTensor),
            "timm": SimpleNamespace(__version__="0.5.4"),
        }
        with patch.dict("sys.modules", modules), patch.object(
            spatial, "_historical_model_class", return_value=factory
        ):
            result = spatial._strict_model_load(self.meta, state, ROOT)
        self.assertIs(result, model)
        self.assertEqual(factory.call_args.args[0].target_size, 266)
        self.assertEqual(factory.call_args.args[1], self.meta["model_name"])
        self.assertEqual(factory.call_args.kwargs, {"pretrained": False})
        model.cpu.assert_called_once_with()
        model.load_state_dict.assert_called_once_with(state, strict=True)
        model.eval.assert_called_once_with()
        model.forward.assert_not_called()

    def test_wrong_timm_never_builds_model(self):
        with patch.dict(
            "sys.modules",
            {"torch": SimpleNamespace(), "timm": SimpleNamespace(__version__="1.0.0")},
        ), patch.object(spatial, "_historical_model_class") as constructor:
            with self.assertRaisesRegex(ValueError, "timm==0.5.4"):
                spatial._strict_model_load(self.meta, {}, ROOT)
            constructor.assert_not_called()

    def test_missing_historical_code_is_explained(self):
        with self.assertRaisesRegex(FileNotFoundError, "project-root"):
            spatial._historical_model_class(self.root)

    def test_python_api_returns_model_and_meta(self):
        model = Mock()
        with patch.object(spatial, "_safe_load", return_value=self.obj), patch.object(
            spatial, "_strict_model_load", return_value=model
        ):
            loaded, metadata = spatial.load_spatial_checkpoint(
                self.asset_id, self.root, 0, self.manifest(), ROOT
            )
        self.assertIs(loaded, model)
        self.assertIs(metadata, self.meta)

    def test_cli_json_has_no_prediction_claim(self):
        with patch.object(spatial, "_safe_load", return_value=self.obj), patch.object(
            spatial, "_strict_model_load"
        ), patch("sys.stdout", new_callable=io.StringIO) as output:
            status = spatial.main(
                [
                    "--asset",
                    self.asset_id,
                    "--root",
                    str(self.root),
                    "--fold",
                    "0",
                    "--manifest",
                    str(self.manifest()),
                    "--project-root",
                    str(ROOT),
                ]
            )
        self.assertEqual(status, 0)
        result = json.loads(output.getvalue())
        self.assertFalse(result["inference_run"])
        self.assertEqual(result["num_classes"], 266)

    def test_unknown_asset_or_invalid_fold_rejected(self):
        for asset, fold in (("unknown", 0), (self.asset_id, True), (self.asset_id, 5)):
            with self.subTest(asset=asset, fold=fold), self.assertRaises(ValueError):
                spatial.inspect_checkpoint(
                    asset, self.root, fold, self.root / "missing.json", ROOT
                )


if __name__ == "__main__":
    unittest.main()
