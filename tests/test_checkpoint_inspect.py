"""Inspect gates with synthetic files and mocked tensor loading; no PyTorch needed."""

import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from curconmix_release import checkpoint

ROOT = Path(__file__).resolve().parents[1]


class CheckpointInspectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="checkpoint-inspect-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.payload = b"synthetic checkpoint stand-in; not a model"
        self.asset = {
            "id": "lls48-mrtt-b224",
            "kind": "checkpoint_bundle",
            "dataset": "lls48",
            "variant": "swinb224",
            "temporal_model": "mr2",
            "folds": list(range(5)),
            "license": "CC-BY-NC-4.0",
            "license_status": "approved",
            "status": "published",
            "available": True,
            "files": [
                {
                    "filename": f"lls48-mrtt-b224-fold{fold}.pth",
                    "fold": fold,
                    "format": "file",
                    "install_path": f"lls48/b224/best_model_fold{fold}.pth",
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
        self.config = json.loads(
            (ROOT / "configs/temporal/lls48_base224_plus.json").read_text()
        )
        self.meta = {
            "model_type": "mr2",
            "embed_dim": 1024,
            "num_classes": 266,
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
            "fold": 0,
            "license": "CC-BY-NC-4.0",
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
                "dataset": "LLS48",
                "variant": "b224",
                "fold": 0,
                "license": "CC-BY-NC-4.0",
            },
        }

    def manifest(self):
        path = self.root / "manifest.json"
        path.write_text(
            json.dumps({"schema_version": 2, "assets": [self.asset]}), encoding="utf-8"
        )
        return path

    def inspect(self):
        return checkpoint.inspect_checkpoint(
            "lls48-mrtt-b224", self.root, 0, self.manifest(), ROOT / "configs/temporal"
        )

    def test_matching_file_loads_strictly_without_inference(self):
        with patch.object(
            checkpoint, "_safe_load", return_value=self.obj
        ) as loader, patch.object(checkpoint, "_strict_model_load") as strict:
            result = self.inspect()
        loader.assert_called_once()
        strict.assert_called_once_with(self.meta, self.obj["model"])
        self.assertFalse(result["inference_run"])
        self.assertEqual(result["embedding_dim"], 1024)
        self.assertEqual(result["strict_loading_status"], "PASS")

    def test_unpublished_never_loads_checkpoint(self):
        self.asset["status"] = "unpublished"
        with patch.object(checkpoint, "_safe_load") as loader:
            with self.assertRaisesRegex(ValueError, "not published"):
                self.inspect()
            loader.assert_not_called()

    def test_wrong_checksum_never_loads_checkpoint(self):
        self.asset["files"][0]["sha256"] = "0" * 64
        with patch.object(checkpoint, "_safe_load") as loader:
            with self.assertRaisesRegex(ValueError, "mismatch"):
                self.inspect()
            loader.assert_not_called()

    def test_wrong_manifest_fold_or_install_path_rejected(self):
        for change in ({"fold": 1}, {"install_path": "wrong/fold.pth"}):
            with self.subTest(change=change):
                original = copy.deepcopy(self.asset["files"][0])
                self.asset["files"][0].update(change)
                with patch.object(checkpoint, "_safe_load") as loader:
                    with self.assertRaises(ValueError):
                        self.inspect()
                    loader.assert_not_called()
                self.asset["files"][0] = original

    def test_wrong_checkpoint_metadata_never_builds_model(self):
        for change in (
            {"fold": 1},
            {"embed_dim": 1536},
            {"mapping_sha256": "0" * 64},
            {"license": "MIT"},
        ):
            with self.subTest(change=change):
                obj = {**self.obj, "meta": {**self.meta, **change}}
                with patch.object(
                    checkpoint, "_safe_load", return_value=obj
                ), patch.object(checkpoint, "_strict_model_load") as strict:
                    with self.assertRaises(ValueError):
                        self.inspect()
                    strict.assert_not_called()

    def test_optimizer_payload_rejected(self):
        with patch.object(
            checkpoint, "_safe_load", return_value={**self.obj, "optimizer": {}}
        ):
            with self.assertRaisesRegex(ValueError, "model-only"):
                self.inspect()

    def test_release_identity_mismatch_rejected(self):
        self.obj["release"]["variant"] = "l224"
        with patch.object(checkpoint, "_safe_load", return_value=self.obj):
            with self.assertRaisesRegex(ValueError, "release identity"):
                self.inspect()

    def test_safe_loading_failure_has_no_fallback(self):
        with patch.object(
            checkpoint, "_safe_load", side_effect=ValueError("unsafe global")
        ) as loader, patch.object(checkpoint, "_strict_model_load") as strict:
            with self.assertRaisesRegex(ValueError, "unsafe global"):
                self.inspect()
            loader.assert_called_once()
            strict.assert_not_called()

    def test_safe_load_requests_cpu_and_weights_only(self):
        loader = Mock(return_value=self.obj)
        path = self.root / "synthetic.pth"
        with patch.dict("sys.modules", {"torch": SimpleNamespace(load=loader)}):
            self.assertIs(checkpoint._safe_load(path), self.obj)
        loader.assert_called_once_with(path, map_location="cpu", weights_only=True)

    def test_unknown_asset_and_invalid_fold_rejected(self):
        for asset, fold in (
            ("unknown", 0),
            ("lls48-mrtt-b224", True),
            ("lls48-mrtt-b224", 5),
        ):
            with self.subTest(asset=asset, fold=fold), self.assertRaises(ValueError):
                checkpoint.inspect_checkpoint(
                    asset, self.root, fold, self.root / "missing.json", self.root
                )

    def test_missing_local_file_is_not_downloaded_or_loaded(self):
        with patch.object(checkpoint, "_safe_load") as loader:
            with self.assertRaises(FileNotFoundError):
                checkpoint.inspect_checkpoint(
                    "lls48-mrtt-b224",
                    self.root,
                    4,
                    self.manifest(),
                    ROOT / "configs/temporal",
                )
            loader.assert_not_called()

    def test_cli_returns_json(self):
        with patch.object(
            checkpoint, "_safe_load", return_value=self.obj
        ), patch.object(checkpoint, "_strict_model_load"), patch(
            "sys.stdout", new_callable=io.StringIO
        ) as out:
            status = checkpoint.main(
                [
                    "--asset",
                    "lls48-mrtt-b224",
                    "--root",
                    str(self.root),
                    "--fold",
                    "0",
                    "--manifest",
                    str(self.manifest()),
                    "--config-root",
                    str(ROOT / "configs/temporal"),
                ]
            )
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(out.getvalue())["num_classes"], 266)
