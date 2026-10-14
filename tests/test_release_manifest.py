"""Synthetic manifest derivation tests; no model, clinical data, or network use."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "release_manifest_builder", ROOT / "tools/build_release_manifest.py"
)
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


class ReleaseManifestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="curconmix-manifest-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.base = self.root / "base.json"
        self.base.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "assets": [
                        {
                            "id": asset_id,
                            "license": "CC-BY-NC-4.0",
                            "license_status": "approved",
                            "status": "unpublished",
                            "available": False,
                            "evaluation_scope": "independent_five_frame_clips"
                            if asset_id == "lls48-dataset"
                            else "action_temporal",
                            "url": None,
                            "sha256": None,
                            "size_bytes": None,
                        }
                        for asset_id in sorted(BUILDER.EXPECTED_IDS)
                    ],
                }
            ),
            encoding="utf-8",
        )
        self.checkpoints = self.root / "checkpoints"
        self.checkpoints.mkdir()
        self.checkpoint_manifest = self.root / "checkpoint_manifest.json"
        records = []
        for variant in BUILDER.VARIANTS:
            for fold in range(5):
                filename = f"lls48-mrtt-{variant}-fold{fold}.pth"
                payload = f"synthetic-{variant}-{fold}".encode()
                (self.checkpoints / filename).write_bytes(payload)
                records.append(
                    {
                        "id": filename[:-4],
                        "variant": variant,
                        "fold": fold,
                        "filename": filename,
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "size_bytes": len(payload),
                        "license": BUILDER.LICENSE,
                        "target_path": f"artifacts/checkpoints/lls48/{variant}/best_model_fold{fold}.pth",
                        "tensor_equality": "PASS",
                        "strict_model_load": "PASS",
                        "architecture_metadata": "PASS",
                        "source_preserved": True,
                    }
                )
        self.checkpoint_source = {"license": BUILDER.LICENSE, "artifacts": records}
        self.write_checkpoints()

    def write_checkpoints(self):
        self.checkpoint_manifest.write_text(
            json.dumps(self.checkpoint_source), encoding="utf-8"
        )

    def build(self, **kwargs):
        return BUILDER.build(
            self.base,
            checkpoint_dir=self.checkpoints,
            checkpoint_manifest=self.checkpoint_manifest,
            **kwargs,
        )

    def dataset(self):
        directory = self.root / "dataset"
        directory.mkdir()
        payload = b"synthetic archive placeholder; QA metadata is fixture"
        (directory / "synthetic.zip").write_bytes(payload)
        sha = hashlib.sha256(b"x").hexdigest()
        names = [
            "dataframe/LLS48_TRI195.csv",
            "splits/folds.json",
            "manifests/frames.json",
            "manifests/checksums.json",
            "DATA_AND_MODEL_LICENSE.md",
        ] + [f"data/VID01/synthetic/frame_{index:05}.png" for index in range(9560)]
        members = [{"path": name, "size_bytes": 1, "sha256": sha} for name in names]
        source = {
            "summary": {
                "frames": 9560,
                "clips": 1912,
                "procedures": 48,
                "annotation_sha256": sha,
            },
            "validation": {
                "zip_crc": "passed",
                "membership": "passed",
                "per_member_sha256": "passed",
                "source_annotation_unchanged": True,
            },
            "files": [
                {
                    "filename": "synthetic.zip",
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "size_bytes": len(payload),
                    "format": "zip",
                    "install_path": ".",
                    "expanded_size_bytes": len(members),
                    "members": members,
                }
            ],
        }
        (directory / "release_files.json").write_text(
            json.dumps(source), encoding="utf-8"
        )
        return directory, source

    def test_checkpoint_only_plan_is_unpublished_and_preserves_base(self):
        original = self.base.read_bytes()
        manifest, plan = self.build(verify_file_hashes=True)
        self.assertEqual(manifest["schema_version"], 2)
        self.assertEqual(len(manifest["assets"]), 7)
        self.assertEqual(len(plan["files"]), 15)
        self.assertEqual(plan["repository"], "yongyong98/CurConMix-TMI")
        self.assertEqual(plan["tag"], "lls48-v1.0.0")
        for asset in manifest["assets"]:
            self.assertEqual(asset["status"], "unpublished")
            self.assertFalse(asset["available"])
            self.assertTrue(all(p["url"] is None for p in asset["files"]))
        self.assertEqual(self.base.read_bytes(), original)

    def test_published_flag_enables_only_prepared_lls48_bundles(self):
        manifest, _ = self.build(published=True)
        for asset in manifest["assets"]:
            if asset["id"].startswith("lls48-mrtt"):
                self.assertTrue(asset["available"])
                for part in asset["files"]:
                    self.assertTrue(
                        part["url"].startswith(
                            "https://github.com/yongyong98/CurConMix-TMI/releases/download/lls48-v1.0.0/"
                        )
                    )
                    self.assertRegex(
                        part["install_path"],
                        r"^lls48/(b224|b384|l224)/best_model_fold[0-4]\.pth$",
                    )
            else:
                self.assertFalse(asset["available"])

    def test_full_plan_keeps_inline_dataset_members_and_boundary(self):
        dataset, _ = self.dataset()
        manifest, plan = self.build(dataset_dir=dataset, published=True)
        self.assertEqual(len(plan["files"]), 16)
        asset = next(a for a in manifest["assets"] if a["id"] == "lls48-dataset")
        self.assertEqual(asset["evaluation_scope"], "independent_five_frame_clips")
        self.assertEqual(len(asset["files"][0]["members"]), 9565)
        self.assertEqual(asset["files"][0]["member_count"], 9565)
        self.assertNotIn(str(self.root), json.dumps(manifest))
        self.assertIn(str(self.root), plan["files"][0]["path"])

    def test_bad_checkpoint_qa_or_license_is_rejected(self):
        original = copy.deepcopy(self.checkpoint_source)
        for key, value in (
            ("tensor_equality", "FAIL"),
            ("strict_model_load", "FAIL"),
            ("architecture_metadata", "FAIL"),
            ("source_preserved", False),
            ("license", "MIT"),
        ):
            with self.subTest(key=key):
                self.checkpoint_source = copy.deepcopy(original)
                self.checkpoint_source["artifacts"][0][key] = value
                self.write_checkpoints()
                with self.assertRaises(ValueError):
                    self.build()

    def test_duplicate_fold_or_wrong_target_is_rejected(self):
        original = copy.deepcopy(self.checkpoint_source)
        self.checkpoint_source["artifacts"][1]["fold"] = 0
        self.write_checkpoints()
        with self.assertRaisesRegex(ValueError, "15 unique"):
            self.build()
        self.checkpoint_source = original
        self.checkpoint_source["artifacts"][0]["target_path"] = "../escape.pth"
        self.write_checkpoints()
        with self.assertRaisesRegex(ValueError, "identity"):
            self.build()

    def test_missing_changed_file_and_hash_mismatch_are_rejected(self):
        file = self.checkpoints / self.checkpoint_source["artifacts"][0]["filename"]
        original = file.read_bytes()
        file.write_bytes(b"x" * len(original))
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.build(verify_file_hashes=True)
        file.write_bytes(b"bad")
        with self.assertRaisesRegex(ValueError, "size"):
            self.build()
        file.unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            self.build()

    def test_failed_dataset_qa_and_noncanonical_selection_rejected(self):
        directory, source = self.dataset()
        original = copy.deepcopy(source)
        for kind in ("qa", "count"):
            source = copy.deepcopy(original)
            if kind == "qa":
                source["validation"]["zip_crc"] = "failed"
            else:
                source["summary"]["frames"] = 1
            (directory / "release_files.json").write_text(
                json.dumps(source), encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                self.build(dataset_dir=directory)

    def test_duplicate_dataset_member_rejected(self):
        directory, source = self.dataset()
        source["files"][0]["members"].append(source["files"][0]["members"][0])
        (directory / "release_files.json").write_text(
            json.dumps(source), encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "Duplicate dataset"):
            self.build(dataset_dir=directory)

    def test_private_paths_in_base_metadata_rejected(self):
        source = json.loads(self.base.read_text())
        source["assets"][0]["internal_path"] = "C:/Users/private/source"
        self.base.write_text(json.dumps(source), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Private absolute"):
            self.build()

    def test_default_output_preserved_explicit_replacement_atomic(self):
        output = self.root / "output.json"
        BUILDER.save_json(output, {"old": True})
        with self.assertRaises(FileExistsError):
            BUILDER.save_json(output, {"new": True})
        self.assertEqual(json.loads(output.read_text()), {"old": True})
        BUILDER.save_json(output, {"new": True}, replace=True)
        self.assertEqual(json.loads(output.read_text()), {"new": True})
        self.assertFalse(list(self.root.glob(".manifest-*")))

    def test_checksum_lines_contain_exact_basenames_and_hashes_only(self):
        _, plan = self.build()
        content = BUILDER.checksum_text(plan)
        actual = {
            name: sha
            for sha, name in (line.split("  ") for line in content.splitlines())
        }
        expected = {
            Path(record["path"]).name: record["sha256"] for record in plan["files"]
        }
        self.assertEqual(actual, expected)
        self.assertEqual(len(actual), 15)
        self.assertNotIn(str(self.root), content)
        self.assertNotIn("\\", content)
        self.assertNotIn("/", content)
        self.assertEqual(list(actual), sorted(actual))

    def test_checksum_windows_paths_are_reduced_on_all_platforms(self):
        content = BUILDER.checksum_text(
            {
                "files": [
                    {
                        "path": r"C:\private\example.zip",
                        "size_bytes": 1,
                        "sha256": "a" * 64,
                    }
                ]
            }
        )
        self.assertEqual(content, "a" * 64 + "  example.zip\n")

    def test_checksum_duplicates_and_unsafe_names_fail(self):
        for names in (("A.zip", "a.zip"), ("bad\nname.zip",), ("CON.zip",)):
            with self.subTest(names=names):
                plan = {
                    "files": [
                        {"path": name, "sha256": "a" * 64, "size_bytes": 1}
                        for name in names
                    ]
                }
                with self.assertRaises(ValueError):
                    BUILDER.checksum_text(plan)


if __name__ == "__main__":
    unittest.main()
