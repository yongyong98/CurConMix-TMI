"""Synthetic spatial manifest fixtures; no model execution or network requests."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "spatial_manifest_builder", ROOT / "tools/build_spatial_manifest.py"
)
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)
PENDING_BANNER = (
    "> Spatial release status: preparation in progress. Downloads are disabled until\n"
    "> all spatial assets pass local loading and server-side integrity checks."
)


def assert_publication_consistent(manifest, document):
    """Check the checked-in publication boundary without reading any checkpoints."""
    assets = manifest["assets"]
    expected = {
        f"{dataset}-spatial-{variant}"
        for dataset, variants in BUILDER.VARIANTS.items()
        for variant in variants
    }
    assert len(assets) == 3 and {asset["id"] for asset in assets} == expected
    assert all(type(asset.get("available")) is bool for asset in assets)
    states = {asset["available"] for asset in assets}
    assert len(states) == 1, "Mixed spatial publication is not allowed"
    published = states == {True}
    assert (PENDING_BANNER in document) is not published
    for asset in assets:
        assert asset["status"] == ("published" if published else "unpublished")
        assert len(asset["files"]) == 5
        assert {record["fold"] for record in asset["files"]} == set(range(5))
        for record in asset["files"]:
            assert record["url"] == BUILDER.release_url(record["filename"], published)
    assert manifest["checksums"]["url"] == BUILDER.release_url(
        BUILDER.CHECKSUM_FILENAME, published
    )
    BUILDER.public_paths_only(manifest)


class SpatialManifestTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(
            prefix="curconmix-spatial-manifest-test-"
        )
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.directory = self.root / "weights"
        self.source_path = self.root / "source.json"
        self.source = self.fixture("lls48", self.directory)
        self.cholect_directory = self.root / "cholect_weights"
        self.cholect_source_path = self.root / "cholect_source.json"
        self.cholect_source = self.fixture("cholect45", self.cholect_directory)
        self.cholect_source_path.write_text(
            json.dumps(self.cholect_source), encoding="utf-8"
        )
        self.write_source()

    def fixture(self, dataset, directory):
        directory.mkdir()
        records = []
        for variant in BUILDER.VARIANTS[dataset]:
            for fold in range(5):
                artifact_id = f"{dataset}-spatial-{variant}-fold{fold}"
                filename = f"{artifact_id}.pth"
                content = f"synthetic-{variant}-{fold}".encode()
                (directory / filename).write_bytes(content)
                records.append(
                    {
                        "id": artifact_id,
                        "filename": filename,
                        "variant": variant,
                        "fold": fold,
                        "dataset": dataset,
                        "license": BUILDER.LICENSE,
                        "sha256": hashlib.sha256(content).hexdigest(),
                        "size_bytes": len(content),
                        "source_sha256": "a" * 64,
                        "canonical_prediction_sha256": (
                            "b" if variant == "b224" else "c"
                        )
                        * 64,
                        "mapping_sha256": "d" * 64,
                        "tensor_equality": "PASS",
                        "strict_model_load": "PASS",
                        "source_preserved": True,
                        "prediction_pairing": "EXACT_FLOAT32",
                        "install_path": f"{dataset}/{variant}/fold{fold}.pth",
                        "model_tensor_count": 1,
                        "model_tensor_bytes": 4,
                        "metadata": {
                            "schema_version": 1,
                            "dataset": dataset,
                            "variant": variant,
                            "fold": fold,
                            "model_class": "TripletModel",
                            "num_classes": sum(BUILDER.CLASS_COUNTS[dataset].values()),
                            "triplet_size": BUILDER.CLASS_COUNTS[dataset]["triplet"],
                            "class_counts": BUILDER.CLASS_COUNTS[dataset],
                            "mapping_sha256": "d" * 64,
                            "license": BUILDER.LICENSE,
                            "internal_ignored": "C:/private/synthetic-source.pth",
                        },
                    }
                )
        return {
            "schema_version": 1,
            "license": BUILDER.LICENSE,
            "artifacts": records,
        }

    def write_source(self):
        self.source_path.write_text(json.dumps(self.source), encoding="utf-8")

    def build(self, **kwargs):
        return BUILDER.build(
            self.directory,
            self.source_path,
            self.cholect_directory,
            self.cholect_source_path,
            **kwargs,
        )

    def arguments(self):
        return [
            "--checkpoint-dir",
            str(self.directory),
            "--checkpoint-manifest",
            str(self.source_path),
            "--cholect45-checkpoint-dir",
            str(self.cholect_directory),
            "--cholect45-checkpoint-manifest",
            str(self.cholect_source_path),
        ]

    def test_default_manifest_is_fail_closed_exactly_three_complete_bundles(self):
        original = self.source_path.read_bytes()
        manifest, plan = self.build()
        self.assertEqual(manifest["schema_version"], 2)
        self.assertEqual(len(manifest["assets"]), 3)
        self.assertEqual(plan["repository"], "yongyong98/CurConMix-TMI")
        self.assertEqual(plan["tag"], "spatial-encoders-v1.0.0")
        self.assertEqual(len(plan["files"]), 15)
        self.assertEqual(self.source_path.read_bytes(), original)
        self.assertNotIn("private", json.dumps(manifest))
        self.assertNotIn(str(self.root), json.dumps(manifest))
        self.assertIsNone(manifest["checksums"]["url"])
        for asset in manifest["assets"]:
            self.assertEqual(asset["kind"], "spatial_checkpoint_bundle")
            self.assertEqual(asset["architecture"], "TripletModel")
            self.assertIn(asset["variant"], ("swinb224", "swinb384", "swinl224"))
            self.assertEqual(asset["folds"], list(range(5)))
            self.assertEqual(asset["status"], "unpublished")
            self.assertFalse(asset["available"])
            self.assertEqual(
                [record["fold"] for record in asset["files"]], list(range(5))
            )
            self.assertTrue(all(record["url"] is None for record in asset["files"]))

    def test_explicit_published_flag_uses_only_fixed_repository_and_tag(self):
        manifest, _ = self.build(published=True)
        for asset in manifest["assets"]:
            self.assertTrue(asset["available"])
            self.assertEqual(asset["status"], "published")
            for record in asset["files"]:
                self.assertEqual(
                    record["url"],
                    f"https://github.com/{BUILDER.REPOSITORY}/releases/download/{BUILDER.TAG}/{record['filename']}",
                )
                self.assertRegex(
                    record["install_path"],
                    r"^(lls48/b(224|384)|cholect45/l224)/fold[0-4]\.pth$",
                )
        self.assertTrue(
            manifest["checksums"]["url"].endswith(BUILDER.CHECKSUM_FILENAME)
        )

    def test_every_export_gate_is_required(self):
        original = copy.deepcopy(self.source)
        for key, value in (
            ("tensor_equality", "FAIL"),
            ("strict_model_load", "FAIL"),
            ("prediction_pairing", "APPROXIMATE"),
            ("source_preserved", False),
        ):
            with self.subTest(key=key):
                self.source = copy.deepcopy(original)
                self.source["artifacts"][0][key] = value
                self.write_source()
                with self.assertRaisesRegex(ValueError, "QA"):
                    self.build()

    def test_missing_duplicate_bool_or_foreign_variant_fold_is_rejected(self):
        original = copy.deepcopy(self.source)
        changes = (
            lambda records: records.pop(),
            lambda records: records.__setitem__(1, records[0]),
            lambda records: records[0].__setitem__("fold", False),
            lambda records: records[0].__setitem__("variant", "l224"),
        )
        for change in changes:
            self.source = copy.deepcopy(original)
            change(self.source["artifacts"])
            self.write_source()
            with self.assertRaises(ValueError):
                self.build()

    def test_identity_license_and_unsafe_paths_rejected(self):
        original = copy.deepcopy(self.source)
        for key, value in (
            ("dataset", "cholect45"),
            ("license", "MIT"),
            ("id", "unverified"),
            ("filename", "../escape.pth"),
            ("install_path", "C:/private/checkpoint.pth"),
        ):
            with self.subTest(key=key):
                self.source = copy.deepcopy(original)
                self.source["artifacts"][0][key] = value
                self.write_source()
                with self.assertRaisesRegex(ValueError, "identity"):
                    self.build()

    def test_export_sha_is_always_recomputed(self):
        record = self.source["artifacts"][0]
        path = self.directory / record["filename"]
        path.write_bytes(b"x" * record["size_bytes"])
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.build()

    def test_export_size_and_missing_file_rejected(self):
        path = self.directory / self.source["artifacts"][0]["filename"]
        path.write_bytes(b"truncated")
        with self.assertRaisesRegex(ValueError, "size"):
            self.build()
        path.unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            self.build()

    def test_provenance_hashes_and_variant_consistency_required(self):
        original = copy.deepcopy(self.source)
        for key, value in (
            ("source_sha256", "bad"),
            ("canonical_prediction_sha256", "e" * 64),
            ("mapping_sha256", "e" * 64),
            ("model_tensor_count", 0),
            ("model_tensor_bytes", True),
        ):
            with self.subTest(key=key):
                self.source = copy.deepcopy(original)
                self.source["artifacts"][0][key] = value
                self.write_source()
                with self.assertRaises(ValueError):
                    self.build()

    def test_cholect45_requires_complete_l224_and_all_gates(self):
        original = copy.deepcopy(self.cholect_source)
        for change in (
            lambda records: records.pop(),
            lambda records: records[0].__setitem__("variant", "b224"),
            lambda records: records[0].__setitem__("prediction_pairing", "APPROXIMATE"),
            lambda records: records[0]["metadata"].__setitem__("num_classes", 266),
        ):
            source = copy.deepcopy(original)
            change(source["artifacts"])
            self.cholect_source_path.write_text(json.dumps(source), encoding="utf-8")
            with self.assertRaises(ValueError):
                self.build()

    def test_partial_dataset_selection_and_bad_source_schema_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Both complete"):
            BUILDER.build(self.directory, self.source_path, None, None)
        self.source["schema_version"] = 2
        self.write_source()
        with self.assertRaisesRegex(ValueError, "schema/license"):
            self.build()

    def test_publication_boundary_and_private_paths_are_checked(self):
        pending, _ = self.build()
        assert_publication_consistent(pending, PENDING_BANNER)
        published, _ = self.build(published=True)
        assert_publication_consistent(published, "All spatial assets are published.")
        with self.assertRaises(AssertionError):
            assert_publication_consistent(pending, "No publication boundary")
        with self.assertRaises(AssertionError):
            assert_publication_consistent(published, PENDING_BANNER)
        pending["assets"][0]["available"] = True
        with self.assertRaisesRegex(AssertionError, "Mixed"):
            assert_publication_consistent(pending, PENDING_BANNER)
        published["private_path"] = "C:/Users/private/checkpoint.pth"
        with self.assertRaisesRegex(ValueError, "Private absolute"):
            assert_publication_consistent(
                published, "All spatial assets are published."
            )

    def test_cli_emits_exact_sixteen_asset_plan_and_fifteen_weight_checksums(self):
        with patch.object(BUILDER, "ROOT", self.root):
            BUILDER.main(self.arguments())
        local = self.root / ".release-local"
        plan = json.loads((local / "spatial_upload_plan.json").read_text())
        manifest = json.loads((local / "pending_spatial_assets.json").read_text())
        checksum_file = local / BUILDER.CHECKSUM_FILENAME
        lines = checksum_file.read_text().splitlines()
        actual = {name: sha for sha, name in (line.split("  ") for line in lines)}
        expected = {
            record["filename"]: record["sha256"]
            for record in self.source["artifacts"] + self.cholect_source["artifacts"]
        }
        self.assertEqual(actual, expected)
        self.assertEqual(len(plan["files"]), 16)
        self.assertEqual(
            {Path(item["path"]).name for item in plan["files"]},
            set(expected) | {BUILDER.CHECKSUM_FILENAME},
        )
        self.assertEqual(
            plan["files"][-1]["sha256"],
            hashlib.sha256(checksum_file.read_bytes()).hexdigest(),
        )
        self.assertEqual(manifest["checksums"]["sha256"], plan["files"][-1]["sha256"])
        self.assertNotIn(BUILDER.CHECKSUM_FILENAME, actual)
        self.assertNotIn(str(self.root), checksum_file.read_text())
        for record in plan["files"]:
            self.assertEqual(Path(record["path"]).stat().st_size, record["size_bytes"])

    def test_existing_default_outputs_are_preserved(self):
        local = self.root / ".release-local"
        local.mkdir()
        target = local / "pending_spatial_assets.json"
        target.write_text("existing", encoding="utf-8")
        with patch.object(BUILDER, "ROOT", self.root), self.assertRaises(
            FileExistsError
        ):
            BUILDER.main(self.arguments())
        self.assertEqual(target.read_text(), "existing")
        self.assertEqual(list(local.iterdir()), [target])

    def test_cli_output_collision_and_input_overwrite_rejected(self):
        before = self.source_path.read_bytes()
        for arguments in (
            ("--output", str(self.source_path)),
            ("--output", str(self.cholect_source_path)),
            (
                "--output",
                str(self.root / "same.json"),
                "--upload-plan",
                str(self.root / "same.json"),
            ),
            ("--checksums-output", str(self.root / "wrong.txt")),
            ("--output", str(self.directory / self.source["artifacts"][0]["filename"])),
        ):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                BUILDER.main(self.arguments() + list(arguments))
        self.assertEqual(self.source_path.read_bytes(), before)

    def test_explicit_output_targets_allow_replacement(self):
        output = self.root / "spatial_assets.json"
        plan = self.root / "private_plan.json"
        checksums = self.root / BUILDER.CHECKSUM_FILENAME
        args = self.arguments() + [
            "--output",
            str(output),
            "--upload-plan",
            str(plan),
            "--checksums-output",
            str(checksums),
        ]
        BUILDER.main(args)
        BUILDER.main(args + ["--published"])
        self.assertTrue(json.loads(output.read_text())["assets"][0]["available"])
        self.assertFalse(list(self.root.glob(".manifest-*")))


class PackagedSpatialPublicationTests(unittest.TestCase):
    def test_packaged_spatial_manifest_and_documentation_agree(self):
        manifest_path = ROOT / "manifests/spatial_assets.json"
        self.assertTrue(
            manifest_path.is_file(),
            "Generate the qualified spatial manifest before release QA",
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        document = (ROOT / "docs/spatial_checkpoints.md").read_text(encoding="utf-8")
        assert_publication_consistent(manifest, document)


if __name__ == "__main__":
    unittest.main()
