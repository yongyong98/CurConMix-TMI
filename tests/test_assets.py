"""Validate public metadata without reading or downloading clinical assets."""

import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from urllib.parse import urlparse
from urllib.request import Request
import zipfile


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests" / "assets.json"


def load_assets_module():
    spec = importlib.util.spec_from_file_location(
        "curconmix_assets_under_test", ROOT / "src" / "curconmix_release" / "assets.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MockHTTPSResponse(io.BytesIO):
    def geturl(self):
        return "https://downloads.invalid/synthetic-test-asset.bin"


class AssetDownloadSafetyTests(unittest.TestCase):
    """Use synthetic bytes and mock every network call."""

    @classmethod
    def setUpClass(cls):
        cls.module = load_assets_module()

    def setUp(self):
        self.payload = b"synthetic unit-test bytes; no clinical data"
        self.asset = {
            "id": "synthetic-test-asset",
            "status": "published",
            "available": True,
            "license_status": "approved",
            "url": "https://downloads.invalid/synthetic-test-asset.bin",
            "sha256": hashlib.sha256(self.payload).hexdigest(),
            "size_bytes": len(self.payload),
        }

    def test_published_approved_asset_passes_validation(self):
        self.module.validate_downloadable(self.asset)

    def test_hardlink_probe_cleans_up_without_creating_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sentinel = root / "keep.txt"
            sentinel.write_bytes(b"preserve")
            self.module._probe_hardlink_support(root / "not-created" / "dataset")
            self.assertEqual(list(root.iterdir()), [sentinel])
            self.assertEqual(sentinel.read_bytes(), b"preserve")

    def test_unsupported_download_filesystem_fails_before_network(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "not-created" / "asset.bin"
            with mock.patch.object(
                self.module.os, "link", side_effect=OSError("not supported")
            ), mock.patch.object(self.module, "urlopen") as network:
                with self.assertRaisesRegex(OSError, "hard-link support"):
                    self.module.download_asset(self.asset, destination)
                network.assert_not_called()
            self.assertEqual(list(root.iterdir()), [])

    def test_redirect_handler_rejects_http_before_network_access(self):
        handler = self.module.HTTPSOnlyRedirect()
        request = Request("https://downloads.invalid/synthetic-test-asset.bin")
        with mock.patch.object(self.module, "urlopen") as network:
            with self.assertRaisesRegex(ValueError, "HTTPS"):
                handler.redirect_request(
                    request,
                    None,
                    302,
                    "Found",
                    {},
                    "http://downloads.invalid/synthetic-test-asset.bin",
                )
            network.assert_not_called()

    def test_unapproved_or_unpublished_asset_never_calls_network(self):
        changes = (
            {"status": "unpublished"},
            {"available": False},
            {"license_status": "pending"},
        )
        for change in changes:
            with self.subTest(change=change):
                asset = dict(self.asset, **change)
                with tempfile.TemporaryDirectory(
                    prefix="curconmix-gate-test-"
                ) as directory:
                    target = Path(directory) / "not-created" / "asset.bin"
                    with mock.patch.object(self.module, "urlopen") as network:
                        with self.assertRaises(ValueError):
                            self.module.download_asset(asset, target)
                        network.assert_not_called()
                    self.assertFalse(target.parent.exists())

    def test_existing_destination_is_preserved_without_network(self):
        with tempfile.TemporaryDirectory(
            prefix="curconmix-overwrite-test-"
        ) as directory:
            target = Path(directory) / "existing.bin"
            original = b"preserve this existing synthetic file"
            target.write_bytes(original)
            with mock.patch.object(self.module, "urlopen") as network:
                with self.assertRaises(FileExistsError):
                    self.module.download_asset(self.asset, target)
                network.assert_not_called()
            self.assertEqual(target.read_bytes(), original)
            self.assertEqual(list(Path(directory).iterdir()), [target])

    def test_hash_mismatch_rejects_output_and_removes_temporary_file(self):
        asset = dict(self.asset, sha256="0" * 64)
        with tempfile.TemporaryDirectory(prefix="curconmix-hash-test-") as directory:
            target = Path(directory) / "asset.bin"
            response = MockHTTPSResponse(self.payload)
            with mock.patch.object(
                self.module, "urlopen", return_value=response
            ) as network:
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    self.module.download_asset(asset, target)
                network.assert_called_once()
            self.assertFalse(target.exists())
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_size_mismatch_rejects_short_and_long_downloads(self):
        for delta in (-1, 1):
            with self.subTest(declared_size_delta=delta):
                asset = dict(self.asset, size_bytes=len(self.payload) + delta)
                with tempfile.TemporaryDirectory(
                    prefix="curconmix-size-test-"
                ) as directory:
                    target = Path(directory) / "asset.bin"
                    response = MockHTTPSResponse(self.payload)
                    with mock.patch.object(
                        self.module, "urlopen", return_value=response
                    ):
                        with self.assertRaisesRegex(ValueError, "size"):
                            self.module.download_asset(asset, target)
                    self.assertFalse(target.exists())
                    self.assertEqual(list(Path(directory).iterdir()), [])

    def test_matching_synthetic_download_is_saved(self):
        with tempfile.TemporaryDirectory(prefix="curconmix-valid-test-") as directory:
            target = Path(directory) / "asset.bin"
            response = MockHTTPSResponse(self.payload)
            with mock.patch.object(
                self.module, "urlopen", return_value=response
            ) as network:
                result = self.module.download_asset(self.asset, target)
                network.assert_called_once_with(self.asset["url"], timeout=60)
            self.assertEqual(result, target)
            self.assertEqual(target.read_bytes(), self.payload)
            self.assertEqual(list(Path(directory).iterdir()), [target])


class PublicAssetManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        cls.assets = cls.manifest["assets"]

    def test_schema_and_unique_ids(self):
        self.assertIn(self.manifest["schema_version"], (1, 2))
        ids = [asset["id"] for asset in self.assets]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(ids)
        for asset in self.assets:
            with self.subTest(asset=asset["id"]):
                self.assertRegex(asset["id"], r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
                self.assertIn(asset["kind"], {"dataset", "checkpoint_bundle"})
                self.assertIn(asset["dataset"], {"cholect45", "lls48"})
                for field in (
                    "status",
                    "available",
                    "license",
                    "license_status",
                    "release_note",
                ):
                    self.assertIn(field, asset)
                if "files" not in asset:
                    for field in ("url", "sha256", "size_bytes"):
                        self.assertIn(field, asset)

    def test_download_metadata_fails_closed(self):
        for asset in self.assets:
            with self.subTest(asset=asset["id"]):
                self.assertIn(asset["status"], {"unpublished", "published"})
                if asset["status"] == "unpublished":
                    self.assertIs(asset["available"], False)
                    self.assertIsNone(asset.get("url"))
                    self.assertIsNone(asset.get("sha256"))
                    self.assertIsNone(asset.get("size_bytes"))
                else:
                    self.assertIs(asset["available"], True)
                    module = load_assets_module()
                    module.validate_downloadable(asset)
                    for part in module.asset_files(asset):
                        parsed = urlparse(part["url"])
                        self.assertEqual(parsed.scheme, "https")
                        self.assertTrue(parsed.hostname)
                        self.assertNotIn(parsed.hostname, {"localhost", "example.com"})
                        self.assertIsNone(parsed.username)
                        self.assertIsNone(parsed.password)
                        self.assertRegex(part["sha256"], r"^[0-9a-f]{64}$")
                        self.assertIsInstance(part["size_bytes"], int)
                        self.assertGreater(part["size_bytes"], 0)
                    self.assertEqual(asset["license_status"], "approved")

    def test_approved_license_is_separate_from_asset_publication(self):
        for asset in self.assets:
            with self.subTest(asset=asset["id"]):
                self.assertEqual(asset["license"], "CC-BY-NC-4.0")
                self.assertEqual(asset["license_status"], "approved")
                if asset["status"] == "unpublished":
                    module = load_assets_module()
                    with self.assertRaisesRegex(ValueError, "not published"):
                        module.validate_downloadable(asset)

    def test_canonical_lls48_selection(self):
        dataset = next(asset for asset in self.assets if asset["id"] == "lls48-dataset")
        self.assertEqual(dataset["procedures"], 48)
        self.assertEqual(dataset["clips"], 1912)
        self.assertEqual(dataset["frames_per_clip"], 5)
        self.assertEqual(dataset["images"], 9560)
        self.assertEqual(
            dataset["images"], dataset["clips"] * dataset["frames_per_clip"]
        )
        self.assertEqual(dataset["evaluation_scope"], "independent_five_frame_clips")

    def test_mrtt_inventory_does_not_advertise_spatial_weights(self):
        bundles = [
            asset for asset in self.assets if asset["kind"] == "checkpoint_bundle"
        ]
        self.assertGreaterEqual(len(bundles), 6)
        expected = {
            (dataset, variant)
            for dataset in ("cholect45", "lls48")
            for variant in ("swinb224", "swinb384", "swinl224")
        }
        self.assertTrue(
            expected.issubset(
                {(asset["dataset"], asset["variant"]) for asset in bundles}
            )
        )
        for asset in bundles:
            if "temporal_model" not in asset:
                continue
            self.assertEqual(asset["temporal_model"], "mr2")
            self.assertEqual(asset["folds"], [0, 1, 2, 3, 4])
            self.assertIn("spatial encoder is not included", asset["release_note"])

    def test_no_private_paths_in_public_asset_metadata_or_docs(self):
        forbidden = (
            r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]",
            r"\\\\[A-Za-z0-9]",
            r"(?i)/(?:home|Users|mnt)/",
            r"(?i)\b(?:192\.168|10\.\d{1,3})\.",
        )
        for path in (
            MANIFEST,
            ROOT / "README.md",
            *sorted((ROOT / "docs").glob("*.md")),
        ):
            text = path.read_text(encoding="utf-8")
            for pattern in forbidden:
                with self.subTest(file=path.name, pattern=pattern):
                    self.assertIsNone(re.search(pattern, text))

    def test_documented_publication_status_matches_manifest(self):
        dataset = next(asset for asset in self.assets if asset["id"] == "lls48-dataset")
        for relative in (
            "README.md",
            "docs/data.md",
            "docs/model_zoo.md",
            "docs/release_status.md",
            "docs/quickstart_lls48.md",
        ):
            text = (ROOT / relative).read_text(encoding="utf-8")
            with self.subTest(file=relative):
                self.assertEqual(
                    "Publication status: uploading to a private release draft" in text,
                    not dataset["available"],
                )

    def test_lls48_documentation_preserves_interpretation_limits(self):
        text = (ROOT / "docs" / "data.md").read_text(encoding="utf-8")
        self.assertIn("clip-bounded", text)
        self.assertIn("Event extraction resets at every clip boundary", text)
        self.assertIn("must not be concatenated or interpolated across gaps", text)
        self.assertIn("Procedure-wise AP aggregation does **not**", text)

    def test_unpublished_asset_download_exits_without_creating_target(self):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "src")
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        with tempfile.TemporaryDirectory(prefix="curconmix-assets-test-") as directory:
            target = Path(directory) / "data" / "lls48.zip"
            manifest = Path(directory) / "unpublished.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "assets": [
                            {
                                "id": "lls48-dataset",
                                "status": "unpublished",
                                "available": False,
                                "license_status": "approved",
                                "files": [],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "curconmix_release.assets",
                    "--asset",
                    "lls48-dataset",
                    "--destination",
                    str(target),
                    "--manifest",
                    str(manifest),
                ],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertRegex(
                (result.stdout + result.stderr).lower(), r"unpublished|not published"
            )
            self.assertFalse(target.exists())
            self.assertFalse(target.parent.exists())


class MultipartAssetTests(unittest.TestCase):
    """Installation uses only generated synthetic files and mocked HTTP."""

    @classmethod
    def setUpClass(cls):
        cls.module = load_assets_module()

    def make_part(self, name, payload, **extra):
        return {
            "filename": name,
            "url": f"https://downloads.invalid/{name}",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
            "format": "file",
            **extra,
        }

    def make_asset(self, parts):
        return {
            "id": "synthetic-bundle",
            "status": "published",
            "available": True,
            "license_status": "approved",
            "license": "CC-BY-NC-4.0",
            "files": parts,
        }

    def make_zip(self, entries):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, content in entries:
                archive.writestr(name, content)
        return buffer.getvalue()

    def test_cached_parts_resume_without_redownloading(self):
        parts = [self.make_part("a.bin", b"first"), self.make_part("b.bin", b"second")]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.bin").write_bytes(b"first")
            with mock.patch.object(
                self.module, "urlopen", return_value=MockHTTPSResponse(b"second")
            ) as network:
                self.module.process_asset(self.make_asset(parts), root)
                network.assert_called_once_with(parts[1]["url"], timeout=60)
            with mock.patch.object(self.module, "urlopen") as network:
                self.module.process_asset(
                    self.make_asset(parts), root, verify_only=True
                )
                network.assert_not_called()

    def test_install_filesystem_probe_fails_before_cache_download(self):
        part = self.make_part("a.bin", b"first")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(
                self.module.os, "link", side_effect=OSError("not supported")
            ), mock.patch.object(self.module, "urlopen") as network:
                with self.assertRaisesRegex(OSError, "hard-link support"):
                    self.module.process_asset(
                        self.make_asset([part]),
                        root / "dataset",
                        install=True,
                        cache_dir=root / "cache",
                    )
                network.assert_not_called()
            self.assertEqual(list(root.iterdir()), [])

    def test_separate_cache_filesystem_is_probed_before_network(self):
        part = self.make_part("a.bin", b"first")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination, cache = root / "dataset", root / "separate-cache"

            def fail_cache(path):
                if path == cache:
                    raise OSError("cache hard-link support unavailable")

            with mock.patch.object(
                self.module, "_probe_hardlink_support", side_effect=fail_cache
            ) as probe, mock.patch.object(self.module, "urlopen") as network:
                with self.assertRaisesRegex(OSError, "cache hard-link"):
                    self.module.process_asset(
                        self.make_asset([part]),
                        destination,
                        install=True,
                        cache_dir=cache,
                    )
                self.assertEqual(
                    probe.call_args_list,
                    [
                        mock.call(destination.parent),
                        mock.call(destination),
                        mock.call(cache),
                    ],
                )
                network.assert_not_called()
            self.assertEqual(list(root.iterdir()), [])

    def test_verify_only_never_runs_hardlink_probe(self):
        part = self.make_part("a.bin", b"first")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.bin").write_bytes(b"first")
            with mock.patch.object(
                self.module, "_probe_hardlink_support"
            ) as probe, mock.patch.object(self.module, "urlopen") as network:
                self.module.process_asset(
                    self.make_asset([part]), root, verify_only=True
                )
                self.module.process_asset(
                    self.make_asset([part]), root, install=True, verify_only=True
                )
                probe.assert_not_called()
                network.assert_not_called()

    def test_corrupt_cached_part_is_preserved_and_never_redownloaded(self):
        part = self.make_part("a.bin", b"right")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "a.bin"
            path.write_bytes(b"wrong")
            with mock.patch.object(self.module, "urlopen") as network:
                with self.assertRaisesRegex(ValueError, "preserved"):
                    self.module.process_asset(self.make_asset([part]), Path(directory))
                network.assert_not_called()
            self.assertEqual(path.read_bytes(), b"wrong")

    def test_zip_install_and_offline_verification(self):
        payload = self.make_zip(
            [("images/clip/frame.txt", b"synthetic"), ("labels.csv", b"a,b\n")]
        )
        part = self.make_part(
            "part01.zip", payload, format="zip", expanded_size_bytes=13, member_count=2
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            with mock.patch.object(
                self.module, "urlopen", return_value=MockHTTPSResponse(payload)
            ):
                self.module.process_asset(self.make_asset([part]), root, install=True)
            self.assertEqual(
                (root / "images/clip/frame.txt").read_bytes(), b"synthetic"
            )
            with mock.patch.object(self.module, "urlopen") as network:
                self.module.process_asset(
                    self.make_asset([part]), root, install=True, verify_only=True
                )
                self.module.process_asset(self.make_asset([part]), root, install=True)
                network.assert_not_called()
            (root / "labels.csv").write_bytes(b"tamper")
            with mock.patch.object(self.module, "urlopen") as network:
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    self.module.process_asset(
                        self.make_asset([part]), root, install=True, verify_only=True
                    )
                network.assert_not_called()

    def test_checkpoint_install_path_and_verify_without_cache(self):
        part = self.make_part(
            "fold0.pth",
            b"fake checkpoint",
            install_path="lls48/b224/best_model_fold0.pth",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "weights"
            with mock.patch.object(
                self.module,
                "urlopen",
                return_value=MockHTTPSResponse(b"fake checkpoint"),
            ):
                self.module.process_asset(self.make_asset([part]), root, install=True)
            cache = root.parent / ".curconmix-cache/synthetic-bundle/fold0.pth"
            cache.unlink()
            with mock.patch.object(self.module, "urlopen") as network:
                self.module.process_asset(
                    self.make_asset([part]), root, install=True, verify_only=True
                )
                network.assert_not_called()

    def test_zip_collision_does_not_publish_other_members(self):
        payload = self.make_zip([("new.txt", b"new"), ("existing.txt", b"expected")])
        part = self.make_part("part.zip", payload, format="zip")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            root.mkdir()
            (root / "existing.txt").write_bytes(b"user data")
            with mock.patch.object(
                self.module, "urlopen", return_value=MockHTTPSResponse(payload)
            ):
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    self.module.process_asset(
                        self.make_asset([part]), root, install=True
                    )
            self.assertFalse((root / "new.txt").exists())
            self.assertEqual((root / "existing.txt").read_bytes(), b"user data")
            self.assertFalse(list(root.parent.glob(".curconmix-install-*")))

    def test_malicious_zip_paths_are_rejected_before_publish(self):
        for name in (
            "../escape",
            "/absolute",
            "C:/drive",
            "folder\\escape",
            "folder/../escape",
            "CON.txt",
            "file.",
            "file:stream",
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                payload = self.make_zip([("normal.txt", b"normal"), (name, b"bad")])
                # Windows ZipInfo normalizes separators while constructing a ZIP;
                # restore the malicious on-disk name in both filename headers.
                if "\\" in name:
                    payload = payload.replace(
                        name.replace("\\", "/").encode(), name.encode()
                    )
                part = self.make_part("part.zip", payload, format="zip")
                root = Path(directory) / "dataset"
                with mock.patch.object(
                    self.module, "urlopen", return_value=MockHTTPSResponse(payload)
                ):
                    with self.assertRaisesRegex(ValueError, "Unsafe"):
                        self.module.process_asset(
                            self.make_asset([part]), root, install=True
                        )
                self.assertFalse(root.exists())

    def test_zip_symlink_rejected(self):
        link = zipfile.ZipInfo("link")
        link.create_system = 3
        link.external_attr = 0o120777 << 16
        payload = self.make_zip([(link, "../../outside")])
        part = self.make_part("part.zip", payload, format="zip")
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(
                self.module, "urlopen", return_value=MockHTTPSResponse(payload)
            ):
                with self.assertRaisesRegex(ValueError, "symlink"):
                    self.module.process_asset(
                        self.make_asset([part]),
                        Path(directory) / "dataset",
                        install=True,
                    )

    def test_zip_case_collision_and_file_directory_collision_rejected(self):
        for entries in (
            [("Same.txt", b"a"), ("same.txt", b"b")],
            [("Images/a.txt", b"a"), ("images/b.txt", b"b")],
            [("folder", b"a"), ("folder/child", b"b")],
        ):
            with self.subTest(
                entries=entries
            ), tempfile.TemporaryDirectory() as directory:
                payload = self.make_zip(entries)
                part = self.make_part("part.zip", payload, format="zip")
                with mock.patch.object(
                    self.module, "urlopen", return_value=MockHTTPSResponse(payload)
                ):
                    with self.assertRaisesRegex(ValueError, "colli"):
                        self.module.process_asset(
                            self.make_asset([part]),
                            Path(directory) / "dataset",
                            install=True,
                        )

    def test_zip_declared_expansion_bound_enforced(self):
        payload = self.make_zip([("large.txt", b"A" * 2048)])
        part = self.make_part(
            "part.zip", payload, format="zip", expanded_size_bytes=2047
        )
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(
                self.module, "urlopen", return_value=MockHTTPSResponse(payload)
            ):
                with self.assertRaisesRegex(ValueError, "expanded size"):
                    self.module.process_asset(
                        self.make_asset([part]),
                        Path(directory) / "dataset",
                        install=True,
                    )

    def test_optional_member_hash_mismatch_rejected(self):
        payload = self.make_zip([("one.txt", b"one")])
        part = self.make_part(
            "part.zip",
            payload,
            format="zip",
            members=[
                {
                    "path": "one.txt",
                    "size_bytes": 3,
                    "sha256": "0" * 64,
                }
            ],
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            with mock.patch.object(
                self.module, "urlopen", return_value=MockHTTPSResponse(payload)
            ):
                with self.assertRaisesRegex(ValueError, "member SHA"):
                    self.module.process_asset(
                        self.make_asset([part]), root, install=True
                    )
            self.assertFalse(root.exists())

    def test_optional_members_allow_offline_verification_without_zip_cache(self):
        payload = self.make_zip([("one.txt", b"one")])
        part = self.make_part(
            "part.zip",
            payload,
            format="zip",
            members=[
                {
                    "path": "one.txt",
                    "size_bytes": 3,
                    "sha256": hashlib.sha256(b"one").hexdigest(),
                }
            ],
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "one.txt").write_bytes(b"one")
            with mock.patch.object(self.module, "urlopen") as network:
                self.module.process_asset(
                    self.make_asset([part]), root, install=True, verify_only=True
                )
                network.assert_not_called()

    def test_unsafe_part_filename_install_path_or_url_rejected_without_network(self):
        base = self.make_part("ok.bin", b"test")
        for changes in (
            {"filename": "../escape.bin"},
            {"filename": "folder/a.bin"},
            {"install_path": "../../escape"},
            {"url": "http://downloads.invalid/a"},
            {"url": "https://user:secret@downloads.invalid/a"},
        ):
            with self.subTest(
                changes=changes
            ), tempfile.TemporaryDirectory() as directory:
                with mock.patch.object(self.module, "urlopen") as network:
                    with self.assertRaises(ValueError):
                        self.module.process_asset(
                            self.make_asset([{**base, **changes}]),
                            Path(directory) / "new",
                            install=True,
                        )
                    network.assert_not_called()

    def test_missing_offline_cache_does_not_call_network_or_create_directories(self):
        part = self.make_part("part.bin", b"test")
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "missing"
            with mock.patch.object(self.module, "urlopen") as network:
                with self.assertRaises(FileNotFoundError):
                    self.module.process_asset(
                        self.make_asset([part]), destination, verify_only=True
                    )
                network.assert_not_called()
            self.assertFalse(destination.exists())

    def test_unsafe_redirect_with_credentials_rejected(self):
        handler = self.module.HTTPSOnlyRedirect()
        with self.assertRaisesRegex(ValueError, "credentials"):
            handler.redirect_request(
                Request("https://downloads.invalid/part"),
                None,
                302,
                "Found",
                {},
                "https://user:secret@downloads.invalid/part",
            )

    def test_corrupt_zip_crc_rejected_even_with_matching_archive_sha(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr("one.txt", b"abcdef")
        payload = buffer.getvalue().replace(b"abcdef", b"abcdeg", 1)
        part = self.make_part("part.zip", payload, format="zip")
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(
                self.module, "urlopen", return_value=MockHTTPSResponse(payload)
            ):
                with self.assertRaises(zipfile.BadZipFile):
                    self.module.process_asset(
                        self.make_asset([part]),
                        Path(directory) / "dataset",
                        install=True,
                    )


if __name__ == "__main__":
    unittest.main()
