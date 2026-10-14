"""Synthetic fixtures only: no clinical image, annotation, or network access."""

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from curconmix_release import lls48

ROOT = Path(__file__).resolve().parents[1]


def write_fixture(root):
    (root / "dataframe").mkdir()
    rows = []
    for frame in range(1, 6):
        row = {column: "0" for column in lls48.METADATA_COLUMNS + lls48.LABEL_COLUMNS}
        row.update(
            video="VID01",
            clip_name="01_001",
            image_id=f"0_{frame}",
            image_path=f"VID01/01_001/frame_{frame:05d}.png",
        )
        for column in ("triplets_idx", "inst_idx", "verb_idx", "target_idx"):
            row[column] = "[]"
        rows.append(row)
        path = root / "data" / row["image_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"synthetic image fixture {frame}".encode())
    with (root / lls48.MASTER_RELATIVE).open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=lls48.METADATA_COLUMNS + lls48.LABEL_COLUMNS
        )
        writer.writeheader()
        writer.writerows(rows)
    (root / "dict").mkdir()
    for kind in lls48.MAPPING_COUNTS:
        (root / f"dict/{kind}_class_mapping.txt").write_text(
            "synthetic map\n", encoding="utf-8"
        )
    return rows


def write_verification_manifests(root, rows):
    (root / "manifests").mkdir()
    (root / "splits").mkdir()
    files = [lls48.MASTER_RELATIVE] + [
        f"dict/{kind}_class_mapping.txt" for kind in lls48.MAPPING_COUNTS
    ]
    files += [f"data/{row['image_path']}" for row in rows]
    checksums = {
        "schema_version": 1,
        "dataset": "LLS48",
        "files": [lls48.record(root / path, path) for path in files],
    }
    (root / "manifests/checksums.json").write_bytes(lls48.json_bytes(checksums))
    (root / "splits/folds.json").write_bytes(
        lls48.json_bytes({"folds": {"3": ["VID01"]}})
    )
    frames = []
    for index, row in enumerate(rows):
        entry = {
            key: row[key] for key in ("video", "clip_name", "image_id", "image_path")
        }
        entry.update(row_index=index, frame_in_clip=index + 1, fold=3)
        frames.append(entry)
    (root / "manifests/frames.json").write_bytes(
        lls48.json_bytes(
            {"event_extraction_scope": "independent_five_frame_clip", "frames": frames}
        )
    )


class LLS48PackagingTests(unittest.TestCase):
    def test_unsafe_paths_rejected(self):
        for path in (
            "../x",
            "/root",
            "C:/x",
            "a\\b",
            "a/../b",
            "a//b",
            "a/./b",
            "a/x. ",
        ):
            with self.subTest(path=path), self.assertRaises(ValueError):
                lls48.safe_relative(path)

    def test_canonical_folds_match_existing_training_code(self):
        result = lls48.read_canonical_folds(ROOT / "legacy/spatial/lls48/preprocess.py")
        self.assertEqual(result, lls48.CANONICAL_FOLDS)

    def test_synthetic_annotation_retains_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = write_fixture(root)
            actual, clips, summary = lls48.inspect_annotations(
                root, require_canonical=False
            )
            self.assertEqual(actual, rows)
            self.assertEqual(summary["frames"], 5)
            self.assertEqual(len(clips), 1)

    def test_noncanonical_source_rejected_by_default(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_fixture(root)
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                lls48.inspect_annotations(root)

    def test_clip_cannot_cross_boundary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_fixture(root)
            path = root / lls48.MASTER_RELATIVE
            path.write_text(
                path.read_text().replace(
                    "01_001,VID01/01_001/frame_00005", "01_002,VID01/01_002/frame_00005"
                )
            )
            with self.assertRaisesRegex(ValueError, "exactly five"):
                lls48.inspect_annotations(root, require_canonical=False)

    def test_archive_determinism_and_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = b"immutable synthetic content"
            member = {
                "path": "data/test.bin",
                "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            first = lls48.write_archive(
                root / "one.zip", [member], payloads={member["path"]: data}
            )
            second = lls48.write_archive(
                root / "two.zip", [member], payloads={member["path"]: data}
            )
            self.assertEqual(first["sha256"], second["sha256"])
            with zipfile.ZipFile(root / "one.zip") as archive:
                self.assertIsNone(archive.testzip())
                self.assertEqual(archive.namelist(), ["data/test.bin"])
                self.assertEqual(archive.read("data/test.bin"), data)

    def test_archive_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "existing.zip"
            path.write_bytes(b"preserve")
            with self.assertRaises(FileExistsError):
                lls48.write_archive(path, [], payloads={})
            self.assertEqual(path.read_bytes(), b"preserve")

    def test_archive_source_checksum_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            member = {"path": "data/test.bin", "size_bytes": 1, "sha256": "0" * 64}
            with self.assertRaisesRegex(ValueError, "source checksum"):
                lls48.write_archive(
                    Path(temporary) / "bad.zip",
                    [member],
                    payloads={"data/test.bin": b"x"},
                )

    def test_archive_duplicates_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            member = {"path": "data/test.bin", "size_bytes": 1, "sha256": "0" * 64}
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                lls48.write_archive(
                    Path(temporary) / "bad.zip", [member, member], payloads={}
                )

    def test_shard_limit_is_bounded_and_retains_order(self):
        members = [{"path": f"data/{i}.bin", "size_bytes": 20000} for i in range(10)]
        groups = lls48.shard_members(members, maximum=110000)
        self.assertEqual([item for group in groups for item in group], members)
        self.assertEqual(len(groups), 5)

    def test_oversize_member_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exceeds"):
            lls48.shard_members(
                [{"path": "data/a", "size_bytes": 200000}], maximum=110000
            )

    def test_installed_fixture_verifies(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = write_fixture(root)
            write_verification_manifests(root, rows)
            self.assertEqual(
                lls48.verify_dataset(root, require_canonical=False)["status"],
                "VERIFIED",
            )

    def test_modified_image_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = write_fixture(root)
            write_verification_manifests(root, rows)
            (root / "data" / rows[0]["image_path"]).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                lls48.verify_dataset(root, require_canonical=False)

    def test_reordered_frame_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = write_fixture(root)
            write_verification_manifests(root, rows)
            path = root / "manifests/frames.json"
            manifest = json.loads(path.read_text())
            manifest["frames"].reverse()
            path.write_bytes(lls48.json_bytes(manifest))
            with self.assertRaisesRegex(ValueError, "order"):
                lls48.verify_dataset(root, require_canonical=False)

    def test_missing_clip_scope_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = write_fixture(root)
            write_verification_manifests(root, rows)
            path = root / "manifests/frames.json"
            manifest = json.loads(path.read_text())
            manifest["event_extraction_scope"] = "continuous_video"
            path.write_bytes(lls48.json_bytes(manifest))
            with self.assertRaisesRegex(ValueError, "Clip-bounded"):
                lls48.verify_dataset(root, require_canonical=False)

    def test_unselected_checksum_member_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = write_fixture(root)
            write_verification_manifests(root, rows)
            path = root / "manifests/checksums.json"
            manifest = json.loads(path.read_text())
            manifest["files"].append(
                {"path": "data/extra.png", "size_bytes": 0, "sha256": "0" * 64}
            )
            path.write_bytes(lls48.json_bytes(manifest))
            with self.assertRaisesRegex(ValueError, "exactly"):
                lls48.verify_dataset(root, require_canonical=False)


if __name__ == "__main__":
    unittest.main()
