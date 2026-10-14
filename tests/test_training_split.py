"""New-run split contracts only; no dataset, model, or ML training execution."""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from curconmix_repro.training_split import (
    PROTOCOL,
    checkpoint_provenance,
    fold_partition,
    load_training_split,
    split_digest,
    validate_training_split,
)
from curconmix_repro.validation import expected_videos


def synthetic_split(dataset="lls48", annotation_sha256=None):
    """Declared synthetic split on canonical IDs, not a clinical split proposal."""
    outer = expected_videos(dataset)
    universe = set.union(*outer)
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "dataset": dataset,
        "selection_metric": "AP_IVT",
        "annotation_sha256": annotation_sha256
        or hashlib.sha256(b"SYNTHETIC ANNOTATION").hexdigest(),
        "folds": {
            str(k): {
                "train": sorted(universe - outer[k] - outer[(k + 1) % 5]),
                "inner_validation": sorted(outer[(k + 1) % 5]),
                "outer_test": sorted(outer[k]),
            }
            for k in range(5)
        },
    }


class TrainingSplitContracts(unittest.TestCase):
    def setUp(self):
        self.record = synthetic_split()

    def validate(self):
        return validate_training_split(self.record, "lls48")

    def test_both_datasets_all_folds_are_disjoint_and_complete(self):
        for dataset in ("lls48", "cholect45"):
            record = synthetic_split(dataset)
            before = deepcopy(record)
            self.assertEqual(validate_training_split(record, dataset), record)
            universe = set.union(*expected_videos(dataset))
            for k in range(5):
                partition = fold_partition(record, k)
                sets = [set(partition[name]) for name in partition]
                self.assertEqual(set.union(*sets), universe)
                self.assertEqual(sum(map(len, sets)), len(universe))
                self.assertEqual(
                    set(partition["outer_test"]), expected_videos(dataset)[k]
                )
                partition["train"].clear()
            self.assertEqual(record, before)

    def test_no_missing_or_automatic_fallback(self):
        for missing in (None, "", "None", "   "):
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                load_training_split(missing, "lls48")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                load_training_split(Path(directory) / "missing.json", "lls48")

    def test_missing_and_extra_folds(self):
        original = deepcopy(self.record)
        for key in ("missing", "extra"):
            self.record = deepcopy(original)
            if key == "missing":
                del self.record["folds"]["4"]
            else:
                self.record["folds"]["5"] = deepcopy(self.record["folds"]["0"])
            with self.subTest(case=key), self.assertRaises(ValueError):
                self.validate()

    def test_each_partition_must_be_nonempty_without_duplicate_ids(self):
        original = deepcopy(self.record)
        for role in ("train", "inner_validation", "outer_test"):
            for invalid in ([], [original["folds"]["0"][role][0]] * 2):
                self.record = deepcopy(original)
                self.record["folds"]["0"][role] = invalid
                with self.subTest(role=role, invalid=invalid):
                    with self.assertRaises(ValueError):
                        self.validate()

    def test_unknown_or_missing_procedure_is_rejected(self):
        original = deepcopy(self.record)
        for bad in ("VID999", None):
            self.record = deepcopy(original)
            if bad is None:
                self.record["folds"]["0"]["train"].pop()
            else:
                self.record["folds"]["0"]["train"][0] = bad
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.validate()

    def test_overlap_and_noncanonical_outer_assignment(self):
        original = deepcopy(self.record)
        self.record["folds"]["0"]["train"].append(
            self.record["folds"]["0"]["outer_test"][0]
        )
        with self.assertRaises(ValueError):
            self.validate()
        self.record = original
        fold = self.record["folds"]["0"]
        fold["train"][0], fold["outer_test"][0] = (
            fold["outer_test"][0],
            fold["train"][0],
        )
        with self.assertRaises(ValueError):
            self.validate()

    def test_annotation_universe_must_equal_published_ids(self):
        videos = set.union(*expected_videos("lls48"))
        validate_training_split(self.record, "lls48", procedure_ids=videos)
        videos.pop()
        with self.assertRaises(ValueError):
            validate_training_split(self.record, "lls48", procedure_ids=videos)

    def test_split_identity_and_annotation_hash_are_required(self):
        original = deepcopy(self.record)
        for key, bad in (
            ("annotation_sha256", ""),
            ("annotation_sha256", "a" * 63),
            ("annotation_sha256", "A" * 64),
            ("dataset", "cholect45"),
            ("protocol", "legacy_outer_validation"),
            ("selection_metric", "outer_test_AP"),
            ("schema_version", 2),
            ("schema_version", True),
        ):
            self.record = deepcopy(original)
            self.record[key] = bad
            with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                self.validate()

    def test_invalid_fold_types_are_rejected(self):
        for bad in (-1, 5, "0", True, 0.0):
            with self.subTest(fold=bad), self.assertRaises(ValueError):
                fold_partition(self.record, bad)

    def test_duplicate_json_key_rejected_at_any_depth(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            content = json.dumps(self.record)
            for duplicate in (
                content.replace(
                    '"schema_version": 1', '"schema_version": 1, "schema_version": 1'
                ),
                content.replace('"folds": {', '"folds": {"0": {}, ', 1),
            ):
                path.write_text(duplicate, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "[Dd]uplicate"):
                    load_training_split(path, "lls48")

    def test_split_digest_tracks_order_and_not_dict_key_order(self):
        reordered = dict(reversed(list(self.record.items())))
        self.assertEqual(split_digest(self.record), split_digest(reordered))
        reordered = deepcopy(self.record)
        reordered["folds"]["0"]["train"].reverse()
        self.assertNotEqual(split_digest(self.record), split_digest(reordered))

    def test_patient_mapping_optional_but_cannot_cross_partitions(self):
        universe = set.union(*expected_videos("lls48"))
        self.record["patient_groups"] = {v: "P-" + v for v in universe}
        self.validate()
        fold = self.record["folds"]["0"]
        self.record["patient_groups"][fold["train"][0]] = self.record["patient_groups"][
            fold["outer_test"][0]
        ]
        with self.assertRaisesRegex(ValueError, "Patient overlap"):
            self.validate()

    def test_patient_mapping_cannot_omit_or_disguise_identity(self):
        universe = set.union(*expected_videos("lls48"))
        for bad in ("", " ", " PATIENT", "PATIENT ", None):
            self.record["patient_groups"] = {v: "P-" + v for v in universe}
            self.record["patient_groups"][sorted(universe)[0]] = bad
            with self.subTest(value=bad), self.assertRaises(ValueError):
                self.validate()
        self.record["patient_groups"] = {v: "P-" + v for v in universe}
        self.record["patient_groups"].pop(sorted(universe)[0])
        with self.assertRaises(ValueError):
            self.validate()

    def test_provenance_binds_student_to_validation_only(self):
        meta = checkpoint_provenance(self.record, 0, stage="student")
        self.assertEqual(meta["training_split_sha256"], split_digest(self.record))
        self.assertEqual(meta["annotation_sha256"], self.record["annotation_sha256"])
        self.assertEqual(meta["selection_partition"], "inner_validation")
        self.assertEqual(meta["selection_metric"], "AP_IVT")
        self.assertEqual(
            meta["outer_test_procedures"], self.record["folds"]["0"]["outer_test"]
        )

    def test_checkpoint_provenance_cannot_select_on_training_or_test_scores(self):
        for bad in ("train", "outer_test", None):
            with self.subTest(partition=bad), self.assertRaises(ValueError):
                checkpoint_provenance(
                    self.record, 0, stage="student", selection_partition=bad
                )

    def test_split_import_does_not_load_machine_learning_libraries(self):
        environment = dict(
            os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1"
        )
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                "import sys; import curconmix_repro.training_split; assert not {'torch', 'numpy', 'pandas', 'timm'} & set(sys.modules)",
            ],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
