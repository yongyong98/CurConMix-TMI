"""Synthetic metadata/NumPy checks; never deserialize or execute an ML model."""

import ast
import csv
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from test_training_split import synthetic_split
from curconmix_repro.training_split import (
    PROTOCOL,
    checkpoint_provenance,
    fold_partition,
    split_digest,
)
from curconmix_repro.temporal.training_inputs import (
    annotation_key,
    file_sha256,
    guard_training_loaders,
    label_columns,
    load_training_order,
    restore_training_prediction_keys,
    training_order,
    validate_encoder,
    validate_training_features,
)
from curconmix_repro.validation import expected_videos


class TrainingInputContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.features = self.root / "features"
        self.labels = self.root / "labels"
        self.annotation = self.root / "annotation.csv"
        self.config = {
            "dataset": "lls48",
            "model_name": "swin_bas_266_LLS48_SwinB_Curriculum_Feature_Mixup_e7_Student",
            "dim": 1024,
        }
        self.videos = sorted(set.union(*expected_videos("lls48")))
        columns = label_columns("lls48")
        self.rows = {}
        self.truth = {}
        index = 0
        with self.annotation.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=["video", "image_id", "image_path", "clip_name", *columns],
            )
            writer.writeheader()
            for vid_index, video in enumerate(self.videos):
                truth = np.zeros((5, 266), dtype=np.uint8)
                truth[:, vid_index % 195] = 1
                self.truth[video] = truth
                self.rows[video] = []
                for frame in range(5):
                    row = {
                        "video": video,
                        "image_id": f"{video}-{frame}",
                        "image_path": f"synthetic/{video}/{frame}.png",
                        "clip_name": "clip0",
                        **dict(zip(columns, map(str, truth[frame]))),
                    }
                    writer.writerow(row)
                    self.rows[video].append((index, row))
                    index += 1
        self.split = synthetic_split(annotation_sha256=file_sha256(self.annotation))
        self.manifest = {
            "schema_version": 2,
            "kind": "inner_validation_training_features",
            "dataset": "lls48",
            "model_name": self.config["model_name"],
            "annotation_sha256": self.split["annotation_sha256"],
            "annotation_path": str(self.annotation),
            "training_split_sha256": split_digest(self.split),
            "folds": {},
        }
        for video in self.videos:
            file = self.labels / "lls48/triplet_label" / f"{video}.npy"
            file.parent.mkdir(parents=True, exist_ok=True)
            np.save(file, self.truth[video], allow_pickle=False)
        for fold in range(5):
            checkpoint = self.root / f"encoder_{fold}.fake"
            checkpoint.write_bytes(f"SYNTHETIC NON-MODEL BYTES fold={fold}".encode())
            sidecar = self.root / f"encoder_{fold}.json"
            meta = checkpoint_provenance(self.split, fold, stage="student")
            meta.update(
                dataset="lls48",
                model_class="TripletModel",
                model_name="swin_base_patch4_window7_224",
                input_resolution=224,
                feature_dim=1024,
                num_classes=266,
                triplet_size=195,
            )
            sidecar.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "checkpoint_sha256": file_sha256(checkpoint),
                        "size_bytes": checkpoint.stat().st_size,
                        "meta": meta,
                    }
                ),
                encoding="utf-8",
            )
            entry = {
                "encoder": {
                    "checkpoint_path": str(checkpoint),
                    "sha256": file_sha256(checkpoint),
                    "size_bytes": checkpoint.stat().st_size,
                    "metadata_sidecar_path": str(sidecar),
                    "metadata_sidecar_sha256": file_sha256(sidecar),
                }
            }
            for role in ("train", "inner_validation"):
                entry[role] = {}
                for video in fold_partition(self.split, fold)[role]:
                    relative = (
                        f"lls48/{self.config['model_name']}/{fold}/train/{video}.npy"
                    )
                    file = self.features / relative
                    file.parent.mkdir(parents=True, exist_ok=True)
                    np.save(
                        file, np.zeros((5, 1024), dtype=np.float32), allow_pickle=False
                    )
                    label_relative = f"lls48/triplet_label/{video}.npy"
                    entry[role][video] = {
                        "feature": self.file_record(file, relative),
                        "label": self.file_record(
                            self.labels / label_relative, label_relative
                        ),
                        "frame_keys": [
                            annotation_key(i, row) for i, row in self.rows[video]
                        ],
                    }
            self.manifest["folds"][str(fold)] = entry
        self.path = self.root / "manifest.json"
        self.save_manifest()

    @staticmethod
    def file_record(file, relative):
        return {
            "path": relative,
            "sha256": file_sha256(file),
            "size_bytes": file.stat().st_size,
        }

    def save_manifest(self):
        self.path.write_text(json.dumps(self.manifest), encoding="utf-8")

    def validate(self, fold=0):
        return validate_training_features(
            self.path, self.config, self.features, self.labels, self.split, fold=fold
        )

    def first_item(self, role="train"):
        return next(iter(self.manifest["folds"]["0"][role].items()))

    def mutate_encoder_meta(self, callback):
        file = Path(self.manifest["folds"]["0"]["encoder"]["metadata_sidecar_path"])
        record = json.loads(file.read_text())
        callback(record)
        file.write_text(json.dumps(record), encoding="utf-8")
        self.manifest["folds"]["0"]["encoder"]["metadata_sidecar_sha256"] = file_sha256(
            file
        )
        self.save_manifest()

    def test_all_five_folds_validate_without_model_or_outer_test_arrays(self):
        result = self.validate(fold=None)
        self.assertEqual(result["checked_fit_selection_pairs"], 192)
        self.assertFalse(result["outer_test_arrays_opened"])
        self.assertFalse(result["numerical_paper_reproduction"])
        self.assertEqual(result["training_split_sha256"], split_digest(self.split))

    def test_selected_fold_never_opens_outer_test_arrays(self):
        loaded = []
        original_load = np.load

        def audited(file, *args, **kwargs):
            loaded.append(Path(file))
            return original_load(file, *args, **kwargs)

        with patch("numpy.load", side_effect=audited):
            self.validate()
        self.assertTrue(loaded)
        outer_names = {v + ".npy" for v in fold_partition(self.split, 0)["outer_test"]}
        self.assertFalse(outer_names & {p.name for p in loaded})
        self.assertTrue(all("val" not in p.parts for p in loaded))

    def test_manifest_identity_hash_and_membership_fail_closed(self):
        original = deepcopy(self.manifest)
        for field, value in (
            ("schema_version", 1),
            ("dataset", "cholect45"),
            ("annotation_sha256", "0" * 64),
            ("training_split_sha256", "0" * 64),
            ("model_name", "legacy"),
        ):
            self.manifest = deepcopy(original)
            self.manifest[field] = value
            self.save_manifest()
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.validate()
        self.manifest = original
        self.manifest["folds"]["0"]["train"].pop(
            next(iter(self.manifest["folds"]["0"]["train"]))
        )
        self.save_manifest()
        with self.assertRaises(ValueError):
            self.validate()

    def test_each_fold_record_is_required(self):
        del self.manifest["folds"]["4"]
        self.save_manifest()
        with self.assertRaises(ValueError):
            self.validate()

    def test_wrong_file_hash_size_or_path_is_rejected(self):
        _, item = self.first_item()
        original = deepcopy(item["feature"])
        for key, value in (
            ("sha256", "a" * 64),
            ("size_bytes", 1),
            ("path", "../escape.npy"),
        ):
            item["feature"] = deepcopy(original)
            item["feature"][key] = value
            self.save_manifest()
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate()

    def test_wrong_annotation_hash_is_rejected(self):
        with self.annotation.open("a", encoding="utf-8") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "annotation SHA"):
            self.validate()

    def test_duplicate_annotation_header_is_rejected_even_with_valid_hashes(self):
        with self.annotation.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.reader(stream))
        duplicate_index = rows[0].index("tri0")
        for row in rows:
            row.append(row[duplicate_index])
        with self.annotation.open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerows(rows)
        digest = file_sha256(self.annotation)
        self.split["annotation_sha256"] = digest
        self.manifest["annotation_sha256"] = digest
        self.manifest["training_split_sha256"] = split_digest(self.split)
        self.mutate_encoder_meta(
            lambda record: record["meta"].update(
                checkpoint_provenance(self.split, 0, stage="student")
            )
        )
        with self.assertRaisesRegex(ValueError, "[Dd]uplicate.*(header|column)"):
            self.validate()

    def test_frame_order_is_checked_against_annotation(self):
        _, item = self.first_item()
        item["frame_keys"].reverse()
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "row order"):
            self.validate()

    def test_label_values_checked_even_if_manifest_hash_is_updated(self):
        video, item = self.first_item()
        file = self.labels / item["label"]["path"]
        wrong = self.truth[video].copy()
        wrong[0, 194] = 1
        np.save(file, wrong, allow_pickle=False)
        item["label"] = self.file_record(file, item["label"]["path"])
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "labels differ"):
            self.validate()

    def test_array_shape_and_finiteness_are_checked(self):
        _, item = self.first_item()
        file = self.features / item["feature"]["path"]
        for array in (
            np.zeros((4, 1024), dtype=np.float32),
            np.zeros((5, 3), dtype=np.float32),
            np.full((5, 1024), np.nan, dtype=np.float32),
        ):
            np.save(file, array, allow_pickle=False)
            item["feature"] = self.file_record(file, item["feature"]["path"])
            self.save_manifest()
            with self.subTest(shape=array.shape), self.assertRaises(ValueError):
                self.validate()

    def test_legacy_encoder_or_outer_test_selected_encoder_rejected(self):
        self.mutate_encoder_meta(lambda record: record["meta"].pop("training_protocol"))
        with self.assertRaisesRegex(ValueError, "Encoder"):
            self.validate()

    def test_encoder_outer_test_selection_rejected(self):
        self.mutate_encoder_meta(
            lambda record: record["meta"].update(selection_partition="outer_test")
        )
        with self.assertRaisesRegex(ValueError, "Encoder"):
            self.validate()

    def test_encoder_wrong_selection_metric_rejected(self):
        self.mutate_encoder_meta(
            lambda record: record["meta"].update(selection_metric="accuracy")
        )
        with self.assertRaisesRegex(ValueError, "Encoder"):
            self.validate()

    def test_encoder_wrong_resolution_rejected_even_with_updated_sidecar_hash(self):
        self.mutate_encoder_meta(
            lambda record: record["meta"].update(input_resolution=384)
        )
        with self.assertRaisesRegex(ValueError, "architecture"):
            self.validate()

    def test_missing_encoder_architecture_rejected_on_actual_feature_path(self):
        self.mutate_encoder_meta(lambda record: record["meta"].pop("model_class"))
        with self.assertRaisesRegex(ValueError, "architecture"):
            self.validate()

    def test_b384_or_l224_config_cannot_relabel_new_b224_features(self):
        encoder = self.manifest["folds"]["0"]["encoder"]
        for filename in ("lls48_base384_plus.json", "lls48_large224_plus.json"):
            config = json.loads((ROOT / "configs/temporal" / filename).read_text())
            with (
                self.subTest(config=filename),
                self.assertRaisesRegex(ValueError, "B224"),
            ):
                validate_encoder(encoder, self.split, 0, config=config)

    def test_both_dataset_b224_architecture_contracts(self):
        encoder = self.manifest["folds"]["0"]["encoder"]
        sidecar = Path(encoder["metadata_sidecar_path"])
        for dataset, classes, triplets in (
            ("lls48", 266, 195),
            ("cholect45", 131, 100),
        ):
            split = synthetic_split(dataset)
            meta = checkpoint_provenance(split, 0, stage="student")
            meta.update(
                dataset=dataset,
                model_class="TripletModel",
                model_name="swin_base_patch4_window7_224",
                input_resolution=224,
                feature_dim=1024,
                num_classes=classes,
                triplet_size=triplets,
            )
            record = json.loads(sidecar.read_text())
            record["meta"] = meta
            sidecar.write_text(json.dumps(record))
            encoder["metadata_sidecar_sha256"] = file_sha256(sidecar)
            config = json.loads(
                (ROOT / f"configs/temporal/{dataset}_base224_plus.json").read_text()
            )
            with self.subTest(dataset=dataset):
                self.assertEqual(
                    validate_encoder(encoder, split, 0, config=config), meta
                )

    def test_encoder_checkpoint_hash_is_verified_without_deserialization(self):
        file = Path(self.manifest["folds"]["0"]["encoder"]["checkpoint_path"])
        file.write_bytes(b"CHANGED SYNTHETIC NON-MODEL BYTES")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.validate()

    def test_duplicate_manifest_json_keys_rejected(self):
        text = self.path.read_text()
        self.path.write_text(
            text.replace(
                '"schema_version": 2', '"schema_version": 1, "schema_version": 2', 1
            )
        )
        with self.assertRaisesRegex(ValueError, "[Dd]uplicate"):
            self.validate()

    def test_duplicate_encoder_json_keys_rejected(self):
        encoder = self.manifest["folds"]["0"]["encoder"]
        file = Path(encoder["metadata_sidecar_path"])
        text = file.read_text()
        file.write_text(
            text.replace(
                '"schema_version": 1', '"schema_version": 2, "schema_version": 1', 1
            )
        )
        with self.assertRaisesRegex(ValueError, "[Dd]uplicate"):
            validate_encoder(encoder, self.split, 0)

    def test_order_manifest_is_bound_and_duplicate_keys_rejected(self):
        order = training_order(self.config, self.split)
        path = self.root / "order.json"
        path.write_text(json.dumps(order), encoding="utf-8")
        self.assertEqual(load_training_order(path, self.config, self.split), order)
        order["folds"]["0"]["train"].reverse()
        path.write_text(json.dumps(order), encoding="utf-8")
        with self.assertRaises(ValueError):
            load_training_order(path, self.config, self.split)
        text = json.dumps(training_order(self.config, self.split))
        path.write_text(
            text.replace(
                '"schema_version": 2', '"schema_version": 1, "schema_version": 2', 1
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "[Dd]uplicate"):
            load_training_order(path, self.config, self.split)

    def loaders(self):
        partition = fold_partition(self.split, 0)
        return [
            SimpleNamespace(
                dataset=SimpleNamespace(
                    selection_role=role,
                    training_split_sha256=split_digest(self.split),
                    video_names=[v + ".npy" for v in partition[role]],
                )
            )
            for role in ("train", "inner_validation")
        ]

    def synthetic_prediction_frame(self):
        import pandas as pd

        records = []
        outer = {}
        columns = label_columns("lls48")
        for video in fold_partition(self.split, 0)["outer_test"]:
            relative = f"lls48/triplet_label/{video}.npy"
            outer[video] = {
                "label": self.file_record(self.labels / relative, relative),
                "frame_keys": [
                    annotation_key(index, row) for index, row in self.rows[video]
                ],
            }
            for position, labels in enumerate(self.truth[video]):
                records.append(
                    {
                        "video": video + ".npy",
                        "image_id": f"{video}.npy_{position + 1}",
                        "fold": 0,
                        **dict(zip(columns, labels)),
                        "0": 0.25,
                    }
                )
        self.manifest["folds"]["0"]["outer_test"] = outer
        return pd.DataFrame(records)

    def test_new_training_outer_keys_restore_after_full_gt_alignment(self):
        frame = self.synthetic_prediction_frame()
        before = frame.copy(deep=True)
        restored = restore_training_prediction_keys(
            frame, self.manifest, self.labels, fold=0
        )
        self.assertTrue(frame.equals(before))
        self.assertEqual(list(restored["0"]), [0.25] * len(restored))
        expected = [
            row
            for video in fold_partition(self.split, 0)["outer_test"]
            for row in self.manifest["folds"]["0"]["outer_test"][video]["frame_keys"]
        ]
        for column in ("video", "image_id", "image_path", "clip_name"):
            self.assertEqual(list(restored[column]), [row[column] for row in expected])
        self.assertEqual(
            list(restored["source_row_index"]), [row["row_index"] for row in expected]
        )

    def test_new_training_key_restore_rejects_auxiliary_gt_mismatch(self):
        frame = self.synthetic_prediction_frame()
        frame.loc[0, "inst0"] = 1
        with self.assertRaisesRegex(ValueError, "full ground-truth alignment"):
            restore_training_prediction_keys(frame, self.manifest, self.labels, fold=0)

    def test_new_training_key_restore_rejects_nonconsecutive_synthetic_ids(self):
        frame = self.synthetic_prediction_frame()
        frame.loc[0, "image_id"] = "arbitrary_frame_name"
        with self.assertRaisesRegex(ValueError, "not consecutive"):
            restore_training_prediction_keys(frame, self.manifest, self.labels, fold=0)

    def test_direct_loader_guard_rejects_outer_test_even_with_spoofed_role(self):
        train, validation = self.loaders()
        guard_training_loaders(self.split, 0, train, validation)
        validation.dataset.video_names = [
            v + ".npy" for v in fold_partition(self.split, 0)["outer_test"]
        ]
        with self.assertRaisesRegex(ValueError, "outer-test"):
            guard_training_loaders(self.split, 0, train, validation)

    def test_direct_loader_guard_rejects_missing_or_wrong_provenance(self):
        for field, value in (
            ("selection_role", "outer_test"),
            ("training_split_sha256", "0" * 64),
            ("video_names", []),
        ):
            train, validation = self.loaders()
            setattr(validation.dataset, field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                guard_training_loaders(self.split, 0, train, validation)


class TrainingEntryPointStructure(unittest.TestCase):
    """Inspect entry points without importing torch or running a trainer."""

    @staticmethod
    def function(relative, name):
        tree = ast.parse(
            (ROOT / "src/curconmix_repro" / relative).read_text(encoding="utf-8")
        )
        return next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == name
        )

    @staticmethod
    def calls(function, name):
        return [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == name
        ]

    def test_feature_and_split_validation_precedes_temporal_model_construction(self):
        function = self.function("temporal/all_main.py", "run_single_fold")
        models = self.calls(function, "MultiResTemporalModel") + self.calls(
            function, "MultiResTemporalModel2"
        )
        self.assertTrue(models)
        for guard in (
            "load_training_split",
            "load_training_order",
            "validate_training_features",
        ):
            checks = self.calls(function, guard)
            self.assertEqual(len(checks), 1)
            self.assertLess(checks[0].lineno, min(model.lineno for model in models))
        feature_guard = self.calls(function, "validate_training_features")[0]
        self.assertFalse(
            any(
                keyword.arg == "include_outer_test"
                for keyword in feature_guard.keywords
            )
        )

    def test_temporal_fit_and_selection_loaders_have_explicit_inner_split(self):
        function = self.function("temporal/all_main.py", "run_single_fold")
        datasets = self.calls(function, "VideoDataset")
        self.assertEqual(len(datasets), 2)
        self.assertEqual(
            [node.args[2].value for node in datasets], ["train", "inner_validation"]
        )
        for node in datasets:
            self.assertTrue(
                any(
                    keyword.arg == "training_split"
                    and isinstance(keyword.value, ast.Name)
                    and keyword.value.id == "split"
                    for keyword in node.keywords
                )
            )

    def test_direct_loader_guard_precedes_device_and_optimizer_actions(self):
        function = self.function("temporal/engine1.py", "base_train")
        checks = self.calls(function, "guard_training_loaders")
        self.assertEqual(len(checks), 1)
        protected = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("to", "AdamW", "step", "backward")
        ]
        self.assertTrue(protected)
        self.assertLess(checks[0].lineno, min(node.lineno for node in protected))
        validation = self.calls(function, "base_test")
        self.assertEqual(len(validation), 1)
        self.assertEqual(validation[0].args[2].id, "validation_loader")

    def test_new_feature_and_decoder_lineage_gate_is_bidirectional(self):
        function = self.function("cli.py", "infer")
        guards = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.BoolOp)
            and len(node.test.values) == 2
            and any(
                isinstance(part, ast.Name) and part.id == "new_feature_audit"
                for part in ast.walk(node.test)
            )
            and any(
                isinstance(part, ast.Constant) and part.value == "training_protocol"
                for part in ast.walk(node.test)
            )
        ]
        self.assertEqual(len(guards), 1)
        # Execute only the metadata-only refusal block, never the inference function.
        guard = compile(
            ast.Module(body=guards, type_ignores=[]), "<lineage-gate>", "exec"
        )
        split = synthetic_split()
        digest = "b" * 64
        valid_meta = checkpoint_provenance(split, 0, stage="mrtt")
        valid_meta["training_feature_manifest_sha256"] = digest

        def check(meta, audit):
            scope = {
                "meta": meta,
                "new_feature_audit": audit,
                "new_split": split,
                "split_digest": split_digest,
                "PROTOCOL": PROTOCOL,
            }
            exec(guard, scope)

        check(valid_meta, {"manifest_sha256": digest})
        check({}, None)  # Historical inference remains explicitly separate.
        for meta, audit in (
            ({}, {"manifest_sha256": digest}),
            (valid_meta, None),
            ({**valid_meta, "stage": "student"}, {"manifest_sha256": digest}),
            (
                {**valid_meta, "selection_partition": "outer_test"},
                {"manifest_sha256": digest},
            ),
            (
                {**valid_meta, "training_feature_manifest_sha256": "a" * 64},
                {"manifest_sha256": digest},
            ),
        ):
            with self.subTest(meta=meta, audit=audit), self.assertRaises(ValueError):
                check(meta, audit)


if __name__ == "__main__":
    unittest.main()
