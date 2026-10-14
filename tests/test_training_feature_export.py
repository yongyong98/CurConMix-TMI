"""Synthetic files/arrays and mocked extraction only; no model or image execution."""

import builtins
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from curconmix_release import training_feature_export as exporter
from curconmix_release.lls48 import CANONICAL_FOLDS, LABEL_COLUMNS, MASTER_RELATIVE
from curconmix_release.spatial_protocol import metadata, sidecar_path
from curconmix_repro.temporal.training_inputs import validate_training_features
from curconmix_repro.training_split import PROTOCOL


ROOT = Path(__file__).resolve().parents[1]


def config():
    return json.loads((ROOT / "configs/temporal/lls48_base224_plus.json").read_text())


@pytest.fixture
def synthetic_inputs(tmp_path, monkeypatch):
    """Canonical ID shape, but entirely fake images, labels, and checkpoint bytes."""
    source = tmp_path / "data-source"
    annotation = source / MASTER_RELATIVE
    annotation.parent.mkdir(parents=True)
    universe = [video for videos in CANONICAL_FOLDS.values() for video in videos]
    rows = [
        {
            "video": video,
            "clip_name": f"{video[-2:]}_001",
            "image_id": f"{video[-2:]}_1",
            "image_path": f"{video}/{video[-2:]}_001/frame_00001.png",
            **{name: "0" for name in LABEL_COLUMNS},
        }
        for video in universe
    ]
    with annotation.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        image = source / "data" / row["image_path"]
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"SYNTHETIC NOT AN IMAGE; MUST NEVER BE DECODED")
    digest = exporter.sha256_file(annotation)
    split = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "dataset": "lls48",
        "selection_metric": "AP_IVT",
        "annotation_sha256": digest,
        "folds": {},
    }
    for fold in range(5):
        outer = list(CANONICAL_FOLDS[str(fold)])
        remaining = [video for video in universe if video not in outer]
        split["folds"][str(fold)] = {
            "train": remaining[1:],
            "inner_validation": remaining[:1],
            "outer_test": outer,
        }
    split_path = tmp_path / "synthetic-split.json"
    split_path.write_text(json.dumps(split), encoding="utf-8")
    checkpoints = tmp_path / "synthetic-checkpoints"
    checkpoints.mkdir()
    for fold in range(5):
        path = checkpoints / exporter.DEFAULT_PATTERN.format(fold=fold)
        path.write_bytes(f"SYNTHETIC NOT A MODEL {fold}".encode())
        sidecar_path(path).write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "checkpoint_sha256": exporter.sha256_file(path),
                    "size_bytes": path.stat().st_size,
                    "meta": metadata(split, fold, "student", epoch=1, score=0.1),
                }
            ),
            encoding="utf-8",
        )
    summary = {
        "frames": len(rows),
        "procedures": len(universe),
        "annotation_sha256": digest,
        "fixture": "SYNTHETIC_ONLY",
    }
    monkeypatch.setattr(exporter, "inspect_annotations", lambda _: (rows, {}, summary))
    return SimpleNamespace(
        source=source,
        annotation=annotation,
        checkpoints=checkpoints,
        split_path=split_path,
        split=split,
        rows=rows,
        summary=summary,
        output=tmp_path / "new-features",
        arguments=(
            config(),
            split_path,
            source,
            checkpoints,
            tmp_path / "new-features",
        ),
    )


def forbid_ml_imports(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert name.split(".")[0] not in {
            "torch",
            "timm",
            "numpy",
            "cv2",
            "albumentations",
            "PIL",
        }, f"Unexpected model/image import: {name}"
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


def test_default_plan_is_model_free_and_write_free(tmp_path, monkeypatch):
    forbid_ml_imports(monkeypatch)
    plan = exporter.plan_export(
        config(),
        tmp_path / "absent-split.json",
        tmp_path / "absent-data",
        tmp_path / "absent-checkpoints",
        tmp_path / "new-output",
    )
    assert plan["mode"] == "plan_only"
    assert plan["model_execution"] is False
    assert plan["include_outer_test"] is False
    assert plan["export_partitions"] == ["train", "inner_validation"]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "changes",
    [
        {"dataset": "cholect45"},
        {"triplet_size": 193},
        {"num_classes": 264},
        {"dim": 1536},
        {
            "model_name": "swin_bas_266_LLS48_SwinB384_Curriculum_Feature_Mixup_e7_Student"
        },
    ],
)
def test_unsupported_data_or_architecture_is_not_silently_adapted(tmp_path, changes):
    cfg = {**config(), **changes}
    with pytest.raises(ValueError):
        exporter.plan_export(cfg, "split.json", "data", "weights", tmp_path / "out")


@pytest.mark.parametrize(
    "pattern",
    [
        "../fold{fold}.pth",
        "C:\\fold{fold}.pth",
        "fold.pth",
        "{other}{fold}.pth",
        "{fold}.pkl",
    ],
)
def test_checkpoint_pattern_cannot_escape_root(tmp_path, pattern):
    with pytest.raises(ValueError, match="pattern"):
        exporter.plan_export(
            config(),
            "split.json",
            "data",
            "weights",
            tmp_path / "out",
            checkpoint_pattern=pattern,
        )


def test_preflight_checks_files_without_loading_or_writing(
    synthetic_inputs, monkeypatch
):
    fixture = synthetic_inputs
    forbid_ml_imports(monkeypatch)
    loader = Mock(side_effect=AssertionError("Model loading must not run"))
    monkeypatch.setattr(exporter, "_safe_load", loader)
    report = exporter.preflight_export(*fixture.arguments)
    assert report["status"] == "PASS_MODEL_FREE_INPUT_LINEAGE"
    assert report["model_execution"] is False
    assert report["image_decoding"] is False
    assert report["new_files_written"] is False
    assert len(report["verified_checkpoints"]) == 5
    assert not fixture.output.exists()
    loader.assert_not_called()


@pytest.mark.parametrize(
    "change", ["missing", "fold", "selection", "architecture", "hash", "size"]
)
def test_bad_student_provenance_fails_before_any_model_load(
    synthetic_inputs, monkeypatch, change
):
    fixture = synthetic_inputs
    path = fixture.checkpoints / exporter.DEFAULT_PATTERN.format(fold=0)
    sidecar = sidecar_path(path)
    if change == "missing":
        sidecar.unlink()  # Synthetic test fixture only.
    else:
        record = json.loads(sidecar.read_text())
        if change == "fold":
            record["meta"]["fold"] = 1
        elif change == "selection":
            record["meta"]["selection_partition"] = "outer_test"
        elif change == "architecture":
            record["meta"]["input_resolution"] = 384
        elif change == "hash":
            record["checkpoint_sha256"] = "a" * 64
        elif change == "size":
            record["size_bytes"] += 1
        sidecar.write_text(json.dumps(record))
    forbid_ml_imports(monkeypatch)
    with pytest.raises(ValueError):
        exporter.preflight_export(*fixture.arguments)
    assert not fixture.output.exists()


def test_split_must_match_actual_annotation(synthetic_inputs):
    fixture = synthetic_inputs
    fixture.split["annotation_sha256"] = "a" * 64
    fixture.split_path.write_text(json.dumps(fixture.split))
    with pytest.raises(ValueError, match="annotation SHA"):
        exporter.preflight_export(*fixture.arguments)


def test_outer_test_cannot_overlap_fitting_membership(synthetic_inputs):
    fixture = synthetic_inputs
    partitions = fixture.split["folds"]["0"]
    partitions["train"].append(partitions["outer_test"][0])
    fixture.split_path.write_text(json.dumps(fixture.split))
    with pytest.raises(ValueError, match="Overlapping"):
        exporter.preflight_export(*fixture.arguments)


def test_existing_output_preserved(synthetic_inputs):
    fixture = synthetic_inputs
    fixture.output.mkdir()
    marker = fixture.output / "keep.txt"
    marker.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        exporter.preflight_export(*fixture.arguments)
    assert marker.read_bytes() == b"keep"


def test_duplicate_source_keys_rejected(synthetic_inputs):
    fixture = synthetic_inputs
    fixture.rows.append(dict(fixture.rows[0]))
    with pytest.raises(ValueError, match="Duplicate frame key"):
        exporter.preflight_export(*fixture.arguments)


@pytest.mark.parametrize("include_outer", [False, True])
def test_mocked_export_layout_and_original_keys(
    synthetic_inputs, monkeypatch, tmp_path, include_outer
):
    """No torch, no model loading, no forward, no image decoding; fixed fake arrays."""
    fixture = synthetic_inputs
    project = tmp_path / "synthetic-code"
    for relative in exporter.SOURCE_FILES:
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("SYNTHETIC SOURCE IDENTITY ONLY\n")
    monkeypatch.setattr(exporter, "verify_dataset", lambda _: fixture.summary)
    monkeypatch.setattr(
        exporter, "_runtime", lambda *_: (SimpleNamespace(), np, None, None, {})
    )
    loader = Mock(side_effect=lambda path, info, record, fold, project: Mock(fold=fold))
    monkeypatch.setattr(exporter, "_load_student", loader)
    extractor = Mock(
        side_effect=lambda model, rows, *_: np.full(
            (len(rows), 1024), model.fold, dtype=np.float32
        )
    )
    monkeypatch.setattr(exporter, "_extract_rows", extractor)
    result = exporter.execute_export(
        *fixture.arguments, project_root=project, include_outer_test=include_outer
    )
    assert result["training_executed"] is False
    assert result["numerical_paper_reproduction"] is False
    manifest_path = fixture.output / "training_feature_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert loader.call_count == 5
    assert extractor.call_count == (240 if include_outer else 192)
    for fold in range(5):
        entry = manifest["folds"][str(fold)]
        assert ("outer_test" in entry) is include_outer
        val_dir = (
            fixture.output
            / "features"
            / "lls48"
            / config()["model_name"]
            / str(fold)
            / "val"
        )
        assert val_dir.exists() is include_outer
        for role in ("train", "inner_validation"):
            assert set(entry[role]) == set(fixture.split["folds"][str(fold)][role])
            for video, item in entry[role].items():
                source_index = next(
                    index
                    for index, row in enumerate(fixture.rows)
                    if row["video"] == video
                )
                assert item["frame_keys"][0]["row_index"] == source_index
                assert (
                    item["frame_keys"][0]["image_path"]
                    == fixture.rows[source_index]["image_path"]
                )
                assert "/train/" in item["feature"]["path"]
    checked = validate_training_features(
        manifest_path,
        config(),
        fixture.output / "features",
        fixture.output / "labels",
        fixture.split,
    )
    assert checked["outer_test_arrays_opened"] is False
    assert checked["checked_fit_selection_pairs"] == 192


def test_restricted_payload_must_match_sidecar_before_model_construction(
    synthetic_inputs, monkeypatch
):
    fixture = synthetic_inputs
    path = fixture.checkpoints / exporter.DEFAULT_PATTERN.format(fold=0)
    info = json.loads(sidecar_path(path).read_text())
    wrong = {**info["meta"], "fold": 1}
    monkeypatch.setattr(exporter, "_safe_load", lambda _: {"model": {}, "meta": wrong})
    construct = Mock(side_effect=AssertionError("Must not construct any model"))
    monkeypatch.setattr(exporter, "_strict_model_load", construct)
    with pytest.raises(ValueError, match="Embedded checkpoint metadata"):
        exporter._load_student(path, info, fixture.split, 0, ROOT)
    construct.assert_not_called()


def test_cli_defaults_to_plan_without_requiring_data(tmp_path, capsys):
    assert (
        exporter.main(
            [
                "--config",
                str(ROOT / "configs/temporal/lls48_base224_plus.json"),
                "--training-split",
                str(tmp_path / "not-created.json"),
                "--dataset-root",
                str(tmp_path / "absent-data"),
                "--checkpoint-root",
                str(tmp_path / "absent-weights"),
                "--output",
                str(tmp_path / "new-output"),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["mode"] == "plan_only"
    assert list(tmp_path.iterdir()) == []
