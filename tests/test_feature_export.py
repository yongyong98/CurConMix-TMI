"""Feature-bridge tests use synthetic arrays and mocks; no model runs or images."""

import builtins
import copy
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
from types import FunctionType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from curconmix_release import feature_contract as contract
from curconmix_release import feature_export as exporter
from curconmix_release.lls48 import CANONICAL_FOLDS, LABEL_COLUMNS, MASTER_RELATIVE


ROOT = Path(__file__).resolve().parents[1]


def config_for(variant="224"):
    return json.loads(
        (ROOT / f"configs/temporal/lls48_base{variant}_plus.json").read_text()
    )


def synthetic_rows():
    return [
        {
            "video": video,
            "clip_name": f"{video[-2:]}_{clip:03d}",
            "image_id": f"{video[-2:]}_{(clip - 1) * 5 + frame}",
            "image_path": f"{video}/{video[-2:]}_{clip:03d}/frame_{frame:05d}.png",
            **{
                name: str(int(index == clip))
                for index, name in enumerate(LABEL_COLUMNS)
            },
        }
        for video in ["VID10", "VID04"]
        for clip in [1, 2]
        for frame in range(1, 6)
    ]


@pytest.fixture
def export_fixture(tmp_path, monkeypatch):
    """End-to-end exporter orchestration with every model/image action replaced."""
    rows = synthetic_rows()
    source = tmp_path / "source"
    annotation = source / MASTER_RELATIVE
    annotation.parent.mkdir(parents=True)
    annotation.write_text("synthetic annotation stand-in, not clinical data\n")
    digest = hashlib.sha256(annotation.read_bytes()).hexdigest()
    monkeypatch.setattr(contract, "MASTER_SHA256", digest)
    monkeypatch.setattr(exporter, "MASTER_SHA256", digest)
    monkeypatch.setattr(contract, "inspect_annotations", lambda _: (rows, {}, {}))
    summary = {"status": "VERIFIED_SYNTHETIC"}
    monkeypatch.setattr(exporter, "verify_dataset", lambda _: summary)
    index = [
        {
            **row,
            "row_index": index,
            "image_path": source / "data" / row["image_path"],
            "labels": [int(row[column]) for column in LABEL_COLUMNS],
        }
        for index, row in enumerate(rows)
    ]
    monkeypatch.setattr(exporter, "LLS48Index", lambda _: index)
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    records = []
    for fold in range(5):
        path = checkpoints / f"synthetic-{fold}.pth"
        path.write_bytes(f"not a model {fold}".encode())
        records.append(
            (
                path,
                {
                    "sha256": exporter.sha256_file(path),
                    "fold": fold,
                    "source_sha256": "1" * 64,
                    "canonical_prediction_sha256": "2" * 64,
                    "mapping_sha256": "3" * 64,
                },
            )
        )
    monkeypatch.setattr(exporter, "_validated_file", lambda a, r, f, m: records[f])
    manifest = tmp_path / "spatial.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "assets": [
                    {
                        "id": "lls48-spatial-b224",
                        "files": [record for _, record in records],
                    }
                ],
            }
        )
    )
    monkeypatch.setattr(contract, "DEFAULT_SPATIAL_MANIFEST", manifest)
    model_loader = Mock(return_value=(Mock(), {}))
    monkeypatch.setattr(exporter, "load_spatial_checkpoint", model_loader)
    monkeypatch.setattr(
        exporter,
        "_runtime",
        lambda *args: (
            SimpleNamespace(),
            np,
            None,
            None,
            contract.RUNTIME_VERSIONS.copy(),
        ),
    )
    extraction = Mock(
        side_effect=lambda model, group, *args: np.full(
            (len(group), 1024), 0.25, dtype=np.float32
        )
    )
    monkeypatch.setattr(exporter, "_extract_rows", extraction)
    output = tmp_path / "output"
    result = exporter.execute_export(
        config_for(),
        "lls48-spatial-b224",
        source,
        checkpoints,
        output,
        spatial_manifest=manifest,
        project_root=ROOT,
    )
    return output, rows, result, model_loader, extraction


def test_plan_does_not_import_image_or_model_packages_or_write(
    tmp_path, monkeypatch, capsys
):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert name.split(".")[0] not in {
            "torch",
            "timm",
            "cv2",
            "albumentations",
            "PIL",
        }
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    output = tmp_path / "uncreated"
    assert (
        exporter.main(
            [
                "--config",
                str(ROOT / "configs/temporal/lls48_base224_plus.json"),
                "--asset",
                "lls48-spatial-b224",
                "--dataset-root",
                str(tmp_path / "missing-data"),
                "--checkpoint-root",
                str(tmp_path / "missing-weights"),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    plan = json.loads(capsys.readouterr().out)
    assert plan["mode"] == "plan_only" and plan["model_execution"] is False
    assert not output.exists()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("variant", ["224", "384"])
def test_plan_matches_only_explicit_released_variant(variant, tmp_path):
    plan = exporter.plan_export(
        config_for(variant),
        f"lls48-spatial-b{variant}",
        "data",
        "weights",
        tmp_path / "out",
        2,
        "cpu",
    )
    assert plan["preprocessing"]["resize"] == [int(variant), int(variant)]
    assert plan["preprocessing"]["random_augmentation"] is False


@pytest.mark.parametrize("asset", ["lls48-spatial-b384", "lls48-spatial-l224"])
def test_wrong_variant_is_not_silently_substituted(asset, tmp_path):
    with pytest.raises(ValueError):
        exporter.plan_export(
            config_for(), asset, "data", "weights", tmp_path / "out", 2, "cpu"
        )


def test_unpublished_weights_fail_before_runtime_or_output(tmp_path, monkeypatch):
    runtime = Mock(side_effect=AssertionError("Runtime must not be loaded"))
    monkeypatch.setattr(exporter, "_runtime", runtime)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "assets": [
                    {
                        "id": "lls48-spatial-b224",
                        "status": "prepared",
                        "available": False,
                        "license": "CC-BY-NC-4.0",
                        "license_status": "approved",
                        "files": [],
                    }
                ],
            }
        )
    )
    output = tmp_path / "new-output"
    with pytest.raises((ValueError, RuntimeError)):
        exporter.execute_export(
            config_for(),
            "lls48-spatial-b224",
            tmp_path / "data",
            tmp_path / "weights",
            output,
            spatial_manifest=manifest,
        )
    assert not runtime.called and not output.exists()
    assert not list(tmp_path.glob(".*.staging-*"))


def test_mocked_export_preserves_order_and_five_frame_boundaries(export_fixture):
    output, rows, result, models, extraction = export_fixture
    manifest = contract.validate_feature_manifest(
        output / "feature_manifest.json",
        config_for(),
        output / "features",
        output / "labels",
    )
    assert models.call_count == 5 and extraction.call_count == 10
    assert result["reproduction_status"] == "NOT_NUMERICALLY_QUALIFIED"
    for video, procedure in manifest["procedures"].items():
        keys = procedure["frames"]
        assert [key["frame_in_clip"] for key in keys] == [1, 2, 3, 4, 5] * 2
        assert len({key["clip_name"] for key in keys}) == 2
        assert keys == [
            contract.frame_key(i, row)
            for i, row in enumerate(rows)
            if row["video"] == video
        ]
    assert set(manifest["folds"]) == set(map(str, range(5)))


def test_copied_annotation_not_modified(export_fixture):
    output, _, _, _, _ = export_fixture
    assert (output / MASTER_RELATIVE).read_bytes() == (
        output.parent / "source" / MASTER_RELATIVE
    ).read_bytes()


def test_existing_output_is_never_overwritten(tmp_path):
    output = tmp_path / "keep"
    output.mkdir()
    marker = output / "marker"
    marker.write_text("preserve")
    with pytest.raises(FileExistsError):
        exporter._new_destination(output, [tmp_path / "source"])
    assert marker.read_text() == "preserve"


def test_atomic_publication_refuses_even_empty_existing_directory(tmp_path):
    source, destination = tmp_path / "stage", tmp_path / "existing"
    source.mkdir()
    destination.mkdir()
    (source / "marker").write_text("keep")
    with pytest.raises(OSError):
        exporter._publish_without_replace(source, destination)
    assert (source / "marker").read_text() == "keep"
    assert destination.is_dir() and not list(destination.iterdir())


@pytest.mark.parametrize("relative", ["source/nested", "source", "."])
def test_output_must_not_overlap_sources(tmp_path, relative):
    with pytest.raises((ValueError, FileExistsError)):
        exporter._new_destination(tmp_path / relative, [tmp_path / "source"])


def test_output_parent_alias_cannot_bypass_source_overlap(tmp_path):
    (tmp_path / "source").mkdir()
    (tmp_path / "sibling").mkdir()
    alias = tmp_path / "sibling" / ".." / "source" / "new-output"
    with pytest.raises(ValueError, match="overlap"):
        exporter._new_destination(alias, [tmp_path / "source"])
    assert not (tmp_path / "source/new-output").exists()


def test_source_code_hash_ignores_only_newline_convention(tmp_path):
    first, second = tmp_path / "lf.py", tmp_path / "crlf.py"
    first.write_bytes(b"# synthetic source\nx = 3\n")
    second.write_bytes(b"# synthetic source\r\nx = 3\r\n")
    assert contract.source_code_hash(first) == contract.source_code_hash(second)
    assert exporter.sha256_file(first) != exporter.sha256_file(second)
    second.write_bytes(b"# synthetic source\r\nx = 4\r\n")
    assert contract.source_code_hash(first) != contract.source_code_hash(second)


@pytest.mark.parametrize(
    "change",
    [
        "keys",
        "split",
        "checkpoint_fold",
        "checkpoint_hash",
        "runtime",
        "source_code",
        "preprocessing",
        "config",
    ],
)
def test_contract_rejects_manifest_drift(export_fixture, change):
    output, _, _, _, _ = export_fixture
    path = output / "feature_manifest.json"
    manifest = json.loads(path.read_text())
    if change == "keys":
        manifest["procedures"]["VID10"]["frames"].reverse()
    elif change == "split":
        manifest["folds"]["0"]["files"][0]["split"] = "train"
    elif change == "checkpoint_fold":
        manifest["folds"]["0"]["checkpoint"]["fold"] = 1
    elif change == "checkpoint_hash":
        manifest["folds"]["0"]["checkpoint"]["sha256"] = manifest["folds"]["1"][
            "checkpoint"
        ]["sha256"]
    elif change == "runtime":
        manifest["runtime"]["timm"] = "1.0.20"
    elif change == "source_code":
        manifest["source_code_sha256"]["legacy/spatial/lls48/augmentation.py"] = (
            "f" * 64
        )
    elif change == "preprocessing":
        manifest["preprocessing"]["random_augmentation"] = True
    else:
        manifest["config_identity_sha256"] = "f" * 64
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        contract.validate_feature_manifest(
            path, config_for(), output / "features", output / "labels"
        )


def test_contract_rejects_array_tampering(export_fixture):
    output, _, _, _, _ = export_fixture
    path = next((output / "features").rglob("*.npy"))
    array = np.load(path, allow_pickle=False)
    array[0, 0] = 9
    np.save(path, array, allow_pickle=False)
    with pytest.raises(ValueError, match="SHA-256"):
        contract.validate_feature_manifest(
            output / "feature_manifest.json",
            config_for(),
            output / "features",
            output / "labels",
        )


def synthetic_predictions(output, manifest):
    rows = []
    for video, entry in manifest["procedures"].items():
        labels = np.load(output / entry["labels"]["path"], allow_pickle=False)
        fold = next(int(f) for f, videos in CANONICAL_FOLDS.items() if video in videos)
        for index, label in enumerate(labels):
            rows.append(
                {
                    "video": video + ".npy",
                    "image_id": f"{video}.npy_{index + 1}",
                    "clip_name": "",
                    "image_path": "",
                    "fold": fold,
                    **dict(zip(LABEL_COLUMNS, label)),
                    "0": 0.42,
                }
            )
    return pd.DataFrame(rows)


def test_restores_original_clip_keys_only_after_full_gt_match(export_fixture):
    output, _, _, _, _ = export_fixture
    manifest = json.loads((output / "feature_manifest.json").read_text())
    frame = synthetic_predictions(output, manifest)
    original = frame.copy(deep=True)
    restored = contract.restore_prediction_keys(frame, manifest, output / "labels")
    pd.testing.assert_frame_equal(frame, original)
    assert restored["image_id"].tolist() == [
        key["image_id"]
        for entry in manifest["procedures"].values()
        for key in entry["frames"]
    ]
    assert restored["source_row_index"].tolist() == list(range(20))
    assert restored["0"].equals(original["0"])


@pytest.mark.parametrize("change", ["aux_gt", "fold", "length", "id_order"])
def test_restore_rejects_gt_fold_or_order_mismatch(export_fixture, change):
    output, _, _, _, _ = export_fixture
    manifest = json.loads((output / "feature_manifest.json").read_text())
    frame = synthetic_predictions(output, manifest)
    if change == "aux_gt":
        frame.loc[0, "t37"] = 1
    elif change == "fold":
        frame.loc[0, "fold"] = 4
    elif change == "length":
        frame = frame.iloc[:-1]
    else:
        frame.loc[0, "image_id"] = "VID10.npy_2"
    with pytest.raises(ValueError):
        contract.restore_prediction_keys(frame, manifest, output / "labels")


def test_semantic_config_digest_preserves_precision():
    config = config_for()
    changed = copy.deepcopy(config)
    changed["learning_rate"] += 1e-12
    assert contract.config_identity(config) != contract.config_identity(changed)
    assert contract.config_identity(config) == contract.config_identity(
        dict(reversed(list(config.items())))
    )


def test_feature_extraction_uses_pooled_pre_head_output_only():
    """Exercise orchestration with plain fake tensors and mocked image reads."""

    class ArrayStandIn:
        def __init__(self, values):
            self.values = values

        def to(self, **kwargs):
            return self

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.values

    pool = Mock(side_effect=lambda x: x)
    fake_forward = FunctionType(
        (lambda self, x: None).__code__, {"pool_backbone_feature": pool}
    )
    FakeModel = type("FakeModel", (), {"forward": fake_forward})
    model = FakeModel()
    model.model = Mock(
        side_effect=lambda batch: ArrayStandIn(
            np.ones((len(batch.values), 1024), dtype=np.float32)
        )
    )
    model.head = Mock(side_effect=AssertionError("Classification head must not run"))
    torch = SimpleNamespace(
        inference_mode=nullcontext,
        float32="float32",
        stack=lambda values: ArrayStandIn(np.stack(values)),
    )
    cv2 = SimpleNamespace(
        IMREAD_COLOR=1,
        COLOR_BGR2RGB=4,
        imread=Mock(return_value=np.zeros((3, 4, 3), dtype=np.uint8)),
        cvtColor=Mock(side_effect=lambda value, code: value[..., ::-1]),
    )
    transform = Mock(side_effect=lambda image: {"image": image.astype(np.float32)})
    values = exporter._extract_rows(
        model,
        [{"image_path": Path(f"synthetic-{i}.png")} for i in range(3)],
        transform,
        2,
        "cpu",
        (torch, np, cv2),
    )
    assert values.shape == (3, 1024) and values.dtype == np.float32
    assert model.model.call_count == pool.call_count == 2
    assert not model.head.called
    assert cv2.imread.call_count == transform.call_count == 3
