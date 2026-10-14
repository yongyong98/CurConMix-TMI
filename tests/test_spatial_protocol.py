"""Synthetic nested spatial protocol tests; never construct or execute a model."""

import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from curconmix_release import spatial_protocol as protocol
from curconmix_release import spatial_training as spatial
from curconmix_release import cache_prepare as cache
from curconmix_repro.training_split import split_digest


@pytest.fixture(params=["cholect45", "lls48"])
def fixture(request, tmp_path):
    dataset = request.param
    annotation = tmp_path / "synthetic.csv"
    annotation.write_text("synthetic annotation identity only\n", encoding="utf-8")
    mapping = spatial.fold_map(dataset)
    record = {
        "schema_version": 1,
        "protocol": "nested_procedure_validation_v1",
        "dataset": dataset,
        "annotation_sha256": spatial.sha256_file(annotation),
        "selection_metric": "AP_IVT",
        "folds": {},
    }
    for fold in range(5):
        pool = [v for v, k in mapping.items() if k != fold]
        record["folds"][str(fold)] = {
            "train": pool[2:],
            "inner_validation": pool[:2],
            "outer_test": [v for v, k in mapping.items() if k == fold],
        }
    path = tmp_path / "split.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    cfg = SimpleNamespace(
        training_split=path, parent_path=tmp_path, path_csv=annotation
    )
    frame = pd.DataFrame(
        [
            {
                "video": v,
                "fold": k,
                "image_path": f"{v}/frame.png",
                "image_id": f"{v}_1",
                "label": 1,
            }
            for v, k in mapping.items()
        ]
    )
    return dataset, record, cfg, frame


def test_actual_helper_uses_train_and_inner_only(fixture):
    dataset, record, cfg, frame = fixture
    path = spatial.repository_root() / "legacy" / "spatial" / dataset / "helper.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "get_dataframes"
    )
    namespace = {}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    for fold in range(5):
        train, valid, temp = namespace["get_dataframes"](frame, fold, cfg)
        parts = record["folds"][str(fold)]
        assert set(train.video) == set(parts["train"])
        assert set(valid.video) == set(parts["inner_validation"])
        assert temp.equals(valid) and temp is not valid
        assert not set(train.video) & set(parts["outer_test"])
        assert not set(train.video) & set(valid.video)


def test_outer_labels_cannot_reach_selector(fixture):
    _, record, cfg, frame = fixture
    fold = 0
    valid = protocol.select_frame(frame, record, fold, "inner_validation")
    seen = []

    def metric(values, configuration):
        seen.append(set(values.video))
        return 0.0

    assert protocol.inner_validation_ap(valid, record, fold, metric, cfg) == 0.0
    assert protocol.is_better_inner_ap(0.0, float("-inf"))
    assert not protocol.is_better_inner_ap(0.0, 0.0)
    outer = protocol.select_frame(frame, record, fold, "outer_test")
    outer["label"] = 999999
    with pytest.raises(ValueError, match="non-inner"):
        protocol.inner_validation_ap(outer, record, fold, metric, cfg)
    assert seen == [set(record["folds"]["0"]["inner_validation"])]


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -float("inf")])
def test_no_nonfinite_selection_fallback(score):
    with pytest.raises(ValueError, match="non-finite"):
        protocol.is_better_inner_ap(score, float("-inf"))


def test_missing_split_training_fails_but_legacy_inference_allowed(fixture):
    dataset, _, cfg, _ = fixture
    cfg.training_split = None
    with pytest.raises(ValueError, match="required"):
        protocol.cfg_split(cfg, dataset)
    assert protocol.cfg_split(cfg, dataset, required=False) is None


def test_annotation_and_procedure_overlap_fail(fixture):
    dataset, record, cfg, _ = fixture
    corrupted = copy.deepcopy(record)
    corrupted["folds"]["0"]["train"].append(corrupted["folds"]["0"]["outer_test"][0])
    cfg.training_split.write_text(json.dumps(corrupted), encoding="utf-8")
    with pytest.raises(ValueError, match="Overlapping"):
        protocol.cfg_split(cfg, dataset)
    cfg.training_split.write_text(json.dumps(record), encoding="utf-8")
    cfg.path_csv.write_text("changed annotation", encoding="utf-8")
    with pytest.raises(ValueError, match="annotation_sha256"):
        protocol.cfg_split(cfg, dataset)


def test_new_sidecar_binds_exact_bytes_fold_stage_and_split(fixture, tmp_path):
    _, record, _, _ = fixture
    checkpoint = tmp_path / "synthetic-not-a-checkpoint.pth"
    checkpoint.write_bytes(b"synthetic, never tensor-deserialized")
    with pytest.raises(ValueError, match="inference-only"):
        protocol.verify_sidecar(checkpoint, record, 0, "student")
    meta = protocol.metadata(record, 0, "student", epoch=3, score=0.2)
    protocol.write_sidecar(checkpoint, meta)
    assert protocol.verify_sidecar(checkpoint, record, 0, "student")["meta"] == meta
    assert meta["training_split_sha256"] == split_digest(record)
    for fold, stage in [(1, "student"), (0, "teacher")]:
        with pytest.raises(ValueError, match="lineage"):
            protocol.verify_sidecar(checkpoint, record, fold, stage)
    changed = copy.deepcopy(record)
    changed["folds"]["0"]["train"][0], changed["folds"]["0"]["inner_validation"][0] = (
        changed["folds"]["0"]["inner_validation"][0],
        changed["folds"]["0"]["train"][0],
    )
    with pytest.raises(ValueError, match="lineage"):
        protocol.verify_sidecar(checkpoint, changed, 0, "student")
    checkpoint.write_bytes(b"changed bytes")
    with pytest.raises(ValueError, match="bytes differ"):
        protocol.verify_sidecar(checkpoint, record, 0, "student")


def test_softlabel_row_order_is_bound(fixture, tmp_path):
    _, record, _, frame = fixture
    rows = protocol.select_rows(frame.to_dict("records"), record, 0)
    artifact = tmp_path / "synthetic-softlabels.csv"
    artifact.write_bytes(b"synthetic")
    protocol.write_sidecar(
        artifact,
        protocol.metadata(record, 0, "softlabels"),
        row_order_sha256=spatial.row_order_sha256(rows),
    )
    protocol.verify_sidecar(artifact, record, 0, "softlabels", rows)
    with pytest.raises(ValueError, match="row order"):
        protocol.verify_sidecar(artifact, record, 0, "softlabels", rows[::-1])


def test_dense_cache_estimate_excludes_both_heldout_roles(fixture):
    _, record, _, frame = fixture
    resources = cache.estimate_resources(frame.to_dict("records"), record)
    for fold in range(5):
        n = len(record["folds"][str(fold)]["train"])
        assert resources["folds"][str(fold)]["training_frames"] == n
        assert resources["folds"][str(fold)]["similarity_bytes_float32"] == n * n * 4


def test_training_sources_gate_all_supported_paths_and_do_not_test_select(fixture):
    dataset, _, _, _ = fixture
    source = spatial.repository_root() / "legacy" / "spatial" / dataset / "train.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    for name in ("train_cross_val", "train_cross_val_SSL"):
        node = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name
        )
        calls = [n for n in ast.walk(node) if isinstance(n, ast.Call)]
        assert any(
            isinstance(n.func, ast.Name) and n.func.id == "training_gate" for n in calls
        )
        assert not any(
            isinstance(n.func, ast.Name) and n.func.id == "cholect45_ivtmetrics_mAP"
            for n in calls
        )
        if name == "train_cross_val":
            assert any(
                isinstance(n.func, ast.Name) and n.func.id == "inner_validation_ap"
                for n in calls
            )
            assert not any(
                isinstance(n.func, ast.Name) and n.func.id == "per_epoch_ivtmetrics"
                for n in calls
            )
        else:
            assert not any(
                isinstance(n.func, ast.Name) and n.func.id == "valid_fn" for n in calls
            )
    if dataset == "lls48":
        for name in ("linear_probe_lls48", "train_cross_challenge"):
            node = next(
                n
                for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == name
            )
            assert len(node.body) == 1 and isinstance(node.body[0], ast.Raise)


def test_pretrain_metadata_has_no_validation_selection(fixture):
    _, record, _, _ = fixture
    meta = protocol.metadata(record, 0, "pretrain", epoch=6)
    assert meta["selection_partition"] == "fixed_training_schedule"
    assert meta["selection_metric"] is None


def test_changed_split_cannot_publish_checkpoint(fixture):
    _, record, cfg, _ = fixture
    assert protocol.checkpoint_metadata(cfg, record, 0, "student")[
        "training_split_sha256"
    ] == split_digest(record)
    changed = copy.deepcopy(record)
    parts = changed["folds"]["0"]
    parts["train"][0], parts["inner_validation"][0] = (
        parts["inner_validation"][0],
        parts["train"][0],
    )
    cfg.training_split.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="changed during execution"):
        protocol.checkpoint_metadata(cfg, record, 0, "student")


def test_stage_command_forwards_explicit_split(fixture, tmp_path):
    _, _, cfg, _ = fixture
    args = SimpleNamespace(
        stage="teacher",
        data_root=tmp_path,
        output_dir=tmp_path / "run",
        annotation_csv=None,
        device="cpu",
        workers=0,
        training_split=cfg.training_split,
    )
    command = spatial.build_command(args)
    assert "training_split=" + json.dumps(str(cfg.training_split.resolve())) in command
