"""Synthetic file-only preflight tests. No models, GPUs, or pickle loads."""

import csv
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from curconmix_release import spatial_training as spatial
from curconmix_release import cache_prepare as cache
from curconmix_release import spatial_protocol as protocol
from curconmix_repro.training_split import split_digest


@pytest.fixture
def lls_inputs(tmp_path):
    root = tmp_path / "synthetic-dataset"
    annotation = root / "dataframe" / "LLS48_TRI195.csv"
    annotation.parent.mkdir(parents=True)
    columns = [
        f"{prefix}{i}"
        for prefix, count in zip(("tri", "inst", "v", "t"), (195, 17, 16, 38))
        for i in range(count)
    ]
    rows = []
    for video in spatial.fold_map("lls48"):
        clip = video[3:] + "_001"
        for i in range(1, 6):
            relative = f"{video}/{clip}/frame_{i:05d}.png"
            path = root / "data" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"synthetic bytes, not a real image")
            rows.append(
                {
                    "video": video,
                    "clip_name": clip,
                    "image_path": relative,
                    "image_id": f"{video}_{i}",
                    **{c: "0" for c in columns},
                }
            )
    with annotation.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return root, annotation, rows


def args_for(tmp_path, root, stage="pretrain"):
    annotation = root / "dataframe" / "LLS48_TRI195.csv"
    split_path = None
    if annotation.is_file() and stage != "predict":
        split_path = tmp_path / "synthetic-training-split.json"
        mapping = spatial.fold_map("lls48")
        record = {
            "schema_version": 1,
            "protocol": "nested_procedure_validation_v1",
            "dataset": "lls48",
            "annotation_sha256": spatial.sha256_file(annotation),
            "selection_metric": "AP_IVT",
            "folds": {},
        }
        for fold in range(5):
            pool = [v for v, k in mapping.items() if k != fold]
            record["folds"][str(fold)] = {
                "train": pool[1:],
                "inner_validation": pool[:1],
                "outer_test": [v for v, k in mapping.items() if k == fold],
            }
        split_path.write_text(json.dumps(record), encoding="utf-8")
    return SimpleNamespace(
        stage=stage,
        data_root=root,
        output_dir=tmp_path / "new-run",
        annotation_csv=None,
        feature_dir=tmp_path / "caches",
        model_tag="SwinB224",
        device="cuda:0",
        workers=4,
        training_split=split_path,
    )


def make_stage_files(args, dataset="lls48"):
    for path in spatial.required_artifacts(dataset, args):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic-not-a-checkpoint-or-pickle")


def test_literal_folds_are_canonical():
    assert len(spatial.fold_map("lls48")) == 48
    assert len(spatial.fold_map("cholect45")) == 45
    assert set(spatial.fold_map("lls48").values()) == set(range(5))
    assert spatial.fold_map("lls48")["VID10"] == 0
    assert spatial.fold_map("cholect45")["VID79"] == 0


def test_synthetic_annotation_matches_layout_only(lls_inputs):
    root, _, _ = lls_inputs
    rows, report = spatial.inspect_inputs("lls48", root)
    assert len(rows) == 240
    assert report["selected_procedures"] == 48
    assert not report["errors"]
    assert not report["missing_files"]


@pytest.mark.parametrize(
    "stage,count",
    [
        ("pretrain", 10),
        ("teacher", 5),
        ("softlabels", 5),
        ("student", 5),
        ("predict", 5),
    ],
)
def test_all_stages_report_exact_missing_artifacts(tmp_path, lls_inputs, stage, count):
    args = args_for(tmp_path, lls_inputs[0], stage)
    report = spatial.preflight("lls48", args)
    assert report["status"] == "BLOCKED"
    assert len(report["missing_files"]) == count
    assert report["model_executed"] is False


def test_teacher_checkpoint_names_match_historical_stage(tmp_path):
    args = args_for(tmp_path, tmp_path, "teacher")
    assert (
        spatial.required_artifacts("lls48", args)[0].name
        == "fold0_swin_bas_CurConMix_B224_Pretrain_6.pth"
    )
    assert (
        spatial.required_artifacts("cholect45", args)[0].name
        == "fold0_swin_bas_CurConMix_B224_Pretrain_2.pth"
    )


def test_teacher_rejects_legacy_and_accepts_split_bound_warmstarts(
    tmp_path, lls_inputs
):
    args = args_for(tmp_path, lls_inputs[0], "teacher")
    make_stage_files(args)
    assert spatial.preflight("lls48", args)["status"] == "BLOCKED"
    split = protocol.load_split(args.training_split, "lls48", lls_inputs[1])
    for fold, path in enumerate(spatial.required_artifacts("lls48", args)):
        protocol.write_sidecar(
            path, protocol.metadata(split, fold, "pretrain", epoch=6)
        )
    assert spatial.preflight("lls48", args)["status"] == "REQUIRED_FILES_PRESENT"
    args.training_split = None
    assert spatial.preflight("lls48", args)["status"] == "BLOCKED"


def test_cache_preflight_never_unpickles(lls_inputs, tmp_path, monkeypatch):
    import pickle

    monkeypatch.setattr(
        pickle, "load", lambda *a, **k: pytest.fail("must not load pickle")
    )
    args = args_for(tmp_path, lls_inputs[0])
    make_stage_files(args)
    report = spatial.preflight("lls48", args)
    assert report["status"] == "BLOCKED"
    assert any("No cache provenance" in error for error in report["errors"])
    (args.feature_dir / "INCOMPLETE.json").write_text("{}")
    assert spatial.preflight("lls48", args)["status"] == "BLOCKED"


def test_missing_image_is_reported(lls_inputs, tmp_path):
    root, _, rows = lls_inputs
    path = root / "data" / rows[0]["image_path"]
    path.unlink()
    args = args_for(tmp_path, root, "predict")
    make_stage_files(args)
    assert str(path) in spatial.preflight("lls48", args)["missing_files"]


@pytest.mark.parametrize(
    "relative", ["../raw.png", "C:/raw.png", "/raw.png", "a//raw.png", "a/./raw.png"]
)
def test_image_paths_cannot_escape(tmp_path, relative):
    with pytest.raises(ValueError):
        spatial.image_path(tmp_path, relative)


def test_softlabels_require_exact_training_keys_and_probabilities(tmp_path, lls_inputs):
    root, _, _ = lls_inputs
    rows, _ = spatial.inspect_inputs("lls48", root)
    args = args_for(tmp_path, root, "student")
    split = protocol.load_split(args.training_split, "lls48", lls_inputs[1])
    headers = ["image_path"] + [str(i) for i in range(266)]
    for fold, path in enumerate(spatial.required_artifacts("lls48", args)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(headers)
            writer.writerows(
                [
                    [row["image_path"]] + [0.5] * 266
                    for row in protocol.select_rows(rows, split, fold)
                ]
            )
        protocol.write_sidecar(
            path,
            protocol.metadata(split, fold, "softlabels"),
            row_order_sha256=spatial.row_order_sha256(
                protocol.select_rows(rows, split, fold)
            ),
            teacher_checkpoint_sha256="a" * 64,
        )
    assert spatial.preflight("lls48", args)["status"] == "REQUIRED_FILES_PRESENT"
    with spatial.required_artifacts("lls48", args)[0].open("a", newline="") as handle:
        csv.writer(handle).writerow(["not-a-training-frame"] + [float("nan")] * 266)
    assert spatial.preflight("lls48", args)["status"] == "BLOCKED"


def test_invalid_cache_sidecar_blocks(tmp_path, lls_inputs):
    args = args_for(tmp_path, lls_inputs[0])
    make_stage_files(args)
    (args.feature_dir / "cache_provenance.json").write_text(
        json.dumps({"complete": False})
    )
    report = spatial.preflight("lls48", args)
    assert report["status"] == "BLOCKED"
    assert any("provenance" in e for e in report["errors"])


@pytest.mark.parametrize("dataset", ["cholect45", "lls48"])
def test_default_preview_is_model_free_without_installed_dependencies(dataset):
    script = spatial.repository_root() / "legacy" / "spatial" / dataset / "run_stage.py"
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            str(script),
            "pretrain",
            "--data-root",
            "missing-dataset",
            "--output-dir",
            "missing-output",
            "--feature-dir",
            "missing-cache",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Preview only" in result.stdout


def test_failed_execute_preflight_never_starts_subprocess(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stage",
            "teacher",
            "--data-root",
            str(tmp_path / "missing"),
            "--output-dir",
            str(tmp_path / "run"),
            "--execute",
        ],
    )
    monkeypatch.setattr(
        spatial.subprocess, "run", lambda *a, **k: pytest.fail("must not start stage")
    )
    with pytest.raises(SystemExit) as error:
        spatial.main("lls48")
    assert error.value.code == 2


def test_cache_estimate_is_dense_and_fold_matched(lls_inputs):
    rows, _ = spatial.inspect_inputs("lls48", lls_inputs[0])
    estimate = cache.estimate_resources(rows)
    for fold in range(5):
        n = sum(r["fold"] != fold for r in rows)
        assert estimate["folds"][str(fold)]["similarity_bytes_float32"] == n * n * 4
        assert estimate["folds"][str(fold)]["feature_bytes_float32"] == n * 1024 * 4


def test_cache_plan_rejects_existing_directory_before_model(
    tmp_path, lls_inputs, monkeypatch
):
    args = SimpleNamespace(
        dataset="lls48",
        data_root=lls_inputs[0],
        annotation_csv=None,
        output_dir=tmp_path / "cache",
    )
    args.output_dir.mkdir()
    monkeypatch.setattr(cache, "available_memory", lambda: 10**15)
    _, report = cache.check_plan(args)
    assert report["status"] == "BLOCKED"
    assert any("must not exist" in e for e in report["errors"])


def test_cache_download_opt_in_is_required_without_importing_model():
    args = SimpleNamespace(allow_pretrained_download=False)
    with pytest.raises(ValueError, match="allow-pretrained-download"):
        cache.execute(args, [], {"status": "INPUT_FILES_PRESENT"})


def test_cache_publish_refuses_overwrite(tmp_path):
    staging, target = tmp_path / "staging", tmp_path / "target"
    staging.mkdir()
    target.mkdir()
    (staging / "fold0.pkl").write_bytes(b"new")
    (target / "fold0.pkl").write_bytes(b"preserved")
    with pytest.raises(FileExistsError):
        cache.publish_staged(staging, target, ["fold0.pkl"])
    assert (target / "fold0.pkl").read_bytes() == b"preserved"


def test_cache_cli_preview_imports_no_torch():
    source = str(spatial.repository_root() / "src")
    code = f"import sys; sys.path.insert(0, {source!r}); from curconmix_release.cache_prepare import main; sys.argv=['cache','--dataset','lls48','--data-root','absent','--output-dir','absent-new']; main(); assert 'torch' not in sys.modules"
    result = subprocess.run(
        [sys.executable, "-S", "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "Preview only" in result.stdout


def test_complete_cache_provenance_binds_rows_and_source_images(tmp_path, lls_inputs):
    args = args_for(tmp_path, lls_inputs[0])
    make_stage_files(args)
    rows, details = spatial.inspect_inputs("lls48", args.data_root)
    split = protocol.load_split(args.training_split, "lls48", lls_inputs[1])
    provenance = {
        "schema_version": 1,
        "complete": True,
        "dataset": "lls48",
        "annotation_sha256": details["annotation_sha256"],
        "training_split_sha256": split_digest(split),
        "model_name": spatial.MODEL_NAME,
        "model_tag": args.model_tag,
        "folds": {},
    }
    for fold in range(5):
        selected = protocol.select_rows(rows, split, fold)
        files = {}
        for name in spatial.cache_filenames(args.model_tag, fold):
            path = args.feature_dir / name
            files[name] = {
                "size_bytes": path.stat().st_size,
                "sha256": spatial.sha256_file(path),
            }
        provenance["folds"][str(fold)] = {
            "row_order_sha256": spatial.row_order_sha256(selected),
            "files": files,
            "rows": [
                {
                    "key": spatial.row_identity(r),
                    "image_sha256": spatial.sha256_file(
                        spatial.image_path(args.data_root, r["image_path"])
                    ),
                }
                for r in selected
            ],
        }
    (args.feature_dir / "cache_provenance.json").write_text(json.dumps(provenance))
    assert spatial.preflight("lls48", args)["status"] == "REQUIRED_FILES_PRESENT"
    used_row = protocol.select_rows(rows, split, 0)[0]
    spatial.image_path(args.data_root, used_row["image_path"]).write_bytes(
        b"changed synthetic input"
    )
    report = spatial.preflight("lls48", args)
    assert report["status"] == "BLOCKED"
    assert any("Source image differs" in e for e in report["errors"])


def test_check_only_works_without_model_or_dataframe_dependencies(tmp_path, lls_inputs):
    args = args_for(tmp_path, lls_inputs[0], "predict")
    make_stage_files(args)
    script = spatial.repository_root() / "legacy" / "spatial" / "lls48" / "run_stage.py"
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            str(script),
            "predict",
            "--data-root",
            str(args.data_root),
            "--output-dir",
            str(args.output_dir),
            "--check-only",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "No model or stage was run" in result.stdout


def test_hardlink_probe_failure_leaves_no_probe_file(tmp_path, monkeypatch):
    def denied(*args, **kwargs):
        raise OSError("synthetic unsupported filesystem")

    monkeypatch.setattr(cache.os, "link", denied)
    with pytest.raises(ValueError, match="hard-link"):
        cache.probe_hardlinks(tmp_path)
    assert not list(tmp_path.iterdir())


def test_cache_plan_reports_resource_shortage_without_model(
    tmp_path, lls_inputs, monkeypatch
):
    args = SimpleNamespace(
        dataset="lls48",
        data_root=lls_inputs[0],
        annotation_csv=None,
        output_dir=tmp_path / "new-cache",
    )
    monkeypatch.setattr(cache.shutil, "disk_usage", lambda p: SimpleNamespace(free=1))
    monkeypatch.setattr(cache, "available_memory", lambda: 1)
    _, report = cache.check_plan(args)
    assert report["status"] == "BLOCKED"
    assert any("disk" in error for error in report["errors"])
    assert any("RAM" in error for error in report["errors"])


def test_synthetic_bad_clip_order_rejected(lls_inputs):
    root, annotation, rows = lls_inputs
    rows[0], rows[1] = rows[1], rows[0]
    with annotation.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _, report = spatial.inspect_inputs("lls48", root)
    assert any("five ordered" in error for error in report["errors"])
