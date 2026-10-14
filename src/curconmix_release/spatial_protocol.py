"""Model-free lineage and partition gates for new spatial training runs.

Historical inference remains separate. A published legacy checkpoint without
this provenance is not a certified warm start for a new nested-validation run.
"""

import json
import math
from pathlib import Path
from types import SimpleNamespace

from curconmix_repro.training_split import (
    fold_partition,
    load_training_split,
    split_digest,
)


def load_split(path, dataset, annotation):
    from .spatial_training import fold_map, sha256_file

    if not path:
        raise ValueError("An explicit training_split / --training-split is required.")
    mapping = fold_map(dataset)
    outer = {str(f): [v for v, k in mapping.items() if k == f] for f in range(5)}
    record = load_training_split(path, dataset, expected_outer_folds=outer)
    if record["annotation_sha256"] != sha256_file(annotation):
        raise ValueError(
            "Training split annotation_sha256 differs from the actual CSV."
        )
    return record


def cfg_split(cfg, dataset, required=True):
    path = getattr(cfg, "training_split", None)
    if not path and not required:
        return None
    from .spatial_training import annotation_path

    return load_split(
        path, dataset, annotation_path(dataset, cfg.parent_path, cfg.path_csv)
    )


def select_rows(rows, record, fold, partition="train"):
    ids = set(fold_partition(record, fold)[partition])
    selected = [row for row in rows if str(row["video"]) in ids]
    if {str(row["video"]) for row in selected} != ids:
        raise ValueError(f"Missing annotation procedures in fold {fold}/{partition}.")
    return selected


def select_frame(frame, record, fold, partition):
    ids = set(fold_partition(record, fold)[partition])
    selected = (
        frame.loc[frame["video"].astype(str).isin(ids)].reset_index(drop=True).copy()
    )
    if set(selected["video"].astype(str)) != ids:
        raise ValueError(f"Missing annotation procedures in fold {fold}/{partition}.")
    return selected


def training_frames(frame, fold, cfg, dataset, record=None):
    record = record if record is not None else cfg_split(cfg, dataset)
    train = select_frame(frame, record, fold, "train")
    valid = select_frame(frame, record, fold, "inner_validation")
    return train, valid, valid.copy()


def metadata(record, fold, stage, epoch=None, score=None):
    parts = fold_partition(record, fold)
    meta = {
        "training_protocol": record["protocol"],
        "training_split_sha256": split_digest(record),
        "annotation_sha256": record["annotation_sha256"],
        "dataset": record["dataset"],
        "fold": int(fold),
        "stage": stage,
        "train_procedures": parts["train"],
        "inner_validation_procedures": parts["inner_validation"],
        "outer_test_procedures": parts["outer_test"],
        "selection_partition": (
            "fixed_training_schedule" if stage == "pretrain" else "inner_validation"
        ),
        "selection_metric": None if stage == "pretrain" else "AP_IVT",
    }
    if epoch is not None:
        meta["epoch"] = int(epoch)
    if score is not None:
        meta["selection_score"] = float(score)
    if stage in ("teacher", "student", "softlabels"):
        meta.update(
            model_name="swin_base_patch4_window7_224",
            input_resolution=224,
            feature_dim=1024,
            num_classes=266 if record["dataset"] == "lls48" else 131,
            triplet_size=195 if record["dataset"] == "lls48" else 100,
            model_class="TripletModel",
        )
    return meta


def validate_meta(meta, record, fold, stage):
    expected = metadata(record, fold, stage)
    if not isinstance(meta, dict) or any(meta.get(k) != v for k, v in expected.items()):
        raise ValueError(
            f"Artifact lineage does not match fold {fold}/{stage}/training split."
        )
    if stage == "pretrain" and meta.get("epoch") != (
        6 if record["dataset"] == "lls48" else 2
    ):
        raise ValueError(
            "Teacher warm start requires the fixed final pretraining epoch."
        )


def checkpoint_metadata(cfg, record, fold, stage, epoch=None, score=None):
    current = cfg_split(cfg, record["dataset"])
    if split_digest(current) != split_digest(record):
        raise ValueError(
            "Training split changed during execution; refusing checkpoint publication."
        )
    return metadata(record, fold, stage, epoch, score)


def sidecar_path(path):
    return Path(str(path) + ".provenance.json")


def write_sidecar(path, meta, **extra):
    from .spatial_training import sha256_file

    path = Path(path)
    record = {
        "schema_version": 1,
        "checkpoint_sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "meta": meta,
        **extra,
    }
    sidecar = sidecar_path(path)
    temporary = sidecar.with_suffix(sidecar.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
        handle.write("\n")
    temporary.replace(sidecar)


def verify_sidecar(path, record, fold, stage, rows=None):
    from .spatial_training import row_order_sha256, sha256_file

    path = Path(path)
    sidecar = sidecar_path(path)
    if not sidecar.is_file():
        raise ValueError(
            f"Missing training provenance: {sidecar}. Legacy artifacts are inference-only."
        )
    info = json.loads(sidecar.read_text(encoding="utf-8"))
    if info.get("schema_version") != 1:
        raise ValueError("Unsupported artifact provenance schema.")
    validate_meta(info.get("meta"), record, fold, stage)
    if info.get("size_bytes") != path.stat().st_size or info.get(
        "checkpoint_sha256"
    ) != sha256_file(path):
        raise ValueError(f"Artifact bytes differ from provenance: {path}")
    if rows is not None and info.get("row_order_sha256") != row_order_sha256(rows):
        raise ValueError("Soft-label training-row order differs from provenance.")
    return info


def verify_softlabels(path, frame, cfg, dataset, fold):
    record = cfg_split(cfg, dataset)
    rows = frame.to_dict("records")
    info = verify_sidecar(path, record, fold, "softlabels", rows)
    if not info.get("teacher_checkpoint_sha256"):
        raise ValueError("Soft-label provenance lacks the teacher checkpoint identity.")
    from .spatial_training import check_softlabels

    check_softlabels(Path(path), rows, dataset, fold, record)


def is_better_inner_ap(score, best_score):
    value = float(score)
    if not math.isfinite(value):
        raise ValueError(
            "Inner-validation AP is undefined/non-finite; no checkpoint fallback is permitted."
        )
    return value > best_score


def inner_validation_ap(frame, record, fold, evaluator, cfg):
    expected = set(fold_partition(record, fold)["inner_validation"])
    if set(frame["video"].astype(str)) != expected:
        raise ValueError(
            "Checkpoint-selection evaluator received a non-inner-validation partition."
        )
    value = float(evaluator(frame, cfg))
    is_better_inner_ap(
        value, float("-inf")
    )  # Reject undefined scores, never fall back.
    return value


def training_gate(cfg, dataset, stage):
    """Direct legacy entry points also run the model-free preflight first."""
    from .spatial_training import preflight

    selected = list(cfg.trn_fold)
    if (
        int(cfg.n_fold) != 5
        or not 0 <= int(cfg.start_n_fold) < 5
        or not selected
        or len(set(selected)) != len(selected)
        or any(
            type(fold) is not int or fold not in range(int(cfg.start_n_fold), 5)
            for fold in selected
        )
    ):
        raise ValueError(
            "Training must use explicit valid fold IDs within canonical folds 0-4."
        )
    if cfg.split_selector not in (
        "LLS48" if dataset == "lls48" else "cholect45-crossval",
    ):
        raise ValueError(
            "Only the canonical nested cross-validation training path is supported."
        )
    if int(cfg.epochs) < 1 or getattr(cfg, "debug", False):
        raise ValueError(
            "New protocol requires a positive, explicit schedule; debug training is disabled."
        )
    if (
        cfg.model_name != "swin_base_patch4_window7_224"
        or int(cfg.height) != 224
        or int(cfg.width) != 224
        or int(cfg.target_size) != (266 if dataset == "lls48" else 131)
    ):
        raise ValueError(
            "This new spatial training workflow currently supports the packaged B224 configuration only."
        )
    if int(cfg.triplet_size) != (195 if dataset == "lls48" else 100):
        raise ValueError(
            "Triplet class count differs from the packaged dataset taxonomy."
        )
    if stage == "teacher" and not getattr(cfg, "pretrained_ssl", False):
        raise ValueError(
            "Teacher training requires split-matched contrastive pretraining."
        )
    if stage == "student" and getattr(cfg, "pretrained_ssl", False):
        raise ValueError(
            "Student uses ImageNet initialization and split-matched teacher soft labels only."
        )
    expected_exp = f"CurConMix_B224_{stage.title()}"
    if str(cfg.exp) != expected_exp:
        raise ValueError(
            f"Use the packaged stage configuration exp={expected_exp} in a fresh run directory."
        )
    final_epoch = 6 if dataset == "lls48" else 2
    if stage == "pretrain" and int(cfg.epochs) != final_epoch + 1:
        raise ValueError(
            "Contrastive pretraining uses the packaged fixed schedule, not validation/test selection."
        )
    if (
        stage == "teacher"
        and cfg.pretrained_exp != f"CurConMix_B224_Pretrain_{final_epoch}"
    ):
        raise ValueError(
            "Teacher warm start must be the fixed final pretraining epoch."
        )
    if stage == "student" and cfg.teacher_exp != "CurConMix_B224_Teacher":
        raise ValueError(
            "Student soft labels must come from the split-matched teacher stage."
        )
    if stage == "pretrain" and (
        cfg.feature_file_name != "trainset_features_threshold_10.pkl"
        or cfg.cos_sim_matrix_file_name != "similarity_matrix.pkl"
    ):
        raise ValueError(
            "Cache filenames must match the provenance-checked packaged recipe."
        )
    record = cfg_split(cfg, dataset)
    args = SimpleNamespace(
        stage=stage,
        data_root=Path(cfg.parent_path),
        output_dir=Path(cfg.output_dir),
        annotation_csv=cfg.path_csv,
        feature_dir=getattr(cfg, "feature_dir", None),
        model_tag=getattr(cfg, "model_tag", "SwinB224"),
        training_split=cfg.training_split,
    )
    report = preflight(dataset, args)
    if report["status"] == "BLOCKED":
        raise ValueError("Spatial training preflight failed: " + json.dumps(report))
    # Preserve existing results; a new run must not silently replace old weights.
    checkpoint_dir = Path(cfg.output_dir) / "checkpoints"
    for fold in cfg.trn_fold:
        pattern = (
            f"fold{fold}_{cfg.model_name[:8]}_{cfg.exp}_*.pth"
            if stage == "pretrain"
            else f"fold{fold}_{cfg.model_name[:8]}_{cfg.target_size}_{cfg.exp}.pth"
        )
        if any(checkpoint_dir.glob(pattern)):
            raise ValueError(
                "Stage checkpoints already exist; use a fresh run output directory."
            )
    return record
