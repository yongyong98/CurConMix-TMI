"""Model-free validation of newly exported LLS48 features and original row keys.

This checks provenance, array identity and alignment, not numerical equivalence
to the unpublished historical features or reproduction of the reported scores.
"""

import hashlib
import json
from pathlib import Path
import re

from .assets import load_manifest
from .lls48 import (
    CANONICAL_FOLDS,
    LABEL_COLUMNS,
    MASTER_RELATIVE,
    MASTER_SHA256,
    inspect_annotations,
    sha256_file,
    source_file,
)


MODEL_NAMES = {
    "lls48-spatial-b224": (
        "swin_bas_266_LLS48_SwinB_Curriculum_Feature_Mixup_e7_Student"
    ),
    "lls48-spatial-b384": (
        "swin_bas_266_LLS48_SwinB384_Curriculum_Feature_Mixup_e7_Student"
    ),
}
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPATIAL_MANIFEST = PROJECT_ROOT / "manifests/spatial_assets.json"
SOURCE_FILES = (
    "legacy/spatial/lls48/models.py",
    "legacy/spatial/lls48/augmentation.py",
    "src/curconmix_release/feature_export.py",
    "src/curconmix_release/feature_contract.py",
    "src/curconmix_release/dataset.py",
)
SOURCE_HASH_POLICY = "source_bytes_crlf_and_cr_to_lf"
RUNTIME_VERSIONS = {
    "torch": "2.0.1",
    "numpy": "1.22.4",
    "albumentations": "1.0.3",
    "opencv": "4.9.0",
    "timm": "0.5.4",
}
CHECKPOINT_HASH_KEYS = (
    "sha256",
    "source_sha256",
    "canonical_prediction_sha256",
    "mapping_sha256",
)
LIMITATION = (
    "Newly extracted features have not been numerically qualified against the "
    "historical MRTT inputs; published AP reproduction is not established. "
    "Procedure arrays preserve sparse sample order, not a continuous clock. "
    "LLS48 segmental evaluation must remain clip-bounded: reset events at each "
    "independent five-frame clip; never interpolate across unsampled gaps."
)


def config_identity(config):
    """Hash the full semantic config, independent of JSON whitespace."""
    content = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def source_code_hash(path):
    """Ignore source-code newline conventions only; all data hashes stay byte-exact."""
    content = Path(path).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(content).hexdigest()


def preprocessing_spec(resolution):
    return {
        "decoder": "cv2.imread(IMREAD_COLOR), BGR2RGB",
        "augmentation": "legacy/spatial/lls48/augmentation.py:get_transforms(data='valid')",
        "resize": [resolution, resolution],
        "interpolation": "cv2.INTER_LINEAR",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "max_pixel_value": 255.0,
        "tensor_layout": "NCHW float32",
        "random_augmentation": False,
        "feature": "TripletModel.model then historical pool_backbone_feature; pre-head",
    }


def frame_key(row_index, row):
    return {
        "row_index": row_index,
        **{key: row[key] for key in ("video", "clip_name", "image_id", "image_path")},
        "frame_in_clip": int(Path(row["image_path"]).stem.split("_")[1]),
    }


def _trusted_checkpoint_records(path, asset_id):
    manifest = load_manifest(path)
    matches = [asset for asset in manifest["assets"] if asset["id"] == asset_id]
    if len(matches) != 1:
        raise ValueError("Spatial asset absent from trusted public manifest")
    files = matches[0].get("files", [])
    if len(files) != 5 or any(type(item.get("fold")) is not int for item in files):
        raise ValueError(
            "Trusted spatial manifest must contain exactly five integer folds"
        )
    if {item["fold"] for item in files} != set(range(5)):
        raise ValueError("Trusted spatial checkpoint fold set mismatch")
    for item in files:
        if any(
            not re.fullmatch(r"[0-9a-f]{64}", str(item.get(key, "")))
            for key in CHECKPOINT_HASH_KEYS
        ):
            raise ValueError("Trusted checkpoint provenance requires SHA-256 hashes")
    return {item["fold"]: item for item in files}


def _checked_array(root, record, path, shape, dtype):
    import numpy as np

    if record.get("path") != path:
        raise ValueError("Feature/label manifest path does not match its identity")
    resolved = source_file(root, path)
    if record.get("size_bytes") != resolved.stat().st_size or record.get(
        "sha256"
    ) != sha256_file(resolved):
        raise ValueError(f"Feature/label size or SHA-256 mismatch: {path}")
    array = np.load(resolved, allow_pickle=False)
    if (
        list(array.shape) != shape
        or record.get("shape") != shape
        or str(array.dtype) != dtype
        or record.get("dtype") != dtype
        or not np.isfinite(array).all()
    ):
        raise ValueError(
            f"Feature/label shape, dtype or finite-value check failed: {path}"
        )
    return array


def validate_feature_manifest(
    manifest_path,
    config,
    feature_root,
    label_root,
    *,
    fold=None,
    spatial_manifest_path=None,
    project_root=None,
):
    """Verify a complete export without importing torch or executing any model.

    ``fold=None`` hashes all five folds; an integer hashes that fold's features.
    The canonical annotation and all labels/keys are always checked. The roots
    must be the ``features`` and ``labels`` siblings of this manifest.
    """
    import numpy as np

    if fold is not None and (type(fold) is not int or fold not in range(5)):
        raise ValueError("Fold must be 0 through 4")
    manifest_path = Path(manifest_path).resolve()
    root = manifest_path.parent
    if Path(feature_root).resolve() != root / "features" or Path(
        label_root
    ).resolve() != (root / "labels"):
        raise ValueError(
            "Feature and label roots do not belong to this export manifest"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    asset = manifest.get("spatial_asset")
    expected = {
        "schema_version": 1,
        "dataset": "lls48",
        "model_name": config.get("model_name"),
        "config_identity_sha256": config_identity(config),
        "annotation_sha256": MASTER_SHA256,
        "label_columns": LABEL_COLUMNS,
        "dim": 1024,
        "num_classes": 266,
        "event_extraction_scope": "independent_five_frame_clip",
        "reproduction_status": "NOT_NUMERICALLY_QUALIFIED",
    }
    if config.get("dataset") != "lls48" or config.get("dim") != 1024:
        raise ValueError("This feature contract supports LLS48 SwinB only")
    if asset not in MODEL_NAMES or MODEL_NAMES[asset] != config.get("model_name"):
        raise ValueError(
            "Feature asset does not match the temporal model configuration"
        )
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Feature manifest identity/configuration mismatch")
    trusted = _trusted_checkpoint_records(
        spatial_manifest_path or DEFAULT_SPATIAL_MANIFEST, asset
    )
    code_root = Path(project_root) if project_root is not None else PROJECT_ROOT
    source_hashes = {name: source_code_hash(code_root / name) for name in SOURCE_FILES}
    if (
        manifest.get("source_code_hash_policy") != SOURCE_HASH_POLICY
        or manifest.get("source_code_sha256") != source_hashes
    ):
        raise ValueError("Feature exporter/preprocessing source code identity changed")
    runtime = manifest.get("runtime", {})
    if set(runtime) != set(RUNTIME_VERSIONS) or any(
        str(runtime[key]).split("+")[0] != version
        for key, version in RUNTIME_VERSIONS.items()
    ):
        raise ValueError("Feature runtime version provenance mismatch")
    if not re.fullmatch(
        r"[0-9a-f]{64}", str(manifest.get("spatial_manifest_sha256", ""))
    ):
        raise ValueError("Spatial manifest source hash is missing")
    resolution = 224 if asset.endswith("b224") else 384
    if manifest.get("preprocessing") != preprocessing_spec(resolution):
        raise ValueError("Feature preprocessing contract changed")
    annotation = manifest.get("annotation", {})
    annotation_path = source_file(root, MASTER_RELATIVE)
    if annotation != {
        "path": MASTER_RELATIVE,
        "sha256": MASTER_SHA256,
        "size_bytes": annotation_path.stat().st_size,
    }:
        raise ValueError("Canonical annotation record mismatch")
    # inspect_annotations independently hashes the copied original annotation.
    rows, _, _ = inspect_annotations(root)
    canonical = {}
    for index, row in enumerate(rows):
        canonical.setdefault(row["video"], []).append((index, row))
    procedures = manifest.get("procedures", {})
    if set(procedures) != set(canonical):
        raise ValueError("Feature manifest procedure set differs from the annotation")
    for video, source_rows in canonical.items():
        entry = procedures[video]
        if entry.get("frames") != [frame_key(i, row) for i, row in source_rows]:
            raise ValueError(f"Original frame/clip key order mismatch: {video}")
        labels = _checked_array(
            root,
            entry["labels"],
            f"labels/lls48/triplet_label/{video}.npy",
            [len(source_rows), 266],
            "float32",
        )
        expected_labels = np.asarray(
            [[int(row[column]) for column in LABEL_COLUMNS] for _, row in source_rows],
            dtype=np.float32,
        )
        if not np.array_equal(labels, expected_labels):
            raise ValueError(f"Full 266-column ground-truth alignment failed: {video}")
    folds = manifest.get("folds", {})
    if set(folds) != set(map(str, range(5))):
        raise ValueError("Feature manifest must contain exactly five folds")
    for number in range(5):
        entry = folds[str(number)]
        checkpoint = entry.get("checkpoint", {})
        if (
            type(checkpoint.get("fold")) is not int
            or checkpoint.get("fold") != number
            or checkpoint.get("asset") != asset
        ):
            raise ValueError(
                "Spatial checkpoint fold/asset mismatch in feature manifest"
            )
        if any(
            checkpoint.get(key) != trusted[number][key] for key in CHECKPOINT_HASH_KEYS
        ):
            raise ValueError(
                "Spatial checkpoint provenance does not match the trusted fold"
            )
        files = entry.get("files", [])
        if len(files) != len(canonical) or {x.get("video") for x in files} != set(
            canonical
        ):
            raise ValueError("Each fold must contain every procedure exactly once")
        for item in files:
            video = item["video"]
            split = "val" if video in CANONICAL_FOLDS[str(number)] else "train"
            if item.get("split") != split:
                raise ValueError("Feature train/validation membership changed")
            expected_path = (
                f"features/lls48/{config['model_name']}/{number}/{split}/{video}.npy"
            )
            if item.get("path") != expected_path:
                raise ValueError("Feature file path is not bound to its fold and split")
            if fold is None or number == fold:
                _checked_array(
                    root, item, expected_path, [len(canonical[video]), 1024], "float32"
                )
    return manifest


def restore_prediction_keys(frame, manifest, label_root, *, fold=None):
    """Restore keys after temporal inference, checking every GT value first.

    The caller must first obtain ``manifest`` with validate_feature_manifest.
    Historical synthetic IDs must be consecutive *within each procedure*; no
    positional relabeling of arbitrary prediction files is accepted.
    """
    import numpy as np

    if fold is not None and (type(fold) is not int or fold not in range(5)):
        raise ValueError("Fold must be 0 through 4")
    if not frame.index.is_unique or not set(LABEL_COLUMNS + ["video", "fold"]).issubset(
        frame.columns
    ):
        raise ValueError("Predictions require a unique index and all 266 GT columns")
    result = frame.copy(deep=True)
    normalized = result["video"].map(lambda value: str(value).removesuffix(".npy"))
    wanted = (
        set(manifest["procedures"]) if fold is None else set(CANONICAL_FOLDS[str(fold)])
    )
    if set(normalized) != wanted:
        raise ValueError(
            "Prediction procedures do not cover the requested validation set"
        )
    for video in sorted(wanted):
        selected = result.index[normalized == video]
        entry = manifest["procedures"][video]
        keys = entry["frames"]
        expected_fold = next(
            int(f) for f, videos in CANONICAL_FOLDS.items() if video in videos
        )
        if (
            len(selected) != len(keys)
            or not (result.loc[selected, "fold"] == expected_fold).all()
        ):
            raise ValueError(
                "Prediction length or held-out fold differs from feature keys"
            )
        labels_path = Path(label_root) / "lls48/triplet_label" / f"{video}.npy"
        labels = np.load(labels_path, allow_pickle=False)
        if sha256_file(labels_path) != entry["labels"]["sha256"] or not np.array_equal(
            result.loc[selected, LABEL_COLUMNS].to_numpy(), labels
        ):
            raise ValueError("Prediction-to-feature full ground-truth alignment failed")
        for position, index in enumerate(selected):
            old_id = str(result.at[index, "image_id"])
            raw_video = str(result.at[index, "video"])
            if old_id not in (
                f"{raw_video}_{position + 1}",
                f"{video}_{position + 1}",
            ):
                raise ValueError(
                    "Synthetic prediction frame sequence is not consecutive"
                )
            key = keys[position]
            for name in ("video", "clip_name", "image_path", "image_id"):
                result.at[index, name] = key[name]
            result.at[index, "source_row_index"] = key["row_index"]
    result["source_row_index"] = result["source_row_index"].astype("int64")
    return result
