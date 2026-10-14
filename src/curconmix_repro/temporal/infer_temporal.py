import os
import re
import json
import argparse
import logging
import inspect
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 프로젝트 모듈
from .dataset import VideoDataset
from .collate import pad_collate_fn
from .temporal_model import MultiResTemporalModel
from .temporal_model2 import MultiResTemporalModel2
from curconmix_repro.validation import validate_checkpoint_fold

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# --- 데이터셋별 레이블 블록 정의 ---
DATASET_CONFIGS = {
    "lls48":   {"num_classes": 266, "N_TRI": 195, "N_INST": 17, "N_VERB": 16, "N_TARG": 38},
    "cholect45":{"num_classes": 131, "N_TRI": 100, "N_INST": 6,  "N_VERB": 10, "N_TARG": 15},
    "cholect50":{"num_classes": 131, "N_TRI": 100, "N_INST": 6,  "N_VERB": 10, "N_TARG": 15},
}

# --- 저장 디렉토리 결정 규칙 ---
#   - all datasets  → args.prediction_dir
_ENV_ROOT = os.environ.get("CURCONMIX_ROOT")
_PREDICTION_DIR = os.environ.get("CURCONMIX_PREDICTION_DIR")
REPO_ROOT = Path(_ENV_ROOT) if _ENV_ROOT else Path(__file__).resolve().parents[3]
DEFAULT_PREDICTION_DIR = _PREDICTION_DIR or str(REPO_ROOT / "artifacts" / "predictions")

def _decide_save_dir(dataset_name: str, fallback_dir: str) -> str:
    return fallback_dir


def _log_meta(exp_dir: str, meta: Dict, ds_config: Dict):
    """실험별 메타 정보를 보기 좋게 로그로 출력."""
    logging.info("─" * 80)
    logging.info(f"[EXP] {exp_dir}")
    logging.info("[META] "
                 f"name={meta.get('name', 'N/A')}, "
                 f"model_type={meta.get('model_type', 'mr2')}, "
                 f"embed_dim={meta.get('embed_dim', 'N/A')}, "
                 f"dataset={meta.get('dataset', 'N/A')}, "
                 f"num_classes={meta.get('num_classes', ds_config.get('num_classes', 'N/A'))}")
    logging.info("[META] "
                 f"ks={meta.get('ks', 'N/A')}, fusion={meta.get('fusion', 'learned')}, "
                 f"use_spatial_norm={meta.get('use_spatial_norm', False)}, "
                 f"spatial_dropout={meta.get('spatial_dropout', 0.0)}, "
                 f"ks_weights={meta.get('ks_weights', False)}")
    tc = meta.get("temporal_cfg", {})
    logging.info(f"[META] temporal_cfg="
                 f"(nhead={tc.get('nhead','?')}, layers={tc.get('num_layers','?')}, "
                 f"ff={tc.get('dim_feedforward','?')}, dropout={tc.get('dropout','?')})")
    logging.info(f"[DATASET CONFIG] N_TRI={ds_config.get('N_TRI')}, "
                 f"N_INST={ds_config.get('N_INST')}, N_VERB={ds_config.get('N_VERB')}, "
                 f"N_TARG={ds_config.get('N_TARG')} (total={ds_config.get('num_classes')})")
    logging.info("─" * 80)


def get_label_columns(ds_config: Dict):
    N_TRI = ds_config['N_TRI']; N_INST = ds_config['N_INST']
    N_VERB = ds_config['N_VERB']; N_TARG = ds_config['N_TARG']
    num_classes = ds_config['num_classes']

    TRI_COLS  = [f"tri{i}"  for i in range(N_TRI)]
    INST_COLS = [f"inst{i}" for i in range(N_INST)]
    VERB_COLS = [f"v{i}"    for i in range(N_VERB)]
    TARG_COLS = [f"t{i}"    for i in range(N_TARG)]
    PRED_COLS = [str(i) for i in range(num_classes)]

    META_COLS = ["video", "clip_name", "image_path", "image_id"]
    CSV_COL_ORDER = META_COLS + TRI_COLS + INST_COLS + VERB_COLS + TARG_COLS + ["fold"] + PRED_COLS
    return {"TRI_COLS": TRI_COLS, "INST_COLS": INST_COLS, "VERB_COLS": VERB_COLS,
            "TARG_COLS": TARG_COLS, "PRED_COLS": PRED_COLS, "CSV_COL_ORDER": CSV_COL_ORDER}

def _load_ckpt(path: str):
    # Public release: never silently execute arbitrary pickled checkpoint objects.
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as error:
        raise ValueError(
            "Checkpoint is not safe-loadable; use a verified model-only export or a "
            "new safe-format training checkpoint. Unsafe unpickling is never retried."
        ) from error

def _load_meta(ckpt_path: str, expected_fold: Optional[int] = None) -> Dict:
    obj = _load_ckpt(ckpt_path)
    if expected_fold is not None:
        if not isinstance(obj, dict) or not isinstance(obj.get("meta"), dict):
            raise ValueError("An embedded, explicit checkpoint fold is required")
        validate_checkpoint_fold(obj["meta"], expected_fold)
    if isinstance(obj, dict) and "meta" in obj:
        return obj["meta"]
    match = re.fullmatch(r"best_model_fold(\d+)\.pth", Path(ckpt_path).name)
    if match:
        sidecar = Path(ckpt_path).with_name(f"experiment_fold{match.group(1)}.json")
        if sidecar.is_file():
            with sidecar.open(encoding="utf-8") as stream:
                return json.load(stream)
    raise ValueError(f"Checkpoint metadata missing: {ckpt_path}; no architecture will be guessed")

def _build_model_from_meta(meta: Dict):
    model_type = meta.get("model_type", "mr2")
    embed_dim  = int(meta.get("embed_dim", 1024))
    dataset    = meta.get("dataset", "lls48")
    num_classes= int(meta.get("num_classes", 131 if dataset in ["cholect45", "cholect50"] else 266))
    ks         = tuple(meta.get("ks", (4, 6, 9)))
    if model_type == "mr1":
        return MultiResTemporalModel(embed_dim=embed_dim, num_classes=num_classes, ks=ks)
    else:
        return MultiResTemporalModel2(
            embed_dim=embed_dim, num_classes=num_classes, ks=ks,
            fusion=meta.get("fusion", "learned"),
            use_spatial_norm=meta.get("use_spatial_norm", False),
            spatial_dropout=float(meta.get("spatial_dropout", 0.0)),
            ks_weights=bool(meta.get("ks_weights", False)),
            temporal_cfg=meta.get("temporal_cfg", {"nhead":4,"num_layers":3,"dim_feedforward":2048,"dropout":0.1}),
        )

def _list_experiment_dirs(root: str) -> List[str]:
    """root 하위에서 best_model_fold*.pth 가 있는 최상위 폴더만 수집"""
    exp_dirs = []
    for dirpath, _, filenames in os.walk(root):
        if any(re.match(r"best_model_fold\d+\.pth$", f) for f in filenames):
            exp_dirs.append(dirpath)
    exp_dirs = sorted(set(exp_dirs))
    pruned = []
    for d in exp_dirs:
        if not any((d != e and d.startswith(e + os.sep)) for e in exp_dirs):
            pruned.append(d)
    return pruned

# ---------- 라벨/피처 경로 해석 유틸 ----------
def _resolve_feature_path_for_video(ds: VideoDataset, dataset_name: str, vid: str) -> Optional[str]:
    base = os.path.splitext(os.path.basename(str(vid)))[0]
    base_upper = base.upper()

    # 1) Dataset 내부 매핑/속성 우선
    for attr in ["feature_paths", "feature_map", "video_to_feature", "vid2feature"]:
        if hasattr(ds, attr):
            obj = getattr(ds, attr)
            if isinstance(obj, dict):
                for k, v in obj.items():
                    k0 = os.path.splitext(os.path.basename(str(k)))[0].upper()
                    if k0 == base_upper and os.path.isfile(str(v)):
                        return str(v)
            if isinstance(obj, (list, tuple)):
                for p in obj:
                    p0 = os.path.splitext(os.path.basename(str(p)))[0].upper()
                    if p0 == base_upper and os.path.isfile(str(p)):
                        return str(p)

    # 2) feature 디렉토리 후보(데이터셋이 노출하는 속성 우선)
    cand_dirs = []
    for attr in ["feature_dir", "features_dir", "feature_root", "features_root"]:
        if hasattr(ds, attr):
            val = getattr(ds, attr)
            if isinstance(val, str) and os.path.isdir(val):
                cand_dirs.append(val)

    seen, feat_dirs = set(), []
    for d in cand_dirs:
        if d not in seen and os.path.isdir(d):
            feat_dirs.append(d)
        seen.add(d)

    # 3) 파일명 후보
    basename_opts = [base_upper, base]
    for d in feat_dirs:
        for stem in basename_opts:
            for fname in [f"{stem}.npy", f"{stem}.NPY"]:
                cand = os.path.join(d, fname)
                if os.path.isfile(cand):
                    return cand
    return None

def _resolve_label_path_for_video(ds: VideoDataset, dataset_name: str, vid: str) -> Optional[str]:
    base = os.path.splitext(os.path.basename(str(vid)))[0]
    base_upper = base.upper()

    m = re.search(r'(\d+)', base_upper)
    num = int(m.group(1)) if m else None

    # 1) Dataset 내부 매핑 우선
    for attr in ["label_paths", "label_map", "video_to_label", "vid2label"]:
        if hasattr(ds, attr):
            obj = getattr(ds, attr)
            if isinstance(obj, dict):
                for k, v in obj.items():
                    k0 = os.path.splitext(os.path.basename(str(k)))[0].upper()
                    if k0 == base_upper and os.path.isfile(str(v)):
                        return str(v)
            if isinstance(obj, (list, tuple)):
                for p in obj:
                    p0 = os.path.splitext(os.path.basename(str(p)))[0].upper()
                    if p0 == base_upper and os.path.isfile(str(p)):
                        return str(p)

    # 2) 라벨 디렉토리 후보군
    cand_dirs = []
    for attr in ["label_dir", "label_root", "labels_dir", "labels_root"]:
        if hasattr(ds, attr):
            val = getattr(ds, attr)
            if isinstance(val, str) and os.path.isdir(val):
                cand_dirs.append(val)

    # 휴리스틱 디렉토리(필요 시)
    for root_base in [os.getcwd(), os.path.dirname(os.getcwd()), os.path.dirname(os.path.dirname(os.getcwd()))]:
        for ds_name in [dataset_name, dataset_name.lower(), dataset_name.upper()]:
            cand_dirs.append(os.path.join(root_base, "triplet_temporal", "data", ds_name, "triplet_label"))

    # 중복 제거 + 존재하는 것만
    seen, label_dirs = set(), []
    for d in cand_dirs:
        if d not in seen and os.path.isdir(d):
            label_dirs.append(d)
            seen.add(d)

    # 3) 파일명 후보
    basename_opts = [base_upper, base]
    if num is not None:
        basename_opts += [f"VID{num:02d}", f"video{num:02d}", f"VIDEO{num:02d}"]

    for d in label_dirs:
        for stem in basename_opts:
            for fname in [f"{stem}.npy", f"{stem}.NPY"]:
                cand = os.path.join(d, fname)
                if os.path.isfile(cand):
                    return cand
    return None

# -----------------------------------------------

@torch.no_grad()
def _infer_fold_df_like_format(
    model: nn.Module,
    dataset_obj: VideoDataset,
    device: torch.device,
    use_logits: bool,
    fold_index: int,
    ds_config: dict,
    dataset_name: str,
    split: str,
    verbose_labels: bool = False,
    check_label_shape: bool = False,
    verbose_features: bool = False,
    check_feature_shape: bool = False,
) -> pd.DataFrame:
    """
    Convert one fold split to a frame-level DataFrame using the dataset CSV schema.
    verbose_labels=True → 각 비디오의 label.npy 경로/shape 출력
    verbose_features=True → 각 비디오의 feature.npy 경로/shape 출력
    """
    num_classes = ds_config['num_classes']
    cols = get_label_columns(ds_config)
    TRI_COLS, INST_COLS, VERB_COLS, TARG_COLS = cols['TRI_COLS'], cols['INST_COLS'], cols['VERB_COLS'], cols['TARG_COLS']
    CSV_COL_ORDER = cols['CSV_COL_ORDER']

    loader = DataLoader(dataset_obj, batch_size=1, shuffle=False, drop_last=False,
                        num_workers=0, pin_memory=True, collate_fn=pad_collate_fn)
    model.eval()
    rows = []

    for (video, labels, video_names, valid_len) in tqdm(loader, desc=f"Infer(fold={fold_index}, split={split})"):
        video = video.to(device)
        labels = labels.to(device)
        valid_len = valid_len.to(device)

        sig = inspect.signature(model.forward)
        if "valid_len" in sig.parameters:
            logits, _ = model(video, valid_len=valid_len)
        else:
            logits, _ = model(video)

        T = int(valid_len[0].item())
        logits = logits[0, :T]
        labels = labels[0, :T]

        preds = logits if use_logits else logits.sigmoid()
        preds_np = preds.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()

        tri_np  = labels_np[:, 0:ds_config['N_TRI']]
        inst_np = labels_np[:, ds_config['N_TRI']:ds_config['N_TRI']+ds_config['N_INST']]
        verb_np = labels_np[:, ds_config['N_TRI']+ds_config['N_INST']:ds_config['N_TRI']+ds_config['N_INST']+ds_config['N_VERB']]
        targ_np = labels_np[:, ds_config['N_TRI']+ds_config['N_INST']+ds_config['N_VERB']:]

        vid = str(video_names[0])

        # 라벨 npy 경로/shape 로그
        if verbose_labels and split == "val":
            npy_path = _resolve_label_path_for_video(dataset_obj, dataset_name, vid)
            if npy_path is None:
                logging.warning(f"[val][{vid}] label npy NOT FOUND (expose 'label_dir' or mapping on VideoDataset).")
            else:
                if check_label_shape:
                    try:
                        arr = np.load(npy_path, mmap_mode="r")
                        shape_str = f"shape={arr.shape}, dtype={arr.dtype}"
                        if arr.ndim == 2 and arr.shape[1] != num_classes:
                            logging.warning(f"[val][{vid}] label npy: {npy_path}  ({shape_str})  -> WARNING: last dim != num_classes({num_classes})")
                        else:
                            logging.info(f"[val][{vid}] label npy: {npy_path}  ({shape_str})")
                    except Exception as e:
                        logging.warning(f"[val][{vid}] label npy: {npy_path}  (failed to read: {e})")
                else:
                    logging.info(f"[val][{vid}] label npy: {npy_path}")

        # 피처 npy 경로/shape 로그
        if verbose_features and split == "val":
            fpath = _resolve_feature_path_for_video(dataset_obj, dataset_name, vid)
            if fpath is None:
                logging.warning(f"[val][{vid}] feature npy NOT FOUND (expose 'feature_dir' or mapping on VideoDataset).")
            else:
                if check_feature_shape:
                    try:
                        farr = np.load(fpath, mmap_mode="r")
                        logging.info(f"[val][{vid}] feature npy: {fpath}  (shape={farr.shape}, dtype={farr.dtype})")
                    except Exception as e:
                        logging.warning(f"[val][{vid}] feature npy: {fpath}  (failed to read: {e})")
                else:
                    logging.info(f"[val][{vid}] feature npy: {fpath}")

        # DF 행 구성
        for t in range(T):
            row = {"video": vid, "clip_name": "", "image_path": "", "image_id": f"{vid}_{t+1}", "fold": int(fold_index)}
            for i, c in enumerate(TRI_COLS):  row[c] = float(tri_np[t, i])
            for i, c in enumerate(INST_COLS): row[c] = float(inst_np[t, i])
            for i, c in enumerate(VERB_COLS): row[c] = float(verb_np[t, i])
            for i, c in enumerate(TARG_COLS): row[c] = float(targ_np[t, i])
            for k in range(num_classes):      row[str(k)] = float(preds_np[t, k])
            rows.append(row)

    df = pd.DataFrame(rows)
    df = df[CSV_COL_ORDER]
    return df

def main():
    ap = argparse.ArgumentParser("Dump ONE combined CSV (5-fold) per experiment in your required format")
    ap.add_argument("--experiments_root", required=True, type=str, help="Checkpoint root")
    ap.add_argument("--prediction_dir", type=str, default=DEFAULT_PREDICTION_DIR, help="prediction CSV output directory")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--use_logits", action="store_true", help="로짓을 저장(기본은 확률)")
    ap.add_argument("--split", type=str, default="val", choices=["val", "test"], help="fold별 사용할 split")

    # 추가 플래그
    ap.add_argument("--verbose_labels", action="store_true", help="검증에서 각 비디오의 label.npy 경로를 모두 출력")
    ap.add_argument("--check_label_shape", action="store_true", help="label.npy를 열어 shape도 함께 출력(마지막 차원과 num_classes 검증)")
    ap.add_argument("--verbose_features", action="store_true", help="검증에서 각 비디오의 feature.npy 경로를 모두 출력")
    ap.add_argument("--check_feature_shape", action="store_true", help="feature.npy를 열어 shape도 함께 출력")

    # ✅ 새 옵션: 기존 결과 스킵/강제 재추론
    ap.add_argument("--force_reinfer", action="store_true",
                    help="기존 CSV가 있어도 무시하고 처음부터 다시 inference하여 덮어쓰기")

    args = ap.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    exp_dirs = _list_experiment_dirs(args.experiments_root)
    if not exp_dirs:
        logging.error(f"No experiments found under: {args.experiments_root}")
        return

    total_exps = len(exp_dirs)
    logging.info(f"Found {total_exps} experiment(s).")

    saved_count = 0
    skipped_count = 0
    error_count = 0
    saved_paths = []

    # 상위 진행바: 전체 실험 진행도
    pbar = tqdm(total=total_exps, desc="Experiments", unit="exp")

    for idx, exp_dir in enumerate(exp_dirs, 1):
        try:
            logging.info("="*14 + f" [{idx}/{total_exps}] {exp_dir} " + "="*14)

            meta, ckpts = None, []
            for f in range(5):
                p = os.path.join(exp_dir, f"best_model_fold{f}.pth")
                if os.path.exists(p):
                    ckpts.append((f, p))
                    if meta is None:
                        meta = _load_meta(p)

            if meta is None:
                logging.warning(f"[{exp_dir}] no best_model_fold*.pth found; skip.")
                skipped_count += 1
                pbar.update(1)
                continue

            dataset_name = meta.get("dataset", "lls48")
            ds_config = DATASET_CONFIGS.get(dataset_name)
            if not ds_config:
                logging.error(f"[{exp_dir}] Dataset '{dataset_name}' is not supported. Skip.")
                skipped_count += 1
                pbar.update(1)
                continue

            # 저장할 디렉토리 결정 (데이터셋별 서로 다른 위치)
            save_dir = _decide_save_dir(dataset_name, args.prediction_dir)
            os.makedirs(save_dir, exist_ok=True)

            out_name = f"{os.path.basename(exp_dir)}.csv"
            out_path = os.path.join(save_dir, out_name)

            # ✅ 기존 결과 스킵 / 강제 재추론 분기
            if os.path.isfile(out_path) and not args.force_reinfer:
                logging.info(f"⏭️  Skip existing: {out_path}  (재생성하려면 --force_reinfer 사용)")
                skipped_count += 1
                pbar.update(1)
                continue
            elif os.path.isfile(out_path) and args.force_reinfer:
                logging.info(f"♻️  Re-infer (overwrite): {out_path}")

            logging.info(f"[SAVE DIR] dataset={dataset_name} → {save_dir}")
            _log_meta(exp_dir, meta, ds_config)
            model = _build_model_from_meta(meta).to(device)

            fold_dfs = []
            # (선택) fold 진행감도 보여주고 싶다면 아래 주석 해제해서 간단한 진행표시
            # fold_pbar = tqdm(total=len(ckpts), desc=f"Folds ({os.path.basename(exp_dir)})", unit="fold", leave=False)
            for fold_idx, ckpt_path in ckpts:
                logging.info(f"[fold{fold_idx}] Loading checkpoint: {ckpt_path}")
                obj = _load_ckpt(ckpt_path)
                state_dict = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                if missing or unexpected:
                    logging.warning(f"[fold{fold_idx}] load_state_dict: missing={missing}, unexpected={unexpected}")

                ds = VideoDataset(dataset_name, meta.get("name", "224_student"), args.split, fold_idx)

                df_fold = _infer_fold_df_like_format(
                    model, ds, device, args.use_logits, fold_idx, ds_config,
                    dataset_name=dataset_name, split=args.split,
                    verbose_labels=args.verbose_labels, check_label_shape=args.check_label_shape,
                    verbose_features=args.verbose_features, check_feature_shape=args.check_feature_shape
                )
                fold_dfs.append(df_fold)
                logging.info(f"[fold{fold_idx}] frames: {len(df_fold)}")
                # fold_pbar.update(1)
            # fold_pbar.close()

            if not fold_dfs:
                logging.warning(f"[{exp_dir}] no folds dumped; skip combine.")
                skipped_count += 1
                pbar.update(1)
                continue

            all_df = pd.concat(fold_dfs, ignore_index=True)
            all_df.to_csv(out_path, index=False, encoding="utf-8-sig")
            logging.info(f"✅ saved: {out_path}")
            saved_paths.append(out_path)
            saved_count += 1

        except Exception as e:
            logging.exception(f"❌ Error while processing {exp_dir}: {e}")
            error_count += 1

        finally:
            pbar.update(1)

    pbar.close()

    # 요약 출력
    logging.info("─" * 80)
    logging.info(f"SUMMARY | total={total_exps}, saved={saved_count}, skipped={skipped_count}, errors={error_count}")
    if saved_paths:
        logging.info("Saved CSVs:")
        for p in saved_paths:
            logging.info(f" - {p}")
    logging.info("Done.")

if __name__ == "__main__":
    raise SystemExit("Use python -m curconmix_repro.cli infer-temporal for validated inference")
