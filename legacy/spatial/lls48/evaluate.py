# evaluate.py
import os
import sys
import pandas as pd
import hydra
import warnings
import traceback
from typing import Any, Dict
from utils import (
    cholect45_ivtmetrics_mAP,
    cholect45_ivtmetrics_mAP_challenge,
    cholect45_ivtmetrics_mAP_all,
)
from global_var import config_name

warnings.filterwarnings('ignore')

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ----------------------- helpers -----------------------
def _join_prediction_dir(CFG):
    """
    Return the directory that holds evaluation CSVs.
    predictions/ if CFG.inference else oofs/
    """
    folder = "predictions" if getattr(CFG, "inference", False) else "oofs"
    return os.path.join(CFG.output_dir, folder)


def _build_ensemble_name(file_list):
    """
    Build a readable ensemble name from a list of filenames.
    Drops the first underscore token to skip dates/prefixes and joins segments.
    Uses a double underscore between models to reduce collisions.
    """
    parts = []
    for f in file_list:
        stem = os.path.splitext(os.path.basename(f))[0]
        segs = stem.split('_')[1:] if '_' in stem else [stem]
        parts.append('_'.join(segs))
    return "__".join(parts)


def _looks_like_prediction_csv(path, triplet_size=None):
    """
    요약/클래스별 AP CSV 등을 제외하고, '예측 CSV'만 True.
    규칙:
      1) 파일명에 summary/classwise_AP 들어가면 제외
      2) 헤더에 '0' 컬럼이 있어야 함
      3) triplet_size가 주어지면 '0'.. 연속 열이 최소 1개 이상 존재해야 함
    """
    fname = os.path.basename(path).lower()
    if ("summary" in fname) or ("classwise_ap" in fname):
        return False

    try:
        df_head = pd.read_csv(path, nrows=1)
    except Exception:
        return False

    cols = list(df_head.columns)
    if "0" not in cols:
        return False

    if triplet_size is not None and triplet_size > 0:
        return True

    return True


def _list_prediction_csvs(base_dir, triplet_size=None):
    """디렉토리 내에서 '진짜 예측 CSV'만 골라 리턴"""
    files = []
    if not os.path.isdir(base_dir):
        return files
    for f in os.listdir(base_dir):
        if not f.lower().endswith(".csv"):
            continue
        full = os.path.join(base_dir, f)
        if _looks_like_prediction_csv(full, triplet_size=triplet_size):
            files.append(f)
    return files


def _detect_pred_layout(df, triplet_size_hint=None):
    """
    예측 열 레이아웃 가벼운 감지 정보:
    - 첫 예측 컬럼 인덱스(pred0_idx)
    - 예측 열 개수(pred_count) : 넘침 방지 위해 hint가 있으면 min으로 맞춤
    - df.shape
    """
    cols = list(df.columns)
    pred0_idx = cols.index("0") if "0" in cols else None

    pred_count = 0
    if pred0_idx is not None:
        i = pred0_idx
        while i < len(cols) and cols[i].isdigit():
            pred_count += 1
            i += 1

    if triplet_size_hint is not None and pred_count > 0:
        pred_count = min(pred_count, int(triplet_size_hint))

    return pred0_idx, pred_count, df.shape


def _get_triplet_size(CFG) -> int:
    """
    평가/앙상블의 유일 기준: CFG.triplet_size
    """
    if not hasattr(CFG, "triplet_size"):
        raise ValueError("CFG.triplet_size가 설정되어 있지 않습니다. (평가는 triplet_size로 이뤄집니다)")
    t = int(CFG.triplet_size)
    if t <= 0:
        raise ValueError(f"CFG.triplet_size가 0 이하입니다: {t}")
    return t


def _to_score(x: Any) -> float:
    """
    cholect45_ivtmetrics_mAP 반환값 안전 처리:
    - float (score) 또는
    - (score, classwise_df) 튜플
    """
    if isinstance(x, tuple):
        return float(x[0])
    return float(x)


def _get_ensemble_avg(CFG) -> bool:
    """
    앙상블 평균 여부 기본값:
    - CSV가 sigmoid 확률이므로 기본 True 권장
    - CFG에 명시되면 그 값을 따름
    """
    return bool(getattr(CFG, "ensemble_avg", True))


def _append_summary_row(summary_path: str, row: Dict[str, str]):
    """
    요약 CSV에 한 줄 추가(있으면 병합/중복 제거).
    row: {"experiment": name, comp1: "mean ± std", ...}
    """
    if os.path.isfile(summary_path):
        try:
            df_prev = pd.read_csv(summary_path)
        except Exception:
            df_prev = pd.DataFrame(columns=["experiment"])
    else:
        df_prev = pd.DataFrame(columns=["experiment"])

    df_new = pd.DataFrame([row])
    combined = pd.concat([df_prev, df_new], ignore_index=True)
    combined = combined.drop_duplicates(subset=["experiment"], keep="last")
    combined.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"Summary CSV updated: {summary_path} (total experiments: {len(combined)})")


# ----------------------- core evals -----------------------
def evaluate(CFG):
    """
    디렉토리 내 개별 실험 점수(mAP) 출력 + (옵션) 앙상블 점수 출력.
    """
    triplet_size = _get_triplet_size(CFG)
    prediction_dir = _join_prediction_dir(CFG)
    prediction_files = _list_prediction_csvs(prediction_dir, triplet_size=triplet_size)

    # ── Individual File Evaluation ──
    if not getattr(CFG, "ensemble_only", False):
        for pred_file in prediction_files:
            path = os.path.join(prediction_dir, pred_file)
            try:
                df = pd.read_csv(path)
                experiment_parts = pred_file.split(".")[0].split('_')[1:]
                experiment_name = '_'.join(experiment_parts) if experiment_parts else os.path.splitext(pred_file)[0]

                pred0_idx, pred_count, shape = _detect_pred_layout(df, triplet_size_hint=triplet_size)
                print(f"\n[Evaluate] Experiment='{experiment_name}'")
                print(f"  - File         : {path}")
                print(f"  - DF shape     : {shape}")
                print(f"  - Pred cols    : start='{pred0_idx}', count='{pred_count}', triplet_size='{triplet_size}'")

                if 'challenge' in experiment_parts:
                    score = cholect45_ivtmetrics_mAP_challenge(df, CFG)
                    score = _to_score(score)
                    print(f"  => mAP(challenge) = {round(score * 100, 2)}%")
                else:
                    res = cholect45_ivtmetrics_mAP(df, CFG)
                    score = _to_score(res)
                    print(f"  => mAP = {round(score * 100, 2)}%")
                    # 옵션: 클래스별 AP 저장
                    if getattr(CFG, "save_classwise_ap", False) and isinstance(res, tuple) and len(res) == 2:
                        _, classwise_ap_df = res
                        out_csv = os.path.join(prediction_dir, f'{experiment_name}_classwise_AP.csv')
                        classwise_ap_df.to_csv(out_csv, index=False)
                        print(f"  => Saved classwise AP: {out_csv}")

            except Exception as e:
                print("\nEvaluation failed on this experiment/file:")
                print(f"  - File       : {path}")
                print(f"  - Experiment : {pred_file}")
                print("  - Reason     :", repr(e))
                traceback.print_exc()
                # 계속 진행

    # ── Ensemble (optional) ──
    if getattr(CFG, "ensemble", False):
        run_ensemble(CFG, prediction_dir)


def run_ensemble(CFG, prediction_dir: str):
    """
    CFG.ensemble_models에 명시된 CSV들만 읽어서 앙상블 수행.
    CSV는 sigmoid 확률(0~1)이라고 가정.
    evaluate_all=True이면 IVT 각 컴포넌트 mean±std도 함께 계산/출력/저장.
    """
    try:
        if not getattr(CFG, "ensemble_models", None):
            raise ValueError("CFG.ensemble_models가 비어있습니다.")

        triplet_size = _get_triplet_size(CFG)
        use_avg = _get_ensemble_avg(CFG)

        preds = None
        num_models = len(CFG.ensemble_models)
        df_template = None
        pred0_idx = None

        print("\n[Ensemble] Model list:")
        for model in CFG.ensemble_models:
            path = os.path.join(prediction_dir, model)
            print(f"  - {path}")
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Prediction CSV not found: {path}")

            df_m = pd.read_csv(path)

            if pred0_idx is None:
                if "0" not in df_m.columns:
                    raise KeyError(f"Could not find column '0': {path}")
                pred0_idx = df_m.columns.get_loc("0")

            cur = df_m.iloc[:, pred0_idx:pred0_idx + triplet_size].values  # 확률 행렬
            preds = cur if preds is None else preds + cur
            df_template = df_m  # 마지막 것을 템플릿으로 사용

        # 평균/합 선택 (기본: 평균)
        if use_avg:
            preds = preds / num_models

        # 템플릿에 앙상블 결과 주입
        df_template.iloc[:, pred0_idx:pred0_idx + triplet_size] = preds

        # ---------------- Base mAP (전체 1값) ----------------
        res = cholect45_ivtmetrics_mAP(df_template, CFG)
        score = _to_score(res)
        ensemble_experiments = [os.path.splitext(os.path.basename(m))[0] for m in CFG.ensemble_models]
        print(f"  => Ensemble mAP [{ensemble_experiments}] = {round(score * 100, 2)}%")

        # (옵션) 앙상블 결과 CSV 저장
        if getattr(CFG, "save_ensemble_csv", False):
            ens_name = _build_ensemble_name(CFG.ensemble_models)
            out_csv = os.path.join(prediction_dir, f"{ens_name}__ENSEMBLED.csv")
            df_template.to_csv(out_csv, index=False)
            print(f"  => Saved ensembled CSV: {out_csv}")

        # ---------------- Evaluate-All 모드: IVT 컴포넌트별 ----------------
        if getattr(CFG, "evaluate_all", False):
            mean_mAPs, std_mAPs, classwise_AP_dfs = cholect45_ivtmetrics_mAP_all(df_template, CFG)

            print("  => Ensemble (IVT components):")
            for comp in mean_mAPs:
                mean_pct = mean_mAPs[comp] * 100.0
                std_pct = std_mAPs[comp] * 100.0
                print(f"     - {comp} mAP: {mean_pct:.2f}% (±{std_pct:.2f}%)")

            # (옵션) 클래스별 AP 저장
            if getattr(CFG, "save_classwise_ap", False):
                ens_name = _build_ensemble_name(CFG.ensemble_models)
                for comp, classwise_AP_df in classwise_AP_dfs.items():
                    out_csv = os.path.join(prediction_dir, f'{ens_name}_{comp}_classwise_AP.csv')
                    classwise_AP_df.to_csv(out_csv, index=False)
                    print(f"     - Saved ensemble {comp} classwise AP: {out_csv}")

            # (옵션) 요약 CSV에 앙상블 행 추가
            if getattr(CFG, "save_summary_csv", False):
                summary_path = os.path.join(prediction_dir, "all_experiments_summary.csv")
                ens_name = _build_ensemble_name(CFG.ensemble_models)
                row = {"experiment": f"ENSEMBLE__{ens_name}"}
                for comp in mean_mAPs:
                    mean_pct = mean_mAPs[comp] * 100.0
                    std_pct = std_mAPs[comp] * 100.0
                    row[comp] = f"{mean_pct:.2f} ± {std_pct:.2f}"
                _append_summary_row(summary_path, row)

    except Exception as e:
        print("\nEnsemble evaluation failed.")
        print("  - Reason :", repr(e))
        traceback.print_exc()


def evaluate_all(CFG):
    """
    모든 CSV에 대해 cholect45_ivtmetrics_mAP_all을 계산.
    (옵션) 요약/클래스별 AP 저장.
    """
    triplet_size = _get_triplet_size(CFG)
    base_dir = _join_prediction_dir(CFG)
    prediction_files = _list_prediction_csvs(base_dir, triplet_size=triplet_size)

    want_summary_csv = bool(getattr(CFG, "save_summary_csv", False))
    force_recompute = bool(getattr(CFG, "force_recompute", False))
    summary_path = os.path.join(base_dir, "all_experiments_summary.csv")

    # 기존 요약 로드
    existing_exps = set()
    prev_df = None
    if os.path.isfile(summary_path):
        try:
            prev_df = pd.read_csv(summary_path)
            if "experiment" in prev_df.columns and not force_recompute:
                existing_exps = set(prev_df["experiment"].astype(str).tolist())
                print(f"Existing summary CSV found: {summary_path} (already evaluated: {len(existing_exps)}). force_recompute=False, so skipping duplicates.")
            elif "experiment" in prev_df.columns and force_recompute:
                print(f"Existing summary CSV found: {summary_path}. force_recompute=True, so recomputing all entries.")
            else:
                print(f"Warning: existing summary CSV has no 'experiment' column. Recreating it: {summary_path}")
                prev_df = None
        except Exception as e:
            print(f"Warning: failed to load existing summary CSV ({summary_path}): {e}. Recreating it.")

    summary_rows_new = []
    evaluated_any = False

    for pred_df in prediction_files:
        path = os.path.join(base_dir, pred_df)
        try:
            df = pd.read_csv(path)
            experiment = pred_df.split(".")[0].split('_')[1:]
            experiment_name = ('_').join(experiment) if experiment else os.path.splitext(pred_df)[0]

            # 스킵 판단
            if not force_recompute and experiment_name in existing_exps:
                print(f"Skip: already evaluated -> {experiment_name}")
                continue

            pred0_idx, pred_count, shape = _detect_pred_layout(df, triplet_size_hint=triplet_size)
            print(f"\n[EvaluateAll] Experiment='{experiment_name}'")
            print(f"  - File         : {path}")
            print(f"  - DF shape     : {shape}")
            print(f"  - Pred cols    : start='{pred0_idx}', count='{pred_count}', triplet_size='{triplet_size}'")

            mean_mAPs, std_mAPs, classwise_AP_dfs = cholect45_ivtmetrics_mAP_all(df, CFG)

            # (옵션) 클래스별 AP 저장
            if getattr(CFG, "save_classwise_ap", False):
                for comp, classwise_AP_df in classwise_AP_dfs.items():
                    out_csv = os.path.join(base_dir, f'{experiment_name}_{comp}_classwise_AP.csv')
                    classwise_AP_df.to_csv(out_csv, index=False)

            print(f"Results for {experiment_name}:")
            row = {"experiment": experiment_name}
            for comp in mean_mAPs:
                mean_pct = mean_mAPs[comp] * 100.0
                std_pct = std_mAPs[comp] * 100.0
                print(f"  - {comp} mAP: {mean_pct:.2f}% (±{std_pct:.2f}%)")
                row[comp] = f"{mean_pct:.2f} ± {std_pct:.2f}"
            summary_rows_new.append(row)
            evaluated_any = True

        except Exception as e:
            print("\nEvaluation failed on this experiment/file:")
            print(f"  - File       : {path}")
            print(f"  - Experiment : {pred_df}")
            print("  - Reason     :", repr(e))
            traceback.print_exc()
            # 계속 진행

    # 요약 CSV 저장/병합
    if want_summary_csv:
        new_df = pd.DataFrame(summary_rows_new)

        if prev_df is not None and not prev_df.empty:
            if new_df is not None and not new_df.empty:
                if force_recompute:
                    prev_keep = prev_df[~prev_df["experiment"].isin(new_df["experiment"])]
                    combined = pd.concat([prev_keep, new_df], ignore_index=True)
                else:
                    combined = pd.concat([prev_df, new_df], ignore_index=True)
                    combined = combined.drop_duplicates(subset=["experiment"], keep="first")
            else:
                combined = prev_df
        else:
            combined = new_df if new_df is not None and not new_df.empty else pd.DataFrame(columns=["experiment"])

        if not combined.empty:
            combined.to_csv(summary_path, index=False, encoding="utf-8-sig")
            action = "updated" if os.path.isfile(summary_path) else "created"
            print(f"\nSummary CSV {action}: {summary_path} (total experiments: {len(combined)})")
        else:
            print("\nNo summary rows to save.")


# ----------------------- entry -----------------------
@hydra.main(version_base="1.3", config_path=".", config_name=config_name)
def run(CFG):
    """
    사용 예시
    ----
    # 1) 개별 평가만
    python evaluate.py evaluate_all=false inference=true output_dir="/absolute/path/to/outputs" triplet_size=195

    # 2) 전체 평가 + 요약 생성
    python evaluate.py evaluate_all=true save_summary_csv=true inference=true output_dir="/absolute/path/to/outputs" triplet_size=195

    # 3) 기존 요약이 있어도 전부 다시 평가하여 갱신
    python evaluate.py evaluate_all=true save_summary_csv=true force_recompute=true triplet_size=195

    # 4) 앙상블만 (개별/전체 평가는 건너뜀)
    python evaluate.py ensemble=true ensemble_only=true inference=true output_dir="/absolute/path/to/outputs" \
        ensemble_models="[a.csv,b.csv,c.csv]" triplet_size=195
    """
    # 앙상블만 요청 시: evaluate/evaluate_all 스킵하고 바로 앙상블
    if getattr(CFG, "ensemble_only", False) and getattr(CFG, "ensemble", False):
        prediction_dir = _join_prediction_dir(CFG)
        run_ensemble(CFG, prediction_dir)
        return

    if getattr(CFG, "evaluate_all", False):
        evaluate_all(CFG)
        # evaluate_all 후에도 앙상블 원하면 이어서 실행 (IVT components 포함 출력/저장)
        if getattr(CFG, "ensemble", False):
            prediction_dir = _join_prediction_dir(CFG)
            run_ensemble(CFG, prediction_dir)
    else:
        evaluate(CFG)


if __name__ == "__main__":
    run()
