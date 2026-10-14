# evaluate.py (cholect45 전용, triplet_size 기준 / ensemble-all 지원)
import os
import pandas as pd
import hydra
import warnings
import traceback
from typing import Any, Dict
from utils import cholect45_ivtmetrics_mAP, cholect45_ivtmetrics_mAP_all
from global_var import config_name

warnings.filterwarnings('ignore')


# ----------------------- helpers -----------------------
def _prediction_dir(CFG):
    """Return folder where prediction CSVs live."""
    folder = "predictions" if getattr(CFG, "inference", False) else "oofs"
    return os.path.join(CFG.output_dir, folder)


def _build_ensemble_name(file_list):
    """Use the historical LLS48 naming rule for optional ensemble exports."""
    parts = []
    for filename in file_list:
        stem = os.path.splitext(os.path.basename(filename))[0]
        segments = stem.split('_')[1:] if '_' in stem else [stem]
        parts.append('_'.join(segments))
    return "__".join(parts)


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


def _looks_like_prediction_csv(path, triplet_size=None):
    """
    요약/클래스별 AP CSV 등을 제외하고, '예측 CSV'만 True.
      1) 파일명에 summary/classwise_AP 들어가면 제외
      2) 헤더에 '0' 컬럼이 있어야 함
      3) triplet_size가 주어지면 최소한 '0'부터 시작하는 예측 컬럼이 있어야 함
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
    if not os.path.isdir(base_dir):
        return []
    return [
        f for f in os.listdir(base_dir)
        if f.lower().endswith(".csv")
        and _looks_like_prediction_csv(os.path.join(base_dir, f), triplet_size=triplet_size)
    ]


def _detect_pred_layout(df, triplet_size_hint=None):
    """
    예측 열 레이아웃 감지(디버깅용 정보):
      - pred0_idx : '0' 컬럼의 위치
      - pred_count: '0','1','2' ... 연속 열 개수(대략)
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


def _fmt_mean_std(mean_val: float, std_val: float, as_percent: bool = True, decimals: int = 2) -> str:
    """
    mean/std를 'mean ± std' 문자열로 포맷.
    - as_percent=True면 0~1 스케일을 퍼센트(%)로 변환하여 소수 N자리.
    - 예: 0.544 -> '54.40 ± 2.50'
    """
    if as_percent:
        mean_val = mean_val * 100.0
        std_val = std_val * 100.0
    return f"{mean_val:.{decimals}f} ± {std_val:.{decimals}f}"


def _normalize_keys(d: dict):
    """
    dict의 키를 소문자로 정규화하여 접근 편의 제공.
    예: {'IVT': 0.55, 'i': 0.50} -> {'ivt':0.55, 'i':0.50}
    """
    return {str(k).lower(): v for k, v in d.items()}


def _component_order():
    """CSV 컬럼 순서를 고정: i, v, t, iv, it, ivt"""
    return ["i", "v", "t", "iv", "it", "ivt"]


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
    # 보기 좋게 컬럼 순서 고정
    ordered_cols = ["experiment"] + _component_order()
    for col in ordered_cols:
        if col not in combined.columns:
            combined[col] = ""
    combined = combined.reindex(columns=ordered_cols)
    combined.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"📄 요약 CSV 업데이트: {summary_path} (총 {len(combined)}개 실험)")


# ----------------------- core evals -----------------------
def evaluate(CFG):
    """
    개별 실험 점수(mAP) 출력 + (옵션) 앙상블 점수 출력.
    """
    triplet_size = _get_triplet_size(CFG)
    folder = _prediction_dir(CFG)
    prediction_files = _list_prediction_csvs(folder, triplet_size=triplet_size)

    # 개별 실험 평가
    if not getattr(CFG, "ensemble_only", False):
        for pred_df in prediction_files:
            path = os.path.join(folder, pred_df)
            try:
                df = pd.read_csv(path)
                experiment = pred_df.split(".")[0].split('_')[1:]
                experiment_name = ('_').join(experiment) if experiment else os.path.splitext(pred_df)[0]

                pred0_idx, pred_count, shape = _detect_pred_layout(df, triplet_size_hint=triplet_size)
                print(f"\n[Evaluate] Experiment='{experiment_name}'")
                print(f"  - File         : {path}")
                print(f"  - DF shape     : {shape}")
                print(f"  - Pred cols    : start='{pred0_idx}', count='{pred_count}', triplet_size='{triplet_size}'")

                score = cholect45_ivtmetrics_mAP(df, CFG)
                score = _to_score(score)
                print(f"  => mAP = {round(score * 100, 2)}%")

            except Exception as e:
                print("\n❌ Evaluation failed on this experiment/file:")
                print(f"  - File       : {path}")
                print(f"  - Experiment : {pred_df}")
                print("  - Reason     :", repr(e))
                traceback.print_exc()

    # 앙상블 (옵션)
    if getattr(CFG, "ensemble", False):
        run_ensemble(CFG, folder)


def run_ensemble(CFG, folder: str):
    """
    CFG.ensemble_models에 명시된 CSV들만 읽어서 앙상블 수행.
    CSV는 sigmoid 확률(0~1)이라고 가정.
    evaluate_all=True이면 IVT 각 컴포넌트 mean±std도 함께 계산/출력/요약 반영/클래스별 저장.
    """
    try:
        if not getattr(CFG, "ensemble_models", None):
            raise ValueError("CFG.ensemble_models가 비어있습니다.")

        triplet_size = _get_triplet_size(CFG)
        use_avg = _get_ensemble_avg(CFG)
        print(f'triplet_size: {triplet_size}, use_avg: {use_avg}')

        preds = None
        num_models = len(CFG.ensemble_models)
        df_template = None
        pred0_idx = None

        print("\n[Ensemble] 모델 구성:")
        for model in CFG.ensemble_models:
            mpath = os.path.join(folder, model)
            print(f"  - {mpath}")
            if not os.path.isfile(mpath):
                raise FileNotFoundError(f"예측 CSV를 찾을 수 없습니다: {mpath}")
            df_m = pd.read_csv(mpath)
            if pred0_idx is None:
                if "0" not in df_m.columns:
                    raise KeyError(f"'0' 컬럼을 찾을 수 없습니다: {mpath}")
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
            out_csv = os.path.join(folder, f"{ens_name}__ENSEMBLED.csv")
            df_template.to_csv(out_csv, index=False)
            print(f"  => Saved ensembled CSV: {out_csv}")

        # ---------------- Evaluate-All 모드: IVT 컴포넌트별 ----------------
        if getattr(CFG, "evaluate_all", False):
            mean_mAPs, std_mAPs, classwise_AP_dfs = cholect45_ivtmetrics_mAP_all(df_template, CFG)

            print("  => Ensemble (IVT components):")
            for comp in _component_order():
                if comp in mean_mAPs and comp in std_mAPs:
                    mean_pct = mean_mAPs[comp] * 100.0
                    std_pct = std_mAPs[comp] * 100.0
                    print(f"     - {comp} mAP: {mean_pct:.2f}% (±{std_pct:.2f}%)")

            # (옵션) 클래스별 AP 저장
            if getattr(CFG, "save_classwise_ap", False):
                ens_name = _build_ensemble_name(CFG.ensemble_models)
                for comp, classwise_AP_df in classwise_AP_dfs.items():
                    out_csv = os.path.join(folder, f'{ens_name}_{comp}_classwise_AP.csv')
                    classwise_AP_df.to_csv(out_csv, index=False)
                    print(f"     - Saved ensemble {comp} classwise AP: {out_csv}")

            # (옵션) 요약 CSV에 앙상블 행 추가
            if getattr(CFG, "save_summary_csv", False):
                summary_path = os.path.join(folder, "all_experiments_summary.csv")
                ens_name = _build_ensemble_name(CFG.ensemble_models)
                row = {"experiment": f"ENSEMBLE__{ens_name}"}
                means_l = _normalize_keys(mean_mAPs)
                stds_l = _normalize_keys(std_mAPs)
                for comp in _component_order():
                    if comp in means_l and comp in stds_l:
                        row[comp] = _fmt_mean_std(means_l[comp], stds_l[comp], as_percent=True, decimals=2)
                    else:
                        row[comp] = ""
                _append_summary_row(summary_path, row)

    except Exception as e:
        print("\n❌ Ensemble didn't work.")
        print("  - Reason :", repr(e))
        traceback.print_exc()


def evaluate_all(CFG):
    """
    각 실험에 대해 cholect45_ivtmetrics_mAP_all 계산 후 요약 저장(옵션).
    플래그:
      - save_summary_csv=true  (요약 CSV 저장/병합)
      - force_recompute=true   (요약에 있어도 다시 계산)
      - save_classwise_ap=true (컴포넌트별 AP CSV 저장)
    """
    triplet_size = _get_triplet_size(CFG)

    folder_path = _prediction_dir(CFG)
    prediction_files = _list_prediction_csvs(folder_path, triplet_size=triplet_size)

    summary_path = os.path.join(folder_path, "all_experiments_summary.csv")
    want_summary_csv = bool(getattr(CFG, "save_summary_csv", False) or getattr(CFG, "save_summary", False))
    save_classwise_ap = bool(getattr(CFG, "save_classwise_ap", False))
    force_recompute = bool(getattr(CFG, "force_recompute", False))

    # 기존 요약 CSV 로딩 (있으면)
    existing_exps = set()
    prev_df = None
    if os.path.isfile(summary_path):
        try:
            prev_df = pd.read_csv(summary_path)
            if "experiment" in prev_df.columns:
                if not force_recompute:
                    existing_exps = set(prev_df["experiment"].astype(str).tolist())
                    print(f"✅ 기존 요약 CSV 발견: {summary_path} (이미 평가됨 {len(existing_exps)}개) — force_recompute=False 이므로 스킵 적용")
                else:
                    print(f"✅ 기존 요약 CSV 발견: {summary_path} — force_recompute=True 이므로 전부 재평가/갱신")
            else:
                print(f"⚠️ 기존 요약 CSV에 'experiment' 컬럼이 없습니다. 새로 작성합니다: {summary_path}")
                prev_df = None
        except Exception as e:
            print(f"⚠️ 기존 요약 CSV 로드 실패({summary_path}): {e}. 새로 작성합니다.")

    # 평가 루프
    new_rows = []
    evaluated_any = False
    comp_order = _component_order()

    for pred_df in prediction_files:
        path = os.path.join(folder_path, pred_df)
        try:
            df = pd.read_csv(path)
            experiment_parts = pred_df.split(".")[0].split('_')[1:]
            experiment_name = ('_').join(experiment_parts) if experiment_parts else os.path.splitext(pred_df)[0]

            # 스킵 기준: 기존 요약에 있고, force_recompute=False
            if not force_recompute and experiment_name in existing_exps:
                print(f"⏭️  스킵: 이미 평가된 실험입니다 → {experiment_name}")
                continue

            pred0_idx, pred_count, shape = _detect_pred_layout(df, triplet_size_hint=triplet_size)
            print(f"\n[EvaluateAll] Experiment='{experiment_name}'")
            print(f"  - File         : {path}")
            print(f"  - DF shape     : {shape}")
            print(f"  - Pred cols    : start='{pred0_idx}', count='{pred_count}', triplet_size='{triplet_size}'")

            mean_mAPs, std_mAPs, classwise_AP_dfs = cholect45_ivtmetrics_mAP_all(df, CFG)

            # (옵션) 클래스별 AP 저장
            if save_classwise_ap:
                for comp, classwise_AP_df in classwise_AP_dfs.items():
                    out_csv = os.path.join(folder_path, f'{experiment_name}_{comp}_classwise_AP.csv')
                    classwise_AP_df.to_csv(out_csv, index=False)

            # 행 구성: experiment + i,v,t,iv,it,ivt 컬럼(각각 "mean ± std" 포맷, 퍼센트)
            row = {"experiment": experiment_name}
            means_l = _normalize_keys(mean_mAPs)
            stds_l = _normalize_keys(std_mAPs)

            print(f"{experiment_name}의 결과:")
            for comp in comp_order:
                if comp in means_l and comp in stds_l:
                    txt = _fmt_mean_std(means_l[comp], stds_l[comp], as_percent=True, decimals=2)
                    row[comp] = txt
                    print(f"{comp} mAP: {txt}%".replace("%%", "%"))
                else:
                    row[comp] = ""
                    print(f"{comp} mAP: (not available)")

            new_rows.append(row)
            evaluated_any = True

        except Exception as e:
            print("\n❌ Evaluation failed on this experiment/file:")
            print(f"  - File       : {path}")
            print(f"  - Experiment : {pred_df}")
            print("  - Reason     :", repr(e))
            traceback.print_exc()

    # 요약 CSV 저장/갱신
    if want_summary_csv:
        new_df = pd.DataFrame(new_rows)

        # 기존 존재 + 재평가 모드에 따라 병합/덮어쓰기 로직
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
            combined = new_df if new_df is not None and not new_df.empty else pd.DataFrame(columns=["experiment"] + _component_order())

        # 보기 좋게 컬럼 순서 정리: experiment + (i,v,t,iv,it,ivt)
        ordered_cols = ["experiment"] + _component_order()
        for col in ordered_cols:
            if col not in combined.columns:
                combined[col] = ""  # 누락된 컬럼 보강
        combined = combined.reindex(columns=ordered_cols)

        combined.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"\n📄 요약 CSV 저장/업데이트 완료: {summary_path} (총 {len(combined)}개 실험)")
    else:
        print("\nℹ️ save_summary_csv=false: 요약 CSV는 저장하지 않았습니다.")

    if not evaluated_any:
        print("✨ 새로 평가할 실험이 없습니다. (모두 스킵됨)")


# ----------------------- entry -----------------------
@hydra.main(version_base="1.3", config_path=".", config_name=config_name)
def run(CFG):
    """
    사용 예시
    ----
    # 전체 평가 + 요약 생성(새 것만 추가)
    python evaluate.py evaluate_all=true save_summary_csv=true inference=true output_dir="/absolute/path/to/outputs" triplet_size=100

    # 기존 요약이 있어도 전부 다시 평가하여 갱신(덮어쓰기)
    python evaluate.py evaluate_all=true save_summary_csv=true force_recompute=true triplet_size=100

    # 클래스별 AP도 같이 저장
    python evaluate.py evaluate_all=true save_summary_csv=true save_classwise_ap=true triplet_size=100

    # 앙상블만 + IVT 컴포넌트까지 요약 추가
    python evaluate.py ensemble=true ensemble_only=true evaluate_all=true save_summary_csv=true save_classwise_ap=true ^
        ensemble_models="[a.csv,b.csv,c.csv]" output_dir="/absolute/path/to/outputs" triplet_size=100
    """
    # 앙상블만 요청 시: 개별/evaluate_all 루프 건너뛰고 바로 앙상블 (IVT 컴포넌트 포함)
    if getattr(CFG, "ensemble_only", False) and getattr(CFG, "ensemble", False):
        folder = _prediction_dir(CFG)
        run_ensemble(CFG, folder)
        return

    if getattr(CFG, "evaluate_all", False):
        evaluate_all(CFG)
        # evaluate_all 후에도 앙상블 원하면 이어서 실행 (IVT components 포함 출력/저장)
        if getattr(CFG, "ensemble", False):
            folder = _prediction_dir(CFG)
            run_ensemble(CFG, folder)
    else:
        evaluate(CFG)


if __name__ == "__main__":
    print("config_name: ",config_name)
    run()
