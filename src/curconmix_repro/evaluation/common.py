import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from curconmix_repro.temporal.ivtmetrics_cholec.recognition import Recognition as CholecRecognition
from curconmix_repro.temporal.ivtmetrics_lls.recognition import Recognition as LLSRecognition


COMPONENTS = ["i", "v", "t", "iv", "it", "ivt"]


def _format_metric(mean_value: float, std_value: float) -> str:
    return f"{mean_value * 100:.2f} +- {std_value * 100:.2f}"


def _recognition_for_dataset(dataset: str, triplet_size: int):
    dataset = dataset.lower()
    if dataset in {"cholect45", "cholect50"}:
        return CholecRecognition(num_class=triplet_size)
    if dataset == "lls48":
        return LLSRecognition(num_class=triplet_size)
    raise ValueError(f"Unsupported dataset: {dataset}")


def evaluate_prediction_csv_detailed(csv_path: Path, dataset: str, triplet_size: int, n_fold: int, ignore_null: bool):
    df = pd.read_csv(csv_path)
    tri0_idx = int(df.columns.get_loc("tri0"))
    pred0_idx = int(df.columns.get_loc("0"))

    fold_scores = {component: [] for component in COMPONENTS}

    print(f"[evaluate] file={csv_path}")
    print(
        f"[evaluate] shape={df.shape}, dataset={dataset}, "
        f"triplet_size={triplet_size}, ignore_null={ignore_null}"
    )

    for fold in tqdm(range(n_fold), desc="Processing folds"):
        rec = _recognition_for_dataset(dataset, triplet_size)
        fold_df = df[df["fold"] == fold]
        for video_name in fold_df.video.unique():
            video_df = fold_df[fold_df["video"] == video_name]
            rec.update(
                video_df.iloc[:, tri0_idx : tri0_idx + triplet_size].values,
                video_df.iloc[:, pred0_idx : pred0_idx + triplet_size].values,
            )
            rec.video_end()

        print(f"\n--- Fold {fold} ---")
        for component in COMPONENTS:
            result = rec.compute_video_AP(component, ignore_null=ignore_null)
            fold_scores[component].append(result["mAP"])
            print(f"{component}: {result['mAP'] * 100:.2f}")

    mean_scores = {component: float(np.nanmean(values)) for component, values in fold_scores.items()}
    std_scores = {component: float(np.nanstd(values)) for component, values in fold_scores.items()}

    print("\n[result]")
    for component in COMPONENTS:
        print(f"{component}: {_format_metric(mean_scores[component], std_scores[component])}")

    return fold_scores, mean_scores, std_scores


def evaluate_prediction_csv(csv_path: Path, dataset: str, triplet_size: int, n_fold: int, ignore_null: bool):
    _, mean_scores, std_scores = evaluate_prediction_csv_detailed(
        csv_path=csv_path,
        dataset=dataset,
        triplet_size=triplet_size,
        n_fold=n_fold,
        ignore_null=ignore_null,
    )
    return mean_scores, std_scores


def update_summary(summary_csv: Path, experiment_name: str, mean_scores, std_scores):
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    row = {"experiment": experiment_name}
    for component in COMPONENTS:
        row[component] = _format_metric(mean_scores[component], std_scores[component])

    rows = []
    if summary_csv.exists():
        with summary_csv.open("r", encoding="utf-8-sig", newline="") as fp:
            rows = list(csv.DictReader(fp))
        rows = [existing for existing in rows if existing.get("experiment") != experiment_name]

    rows.append(row)
    rows.sort(key=lambda item: item["experiment"])

    with summary_csv.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["experiment", *COMPONENTS])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[summary] updated {summary_csv}")


def read_summary_row(summary_csv: Path, experiment_name: str):
    with summary_csv.open("r", encoding="utf-8-sig", newline="") as fp:
        for row in csv.DictReader(fp):
            if row.get("experiment") == experiment_name:
                return row
    raise ValueError(f"Experiment '{experiment_name}' not found in {summary_csv}")


def parse_mean(metric_value: str) -> float:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", metric_value)
    if not match:
        raise ValueError(f"Could not parse metric value from '{metric_value}'")
    return float(match.group())


def compare_to_reference_summary(
    reference_summary_csv: Path,
    reference_experiment: str,
    repro_summary_csv: Path,
    repro_experiment: str,
):
    reference_row = read_summary_row(reference_summary_csv, reference_experiment)
    repro_row = read_summary_row(repro_summary_csv, repro_experiment)

    print("\n[compare] reference summary vs reproduced")
    for component in COMPONENTS:
        reference_value = parse_mean(reference_row[component])
        repro_value = parse_mean(repro_row[component])
        print(
            f"{component}: reference={reference_value:.2f}, "
            f"reproduced={repro_value:.2f}, delta={repro_value - reference_value:+.2f}"
        )


def compare_to_reference_metrics(reference_metrics, mean_scores):
    print("\n[compare] reference metrics vs reproduced")
    for component in COMPONENTS:
        reference_value = float(reference_metrics[component])
        repro_value = float(mean_scores[component] * 100.0)
        print(
            f"{component}: reference={reference_value:.2f}, "
            f"reproduced={repro_value:.2f}, delta={repro_value - reference_value:+.2f}"
        )


def build_ensemble_prediction(member_csvs, output_csv: Path, triplet_size: int):
    if not member_csvs:
        raise ValueError("member_csvs must not be empty")

    base_df = None
    pred0_idx = None
    summed = None
    key_columns = None

    for csv_path in member_csvs:
        if not csv_path.exists():
            raise FileNotFoundError(f"Member prediction CSV not found: {csv_path}")
        df = pd.read_csv(csv_path)

        if pred0_idx is None:
            pred0_idx = int(df.columns.get_loc("0"))
            key_columns = [column for column in ["fold", "video", "frame", "frame_id"] if column in df.columns]
            base_df = df.copy()
            summed = df.iloc[:, pred0_idx : pred0_idx + triplet_size].values.astype(np.float64)
            continue

        if key_columns and not base_df[key_columns].equals(df[key_columns]):
            raise ValueError(f"Prediction row mismatch detected in {csv_path}")

        summed += df.iloc[:, pred0_idx : pred0_idx + triplet_size].values

    averaged = summed / float(len(member_csvs))
    base_df.iloc[:, pred0_idx : pred0_idx + triplet_size] = averaged
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    base_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"[ensemble] saved {output_csv}")
