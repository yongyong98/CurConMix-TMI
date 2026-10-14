import os
import random
import numpy as np
import torch
from .ivtmetrics_cholec.recognition import Recognition as Recognition_cholec
from .ivtmetrics_lls.recognition import Recognition as Recognition_lls
from tqdm import tqdm
import pandas as pd

def cholect45_ivtmetrics_mAP(df):
    """
    Compute the official CholecT50 mAP score.

    Takes a dataframe with ground truth triplets and predictions.

    Metric calculation:
    - Aggregate per video over each fold
    - Mean of 5 folds

    Parameters:
    - df (pd.DataFrame): DataFrame with ground truth triplets and predictions.
    - CFG (object): Configuration object containing hyperparameters.

    Returns:
    float: Mean mAP value over 5 folds.

    """
    # Get the indexes of the 1st triplet/prediction columns
    tri0_idx = int(df.columns.get_loc("tri0"))
    pred0_idx = int(df.columns.get_loc("0"))

    # Initiate empty list to store the folds mAP
    ivt = []

    # Loop over the 5 folds
    for fold in range(5):
        # Initialize the ivt metric
        rec = Recognition_cholec(num_class=100)

        # Filter the fold and its corresponding videos
        fold_df = df[df["fold"] == fold]
        vids = fold_df.video.unique()

        # Loop over the videos
        for i, v in enumerate(vids):
            # Filter the video
            vid_df = fold_df[fold_df["video"] == v]

            rec.update(
                vid_df.iloc[:, tri0_idx : tri0_idx + 100].values,
                vid_df.iloc[:, pred0_idx : pred0_idx + 100].values,
            )

            rec.video_end()

        # Get the final mAP score for the fold
        ivt.append(rec.compute_video_AP("ivt", ignore_null=True)["mAP"])

    # Return the mean mAP value over 5 folds
    return np.mean(ivt)


def per_epoch_ivtmetrics(fold_df):
    """
    Compute per-epoch ivtmetrics.

    Parameters:
    - fold_df (pd.DataFrame): DataFrame with ground truth triplets and predictions for a fold.


    Returns:
    float: mAP score for the given fold.

    Example:
    ```python
    epoch_mAP = per_epoch_ivtmetrics(fold_df)
    ```

    """
    # Get the indexes of the 1st triplet/prediction columns
    tri0_idx = int(fold_df.columns.get_loc("tri0"))
    pred0_idx = int(fold_df.columns.get_loc("0"))

    # Initialize the ivt metric
    rec = Recognition_cholec(num_class=100)

    # Get unique videos
    vids = fold_df.video.unique()

    # Loop over the videos
    for i, v in enumerate(vids):
        # Filter the video
        vid_df = fold_df[fold_df["video"] == v]

        rec.update(
            vid_df.iloc[:, tri0_idx : tri0_idx + 100].values,
            vid_df.iloc[:, pred0_idx : pred0_idx + 100].values,
        )

        rec.video_end()

    # Get the final mAP score
    mAP = rec.compute_video_AP("ivt")["mAP"]

    return mAP


def lls48_ivtmetrics_mAP(df):
    """
    Compute the official CholecT50 mAP score.

    Takes a dataframe with ground truth triplets and predictions.

    Metric calculation:
    - Aggregate per video over each fold
    - Mean of 5 folds

    Parameters:
    - df (pd.DataFrame): DataFrame with ground truth triplets and predictions.
    - CFG (object): Configuration object containing hyperparameters.

    Returns:
    float: Mean mAP value over 5 folds.

    """
    # Get the indexes of the 1st triplet/prediction columns
    tri0_idx = int(df.columns.get_loc("tri0"))
    pred0_idx = int(df.columns.get_loc("0"))

    # Initiate empty list to store the folds mAP
    ivt = []

    # Loop over the 5 folds
    for fold in range(5):
        # Initialize the ivt metric
        rec = Recognition_lls(num_class=195)

        # Filter the fold and its corresponding videos
        fold_df = df[df["fold"] == fold]
        vids = fold_df.video.unique()

        # Loop over the videos
        for i, v in enumerate(vids):
            # Filter the video
            vid_df = fold_df[fold_df["video"] == v]

            rec.update(
                vid_df.iloc[:, tri0_idx : tri0_idx + 195].values,
                vid_df.iloc[:, pred0_idx : pred0_idx + 195].values,
            )

            rec.video_end()

        # Get the final mAP score for the fold
        ivt.append(rec.compute_video_AP("ivt")["mAP"])

    # Return the mean mAP value over 5 folds
    return np.mean(ivt)


def per_epoch_ivtmetrics_lls(fold_df):
    """
    Compute per-epoch ivtmetrics.

    Parameters:
    - fold_df (pd.DataFrame): DataFrame with ground truth triplets and predictions for a fold.


    Returns:
    float: mAP score for the given fold.

    Example:
    ```python
    epoch_mAP = per_epoch_ivtmetrics(fold_df)
    ```

    """
    # Get the indexes of the 1st triplet/prediction columns
    tri0_idx = int(fold_df.columns.get_loc("tri0"))
    pred0_idx = int(fold_df.columns.get_loc("0"))

    # Initialize the ivt metric
    rec = Recognition_lls(num_class=195)

    # Get unique videos
    vids = fold_df.video.unique()

    # Loop over the videos
    for i, v in enumerate(vids):
        # Filter the video
        vid_df = fold_df[fold_df["video"] == v]

        rec.update(
            vid_df.iloc[:, tri0_idx : tri0_idx + 195].values,
            vid_df.iloc[:, pred0_idx : pred0_idx + 195].values,
        )

        rec.video_end()

    # Get the final mAP score
    mAP = rec.compute_video_AP("ivt", ignore_null=True)["mAP"]

    return mAP


def cholect50_ivtmetrics_all_components(df, num_class=100, n_fold=5, ignore_null=False):
    """
    모든 component(i, v, t, iv, it, ivt)에 대해 CholecT50 데이터셋 기준 mAP 및 classwise AP 계산.

    Args:
        df (pd.DataFrame): 평가 대상 예측 결과.
        num_class (int): triplet size.
        n_fold (int): 총 fold 수.
        ignore_null (bool): null class 무시 여부.

    Returns:
        mean_mAPs (dict): 각 component 별 평균 mAP.
        std_mAPs (dict): 각 component 별 표준편차.
        classwise_AP_dfs (dict): 각 component 별 classwise AP dataframe.
    """
    tri0_idx = df.columns.get_loc("tri0")
    pred0_idx = df.columns.get_loc("0")

    components = ['i', 'v', 't', 'iv', 'it', 'ivt']
    mAPs = {comp: [] for comp in components}
    classwise_APs = {comp: [] for comp in components}

    for fold in tqdm(range(1, n_fold + 1), desc="Evaluating CholecT50 folds"):
        rec = Recognition_cholec(num_class=num_class)

        fold_df = df[df["fold"] == fold]
        for video in fold_df.video.unique():
            vid_df = fold_df[fold_df["video"] == video]

            rec.update(
                vid_df.iloc[:, tri0_idx:tri0_idx + num_class].values,
                vid_df.iloc[:, pred0_idx:pred0_idx + num_class].values,
            )
            rec.video_end()

        for comp in components:
            result = rec.compute_video_AP(comp, ignore_null=ignore_null) ## False로 설정 (null class도 같이 계산(baseline임))
            mAPs[comp].append(result["mAP"])
            classwise_APs[comp].append(result["AP"])

    mean_mAPs = {comp: np.nanmean(mAPs[comp]) for comp in components}
    std_mAPs = {comp: np.nanstd(mAPs[comp]) for comp in components}

    classwise_AP_dfs = {}
    # for comp in components:
    #     classwise_AP_df = pd.DataFrame({
    #         "class": [f"class_{i}" for i in range(num_class)],
    #         "AP": np.nanmean(classwise_APs[comp], axis=0)
    #     })
    #     classwise_AP_dfs[comp] = classwise_AP_df

    return mean_mAPs, std_mAPs, classwise_AP_dfs

from .ivtmetrics_lls.recognition import Recognition as Recognition_lls


def lls48_ivtmetrics_all_components(df, num_class=195, n_fold=5, ignore_null=False):
    """
    모든 component(i, v, t, iv, it, ivt)에 대해 LLS48 데이터셋 기준 mAP 및 classwise AP 계산.

    Args:
        df (pd.DataFrame): 평가 대상 예측 결과.
        num_class (int): triplet size.
        n_fold (int): 총 fold 수.
        ignore_null (bool): null class 무시 여부.

    Returns:
        mean_mAPs (dict): 각 component 별 평균 mAP.
        std_mAPs (dict): 각 component 별 표준편차.
        classwise_AP_dfs (dict): 각 component 별 classwise AP dataframe.
    """
    tri0_idx = df.columns.get_loc("tri0")
    pred0_idx = df.columns.get_loc("0")

    components = ['i', 'v', 't', 'iv', 'it', 'ivt']
    mAPs = {comp: [] for comp in components}
    classwise_APs = {comp: [] for comp in components}

    for fold in tqdm(range(1, n_fold + 1), desc="Evaluating LLS48 folds"):
        rec = Recognition_lls(num_class=num_class)

        fold_df = df[df["fold"] == fold]
        for video in fold_df.video.unique():
            vid_df = fold_df[fold_df["video"] == video]

            rec.update(
                vid_df.iloc[:, tri0_idx:tri0_idx + num_class].values,
                vid_df.iloc[:, pred0_idx:pred0_idx + num_class].values,
            )
            rec.video_end()

        for comp in components:
            result = rec.compute_video_AP(comp, ignore_null=ignore_null)
            mAPs[comp].append(result["mAP"])
            classwise_APs[comp].append(result["AP"])

    mean_mAPs = {comp: np.nanmean(mAPs[comp]) for comp in components}
    std_mAPs = {comp: np.nanstd(mAPs[comp]) for comp in components}

    classwise_AP_dfs = {}
    # for comp in components:
    #     classwise_AP_df = pd.DataFrame({
    #         "class": [f"class_{i}" for i in range(num_class)],
    #         "AP": np.nanmean(classwise_APs[comp], axis=0)
    #     })
    #     classwise_AP_dfs[comp] = classwise_AP_df

    return mean_mAPs, std_mAPs, classwise_AP_dfs
