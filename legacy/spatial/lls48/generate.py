import torch
import os
from torch.utils.data import DataLoader
import hydra
import warnings
import pandas as pd
import pickle
from models import TripletModel
from preprocess import get_folds
from augmentation import get_transforms
from dataset import TrainDataset
from helper import *
from global_var import config_name
from curconmix_release.spatial_protocol import (
    cfg_split,
    select_frame,
    metadata,
    verify_sidecar,
    validate_meta,
    write_sidecar,
)
from curconmix_release.spatial_training import row_order_sha256, sha256_file

os.environ["HYDRA_FULL_ERROR"] = "1"
warnings.filterwarnings("ignore")
from helper import *


def get_weight_path(CFG, fold=None):
    if CFG.split_selector == "LLS48-challenge":
        ckpt_dir = os.path.join(CFG.output_dir, "linear_probe_ckpts")
    else:
        ckpt_dir = os.path.join(CFG.output_dir, "checkpoints")

    if "challenge" in CFG.split_selector and fold is None:
        # 일반적인 challenge 모델 로딩 (fold 없이)
        filename = f"{CFG.model_name[:8]}_{CFG.target_size}_{CFG.exp}.pth"
    elif fold is not None:
        filename = f"fold{fold}_{CFG.model_name[:8]}_{CFG.target_size}_{CFG.exp}.pth"
    else:
        # fallback: 일반 challenge 모델 (fold 없이)
        filename = f"{CFG.model_name[:8]}_{CFG.target_size}_{CFG.exp}.pth"

    return os.path.join(ckpt_dir, filename)


def run_inference_loop(model, inference_folds, CFG):
    """Run inference and return predictions."""
    dataset = TrainDataset(
        inference_folds,
        transform=get_transforms(data="valid", CFG=CFG),
        inference=True,
        CFG=CFG,
    )
    loader = DataLoader(
        dataset,
        batch_size=CFG.valid_batch_size,
        shuffle=False,
        num_workers=CFG.nworkers,
        pin_memory=False,
        drop_last=False,
    )
    preds = inference_fn(loader, model, CFG.device, CFG)
    return preds


def inference(CFG):
    """
    Run inference and generate soft-labels or final predictions.

    Side Effects:
        - Saves predictions to CSV if CFG.inference is True
        - Saves soft-labels to CSV if CFG.inference is False
    """
    training_split = cfg_split(CFG, "lls48", required=not CFG.inference)
    if not CFG.inference and "challenge" in CFG.split_selector:
        raise ValueError(
            "Soft-label generation requires the nested cross-validation protocol."
        )
    if not CFG.inference:
        for fold in range(CFG.start_n_fold, CFG.n_fold):
            destination = os.path.join(
                CFG.output_dir,
                f"softlabels/sl_f{fold}_{CFG.model_name[:8]}_{CFG.target_size}_{CFG.exp}.csv",
            )
            if os.path.exists(destination) or os.path.exists(
                destination + ".provenance.json"
            ):
                raise ValueError(
                    "Soft-label output already exists; use a fresh run directory."
                )
    folds = get_folds(CFG)

    # Ensure output folders exist
    softlabels_dir = os.path.join(CFG.output_dir, "softlabels")
    predictions_dir = os.path.join(CFG.output_dir, "predictions")
    os.makedirs(softlabels_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)

    pred_df = pd.DataFrame()

    print(
        "\033[94m{}\033[0m".format(
            "Starting inference" if CFG.inference else "Generating soft-labels"
        )
    )

    if CFG.split_selector == "LLS48-challenge":
        for fold in range(CFG.start_n_fold, CFG.n_fold):
            model = TripletModel(CFG, model_name=CFG.model_name, pretrained=True).to(
                CFG.device
            )
            weights_path = get_weight_path(CFG, fold)
            model.load_state_dict(torch.load(weights_path), strict=True)
            print(f"Fold {fold}: Loaded weights from {weights_path}")

            trn_idx = folds[folds["fold"] == "train"].index
            test_idx = folds[folds["fold"] == "test"].index

            inference_folds = folds.loc[
                test_idx if CFG.inference else trn_idx
            ].reset_index(drop=True)
            preds = run_inference_loop(model, inference_folds, CFG)
            inference_folds[[str(c) for c in range(CFG.target_size)]] = preds

            if CFG.inference:
                # 🔥 각 fold별 prediction을 별도 CSV로 저장
                save_path = os.path.join(
                    predictions_dir,
                    f"fold_{fold}_{CFG.model_name[:8]}_{CFG.target_size}_{CFG.exp}.csv",
                )
                inference_folds.to_csv(save_path, index=False)
                print(f"Fold {fold} predictions saved at {save_path}")
            else:
                save_path = os.path.join(
                    softlabels_dir,
                    f"sl_f{fold}_{CFG.model_name[:8]}_{CFG.target_size}_{CFG.exp}.csv",
                )
                inference_folds.to_csv(save_path, index=False)
                print(f"Soft labels saved at {save_path}")

        # 🔥 fold별로 저장하고, 전체 합쳐서도 저장 (선택 사항)
        if CFG.inference:
            # 전체 fold 합쳐서 저장 (원하면)
            save_path = os.path.join(
                predictions_dir,
                f"{CFG.model_name[:8]}_{CFG.target_size}_{CFG.exp}_allfolds.csv",
            )
            print(f"Final predictions (all folds combined) saved at {save_path}")

    elif "challenge" in CFG.split_selector:
        model = TripletModel(CFG, model_name=CFG.model_name, pretrained=False).to(
            CFG.device
        )
        weights_path = get_weight_path(CFG)
        model.load_state_dict(torch.load(weights_path)["model"])
        print(f"Loaded model weights from {weights_path}")

        trn_idx = folds[folds["fold"] == "train"].index
        test_idx = folds[folds["fold"] == "test"].index

        inference_folds = folds.loc[test_idx if CFG.inference else trn_idx].reset_index(
            drop=True
        )
        preds = run_inference_loop(model, inference_folds, CFG)
        inference_folds[[str(c) for c in range(CFG.target_size)]] = preds

        if CFG.inference:
            save_path = os.path.join(
                predictions_dir, f"{CFG.model_name[:8]}_{CFG.target_size}_{CFG.exp}.csv"
            )
            inference_folds.to_csv(save_path, index=False)
            print(f"Predictions saved at {save_path}")
        else:
            save_path = os.path.join(
                softlabels_dir,
                f"sl_{CFG.model_name[:8]}_{CFG.target_size}_{CFG.exp}.csv",
            )
            inference_folds.to_csv(save_path, index=False)
            print(f"Soft labels saved at {save_path}")
    else:
        for fold in range(CFG.start_n_fold, CFG.n_fold):
            weights_path = get_weight_path(CFG, fold)
            if training_split:
                verify_sidecar(
                    weights_path,
                    training_split,
                    fold,
                    "student" if CFG.inference else "teacher",
                )
            model = TripletModel(CFG, model_name=CFG.model_name, pretrained=False).to(
                CFG.device
            )
            checkpoint = torch.load(
                weights_path,
                map_location=CFG.device,
                weights_only=training_split is not None,
            )
            if training_split:
                validate_meta(
                    checkpoint.get("meta"),
                    training_split,
                    fold,
                    "student" if CFG.inference else "teacher",
                )
            model.load_state_dict(checkpoint["model"], strict=True)
            print(f"Fold {fold}: Loaded weights from {weights_path}")

            trn_idx = folds[folds["fold"] != fold].index
            vld_idx = folds[folds["fold"] == fold].index

            inference_folds = folds.loc[
                vld_idx if CFG.inference else trn_idx
            ].reset_index(drop=True)
            if training_split:
                inference_folds = select_frame(
                    folds,
                    training_split,
                    fold,
                    "outer_test" if CFG.inference else "train",
                )
            preds = run_inference_loop(model, inference_folds, CFG)
            inference_folds[[str(c) for c in range(CFG.target_size)]] = preds

            if CFG.inference:
                pred_df = pd.concat([pred_df, inference_folds], ignore_index=True)
            else:
                save_path = os.path.join(
                    softlabels_dir,
                    f"sl_f{fold}_{CFG.model_name[:8]}_{CFG.target_size}_{CFG.exp}.csv",
                )
                inference_folds.to_csv(save_path, index=False)
                write_sidecar(
                    save_path,
                    metadata(training_split, fold, "softlabels"),
                    row_order_sha256=row_order_sha256(
                        inference_folds.to_dict("records")
                    ),
                    teacher_checkpoint_sha256=sha256_file(weights_path),
                )
                print(f"Soft labels for fold {fold} saved at {save_path}")

        if CFG.inference:
            save_path = os.path.join(
                predictions_dir, f"{CFG.model_name[:8]}_{CFG.target_size}_{CFG.exp}.csv"
            )
            pred_df.to_csv(save_path, index=False)
            print(f"Final predictions saved at {save_path}")


# Run the code
@hydra.main(version_base="1.3", config_path=".", config_name=config_name)
def generate(CFG):
    """
    Main function to run the inference.

    Args:
        CFG (OmegaConf): Configuration object.

    Returns:
        None
    """
    inference(CFG)


if __name__ == "__main__":
    generate()
