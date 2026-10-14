# Empty dataframe to store oofs and metrics
import gc
import os
import time
import torch
import pandas as pd
import neptune.new as neptune
from sklearn.metrics import average_precision_score
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR
from torch.cuda import amp
from sklearn.utils.class_weight import compute_class_weight
import torch.backends.cudnn as cudnn

from augmentation import *
from models import *
from preprocess import get_folds
from utils import *
from helper import *
from dataset import *
from losses import *
from losses_weighted import *
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from safe_wandb import init_safe_wandb_from_cfg
from curconmix_release.spatial_protocol import (
    training_gate,
    checkpoint_metadata,
    verify_sidecar,
    validate_meta,
    write_sidecar,
    is_better_inner_ap,
    inner_validation_ap,
)


def _infer_model_tag(model_name: str) -> str:
    if not model_name:
        return "model"
    tag = os.path.basename(str(model_name)).split(".")[0]
    return tag.replace(" ", "_").replace("/", "_").replace("\\", "_")


def _ensure_pkl(name: str) -> str:
    return name if str(name).endswith(".pkl") else f"{name}.pkl"


def _as_score(value):
    if isinstance(value, tuple):
        return float(value[0])
    return float(value)


def select_feature_matrix_paths(CFG, fold: int):
    """Resolve user-supplied, fold-matched contrastive caches without local fallbacks."""
    feature_dir = getattr(CFG, "feature_dir", None)
    model_tag = getattr(CFG, "model_tag", None)
    if not feature_dir or not str(feature_dir).strip():
        raise ValueError(
            "Set feature_dir to the directory containing fold-matched caches."
        )
    if not model_tag or not str(model_tag).strip():
        raise ValueError("Set model_tag to the filename prefix of the supplied caches.")
    feature_dir = str(feature_dir).replace("\\", "/")
    feat_base = _ensure_pkl(
        getattr(CFG, "feature_file_name", "trainset_features_threshold_10.pkl")
    )
    sim_base = _ensure_pkl(
        getattr(CFG, "cos_sim_matrix_file_name", "similarity_matrix.pkl")
    )
    return (
        os.path.join(feature_dir, f"{model_tag}_fold{fold}_{feat_base}"),
        os.path.join(feature_dir, f"{model_tag}_fold_{fold}_{sim_base}"),
    )


def validate_model_image_size(CFG):
    model_name = str(getattr(CFG, "model_name", "")).lower()
    height = int(getattr(CFG, "height", 0))
    width = int(getattr(CFG, "width", 0))

    if "224" in model_name and (height != 224 or width != 224):
        raise ValueError(
            f"Model '{CFG.model_name}' expects 224x224 inputs, "
            f"but CFG.height={height}, CFG.width={width}."
        )
    if "384" in model_name and (height != 384 or width != 384):
        raise ValueError(
            f"Model '{CFG.model_name}' expects 384x384 inputs, "
            f"but CFG.height={height}, CFG.width={width}."
        )


def infer_wandb_phase(CFG, ssl_phase=False):
    if ssl_phase or getattr(CFG, "ssl", False):
        return "ssl_pretrain"
    if getattr(CFG, "distill", False):
        return "student"
    if getattr(CFG, "pretrained_ssl", False):
        return "teacher"
    return "train"


def log_neptune(run, key, value, fold=None):
    """Neptune 로그 기록을 간소화하는 함수"""
    if fold is not None:
        key = f"{key}{fold}"
    run[key].log(value)


def train_cross_val(CFG):
    stage = "student" if CFG.distill else "teacher"
    training_split = training_gate(CFG, "lls48", stage)
    start_time = time.time()
    seed_torch(seed=CFG.seed)
    validate_model_image_size(CFG)

    summary_dir = Path(CFG.output_dir) / "summary_dir"
    log_folder_dir = Path(CFG.output_dir) / CFG.exp
    checkpoint_dir = Path(CFG.output_dir) / "checkpoints"
    summary_dir.mkdir(parents=True, exist_ok=True)
    log_folder_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if CFG.debug:
        CFG.epochs = 1
        CFG.neplog = False

    wandb_run = init_safe_wandb_from_cfg(CFG, infer_wandb_phase(CFG))

    if CFG.neplog:
        run = neptune.init(project=CFG.neptune_project, api_token=CFG.neptune_api_token)
        for key, value in {
            "Model": CFG.model_name,
            "imsize": CFG.height,
            "LR": CFG.lr,
            "bs": CFG.batch_size,
            "Epochs": CFG.epochs,
            "SD": CFG.SD,
            "T_0": CFG.T_0,
            "min_lr": CFG.min_lr,
            "seed": str(CFG.seed),
            "split": CFG.challenge_split,
            "tsize": CFG.target_size,
            "smooth": CFG.smooth,
            "exp": CFG.exp,
        }.items():
            run[key].log(value)

    folds = get_folds(CFG)
    transform = select_transforms(data="train", CFG=CFG)
    print_training_info(folds, CFG)

    for fold in range(CFG.start_n_fold, CFG.n_fold):
        if fold not in CFG.trn_fold:
            continue

        print(f"\033[92m{'-' * 8} Fold {fold + 1} / {CFG.n_fold}\033[0m")
        log_path = log_folder_dir / f"{CFG.exp}_fold_{fold}.csv"
        log_records = []

        model = TripletModel(CFG, CFG.model_name, pretrained=CFG.pretrained).to(
            CFG.device
        )
        if CFG.pretrained_ssl and CFG.pretrained_exp != "myexp":
            weights_path = (
                Path(CFG.output_dir)
                / f"checkpoints/fold{fold}_{CFG.model_name[:8]}_{CFG.pretrained_exp}.pth"
            )
            print(f"Using Pre-trained Checkpoint weight file: {weights_path}")
            verify_sidecar(weights_path, training_split, fold, "pretrain")
            ssl_checkpoint = torch.load(
                weights_path, map_location=CFG.device, weights_only=True
            )
            validate_meta(ssl_checkpoint.get("meta"), training_split, fold, "pretrain")
            ssl_state_dict = ssl_checkpoint["model"]
            model.load_state_dict(ssl_state_dict, strict=False)

        train_folds, valid_folds, valid_folds_temp = get_dataframes(
            folds, fold, CFG, training_split
        )

        if CFG.distill:
            train_folds = apply_self_distillation(fold, train_folds, CFG)
        else:
            print("Using default Train dataset")

        train_dataset = TrainDataset(
            train_folds, CFG, transform=get_transforms(data="train", CFG=CFG), fold=fold
        )
        valid_dataset = TrainDataset(
            valid_folds, CFG, transform=get_transforms(data="valid", CFG=CFG), fold=fold
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=CFG.batch_size,
            shuffle=CFG.shuffle,
            num_workers=CFG.nworkers,
            pin_memory=True,
            drop_last=True,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=CFG.batch_size,
            shuffle=False,
            num_workers=CFG.nworkers,
            pin_memory=False,
            drop_last=False,
        )

        optimizer = AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=CFG.epochs + 1, T_mult=1000, eta_min=CFG.min_lr
        )
        # criterion = nn.BCEWithLogitsLoss(reduction="sum").to(CFG.device) if CFG.loss == 'bce' \
        #             else nn.MultiLabelSoftMarginLoss(reduction="sum").to(CFG.device)
        criterion = nn.BCEWithLogitsLoss(reduction="sum").to(CFG.device)

        best_score = float("-inf")
        scaler = amp.GradScaler()

        print("Start training")
        if CFG.mixup:
            print("Using Mixup")

        for epoch in range(CFG.epochs):
            epoch_start = time.time()
            if CFG.mixup:
                avg_loss = train_mixup(
                    train_loader,
                    model,
                    CFG,
                    criterion,
                    optimizer,
                    epoch,
                    scheduler,
                    CFG.device,
                    scaler=scaler,
                )
            else:
                print("Default training without Mixup")
                avg_loss = train_fn(
                    train_loader,
                    model,
                    CFG,
                    criterion,
                    optimizer,
                    epoch,
                    scheduler,
                    CFG.device,
                    scaler=scaler,
                )

            avg_val_loss, preds = valid_fn(
                valid_loader, model, CFG, criterion, CFG.device
            )
            scheduler.step()

            valid_folds_temp[[str(c) for c in range(CFG.target_size)]] = preds
            valid_folds_temp["fold"] = fold  # fold 정보 추가

            cholect45_epoch_CV = inner_validation_ap(
                valid_folds_temp, training_split, fold, per_epoch_ivtmetrics, CFG
            )

            log_records.append(
                {
                    "epoch": epoch,
                    "train_loss": avg_loss,
                    "val_loss": avg_val_loss,
                    "cmAP": cholect45_epoch_CV,
                }
            )

            if CFG.neplog:
                log_neptune(run, "tloss", avg_loss, fold)
                log_neptune(run, "val_loss", avg_val_loss, fold)
                log_neptune(run, "cmAP", cholect45_epoch_CV, fold)

            wandb_run.log(
                {
                    "phase": infer_wandb_phase(CFG),
                    "fold": fold,
                    "epoch": epoch,
                    "train_loss": avg_loss,
                    "val_loss": avg_val_loss,
                    "cmAP": cholect45_epoch_CV,
                    "lr": scheduler.get_last_lr()[0],
                }
            )

            if is_better_inner_ap(cholect45_epoch_CV, best_score):
                best_score = cholect45_epoch_CV
                save_path = (
                    checkpoint_dir
                    / f"fold{fold}_{CFG.model_name[:8]}_{CFG.target_size}_{CFG.exp}.pth"
                )
                meta = checkpoint_metadata(
                    CFG, training_split, fold, stage, epoch, best_score
                )
                torch.save(
                    {
                        "model": model.state_dict(),
                        "preds": torch.as_tensor(preds),
                        "meta": meta,
                    },
                    save_path,
                )
                write_sidecar(save_path, meta)
                wandb_run.log(
                    {
                        "phase": infer_wandb_phase(CFG),
                        "fold": fold,
                        "best_cmAP": best_score,
                        "best_checkpoint": str(save_path),
                    }
                )

            print(
                raw_line.format(
                    epoch,
                    avg_loss,
                    avg_val_loss,
                    cholect45_epoch_CV,
                    (time.time() - epoch_start) / 60,
                )
            )

        # CSV 로그 저장
        pd.DataFrame(log_records).to_csv(log_path, index=False)
        # Create loss plot directory
        loss_dir = Path(CFG.output_dir) / "loss"
        loss_dir.mkdir(parents=True, exist_ok=True)

        # Plot train/val loss curve
        loss_df = pd.DataFrame(log_records)
        plt.figure(figsize=(8, 5))
        plt.plot(loss_df["epoch"], loss_df["train_loss"], label="Train Loss")
        plt.plot(loss_df["epoch"], loss_df["val_loss"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Fold {fold} - Loss Curve")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(loss_dir / f"{CFG.exp}_fold{fold}_loss.png")
        plt.close()

        del model, train_loader, valid_loader
        torch.cuda.empty_cache()

    print(
        "Training completed. Checkpoints selected on inner-validation AP only; outer-test evaluation was not run."
    )

    wandb_run.update_summary(
        {
            "phase": infer_wandb_phase(CFG),
            "selection_partition": "inner_validation",
            "training_minutes": (time.time() - start_time) / 60,
        }
    )
    wandb_run.finish()

    print(f"Training time: {(time.time() - start_time) / 60:.2f} minutes")


# ================================================================================================
def linear_probe_lls48(CFG):
    raise NotImplementedError(
        "Legacy cross-dataset linear probing is not supported by the nested-validation workflow."
    )


def train_cross_challenge(CFG):
    raise NotImplementedError(
        "Legacy challenge training is disabled; use nested cross-validation."
    )


def select_transforms(data, CFG):
    print("Using Default Augmentation Methods")
    return get_transforms(data="train", CFG=CFG)


def should_save_ssl_checkpoint(CFG, epoch: int) -> bool:
    if epoch == CFG.epochs - 1:
        return True
    return bool(getattr(CFG, "save_all_ssl_epochs", False))


def train_cross_val_SSL(CFG):
    training_split = training_gate(CFG, "lls48", "pretrain")
    start_time = time.time()
    seed_torch(seed=CFG.seed)
    validate_model_image_size(CFG)

    summary_dir = os.path.join(CFG.output_dir, "summary_dir")
    os.makedirs(summary_dir, exist_ok=True)

    log_folder_dir = os.path.join(CFG.output_dir, CFG.exp)
    os.makedirs(log_folder_dir, exist_ok=True)

    checkpoint_dir = os.path.join(CFG.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # DEBUG: Faster iteration
    if CFG.debug:
        CFG.epochs = 1
        CFG.neplog = False

    wandb_run = init_safe_wandb_from_cfg(CFG, infer_wandb_phase(CFG, ssl_phase=True))

    if CFG.neplog:
        run = neptune.init(
            project=CFG.neptune_project,
            api_token=CFG.neptune_api_token,
        )
        run["Model"].log(CFG.model_name)
        run["imsize"].log(CFG.height)
        run["LR"].log(CFG.lr)
        run["bs"].log(CFG.batch_size)
        run["Epochs"].log(CFG.epochs)
        run["SD"].log(CFG.SD)
        run["T_0"].log(CFG.T_0)
        run["min_lr"].log(CFG.min_lr)
        run["seed"].log(str(CFG.seed))
        run["split"].log(CFG.challenge_split)
        run["tsize"].log(CFG.target_size)
        run["smooth"].log(CFG.smooth)
        run["exp"].log(CFG.exp)

    folds = get_folds(CFG)  ## dataset에 따른 fold 나누고, label setting
    transform = select_transforms(data="train", CFG=CFG)
    print_training_info(folds, CFG)
    label_key_list = CFG.label_order

    for fold in range(CFG.start_n_fold, CFG.n_fold):
        if fold in CFG.trn_fold:
            if CFG.feature_batch and (
                CFG.method == "supcon" or CFG.method == "curriculum_supcon"
            ):
                print("Using separated head Supcon model")
                model = FeatureSupConModel(
                    CFG, CFG.model_name, pretrained=CFG.pretrained
                ).to(CFG.device)
            elif CFG.method == "supcon" or CFG.method == "curriculum_supcon":
                print("Using Default Supcon Model")
                model = supcon_Model(CFG, CFG.model_name, pretrained=CFG.pretrained).to(
                    CFG.device
                )

            print("\033[92m" + f"{'-' * 8} Fold {fold + 1} / {CFG.n_fold}" + "\033[0m")
            summary_dir_fold = os.path.join(
                log_folder_dir, f"{CFG.exp}_fold_{fold}.csv"
            )
            log_training = open(summary_dir_fold, "w")

            train_folds, _, _ = get_dataframes(folds, fold, CFG, training_split)

            if (
                (CFG.method == "supcon" or CFG.method == "curriculum_supcon")
                and CFG.feature_batch
                and CFG.feature_mixup
            ):
                print("Feature Batch and Feature Mixup during Pre-training")
                feature_file_name, matrix_file_name = select_feature_matrix_paths(
                    CFG, fold
                )

                if not os.path.exists(feature_file_name):
                    raise FileNotFoundError(
                        f"[split={getattr(CFG, 'split_selector', '?')}] Feature file not found:\n{feature_file_name}\n"
                        f"-> Check large_model/swinv2_large flags, model_tag, and feature_dir."
                    )
                if not os.path.exists(matrix_file_name):
                    raise FileNotFoundError(
                        f"[split={getattr(CFG, 'split_selector', '?')}] Matrix file not found:\n{matrix_file_name}\n"
                        f"-> Check large_model/swinv2_large flags, model_tag, and feature_dir."
                    )

                with open(feature_file_name, "rb") as f:
                    feature_list = pickle.load(f)
                with open(matrix_file_name, "rb") as f:
                    cos_sim_matrix = pickle.load(f)
                print(
                    f"train_set samples: {len(train_folds)}, "
                    f"features: {feature_list.shape}, matrix: {cos_sim_matrix.shape}"
                )
                print(
                    f"Successfully Load {feature_file_name}, Successfully Load Matrix File {matrix_file_name}"
                )
                # label_list = {'t': train_folds['target'].tolist(), 'it': train_folds['inst_target'].tolist(), 'ivt': train_folds['triplet'].tolist()}
                label_list = {
                    "i": train_folds["instrument"].tolist(),
                    "t": train_folds["target"].tolist(),
                    "v": train_folds["verb"].tolist(),
                    "it": train_folds["inst_target"].tolist(),
                    "iv": train_folds["inst_verb"].tolist(),
                    "tv": train_folds["target_verb"].tolist(),
                    "ivt": train_folds["triplet"].tolist(),
                }

                train_dataset = SupConFeatureMixupBatchDataset(
                    train_folds,
                    CFG,
                    features=feature_list,
                    labels=label_list,
                    cos_sim_matrix=cos_sim_matrix,
                    transform=transform,
                )

            elif (
                CFG.method == "supcon" or CFG.method == "curriculum_supcon"
            ) and CFG.feature_batch:
                print("Feature Batch during Pre-training")
                feature_file_name, matrix_file_name = select_feature_matrix_paths(
                    CFG, fold
                )

                if not os.path.exists(feature_file_name):
                    raise FileNotFoundError(
                        f"[split={getattr(CFG, 'split_selector', '?')}] Feature file not found:\n{feature_file_name}\n"
                        f"-> Check large_model/swinv2_large flags, model_tag, and feature_dir."
                    )
                if not os.path.exists(matrix_file_name):
                    raise FileNotFoundError(
                        f"[split={getattr(CFG, 'split_selector', '?')}] Matrix file not found:\n{matrix_file_name}\n"
                        f"-> Check large_model/swinv2_large flags, model_tag, and feature_dir."
                    )

                with open(feature_file_name, "rb") as f:
                    feature_list = pickle.load(f)
                with open(matrix_file_name, "rb") as f:
                    cos_sim_matrix = pickle.load(f)
                print(
                    f"train_set samples: {len(train_folds)}, "
                    f"features: {feature_list.shape}, matrix: {cos_sim_matrix.shape}"
                )
                print(
                    f"Successfully Load {feature_file_name}, Successfully Load Matrix File {matrix_file_name}"
                )

                label_list = {
                    "i": train_folds["instrument"].tolist(),
                    "t": train_folds["target"].tolist(),
                    "v": train_folds["verb"].tolist(),
                    "it": train_folds["inst_target"].tolist(),
                    "iv": train_folds["inst_verb"].tolist(),
                    "tv": train_folds["target_verb"].tolist(),
                    "ivt": train_folds["triplet"].tolist(),
                }

                train_dataset = SupConFeatureBatchDataset(
                    train_folds,
                    CFG,
                    features=feature_list,
                    labels=label_list,
                    cos_sim_matrix=cos_sim_matrix,
                    transform=transform,
                )
            else:
                print("Using default Supcon Dataset")
                train_dataset = Supcon_TrainDataset(
                    train_folds, CFG, transform=transform
                )

            print("Using Defalut Dataloader")
            train_loader = DataLoader(
                train_dataset,
                batch_size=CFG.batch_size,
                shuffle=True,
                num_workers=CFG.nworkers,
                pin_memory=True,
                drop_last=True,
            )
            optimizer = AdamW(
                model.parameters(),
                lr=CFG.lr,
                weight_decay=CFG.weight_decay,
                amsgrad=False,
            )
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=CFG.epochs + 1,
                T_mult=1,
                eta_min=CFG.min_lr,
                last_epoch=-1,
            )

            if CFG.ssl_loss == "supcon":
                if CFG.feature_batch:
                    print("Using Feature batch Supcon")
                    criterion = FeatureBatchdSupConLoss(CFG, temperature=CFG.temp)
                else:
                    print("Using Default Supcon Loss")
                    criterion = SupConLoss(CFG, temperature=CFG.temp)
            else:
                criterion = SupConLoss(CFG, temperature=CFG.temp)

            scaler = amp.GradScaler()
            print(header_ssl)

            for epoch in range(CFG.epochs):
                epoch_start = time.time()

                if CFG.method == "supcon" and CFG.feature_batch:
                    print("using triplet supcon with feature batch")
                    train_dataset.set_label_key("ivt")
                    avg_loss = train_feature_batch_supcon(
                        train_loader,
                        model,
                        CFG,
                        criterion,
                        optimizer,
                        epoch,
                        scheduler,
                        CFG.device,
                        scaler=scaler,
                    )

                elif CFG.method == "supcon":
                    print("using default supcon training")
                    avg_loss = train_supcon(
                        train_loader,
                        model,
                        CFG,
                        criterion,
                        optimizer,
                        epoch,
                        scheduler,
                        CFG.device,
                        scaler=scaler,
                    )

                elif CFG.method == "curriculum_supcon" and CFG.feature_batch:
                    print("curriculum supcon with sampling")
                    interval = 2 if CFG.epochs == 6 else 1
                    label_index = min(epoch // interval, len(label_key_list) - 1)
                    train_dataset.set_label_key(label_key_list[label_index])
                    avg_loss = train_feature_batch_supcon(
                        train_loader,
                        model,
                        CFG,
                        criterion,
                        optimizer,
                        epoch,
                        scheduler,
                        CFG.device,
                        scaler=scaler,
                    )

                elif CFG.method == "curriculum_supcon":
                    print("Using default Curriculum Supcon")
                    avg_loss = train_Naive_Curriculum_Supcon(
                        train_loader,
                        model,
                        CFG,
                        criterion,
                        optimizer,
                        epoch,
                        scheduler,
                        CFG.device,
                        scaler=scaler,
                    )

                log_training.write(f"Epoch: {epoch}  Loss: {avg_loss}  \n")

                scheduler.step()
                cur_lr = scheduler.get_last_lr()

                if CFG.neplog:
                    run[f"tloss{fold}"].log(avg_loss)
                    run[f"cLR_{fold}"].log(cur_lr)

                wandb_payload = {
                    "phase": infer_wandb_phase(CFG, ssl_phase=True),
                    "fold": fold,
                    "epoch": epoch,
                    "train_loss": avg_loss,
                    "lr": cur_lr[0] if isinstance(cur_lr, (list, tuple)) else cur_lr,
                }
                if CFG.method == "curriculum_supcon" and CFG.feature_batch:
                    wandb_payload["curriculum_label"] = label_key_list[label_index]
                wandb_run.log(wandb_payload)

                print(f"epoch: {epoch}, avg_loss: {avg_loss}")
                print(
                    raw_line_ssl.format(
                        epoch, avg_loss, (time.time() - epoch_start) / 60
                    )
                )

                if should_save_ssl_checkpoint(CFG, epoch):
                    save_checkpoint_path = os.path.join(
                        checkpoint_dir,
                        f"fold{fold}_{CFG.model_name[:8]}_{CFG.exp}_{epoch}.pth",
                    )
                    meta = checkpoint_metadata(
                        CFG, training_split, fold, "pretrain", epoch
                    )
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "epoch": epoch,
                            "fold": fold,
                            "exp": CFG.exp,
                            "meta": meta,
                        },
                        save_checkpoint_path,
                    )
                    write_sidecar(save_checkpoint_path, meta)
                    print(f"Model saved at epoch: {epoch}\n")
                    wandb_run.log(
                        {
                            "phase": infer_wandb_phase(CFG, ssl_phase=True),
                            "fold": fold,
                            "saved_epoch": epoch,
                            "checkpoint_path": save_checkpoint_path,
                        }
                    )

            del model, train_loader, train_folds
            torch.cuda.empty_cache()
            gc.collect()

    wandb_run.update_summary(
        {
            "phase": infer_wandb_phase(CFG, ssl_phase=True),
            "training_minutes": (time.time() - start_time) / 60,
        }
    )
    wandb_run.finish()

    print(f"Training time: {(time.time() - start_time) / 60}")


# ================================================================================================
