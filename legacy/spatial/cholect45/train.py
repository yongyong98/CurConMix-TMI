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

from augmentation import *
from models import *
from preprocess import get_folds
from utils import *
from helper import *
from dataset import *
from losses import *
from losses_weighted import *
import pickle
from typing import Tuple
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


def infer_wandb_phase(CFG, ssl_phase=False):
    if ssl_phase or getattr(CFG, "ssl", False):
        return "ssl_pretrain"
    if getattr(CFG, "distill", False):
        return "student"
    if getattr(CFG, "pretrained_ssl", False):
        return "teacher"
    return "train"


def _as_score(value):
    if isinstance(value, tuple):
        return float(value[0])
    return float(value)


def log_neptune(run, key, value, fold=None):
    if fold is not None:
        key = f"{key}{fold}"
    run[key].log(value)


def train_cross_val(CFG):
    stage = "student" if CFG.distill else "teacher"
    training_split = training_gate(CFG, "cholect45", stage)
    start_time = time.time()
    seed_torch(seed=CFG.seed)
    summary_dir = os.path.join(CFG.output_dir, "summary_dir")
    log_folder_dir = os.path.join(CFG.output_dir, CFG.exp)
    checkpoints_dir = os.path.join(CFG.output_dir, "checkpoints")
    oofs_dir = os.path.join(CFG.output_dir, "oofs")
    os.makedirs(summary_dir, exist_ok=True)
    os.makedirs(log_folder_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(oofs_dir, exist_ok=True)

    if CFG.debug:
        CFG.epochs = 1
        CFG.neplog = False

    wandb_run = init_safe_wandb_from_cfg(CFG, infer_wandb_phase(CFG))

    if CFG.neplog:
        run = neptune.init(
            project=CFG.neptune_project,
            api_token=CFG.neptune_api_token,
        )
        log_params = {
            "Model": CFG.model_name,
            "imsize": CFG.height,
            "LR": CFG.lr,
            "bs": CFG.batch_size,
            "Epochs": CFG.epochs,
            "SD": CFG.SD,
            "T_0": CFG.T_0,
            "min_lr": CFG.min_lr,
            "seed": str(CFG.seed),
            "tsize": CFG.target_size,
            "smooth": CFG.smooth,
            "exp": CFG.exp,
        }
        for key, value in log_params.items():
            run[key].log(value)

    folds = get_folds(CFG)
    print_training_info(folds, CFG)

    for fold in range(CFG.start_n_fold, CFG.n_fold):
        if fold not in CFG.trn_fold:
            continue
        print(f"\033[92m{'-' * 8} Fold {fold + 1} / {CFG.n_fold}\033[0m")
        summary_dir_fold = os.path.join(log_folder_dir, f"{CFG.exp}_fold_{fold}.csv")

        with open(summary_dir_fold, "w") as log_training:
            model = TripletModel(CFG, CFG.model_name, pretrained=CFG.pretrained).to(
                CFG.device
            )
            if CFG.pretrained_ssl:
                weights_path = os.path.join(
                    CFG.output_dir,
                    f"checkpoints/fold{fold}_{CFG.model_name[:8]}_{CFG.pretrained_exp}.pth",
                )
                print(f"Using Pre-trained Checkpoint: {weights_path}")
                verify_sidecar(weights_path, training_split, fold, "pretrain")
                ssl_checkpoint = torch.load(
                    weights_path, map_location=CFG.device, weights_only=True
                )
                validate_meta(
                    ssl_checkpoint.get("meta"), training_split, fold, "pretrain"
                )
                ssl_state_dict = ssl_checkpoint["model"]
                # Load weights with non-strict mode
                load_result = model.load_state_dict(ssl_state_dict, strict=False)
                print("Weight Loading Summary:")
                if load_result.missing_keys:
                    print("[Missing Keys] not loaded into model:")
                    for k in load_result.missing_keys:
                        print(f"   - {k}")
                else:
                    print("No missing keys.")

                if load_result.unexpected_keys:
                    print("[Unexpected Keys] in checkpoint but not in model:")
                    for k in load_result.unexpected_keys:
                        print(f"   - {k}")
                else:
                    print("No unexpected keys.")

            train_folds, valid_folds, valid_folds_temp = get_dataframes(
                folds, fold, CFG, training_split
            )
            if CFG.distill:
                train_folds = apply_self_distillation(fold, train_folds, CFG)
            else:
                print("Using default Train dataset")

            train_dataset = TrainDataset(
                train_folds,
                CFG,
                transform=get_transforms(data="train", CFG=CFG),
                fold=fold,
            )
            valid_dataset = TrainDataset(
                valid_folds,
                CFG,
                transform=get_transforms(data="valid", CFG=CFG),
                fold=fold,
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

            # Define optimizer, scheduler, and loss function
            optimizer = AdamW(
                model.parameters(),
                lr=CFG.lr,
                weight_decay=CFG.weight_decay,
                amsgrad=False,
            )
            scheduler = CosineAnnealingWarmRestarts(
                optimizer, T_0=CFG.epochs + 1, T_mult=1000, eta_min=CFG.min_lr
            )
            if CFG.loss == "bce":
                criterion = nn.BCEWithLogitsLoss(reduction="sum").to(CFG.device)

            best_score = float("-inf")
            scaler = amp.GradScaler()
            print("start training")
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
                current_lr = scheduler.get_last_lr()[0]

                valid_folds_temp[[str(c) for c in range(CFG.target_size)]] = preds
                cholect45_epoch_CV = inner_validation_ap(
                    valid_folds_temp, training_split, fold, per_epoch_ivtmetrics, CFG
                )

                log_training.write(
                    f"Epoch: {epoch} Validation Loss: {avg_val_loss} CholecT45 mAP: {cholect45_epoch_CV} \n"
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
                        "lr": current_lr,
                    }
                )

                if is_better_inner_ap(cholect45_epoch_CV, best_score):
                    best_score = cholect45_epoch_CV
                    save_checkpoint_path = os.path.join(
                        CFG.output_dir,
                        f"checkpoints/fold{fold}_{CFG.model_name[:8]}_{CFG.target_size}_{CFG.exp}.pth",
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
                        save_checkpoint_path,
                    )
                    write_sidecar(save_checkpoint_path, meta)
                    wandb_run.log(
                        {
                            "phase": infer_wandb_phase(CFG),
                            "fold": fold,
                            "best_cmAP": best_score,
                            "best_checkpoint": save_checkpoint_path,
                        }
                    )
                print(
                    raw_line.format(
                        epoch,
                        avg_loss,
                        avg_val_loss,
                        cholect45_epoch_CV,
                        (time.time() - epoch_start) / 60**1,
                    )
                )
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

# ================================================================================================


def select_transforms(data, CFG):
    print("Using Default Augmentation Methods")
    return get_transforms(data="train", CFG=CFG)


def _infer_model_tag(model_name: str) -> str:
    """model_tag 미지정 시, model_name으로부터 안전한 기본 접두사 생성."""
    if not model_name:
        return "model"
    tag = os.path.basename(str(model_name)).split(".")[0]
    return tag.replace(" ", "_").replace("/", "_").replace("\\", "_")


def _ensure_pkl(name: str) -> str:
    return name if str(name).endswith(".pkl") else f"{name}.pkl"


def should_save_ssl_checkpoint(CFG, epoch: int) -> bool:
    if epoch == CFG.epochs - 1:
        return True
    return bool(getattr(CFG, "save_all_ssl_epochs", False))


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


def train_cross_val_SSL(CFG):
    training_split = training_gate(CFG, "cholect45", "pretrain")
    start_time = time.time()
    seed_torch(seed=CFG.seed)
    print("Starting Pre-Training with SSL...")
    summary_dir = os.path.join(CFG.output_dir, "summary_dir")
    os.makedirs(summary_dir, exist_ok=True)
    log_folder_dir = os.path.join(CFG.output_dir, CFG.exp)
    os.makedirs(log_folder_dir, exist_ok=True)
    summary_dir_total = os.path.join(log_folder_dir, f"{CFG.exp}_total.csv")
    log_total = open(summary_dir_total, "w")

    # DEBUG 단축
    if getattr(CFG, "debug", False):
        CFG.epochs = 1
        CFG.neplog = False

    wandb_run = init_safe_wandb_from_cfg(CFG, infer_wandb_phase(CFG, ssl_phase=True))

    # Neptune 로깅
    if getattr(CFG, "neplog", False):
        run = neptune.init(project=CFG.neptune_project, api_token=CFG.neptune_api_token)
        run["Model"].log(CFG.model_name)
        run["imsize"].log(CFG.height)
        run["LR"].log(CFG.lr)
        run["bs"].log(CFG.batch_size)
        run["Epochs"].log(CFG.epochs)
        run["SD"].log(CFG.SD)
        run["T_0"].log(CFG.T_0)
        run["min_lr"].log(CFG.min_lr)
        run["seed"].log(str(CFG.seed))
        run["tsize"].log(CFG.target_size)
        run["smooth"].log(CFG.smooth)
        run["exp"].log(CFG.exp)

    # OOF 저장 디렉토리
    os.makedirs(os.path.join(CFG.output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(CFG.output_dir, "oofs"), exist_ok=True)

    # 데이터/변환/정보
    folds = get_folds(CFG)
    transform = select_transforms(data="train", CFG=CFG)
    print_training_info(folds, CFG)

    label_key_list = CFG.label_order  # 예: ['t', 'it', 'ivt']

    # 폴드 루프
    for fold in range(CFG.start_n_fold, CFG.n_fold):
        if fold in CFG.trn_fold:
            # 모델 선택
            if getattr(CFG, "feature_batch", False) and (
                CFG.method in ("supcon", "curriculum_supcon")
            ):
                print("Using separated head Supcon model")
                model = FeatureSupConModel(
                    CFG, CFG.model_name, pretrained=CFG.pretrained
                ).to(CFG.device)
            elif CFG.method in ("supcon", "curriculum_supcon"):
                print("Using Default Supcon Model")
                model = supcon_Model(CFG, CFG.model_name, pretrained=CFG.pretrained).to(
                    CFG.device
                )
            else:
                raise ValueError(f"Unsupported method: {CFG.method}")

            print("\033[92m" + f"{'-' * 8} Fold {fold + 1} / {CFG.n_fold}" + "\033[0m")
            summary_dir_fold = os.path.join(
                log_folder_dir, f"{CFG.exp}_fold_{fold}.csv"
            )
            log_training = open(summary_dir_fold, "w")

            train_folds, _, _ = get_dataframes(folds, fold, CFG, training_split)

            # -----------------------------
            # CurConMix: Feature-batch + Mixup
            # -----------------------------
            if (
                (CFG.method in ("supcon", "curriculum_supcon"))
                and CFG.feature_batch
                and CFG.feature_mixup
            ):
                print("Feature Batch and Feature Mixup during Pre-training")

                feature_file_name, matrix_file_name = select_feature_matrix_paths(
                    CFG, fold
                )

                if not os.path.exists(feature_file_name):
                    raise FileNotFoundError(
                        f"[split={getattr(CFG,'split_selector','?')}] Feature file not found:\n{feature_file_name}\n"
                        f"-> Check split_selector, flags(Base384/tiny/base/large), model_tag/use_hardcoded_50, and feature_dir."
                    )
                if not os.path.exists(matrix_file_name):
                    raise FileNotFoundError(
                        f"[split={getattr(CFG,'split_selector','?')}] Matrix file not found:\n{matrix_file_name}\n"
                        f"-> Check split_selector, flags(Base384/tiny/base/large), model_tag/use_hardcoded_50, and feature_dir."
                    )

                with open(feature_file_name, "rb") as f:
                    feature_list = pickle.load(f)
                with open(matrix_file_name, "rb") as f:
                    cos_sim_matrix = pickle.load(f)

                print(f"Successfully Load {feature_file_name}")
                print(f"Successfully Load Matrix File {matrix_file_name}")
                print(
                    f"Loaded features: {len(feature_list)} samples, Loaded Similarity Matrix: {cos_sim_matrix.shape}"
                )

                # 라벨 사전
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
            else:
                print("Using default Supcon Dataset")
                train_dataset = Supcon_TrainDataset(
                    train_folds, CFG, transform=transform
                )

            # DataLoader / Optim / Sched
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

            # Loss
            if CFG.ssl_loss == "supcon":
                if getattr(CFG, "feature_batch", False):
                    print("Using Feature batch Supcon")
                    criterion = FeatureBatchdSupConLoss(CFG, temperature=CFG.temp)
                else:
                    print("Using Default Supcon Loss")
                    criterion = SupConLoss(CFG, temperature=CFG.temp)
            else:
                raise ValueError(f"Unsupported ssl_loss: {CFG.ssl_loss}")

            scaler = amp.GradScaler()
            print(header_ssl)

            # Epoch loop
            for epoch in range(CFG.epochs):
                epoch_start = time.time()

                if CFG.method == "supcon":
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
                elif CFG.method == "curriculum_supcon" and getattr(
                    CFG, "feature_batch", False
                ):
                    print("curriculum supcon with sampling")
                    # 커리큘럼 스케줄(예: 에폭 6이면 2에폭 간격)
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
                else:
                    raise ValueError(
                        f"Unsupported combination: method={CFG.method}, feature_batch={CFG.feature_batch}"
                    )

                log_output = f"Epoch: {epoch} Training Loss: {avg_loss} \n"
                log_training.write(log_output)
                scheduler.step()
                cur_lr = scheduler.get_last_lr()

                log_val_output = f"Epoch: {epoch}  Loss: {avg_loss}  \n"
                log_training.write(log_val_output)

                if getattr(CFG, "neplog", False):
                    run[f"tloss{fold}"].log(avg_loss)
                    run[f"cLR_{fold}"].log(cur_lr)

                wandb_payload = {
                    "phase": infer_wandb_phase(CFG, ssl_phase=True),
                    "fold": fold,
                    "epoch": epoch,
                    "train_loss": avg_loss,
                    "lr": cur_lr[0] if isinstance(cur_lr, (list, tuple)) else cur_lr,
                }
                if CFG.method == "curriculum_supcon" and getattr(
                    CFG, "feature_batch", False
                ):
                    wandb_payload["curriculum_label"] = label_key_list[label_index]
                wandb_run.log(wandb_payload)

                print(f"epoch: {epoch}, avg_loss: {avg_loss}")
                print(
                    raw_line_ssl.format(
                        epoch, avg_loss, (time.time() - epoch_start) / 60**1
                    )
                )

                if should_save_ssl_checkpoint(CFG, epoch):
                    ckpt_path = os.path.join(
                        CFG.output_dir,
                        f"checkpoints/fold{fold}_{CFG.model_name[:8]}_{CFG.exp}_{epoch}.pth",
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
                        ckpt_path,
                    )
                    write_sidecar(ckpt_path, meta)
                    print(f"Model saved at epoch: {epoch}\n")
                    wandb_run.log(
                        {
                            "phase": infer_wandb_phase(CFG, ssl_phase=True),
                            "fold": fold,
                            "saved_epoch": epoch,
                            "checkpoint_path": ckpt_path,
                        }
                    )

            # cleanup
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
