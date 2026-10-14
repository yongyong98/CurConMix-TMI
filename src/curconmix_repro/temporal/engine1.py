# engine.py
import os
import json
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from .metric import *
from .training_checkpoint import training_checkpoint
from .training_inputs import guard_training_loaders
from curconmix_repro.training_split import checkpoint_provenance, load_training_split

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# 기존 유지: 마스크 미적용 BCE (요청: 1~2 미적용)
loss_layer = nn.BCEWithLogitsLoss()


def base_test(mode, model, test_loader, device, description="Validating"):
    """
    mAP 평가 수행 + (검증 시) 가중치 1회 로깅.
    적용 변경(3~5):
      - [3] squeeze() 제거 → 명시적 reshape(-1, K)
      - [4] video 이름을 프레임 단위로 정확히 전개 (batch 대응)
      - [5] 중복 forward 제거(로깅과 손실/예측에 동일 outputs/parts 재사용)
    """
    model.eval()
    all_preds, all_labels, all_videos = [], [], []
    loss_item, total_samples, logged_weights = 0.0, 0, False

    with torch.no_grad():
        for video, labels, video_name in tqdm(test_loader, desc=description):
            # 텐서 변환
            labels_tensor = torch.as_tensor(labels, dtype=torch.float32)
            video, labels_tensor = video.to(device), labels_tensor.to(device)

            # [5] 한 번만 forward해서 parts/outputs 모두 사용
            outputs, parts = model(video)

            # (검증 시) 가중치 1회 로깅
            if (not logged_weights) and (description == "Validating"):
                try:
                    if isinstance(parts, dict):
                        if "alpha" in parts:
                            alpha = float(parts["alpha"].item())
                            print(
                                f"  - Fusion Alpha: {alpha:.4f} (Spatial: {alpha*100:.1f}%, Temporal: {(1-alpha)*100:.1f}%)"
                            )
                        if "video_gate" in parts:
                            gate = parts["video_gate"]
                            if torch.is_tensor(gate):
                                gate = gate.detach().cpu().numpy()
                            gate = np.asarray(gate, dtype=np.float32).reshape(-1)
                            print(
                                "  - Video Gate: "
                                f"mean={gate.mean():.4f}, min={gate.min():.4f}, max={gate.max():.4f}"
                            )
                        if "ks_weights" in parts:
                            ks_w = parts["ks_weights"]
                            if torch.is_tensor(ks_w):
                                ks_w = ks_w.detach().cpu().numpy()
                            weights_str = " / ".join(
                                [
                                    f"k={k}: {w:.2f}"
                                    for k, w in zip(getattr(model, "ks", []), ks_w)
                                ]
                            )
                            print(f"  - Temporal ks_weights: [ {weights_str} ]")
                except Exception as e:
                    print(f"Could not log weights: {e}")
                logged_weights = True

            # 손실
            loss = loss_layer(outputs, labels_tensor)
            B = video.size(0)
            loss_item += loss.item() * B
            total_samples += B

            # [3] squeeze() 금지 → 명시적 2D 평탄화
            # outputs/labels: [B, T, K] → [B*T, K]
            K = outputs.shape[-1]
            all_preds.append(outputs.reshape(-1, K).detach().cpu().numpy())
            all_labels.append(labels_tensor.reshape(-1, K).detach().cpu().numpy())

            # [4] video 이름을 프레임 단위로 전개
            # - video_name이 리스트/튜플이면 배치 길이와 동일하다고 가정
            # - 배치 내 모든 샘플의 T가 동일하다는 가정(기존 DataLoader 사용 전제)
            #   (batch_size=1이면 항상 안전)
            T = outputs.shape[1]
            if isinstance(video_name, (list, tuple)):
                # 배치 내 각 샘플에 대해 T번 반복
                names_flat = []
                for b in range(len(video_name)):
                    names_flat.extend([str(video_name[b])] * T)
                all_videos.append(names_flat)
            else:
                # 단일 샘플 이름
                all_videos.append([str(video_name)] * T)

    avg_loss = (loss_item / total_samples) if total_samples > 0 else 0.0

    labels_list = np.concatenate(all_labels, axis=0)  # [N_frames, K]
    preds_list = np.concatenate(all_preds, axis=0)  # [N_frames, K]
    vid_names = sum([list(x) for x in all_videos], [])  # 평탄화

    # num_classes는 모드에 따라 유지 (기존 로직)
    num_classes = 131 if mode in ["cholect45", "cholect50"] else 266

    # 안전 체크: 실제 K와 불일치 시 K를 신뢰할 수도 있으나, 기존 규약을 유지
    if preds_list.shape[1] != num_classes or labels_list.shape[1] != num_classes:
        # 불일치 시 경고만 출력하고, 실제 K로 컬럼 생성
        print(
            f"[Warn] num_classes mismatch: expected {num_classes}, got {preds_list.shape[1]}. Using detected K."
        )
        num_classes = preds_list.shape[1]

    if mode in ["lls48"]:  # 또는 args.dataset == 'lls48'
        tri_cols = [f"tri{i}" for i in range(195)]
        inst_cols = [f"inst{i}" for i in range(17)]
        v_cols = [f"v{i}" for i in range(16)]
        t_cols = [f"t{i}" for i in range(38)]
        label_cols = tri_cols + inst_cols + v_cols + t_cols  # 총 266
    else:
        # cholect 계열 131
        tri_cols = [f"tri{i}" for i in range(100)]
        inst_cols = [f"inst{i}" for i in range(6)]
        v_cols = [f"v{i}" for i in range(10)]
        t_cols = [f"t{i}" for i in range(15)]
        label_cols = tri_cols + inst_cols + v_cols + t_cols  # 총 131

    # pred_cols = [f"{i}" for i in range(len(label_cols))]  # 기존 메트릭 호환을 유지
    pred_cols = [f"{i}" for i in range(num_classes)]
    df = pd.DataFrame(
        data=np.concatenate([labels_list, preds_list], axis=1),
        columns=label_cols + pred_cols,
    )
    # video 컬럼 길이 보정(혹시 불일치하면 최대한 맞추기)
    if len(vid_names) != df.shape[0]:
        print(
            f"[Warn] video name count mismatch: expected {df.shape[0]}, got {len(vid_names)}. Adjusting..."
        )
        if len(vid_names) < df.shape[0]:
            vid_names = vid_names + ["unknown"] * (df.shape[0] - len(vid_names))
        else:
            vid_names = vid_names[: df.shape[0]]

    df["video"] = vid_names
    df = df[["video"] + label_cols + pred_cols]

    epoch_mAP = (
        per_epoch_ivtmetrics(df)
        if mode in ["cholect45", "cholect50"]
        else per_epoch_ivtmetrics_lls(df)
    )
    return epoch_mAP, avg_loss


def base_train(
    args,
    model,
    train_loader,
    validation_loader,
    device,
    save_dir,
    fold_num,
    debug=False,
    wandb_run=None,
):
    """
    fold_num 인자를 받도록 수정된 훈련 함수.
    적용 변경(3~5는 검증 함수에 반영됨) + 체크포인트에 model meta 저장.
    (요청대로 1~2는 유지: 손실/마스킹 기존과 동일)
    """
    split = load_training_split(getattr(args, "training_split", None), args.dataset)
    guard_training_loaders(split, fold_num, train_loader, validation_loader)
    if not debug:
        raise ValueError("Independent inner-validation selection must be enabled")
    model.to(device)
    log_save_path = os.path.join(save_dir, "logs")
    os.makedirs(log_save_path, exist_ok=True)

    best_epoch, best_map = 0, -float("inf")

    # 특수 파라미터 LR 분리 (기존 유지)
    special_params, base_params = [], []
    for name, param in model.named_parameters():
        if "fusion_logit" in name or "ks_logits" in name:
            special_params.append(param)
        else:
            base_params.append(param)

    special_lr_multiplier = getattr(args, "special_lr_mult", 1.0)
    param_groups = [
        {"params": base_params},
        {"params": special_params, "lr": args.learning_rate * special_lr_multiplier},
    ]
    print(
        f"Base LR: {args.learning_rate}, Special Params LR: {args.learning_rate * special_lr_multiplier}"
    )

    optimizer = torch.optim.AdamW(
        param_groups, lr=args.learning_rate, weight_decay=1e-2
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        train_loss_sum, total_samples = 0.0, 0

        for video, labels, _ in tqdm(train_loader, desc=f"Training Epoch {epoch}"):
            labels = torch.as_tensor(labels, dtype=torch.float32)
            video, labels = video.to(device), labels.to(device)

            outputs, _ = model(video)
            loss = loss_layer(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * video.size(0)
            total_samples += video.size(0)

        scheduler.step()
        avg_train_loss = train_loss_sum / total_samples if total_samples > 0 else 0.0

        if debug:
            valid_map, valid_loss = base_test(
                args.dataset, model, validation_loader, device, description="Validating"
            )
            if not np.isfinite(valid_map):
                raise ValueError(
                    "Inner-validation AP is undefined; checkpoint selection is not possible"
                )
            print(
                f"Epoch {epoch}: Train Loss {avg_train_loss:.4f} | Valid Loss {valid_loss:.4f}, Valid mAP {valid_map:.4f} | LR {optimizer.param_groups[0]['lr']:.6f}"
            )

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "phase": "mrtt_train",
                        "fold": fold_num,
                        "epoch": epoch,
                        "train_loss": avg_train_loss,
                        "valid_loss": valid_loss,
                        "valid_mAP": valid_map,
                        "lr": optimizer.param_groups[0]["lr"],
                    }
                )

            valid_metric = {
                "epoch": epoch,
                "Train_Loss": avg_train_loss,
                "Valid_Loss": valid_loss,
                "Valid_mAP": valid_map,
            }
            with open(
                os.path.join(log_save_path, f"log_fold{fold_num}.txt"),
                "a",
                encoding="utf-8",
            ) as f:
                f.write(json.dumps(valid_metric) + "\n")

            if valid_map > best_map:
                best_map = valid_map
                best_epoch = epoch
                checkpoint_filename = f"best_model_fold{fold_num}.pth"

                # --- 모델 메타 포함 저장 (요청 추가) ---
                # 가능한 메타(모델/실험 설정) 최대한 담기
                meta = {
                    **checkpoint_provenance(split, fold_num, stage="mrtt"),
                    "model_type": getattr(args, "model_type", None),
                    "embed_dim": getattr(
                        model, "embed_dim", getattr(args, "dim", None)
                    ),
                    "num_classes": getattr(
                        model,
                        "num_classes",
                        (131 if args.dataset == "cholect50" else 266),
                    ),
                    "ks": (
                        list(getattr(model, "ks", [])) if hasattr(model, "ks") else None
                    ),
                    "fusion": getattr(model, "fusion", None),
                    "use_spatial_norm": (
                        bool(
                            getattr(model, "use_spatial_branch", True)
                            and getattr(model, "spatial_norm", None) is not None
                        )
                        if hasattr(model, "use_spatial_branch")
                        else None
                    ),
                    "spatial_dropout": (
                        float(
                            getattr(
                                getattr(model, "spatial_drop", nn.Identity()), "p", 0.0
                            )
                        )
                        if hasattr(model, "spatial_drop")
                        else None
                    ),
                    "ks_weights": (
                        bool(getattr(model, "ks_weights", False))
                        if hasattr(model, "ks_weights")
                        else None
                    ),
                    "temporal_cfg": {
                        "nhead": 4,
                        "num_layers": 3,
                        "dim_feedforward": 2048,
                        "dropout": 0.1,
                    },
                    "dataset": args.dataset,
                    "name": args.name,
                    "fold": fold_num,
                    "procedure_order_sha256": args.procedure_order_sha256,
                    "feature_manifest_sha256": args.feature_manifest_sha256,
                    "training_feature_manifest_sha256": args.training_feature_manifest_sha256,
                }

                ckpt = training_checkpoint(
                    epoch=epoch,
                    model_state=model.state_dict(),
                    optimizer_state=optimizer.state_dict(),
                    scheduler_state=scheduler.state_dict(),
                    best_map=best_map,
                    arguments=vars(args),
                    metadata=meta,
                    fold=fold_num,
                )
                torch.save(ckpt, os.path.join(save_dir, checkpoint_filename))

                # 사람이 보기 좋은 메타 json도 함께
                try:
                    with open(
                        os.path.join(save_dir, f"experiment_fold{fold_num}.json"),
                        "w",
                        encoding="utf-8",
                    ) as f:
                        json.dump(meta, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    print(f"[Warn] could not write experiment json: {e}")

                print(
                    f"✨ New Best Model saved: {checkpoint_filename} with mAP {best_map:.4f} at Epoch {epoch}"
                )

                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "phase": "mrtt_train",
                            "fold": fold_num,
                            "best_valid_mAP": best_map,
                            "best_epoch": best_epoch,
                            "best_checkpoint": os.path.join(
                                save_dir, checkpoint_filename
                            ),
                        }
                    )

    return best_map
