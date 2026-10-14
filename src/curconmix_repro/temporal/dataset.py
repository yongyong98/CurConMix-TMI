import os
from pathlib import Path

import numpy as np

import torch
from torch.utils.data import Dataset
from .procedure_order import load_order, ordered_files
from curconmix_repro.training_split import (
    fold_partition,
    split_digest,
    validate_training_split,
)

_ENV_ROOT = os.environ.get("CURCONMIX_ROOT")
_FEATURE_ROOT = os.environ.get("CURCONMIX_FEATURE_ROOT")
_LABEL_ROOT = os.environ.get("CURCONMIX_LABEL_ROOT")
REPO_ROOT = Path(_ENV_ROOT) if _ENV_ROOT else Path(__file__).resolve().parents[3]
FEATURE_ROOT = Path(_FEATURE_ROOT) if _FEATURE_ROOT else REPO_ROOT / "data" / "feature"
LABEL_ROOT = Path(_LABEL_ROOT) if _LABEL_ROOT else REPO_ROOT / "data" / "label"


class VideoDataset(Dataset):
    def __init__(
        self, dataset, name, mode, fold, procedure_order=None, training_split=None
    ):
        self.mode = mode

        self.videos = []
        self.labels = []
        self.video_names = []
        if mode in ("train", "inner_validation"):
            if training_split is None:
                raise ValueError(
                    "Training datasets require an independent inner-validation split"
                )
            validate_training_split(training_split, dataset)
            partition = fold_partition(training_split, fold)
            self.selection_role = mode
            self.training_split_sha256 = split_digest(training_split)
            filenames = [f"{video}.npy" for video in partition[mode]]
            # Both fit and inner-validation features originate from the outer
            # development pool. Outer-test arrays remain in the separate val folder.
            storage_mode = "train"
        else:
            if mode != "val":
                raise ValueError("Expected train, inner_validation or inference val")
            storage_mode = mode
        video_feature_folder = FEATURE_ROOT / dataset / name / str(fold) / storage_mode
        label_folder = LABEL_ROOT / dataset / "triplet_label"
        print(f"Loading {mode} dataset from {video_feature_folder}")
        print(f"Loading labels from {label_folder}")
        if training_split is not None and mode in ("train", "inner_validation"):
            pass
        elif procedure_order is not None:
            record = load_order(Path(procedure_order), dataset, name)
            filenames = ordered_files(
                record, dataset, name, fold, mode, video_feature_folder
            )
        else:
            # Compatibility for historical inference only. New training runs
            # always pass a recorded order; frame order inside arrays is untouched.
            filenames = os.listdir(video_feature_folder)
        for v_f in filenames:
            v_f_path = os.path.join(video_feature_folder, v_f)

            label_path = os.path.join(label_folder, v_f)
            label = np.load(label_path, allow_pickle=False)
            videos = np.load(v_f_path, allow_pickle=False)

            self.videos.append(videos)
            self.labels.append(label)
            self.video_names.append(v_f)

        print(f"{mode}dataset: Load {dataset} dataset with {self.__len__()} videos.")

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        video, label, video_name = (
            self.videos[index],
            self.labels[index],
            self.video_names[index],
        )
        return video, label, video_name
