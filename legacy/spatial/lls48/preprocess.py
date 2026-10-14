import os
import re
import pandas as pd
import numpy as np
from pathlib import Path


EXPECTED_COMPONENT_COUNTS = {
    'inst': 17,
    'v': 16,
    't': 38,
}

# ------------------------- Split Selector ------------------------- #
def split_selector(case='cholect50'):
    switcher = {
        'cholect50': {
            'train': [1, 15, 26, 40, 52, 65, 79, 2, 18, 27, 43, 56, 66, 92, 4, 22, 31, 47, 57, 68, 96, 5, 23, 35, 48, 60, 70, 103, 13, 25, 36, 49, 62, 75, 110],
            'val'  : [8, 12, 29, 50, 78],
            'test' : [6, 51, 10, 73, 14, 74, 32, 80, 42, 111]
        },
        'cholect50-challenge': {
            'train': [1, 15, 26, 40, 52, 79, 2, 27, 43, 56, 66, 4, 22, 31, 47, 57, 68, 23, 35, 48, 60, 70, 13, 25, 49, 62, 75, 8, 12, 29, 50, 78, 6, 51, 10, 73, 14, 32, 80, 42],
            'val':   [5, 18, 36, 65, 74],
            'test':  [92, 96, 103, 110, 111]
        },
        'cholect45-crossval': {
            0: [79,  2, 51,  6, 25, 14, 66, 23, 50],
            1: [80, 32,  5, 15, 40, 47, 26, 48, 70],
            2: [31, 57, 36, 18, 52, 68, 10,  8, 73],
            3: [42, 29, 60, 27, 65, 75, 22, 49, 12],
            4: [78, 43, 62, 35, 74,  1, 56,  4, 13],
        },
        'cholect50-crossval': {
            0: [79,  2, 51,  6, 25, 14, 66, 23, 50, 111],
            1: [80, 32,  5, 15, 40, 47, 26, 48, 70,  96],
            2: [31, 57, 36, 18, 52, 68, 10,  8, 73, 103],
            3: [42, 29, 60, 27, 65, 75, 22, 49, 12, 110],
            4: [78, 43, 62, 35, 74,  1, 56,  4, 13,  92],
        },
        'LLS48': {
            0: [10, 15, 22, 27, 30, 39, 42, 46, 47], # 9
            1: [4, 6, 7, 8, 18, 29, 32, 36, 37, 43], # 10
            2: [2, 5, 13, 16, 19, 24, 28, 31, 45], # 9
            3: [1, 3, 20, 23, 26, 35, 38, 40, 44, 48], # 10
            4: [9, 11, 12, 14, 17, 21, 25, 33, 34, 41] # 10
        },
        'LLS48-challenge' : {
            'train': [1, 2, 3, 5, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 28, 29, 30, 31, 33, 35, 36, 37, 38, 39, 40, 41, 42, 44, 45, 46, 48],
            'val': [4, 6, 7, 11],
            'test': [23, 27, 32, 34, 43, 47]}
        }
    return switcher.get(case)

# ------------------------- Label Encoding Helpers ------------------------- #
def find_indices_of_ones(binary_list):
    return tuple(np.nonzero(binary_list)[0])

def index_to_label_mapping(binary_array):
    index_to_label = {}
    current_label = 0
    labels = []

    for binary_list in binary_array:
        indices_tuple = find_indices_of_ones(binary_list)
        if indices_tuple not in index_to_label:
            index_to_label[indices_tuple] = current_label
            current_label += 1
        labels.append(index_to_label[indices_tuple])
    return np.array(labels)


def prefixed_columns(columns, prefix):
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    matched = []
    for col in columns:
        result = pattern.fullmatch(str(col))
        if result:
            matched.append((int(result.group(1)), col))
    matched.sort(key=lambda item: item[0])
    return [col for _, col in matched]


def validate_component_schema(folds):
    component_cols = {
        'inst': prefixed_columns(folds.columns, 'inst'),
        'v': prefixed_columns(folds.columns, 'v'),
        't': prefixed_columns(folds.columns, 't'),
    }
    for prefix, expected_count in EXPECTED_COMPONENT_COUNTS.items():
        actual_count = len(component_cols[prefix])
        if actual_count != expected_count:
            raise ValueError(
                f"Unexpected LLS48 component schema for '{prefix}*': "
                f"expected {expected_count} columns, found {actual_count}. "
                f"Columns found: {component_cols[prefix]}"
            )
    return component_cols

# ------------------------- Main Fold Generator ------------------------- #
def get_folds(CFG):
    print("\033[94mPreparing the data\033[0m")
    # train = pd.read_csv(os.path.join(CFG.parent_path, CFG.path_csv))
    csv_path = (Path(CFG.parent_path) / CFG.path_csv).as_posix()
    train = pd.read_csv(csv_path)

    print("Preprocessing the data...")

    folds = train.copy()
    fold_map = split_selector(CFG.split_selector)
    folds["fold"] = -1

    # 위치 기반 인덱스
    index_no = folds.columns.get_loc(CFG.col0)
    component_cols = validate_component_schema(folds)

    binary_array_triplet = folds.iloc[:, index_no:index_no + CFG.triplet_size].values.astype(int) # Tri 0 ~ Tri 194
    binary_array_target = folds.loc[:, component_cols['t']].values.astype(int)
    binary_array_verb = folds.loc[:, component_cols['v']].values.astype(int)
    binary_array_inst = folds.loc[:, component_cols['inst']].values.astype(int)

    binary_array_inst_target = np.concatenate((binary_array_inst, binary_array_target), axis=1)
    binary_array_inst_verb = np.concatenate((binary_array_inst, binary_array_verb), axis=1)
    binary_array_target_verb = np.concatenate((binary_array_target, binary_array_verb), axis=1)

    trip_num = binary_array_triplet.sum(axis=1).tolist()

    # 라벨 변환
    folds["triplet"] = index_to_label_mapping(binary_array_triplet)
    folds["target"] = index_to_label_mapping(binary_array_target)
    folds["verb"] = index_to_label_mapping(binary_array_verb)
    folds["instrument"] = index_to_label_mapping(binary_array_inst)
    folds["inst_target"] = index_to_label_mapping(binary_array_inst_target)
    folds["inst_verb"] = index_to_label_mapping(binary_array_inst_verb)
    folds["target_verb"] = index_to_label_mapping(binary_array_target_verb)
    folds["num"] = trip_num

    # Fold assignment
    for fold, video_list in fold_map.items():
        video_list = [f"VID{vid:02d}" for vid in video_list]
        folds.loc[folds["video"].isin(video_list), "fold"] = fold

    # Keep only assigned folds.
    folds = folds[folds["fold"] != -1].reset_index(drop=True)

    # Print a compact label summary without Unicode-only console symbols.
    print("\nLabel combinations (unique label counts):")
    print(
        "Validated component schema: "
        f"inst={len(component_cols['inst'])}, "
        f"verb={len(component_cols['v'])}, "
        f"target={len(component_cols['t'])}"
    )
    label_columns = ["triplet", "target", "verb", "instrument",
                     "inst_target", "inst_verb", "target_verb"]
    for col in label_columns:
        unique_labels = folds[col].unique()
        print(f"  - {col}: {len(unique_labels):>4} classes")
        print(f"    Examples: {unique_labels[:3]}\n")

    print("Dataset ready!\n")
    return folds
