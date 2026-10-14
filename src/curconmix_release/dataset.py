"""Small, dependency-free index for the released LLS48 frames and clips.

Images are not decoded or resized here. This is a data-access helper, not the
paper's spatial preprocessing or a replacement for the canonical AP evaluator.
"""

from pathlib import Path

from .lls48 import CANONICAL_FOLDS, LABEL_COLUMNS, inspect_annotations, source_file


class LLS48Index:
    """Index unchanged annotations, optionally selecting a held-out fold.

    ``split='val'`` selects procedures assigned to ``fold``; ``split='train'``
    selects the remaining procedures. These are the historical five-fold split
    names, not an additional independent test split. Indexing returns a fresh
    dictionary with an absolute image path and 266 binary labels ordered as
    triplet, instrument, verb, target. ``clips()`` yields independent five-frame
    tuples, never a concatenated temporal stream.
    """

    def __init__(self, root, *, fold=None, split="all"):
        if split not in ("all", "train", "val"):
            raise ValueError("split must be all, train, or val")
        if split == "all" and fold is not None:
            raise ValueError("fold must be omitted for split='all'")
        if split != "all" and (type(fold) is not int or fold not in range(5)):
            raise ValueError("train/val selection requires fold 0 through 4")
        self.root = Path(root).resolve()
        rows, _, self.summary = inspect_annotations(self.root)
        self._folds = {
            video: int(number)
            for number, videos in CANONICAL_FOLDS.items()
            for video in videos
        }
        self._rows = []
        for index, row in enumerate(rows):
            held_out = self._folds[row["video"]]
            selected = (
                split == "all"
                or (split == "val" and held_out == fold)
                or (split == "train" and held_out != fold)
            )
            if selected:
                self._rows.append((index, row))

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, index):
        row_index, row = self._rows[index]
        return {
            "row_index": row_index,
            "video": row["video"],
            "clip_name": row["clip_name"],
            "image_id": row["image_id"],
            "image_path": source_file(self.root, f"data/{row['image_path']}"),
            "held_out_fold": self._folds[row["video"]],
            "labels": tuple(int(row[column]) for column in LABEL_COLUMNS),
            "step": int(row["step"]),
            "task": int(row["task"]),
        }

    def clips(self):
        for start in range(0, len(self), 5):
            clip = tuple(self[index] for index in range(start, start + 5))
            if len({(row["video"], row["clip_name"]) for row in clip}) != 1:
                raise ValueError("Independent clip boundary was lost")
            yield clip
