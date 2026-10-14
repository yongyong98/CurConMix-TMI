"""Data access tests use synthetic metadata only, never clinical files."""

from pathlib import Path
from unittest.mock import patch

import pytest

from curconmix_release.dataset import LLS48Index
from curconmix_release.lls48 import LABEL_COLUMNS


def rows_for(videos):
    return [
        {
            "video": video,
            "clip_name": f"{video[-2:]}_001",
            "image_id": f"{video[-2:]}_{frame}",
            "image_path": f"{video}/{video[-2:]}_001/frame_{frame:05d}.png",
            "step": "0",
            "task": "1",
            **{name: str(int(index == 0)) for index, name in enumerate(LABEL_COLUMNS)},
        }
        for video in videos
        for frame in range(1, 6)
    ]


def build_index(tmp_path, **kwargs):
    rows = rows_for(["VID10", "VID04"])
    with patch(
        "curconmix_release.dataset.inspect_annotations", return_value=(rows, {}, {})
    ):
        return LLS48Index(tmp_path, **kwargs)


def test_procedure_fold_split_is_disjoint(tmp_path):
    training = build_index(tmp_path, fold=0, split="train")
    validation = build_index(tmp_path, fold=0, split="val")
    assert len(training) == len(validation) == 5
    assert {r[1]["video"] for r in training._rows} == {"VID04"}
    assert {r[1]["video"] for r in validation._rows} == {"VID10"}


def test_clips_preserve_identity_and_labels(tmp_path):
    index = build_index(tmp_path)
    with patch(
        "curconmix_release.dataset.source_file",
        side_effect=lambda root, rel: root / rel,
    ):
        clips = list(index.clips())
    assert len(clips) == 2
    assert [len(clip) for clip in clips] == [5, 5]
    assert clips[0][0]["row_index"] == 0
    assert clips[1][0]["row_index"] == 5
    assert len(clips[0][0]["labels"]) == 266
    assert clips[0][0]["labels"][:2] == (1, 0)
    assert isinstance(clips[0][0]["image_path"], Path)
    assert {row["video"] for row in clips[0]} == {"VID10"}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fold": 0},
        {"split": "test"},
        {"split": "train"},
        {"split": "val", "fold": 5},
        {"split": "val", "fold": True},
    ],
)
def test_ambiguous_or_invalid_split_rejected(tmp_path, kwargs):
    with pytest.raises(ValueError):
        LLS48Index(tmp_path, **kwargs)
