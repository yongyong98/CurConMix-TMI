#!/usr/bin/env python3
"""Regression tests for strict frame and clip alignment."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from curconmix_repro.evaluation.action_temporal_io import (  # noqa: E402
    AlignmentError,
    rehydrate_temporal_metadata,
    temporal_image_ordinal,
    validate_temporal_units,
)


class ActionTemporalIOTests(unittest.TestCase):
    def make_master(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "video": ["VID01"] * 3,
                "clip_name": ["01_001"] * 3,
                "image_path": [
                    "VID01/01_001/frame_00001.png",
                    "VID01/01_001/frame_00002.png",
                    "VID01/01_001/frame_00003.png",
                ],
                "image_id": ["0_1", "0_2", "0_3"],
                "fold": [0, 0, 0],
                "tri0": [1, 0, 1],
                "tri1": [0, 1, 0],
            }
        )

    def make_temporal(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "video": ["VID01.npy"] * 3,
                "image_id": ["VID01.npy_1", "VID01.npy_2", "VID01.npy_3"],
                "fold": [0, 0, 0],
                "tri0": [1, 0, 1],
                "tri1": [0, 1, 0],
                "0": [0.9, 0.1, 0.8],
                "1": [0.1, 0.9, 0.2],
            }
        )

    def test_temporal_suffix_is_converted_to_zero_based(self) -> None:
        self.assertEqual(temporal_image_ordinal("VID02.npy", "VID02.npy_1"), 0)
        self.assertEqual(temporal_image_ordinal("VID02.npy", "VID02.npy_45"), 44)

    def test_rehydrate_exact_alignment(self) -> None:
        result = rehydrate_temporal_metadata(self.make_temporal(), self.make_master(), 2, "LLS48")
        self.assertEqual(result["image_path"].tolist(), self.make_master()["image_path"].tolist())
        self.assertEqual(result["clip_name"].tolist(), ["01_001"] * 3)
        self.assertEqual(result["_video_ordinal"].tolist(), [0, 1, 2])

    def test_rehydrate_rejects_gt_mismatch(self) -> None:
        temporal = self.make_temporal()
        temporal.loc[1, "tri1"] = 0
        with self.assertRaises(AlignmentError):
            rehydrate_temporal_metadata(temporal, self.make_master(), 2, "LLS48")

    def test_rehydrate_rejects_missing_ordinal(self) -> None:
        temporal = self.make_temporal()
        temporal.loc[2, "image_id"] = "VID01.npy_4"
        with self.assertRaises(AlignmentError):
            rehydrate_temporal_metadata(temporal, self.make_master(), 2, "LLS48")

    def test_metadata_rejects_unsorted_frames(self) -> None:
        frame = pd.DataFrame({"video": ["A", "A", "A"], "frame": [0, 2, 1]})
        with self.assertRaises(AlignmentError):
            validate_temporal_units(frame, ["video"], "frame")

    def test_metadata_rejects_duplicate_frames(self) -> None:
        frame = pd.DataFrame({"video": ["A", "A"], "frame": [0, 0]})
        with self.assertRaises(AlignmentError):
            validate_temporal_units(frame, ["video"], "frame")

    def test_metadata_rejects_missing_values(self) -> None:
        frame = pd.DataFrame({"video": ["A", None], "frame": [0, 1]})
        with self.assertRaises(AlignmentError):
            validate_temporal_units(frame, ["video"], "frame")

    def test_metadata_rejects_nonconsecutive_frames(self) -> None:
        frame = pd.DataFrame({"video": ["A", "A"], "frame": [0, 2]})
        with self.assertRaises(AlignmentError):
            validate_temporal_units(frame, ["video"], "frame")

    def test_metadata_rejects_fractional_frames(self) -> None:
        frame = pd.DataFrame({"video": ["A", "A"], "frame": [0.0, 1.5]})
        with self.assertRaises(AlignmentError):
            validate_temporal_units(frame, ["video"], "frame")

    def test_metadata_rejects_reopened_unit(self) -> None:
        frame = pd.DataFrame({"video": ["A", "B", "A"], "frame": [0, 0, 1]})
        with self.assertRaises(AlignmentError):
            validate_temporal_units(frame, ["video"], "frame")

    def test_clip_length_is_enforced(self) -> None:
        frame = pd.DataFrame({"video": ["A"] * 4, "clip": ["c"] * 4, "frame": range(4)})
        with self.assertRaises(AlignmentError):
            validate_temporal_units(frame, ["video", "clip"], "frame", expected_unit_length=5)

    def test_scores_cannot_bridge_video_boundaries(self) -> None:
        # The same class is active at the end/start of two videos.  Evaluating
        # units independently must count two events rather than one bridged run.
        from curconmix_repro.evaluation.segmental import Counts, evaluate_multilabel_unit

        pooled = Counts()
        for gt, probability in (
            (np.array([[0], [1]], dtype=np.int8), np.array([[0.0], [1.0]])),
            (np.array([[1], [0]], dtype=np.int8), np.array([[1.0], [0.0]])),
        ):
            pooled = pooled + evaluate_multilabel_unit(gt, probability, (0.5,))[0.5]["counts"]
        self.assertEqual((pooled.tp, pooled.fp, pooled.fn), (2, 0, 0))

    def test_scores_cannot_bridge_lls48_clip_boundaries(self) -> None:
        from curconmix_repro.evaluation.segmental import Counts, evaluate_multilabel_unit

        pooled = Counts()
        for gt in (
            np.array([[0], [0], [0], [0], [1]], dtype=np.int8),
            np.array([[1], [0], [0], [0], [0]], dtype=np.int8),
        ):
            pooled = pooled + evaluate_multilabel_unit(gt, gt.astype(float), (0.1,))[0.1]["counts"]
        self.assertEqual((pooled.tp, pooled.fp, pooled.fn), (2, 0, 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
