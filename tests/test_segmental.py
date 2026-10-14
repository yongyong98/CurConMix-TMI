#!/usr/bin/env python3
"""Deterministic unit tests for class-wise multi-label segmental F1."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from curconmix_repro.evaluation.segmental import (  # noqa: E402
    Counts,
    Event,
    count_events_across_units,
    decompose_triplets,
    evaluate_multilabel_unit,
    extract_events,
    headline_ivt_class_mask,
    match_events,
    metrics_from_counts,
    temporal_iou,
)
from curconmix_repro.temporal.ivtmetrics_cholec.disentangle import (  # noqa: E402
    Disentangle as CholecDisentangle,
)
from curconmix_repro.temporal.ivtmetrics_lls.disentangle import (  # noqa: E402
    Disentangle as LLSDisentangle,
)


class SegmentalF1Tests(unittest.TestCase):
    def assert_counts(self, actual: Counts, expected: tuple[int, int, int]) -> None:
        self.assertEqual((actual.tp, actual.fp, actual.fn), expected)

    def test_perfect_match(self) -> None:
        events = (Event(1, 5),)
        self.assert_counts(match_events(events, events, 0.50), (1, 0, 0))
        self.assertEqual(metrics_from_counts(Counts(1, 0, 0)).f1, 1.0)

    def test_boundary_shift(self) -> None:
        gt = (Event(1, 5),)
        pred = (Event(2, 6),)
        self.assertAlmostEqual(temporal_iou(gt[0], pred[0]), 3 / 5)
        for threshold in (0.10, 0.25, 0.50):
            self.assert_counts(match_events(gt, pred, threshold), (1, 0, 0))

    def test_fragmentation(self) -> None:
        gt = (Event(0, 8),)
        pred = (Event(0, 3), Event(4, 8))
        self.assert_counts(match_events(gt, pred, 0.25), (1, 1, 0))

    def test_merge_with_deterministic_tie(self) -> None:
        gt = (Event(0, 2), Event(4, 6))
        pred = (Event(0, 6),)
        self.assert_counts(match_events(gt, pred, 0.25), (1, 0, 1))

    def test_simultaneous_overlapping_classes(self) -> None:
        gt = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.int8)
        prob = gt.astype(float)
        result = evaluate_multilabel_unit(gt, prob, (0.25,))[0.25]
        self.assert_counts(result["counts"], (2, 0, 0))

    def test_false_positive_only(self) -> None:
        counts = match_events((), (Event(0, 2),), 0.25)
        self.assert_counts(counts, (0, 1, 0))
        self.assertEqual(metrics_from_counts(counts).f1, 0.0)

    def test_false_negative_only(self) -> None:
        counts = match_events((Event(0, 2),), (), 0.25)
        self.assert_counts(counts, (0, 0, 1))
        self.assertEqual(metrics_from_counts(counts).f1, 0.0)

    def test_empty_stream_is_undefined(self) -> None:
        self.assertEqual(extract_events([]), ())
        score = metrics_from_counts(Counts())
        self.assertTrue(math.isnan(score.precision))
        self.assertTrue(math.isnan(score.recall))
        self.assertTrue(math.isnan(score.f1))

    def test_video_boundary_isolation(self) -> None:
        units = [np.array([0, 1]), np.array([1, 0])]
        self.assertEqual(count_events_across_units(units), 2)

    def test_lls48_clip_boundary_isolation(self) -> None:
        clips = [np.array([0, 0, 1]), np.array([1, 0, 0])]
        self.assertEqual(count_events_across_units(clips), 2)

    def test_exact_iou_threshold_is_accepted(self) -> None:
        self.assertEqual(temporal_iou(Event(0, 4), Event(0, 1)), 0.25)
        self.assert_counts(match_events((Event(0, 4),), (Event(0, 1),), 0.25), (1, 0, 0))

    def test_exclusive_end_index(self) -> None:
        self.assertEqual(Event(0, 2).duration, 2)
        self.assertEqual(temporal_iou(Event(0, 2), Event(2, 4)), 0.0)

    def test_null_class_exclusion(self) -> None:
        cholec = headline_ivt_class_mask("CholecT45", 100)
        lls = headline_ivt_class_mask("LLS48", 195)
        self.assertEqual(np.flatnonzero(~cholec).tolist(), list(range(94, 100)))
        self.assertEqual(np.flatnonzero(~lls).tolist(), list(range(179, 195)))

    def test_decomposition_max_and_or(self) -> None:
        bank = np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [1, 0, 1, 1, 1, 1],
                [2, 1, 1, 0, 2, 2],
            ],
            dtype=int,
        )
        values = np.array([[0.1, 0.7, 0.4], [1.0, 0.0, 1.0]])
        instruments, ids = decompose_triplets(values, bank, "i")
        np.testing.assert_array_equal(ids, [0, 1])
        np.testing.assert_allclose(instruments, [[0.7, 0.4], [1.0, 1.0]])

    def test_decomposition_matches_official_dataset_implementations(self) -> None:
        rng = np.random.default_rng(20260721)
        for official in (CholecDisentangle(), LLSDisentangle()):
            probabilities = rng.random((7, official.bank.shape[0]))
            ground_truth = (probabilities >= 0.67).astype(np.int8)
            for values in (probabilities, ground_truth):
                for component in ("i", "v", "t", "iv", "it", "ivt"):
                    actual, mapping_id = decompose_triplets(values, official.bank, component)
                    expected = np.asarray(official.extract(values, component))
                    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
                    np.testing.assert_array_equal(
                        mapping_id,
                        np.unique(official.bank[:, {"ivt": 0, "i": 1, "v": 2, "t": 3, "iv": 4, "it": 5}[component]]),
                    )

    def test_probability_exactly_half_is_positive(self) -> None:
        gt = np.array([[1]], dtype=np.int8)
        result = evaluate_multilabel_unit(gt, np.array([[0.5]]), (0.25,))[0.25]
        self.assert_counts(result["counts"], (1, 0, 0))

    def test_matched_best_gt_is_fp_without_fallback(self) -> None:
        gt = (Event(0, 4), Event(5, 6))
        pred = (Event(0, 1), Event(2, 6))
        self.assert_counts(match_events(gt, pred, 0.25), (1, 1, 1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
