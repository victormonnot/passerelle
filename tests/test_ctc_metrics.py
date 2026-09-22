"""Synthetic correctness checks; none of these measure BCI model performance."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.ctc_metrics import collapse_ctc, corpus_error_rate, edit_distance


class CTCMetricsTests(unittest.TestCase):
    def test_blanks_separate_repeated_tokens(self):
        self.assertEqual(collapse_ctc([1, 1, 0, 1, 1, 2, 2, 0, 2], 0), [1, 1, 2, 2])

    def test_empty_all_blank_and_nonzero_blank(self):
        self.assertEqual(collapse_ctc([], 0), [])
        self.assertEqual(collapse_ctc([0, 0, 0], 0), [])
        self.assertEqual(collapse_ctc([9, 0, 0, 9, 0, 9], 9), [0, 0])

    def test_id_validation(self):
        for ids, blank in [("101", 0), ([1, 0.0], 0), ([True], 0), ([1], None)]:
            with self.subTest(ids=ids, blank=blank), self.assertRaises(TypeError):
                collapse_ctc(ids, blank)

    def test_insertions_deletions_and_substitutions(self):
        cases = [
            ([], [], 0),
            ([], [1, 2], 2),
            ([1, 2], [], 2),
            ([1, 2], [1, 3], 1),
            ([1, 2], [0, 1, 2], 1),
            ([0, 1, 2], [1, 2], 1),
            ([1, 2, 3], [2, 3, 4], 2),
            (["AA", "T"], ["AA", "D"], 1),
        ]
        for reference, hypothesis, expected in cases:
            with self.subTest(reference=reference, hypothesis=hypothesis):
                self.assertEqual(edit_distance(reference, hypothesis), expected)
                self.assertEqual(edit_distance(hypothesis, reference), expected)

    def test_raw_text_is_not_a_token_sequence(self):
        for reference, hypothesis in [("AA", ["AA"]), (["AA"], "AA")]:
            with self.subTest(reference=reference), self.assertRaises(TypeError):
                edit_distance(reference, hypothesis)

    def test_corpus_weights_reference_tokens_not_trials(self):
        self.assertAlmostEqual(corpus_error_rate([[1], [2] * 9], [[], [2] * 9]), 0.1)

    def test_rate_can_exceed_one(self):
        self.assertEqual(corpus_error_rate([[1]], [[1, 2, 3, 4]]), 3.0)

    def test_empty_individual_reference_counts_insertions(self):
        self.assertEqual(corpus_error_rate([[], [1, 2]], [[3], [1, 2]]), 0.5)

    def test_zero_reference_corpus_is_undefined(self):
        for references, hypotheses in [([], []), ([[]], [[]]), ([[]], [[1]])]:
            with self.subTest(references=references), self.assertRaises(ValueError):
                corpus_error_rate(references, hypotheses)

    def test_corpus_length_mismatch_in_both_directions(self):
        for references, hypotheses in [([[1]], []), ([], [[1]]), ([[1]], [[1], [2]])]:
            with self.subTest(references=references), self.assertRaisesRegex(ValueError, "equal lengths"):
                corpus_error_rate(references, hypotheses)

    def test_corpora_can_be_generators(self):
        self.assertEqual(corpus_error_rate(iter([[1], [2]]), iter([[1], [3]])), 0.5)

    def test_cli_decodes_then_aggregates(self):
        payload = {
            "blank_id": 0,
            "trials": [
                {"reference_ids": [1, 1, 2], "prediction_ids": [1, 1, 0, 1, 2, 2]},
                {"reference_ids": [3], "prediction_ids": [0, 4, 4]},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "trials.json"
            output_path = Path(directory) / "metrics.json"
            input_path.write_text(json.dumps(payload), encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, str(Path(__file__).resolve().parents[1] / "scripts" / "ctc_metrics.py"),
                 "--input", str(input_path), "--output", str(output_path)],
                capture_output=True, text=True, check=True,
            )
            self.assertEqual(completed.stdout, "")
            self.assertEqual(json.loads(output_path.read_text(encoding="utf-8")), {
                "num_trials": 2,
                "total_edits": 1,
                "total_reference_tokens": 4,
                "error_rate": 0.25,
                "decoded_ids": [[1, 1, 2], [4]],
            })


if __name__ == "__main__":
    unittest.main()
