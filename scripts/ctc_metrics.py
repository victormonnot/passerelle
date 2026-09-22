"""Greedy CTC decoding and token error rate, without ML dependencies.

Input predictions are framewise argmax token IDs, already truncated to their
true sequence lengths. Remove batch padding before calling ``collapse_ctc``.
These utilities measure token errors; they do not establish BCI performance.

CLI input example::

    {"blank_id": 0, "trials": [
        {"reference_ids": [1, 1, 2], "prediction_ids": [1, 1, 0, 1, 2, 2]}
    ]}

Run with ``python scripts/ctc_metrics.py --input trials.json --output metrics.json``.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
import json
from itertools import zip_longest
from pathlib import Path
from typing import TypeVar


Token = TypeVar("Token")


def _require_sequence(value: object, name: str) -> None:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be an explicit sequence of tokens, not text")


def _require_ids(ids: Sequence[int], name: str) -> None:
    _require_sequence(ids, name)
    if any(type(token) is not int for token in ids):
        raise TypeError(f"{name} must contain Python integer token IDs")


def collapse_ctc(ids: Sequence[int], blank_id: int) -> list[int]:
    """Merge consecutive repeated IDs, then remove blank IDs.

    A blank separates repeats: ``[1, blank, 1]`` decodes to ``[1, 1]``.
    For tensors or NumPy arrays, pass an already truncated ``.tolist()``.
    """
    _require_ids(ids, "ids")
    if type(blank_id) is not int:
        raise TypeError("blank_id must be an explicit Python integer token ID")
    decoded: list[int] = []
    previous: int | None = None
    for token in ids:
        if token != previous and token != blank_id:
            decoded.append(token)
        previous = token
    return decoded


def edit_distance(reference: Sequence[Token], hypothesis: Sequence[Token]) -> int:
    """Return unit-cost Levenshtein distance using O(min(n, m)) memory.

    Pass explicit token sequences, e.g. ``["AA", "T"]`` or ``[1, 2]``.
    Raw strings are rejected to avoid silently computing character errors.
    """
    _require_sequence(reference, "reference")
    _require_sequence(hypothesis, "hypothesis")
    if len(reference) > len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(reference) + 1))
    for row, predicted in enumerate(hypothesis, start=1):
        current = [row]
        for column, expected in enumerate(reference, start=1):
            current.append(min(
                current[-1] + 1,
                previous[column] + 1,
                previous[column - 1] + (expected != predicted),
            ))
        previous = current
    return previous[-1]


def _corpus_counts(
    references: Iterable[Sequence[Token]],
    hypotheses: Iterable[Sequence[Token]],
) -> tuple[int, int, int]:
    if isinstance(references, (str, bytes)) or isinstance(hypotheses, (str, bytes)):
        raise TypeError("corpora must contain explicit sequences of tokens")
    missing = object()
    total_edits = total_reference_tokens = num_trials = 0
    for reference, hypothesis in zip_longest(references, hypotheses, fillvalue=missing):
        if reference is missing or hypothesis is missing:
            raise ValueError("reference and hypothesis corpora must have equal lengths")
        total_edits += edit_distance(reference, hypothesis)
        total_reference_tokens += len(reference)
        num_trials += 1
    if total_reference_tokens == 0:
        raise ValueError("error rate is undefined without any reference tokens")
    return total_edits, total_reference_tokens, num_trials


def corpus_error_rate(
    references: Iterable[Sequence[Token]],
    hypotheses: Iterable[Sequence[Token]],
) -> float:
    """Return total token edits / total reference tokens, possibly greater than 1.

    This is PER when tokens are phonemes. Individual empty references contribute
    insertions; an empty corpus or a corpus with no reference tokens raises
    ValueError. Different numbers of references and hypotheses also raise.
    """
    edits, reference_tokens, _ = _corpus_counts(references, hypotheses)
    return edits / reference_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSON trials file")
    parser.add_argument("--output", type=Path, help="JSON result file; default: stdout")
    args = parser.parse_args()
    try:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("trials"), list):
            raise ValueError("input must be an object containing a trials list")
        blank_id = payload["blank_id"]
        if type(blank_id) is not int:
            raise TypeError("blank_id must be an explicit Python integer token ID")
        references, hypotheses = [], []
        for trial in payload["trials"]:
            if not isinstance(trial, dict):
                raise ValueError("each trial must be an object")
            reference = trial["reference_ids"]
            _require_ids(reference, "reference_ids")
            if blank_id in reference:
                raise ValueError("reference_ids must not contain the CTC blank ID")
            references.append(reference)
            hypotheses.append(collapse_ctc(trial["prediction_ids"], blank_id))
        edits, reference_tokens, num_trials = _corpus_counts(references, hypotheses)
        result = {
            "num_trials": num_trials,
            "total_edits": edits,
            "total_reference_tokens": reference_tokens,
            "error_rate": edits / reference_tokens,
            "decoded_ids": hypotheses,
        }
        output = json.dumps(result, indent=2) + "\n"
        if args.output:
            args.output.write_text(output, encoding="utf-8")
        else:
            print(output, end="")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
