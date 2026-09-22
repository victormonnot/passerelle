# Passerelle

Tools for reproducible intracortical speech decoding experiments.

The current implementation provides a PyTorch CUDA diagnostic, greedy CTC token
decoding, and corpus-level token error metrics. Neural dataset loading and
pretrained decoder inference are not yet implemented.

## Requirements

- Linux x86_64 or WSL2, with Python 3.12.
- An NVIDIA GPU and a compatible driver for the CUDA diagnostic.

The pinned environment includes PyTorch 2.10.0 with CUDA 12.8, NumPy 2.4.6 and
SciPy 1.18.0. Dependencies require approximately 3.9 GiB of downloads and 7.1 GiB
of installed disk space. CTC metrics and their tests use only the Python standard
library and can run without a GPU.

## Installation

From the repository root:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip --isolated install --require-hashes -r requirements.lock
```

`requirements.in` lists direct dependencies. `requirements.lock` pins their
transitive dependencies and wheel hashes for Linux x86_64 and Python 3.12.

## Hardware diagnostic

```sh
.venv/bin/python scripts/check_hardware.py --output reports/hardware.json
```

The diagnostic inventories CPU, available RAM, cgroup memory limits, disk space
and NVIDIA GPU memory. It runs a seeded FP32 512×512 matrix product and
backpropagation on CUDA, compares outputs and gradients with CPU, and reports
peak GPU allocations. It exits with a nonzero status if CUDA is unavailable,
numerical comparisons fail, or the Python environment is not isolated.

To collect hardware information without importing PyTorch:

```sh
.venv/bin/python scripts/check_hardware.py --inventory-only
```

Diagnostic timings establish functionality; they are not model benchmarks.
Generated reports are excluded from version control.

## CTC decoding and error rate

The metrics utility accepts framewise argmax token IDs and reference token IDs
in JSON. Predictions must be truncated to their actual lengths before evaluation.
References must contain neither padding nor the CTC blank token.

Create `reports/trials.json` using this synthetic example (create the `reports/`
directory first if needed):

```json
{
  "blank_id": 0,
  "trials": [
    {"reference_ids": [1, 1, 2], "prediction_ids": [1, 1, 0, 1, 2, 2]}
  ]
}
```

Then run:

```sh
.venv/bin/python scripts/ctc_metrics.py --input reports/trials.json --output reports/metrics.json
```

CTC decoding merges consecutive repeated IDs before removing blanks. The example
decodes to `[1, 1, 2]`. The report includes decoded sequences, total edit distance,
reference-token count and the aggregate error rate:

```text
error_rate = total_token_edits / total_reference_tokens
```

This ratio is phoneme error rate (PER) when the tokens represent phonemes. It is
not word error rate (WER). It can exceed 1 when there are more edits than
reference tokens; a corpus without reference tokens is rejected.

## Validation

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m pip check
```

The tests cover CTC blank handling, insertions, deletions, substitutions, corpus
weighting, input validation and the JSON command-line interface. They use
synthetic inputs to verify correctness; no neural decoding performance is
claimed.
