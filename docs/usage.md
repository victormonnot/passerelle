# Using the current tools

[Back to Passerelle](../README.md)

Passerelle is at the tooling stage, before V1. This guide covers the implemented
utilities. Neural dataset loading and pretrained decoder inference are not yet
implemented.

## Set up Python

Use Python 3.12 on Linux or WSL2. From the repository root:

```sh
python3.12 -m venv .venv
```

The CTC utility and its tests use only the Python standard library. They can run
without installing the pinned dependencies or using a GPU.

## CTC decoding and error rate

The [metrics utility](../scripts/ctc_metrics.py) accepts framewise argmax token
IDs and reference token IDs in JSON. Predictions must be truncated to their
actual lengths before evaluation. References must contain neither padding nor
the CTC blank token. This utility starts from predicted IDs; it does not run a
neural model.

Create the output directory:

```sh
mkdir -p reports
```

Save this synthetic example as `reports/trials.json`:

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

```json
{
  "num_trials": 1,
  "total_edits": 0,
  "total_reference_tokens": 3,
  "error_rate": 0.0,
  "decoded_ids": [[1, 1, 2]]
}
```

The error rate is calculated across all trials:

```text
error_rate = total_token_edits / total_reference_tokens
```

This ratio is phoneme error rate (PER) when the tokens represent phonemes. It is
not word error rate (WER) unless the tokens represent words. It can exceed 1
when there are more edits than reference tokens; a corpus without reference
tokens is rejected.

The zero error rate above comes from a deliberately simple synthetic example.
It is not a speech decoding result.

## PyTorch environment

The full pinned environment targets **Linux x86_64 and Python 3.12**, including
WSL2. The CUDA diagnostic also requires an NVIDIA GPU and a compatible driver.

After creating `.venv`, install the dependencies from the repository root:

```sh
.venv/bin/python -m pip --isolated install --require-hashes -r requirements.lock
```

The environment includes PyTorch 2.10.0 with CUDA 12.8, NumPy 2.4.6 and
SciPy 1.18.0. Allow approximately **3.9 GiB of downloads and 7.1 GiB of installed
disk space** for the dependencies.

[`requirements.in`](../requirements.in) lists direct dependencies.
[`requirements.lock`](../requirements.lock) pins their transitive dependencies
and wheel hashes for Linux x86_64 and Python 3.12.

## Hardware diagnostic

The [hardware script](../scripts/check_hardware.py) can collect information
without importing PyTorch:

```sh
.venv/bin/python scripts/check_hardware.py --inventory-only
```

With the pinned environment installed, run the CUDA check and save its report:

```sh
.venv/bin/python scripts/check_hardware.py --output reports/hardware.json
```

The diagnostic inventories CPU, available RAM, cgroup memory limits, disk space
and NVIDIA GPU memory. It runs a seeded FP32 512×512 matrix product and
backpropagation on CUDA, compares outputs and gradients with CPU, and reports
peak GPU allocations. It exits with a nonzero status if CUDA is unavailable,
numerical comparisons fail, or the Python environment is not isolated.

Diagnostic timings establish functionality; they are not model benchmarks.
Generated reports are excluded from version control.

## Validation

Run the synthetic tests:

```sh
.venv/bin/python -m unittest discover -s tests -v
```

The tests cover CTC blank handling, insertions, deletions, substitutions, corpus
weighting, input validation and the JSON command-line interface. They use
synthetic inputs to verify correctness; no neural decoding performance is
claimed.

If you installed the PyTorch environment, check its dependencies as well:

```sh
.venv/bin/python -m pip check
```
