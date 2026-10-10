# Passerelle

**Work in progress. V1 isn't ready yet.**

Passerelle is a brain-computer interface (BCI) research project. The idea is to
use brain signals as a way to communicate with software. The first focus is
speech decoding from recordings made by sensors implanted in the brain.

The project is still at the tooling stage. There is no neural dataset loading
or model inference yet, and no speech decoding result to show.

## What exists so far

- **Decoding and error metrics.** Small utilities that turn a sequence of
  predicted token IDs into an output sequence and compare it with a reference.
- **A hardware check.** A script that inventories the machine and verifies that
  a small PyTorch calculation works on the GPU.
- **Synthetic tests.** Checks for the decoding and metrics utilities, including
  repeated tokens, missing tokens and invalid inputs.

These are the first pieces for future experiments. The tests verify the tools;
they don't measure how well a model decodes brain signals.

## What's next

The first goal is to run an existing speech decoder on recorded neural data,
compare its output with the reference, and make the result easy to inspect and
reproduce. That baseline needs to work before moving on to new experiments.

## Try the current tools

Use **Python 3.12 on Linux or WSL2**. The decoding and metrics tests need no GPU
or extra packages.

```sh
git clone https://github.com/victormonnot/passerelle.git
cd passerelle
python3.12 -m venv .venv
.venv/bin/python -m unittest discover -s tests -v
```

To try the metrics on a small example, follow the
[CTC walkthrough](docs/usage.md#ctc-decoding-and-error-rate). It uses synthetic
token IDs and runs without downloading a model or dataset.

## Technical details

The [usage guide](docs/usage.md) covers the JSON input format, how the error rate
is calculated, the optional [PyTorch environment](docs/usage.md#pytorch-environment)
and the [hardware diagnostic](docs/usage.md#hardware-diagnostic).

The implementation is in [scripts/](scripts/); the synthetic checks are in
[tests/test_ctc_metrics.py](tests/test_ctc_metrics.py).
