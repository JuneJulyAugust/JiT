# JiT Section 3.3 practice package

This package is a guided implementation of the toy experiment in Section 3.3 of *Back to Basics: Let Denoising Generative Models Denoise*. The command-line interface and configuration validation work. The tensor operations, model, training loop, sampler, and evaluation remain explicit exercises, so the package does not yet reproduce Figure 2 or provide experimental results.

Read [DESIGN.md](DESIGN.md) before writing numerical code. It derives the comparison from one sample, identifies which behavior comes from the paper and which behavior is a local implementation choice, and follows one training update in execution order. Then use [TODO.md](TODO.md) to complete the exercises in dependency order.

## Set up the package

Run every command from the repository root. Activate the existing environment before installing the package. Activation changes `python`, `pip`, and `jit-toy` for the current shell; repeat it once when opening a new shell.

```bash
source .venv/bin/activate
pip install --no-build-isolation -e ./toy_experiment
```

The editable install uses the existing PyTorch environment and makes source edits visible without reinstalling. It installs Click and Loguru from [pyproject.toml](pyproject.toml). Plotting is optional and can wait until exercise V02:

```bash
pip install --no-build-isolation -e './toy_experiment[plot]'
```

## Inspect the planned comparison

The following command validates the baseline configuration and prints the 12 planned runs: four observed dimensions crossed with clean-data, noise, and velocity prediction.

```bash
jit-toy plan --config toy_experiment/configs/baseline.json --suite
```

The JSON array goes to standard output. A Loguru record containing the source path and line number goes to standard error. This command does not import PyTorch, train a model, or create an experiment directory.

Use the smoke configuration when you need a short end-to-end wiring check after implementing the exercises:

```bash
jit-toy plan --config toy_experiment/configs/smoke.json
```

The smoke configuration requests only ten optimizer updates. Passing it will show that the parts connect; it will not establish model quality.

## Run the current checks

```bash
python -m unittest discover -s toy_experiment/tests -v
```

These tests cover configuration validation, command dispatch, output-path protection, intentional exercise errors, and source-located logging. They do not test the unfinished tensor algebra, gradients, training behavior, or sample quality.

## Implement the exercises

Each unfinished function raises `ExerciseNotImplemented` with an identifier such as D01 or M01. Find the remaining stops with:

```bash
rg -n 'ExerciseNotImplemented' toy_experiment/src/jit_toy
```

Complete the items in [TODO.md](TODO.md) in order. Keep each stop until its stated invariant has an independent check. The first exercise is D01, `sample_spiral`: for fixed count, turns, radius, and seed, return reproducible finite float32 points with shape `[count, 2]` and radius no greater than the configured limit.

## Run the completed experiment

After completing D01 through E03, the intended command sequence is:

```bash
jit-toy --log-file outputs/toy-smoke.log train \
  --config toy_experiment/configs/smoke.json \
  --output outputs/toy-smoke

jit-toy sample \
  --checkpoint outputs/toy-smoke/checkpoint.pt \
  --output outputs/toy-smoke-samples.pt \
  --device cpu

jit-toy plot \
  --samples outputs/toy-smoke-samples.pt \
  --output outputs/toy-smoke-comparison.png
```

These numerical commands currently stop at E01, E02, or E03 with a nonzero exit status. They refuse existing output paths so a practice run cannot silently overwrite earlier work. Keep the Loguru file outside the new training directory because opening the log creates its parent before `train` checks the output path.
