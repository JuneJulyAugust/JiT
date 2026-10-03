# JiT Section 3.3 practice package

The package validates configurations and dispatches commands, but all numerical functions remain exercises. It has produced no trained model or generated samples. The goal is to implement the toy comparison in Section 3.3 of *Back to Basics: Let Denoising Generative Models Denoise*: predict clean data, noise, or velocity while holding the data, network, training budget, and velocity loss fixed.

Read [DESIGN.md](DESIGN.md) for the equations, controls, and implementation contracts. Use [TODO.md](TODO.md) to implement and check one exercise at a time.

## Contents

- [1. Set up the package](#1-set-up-the-package)
- [1.1 View the equations](#11-view-the-equations)
- [2. Inspect the planned comparison](#2-inspect-the-planned-comparison)
- [3. Check the working infrastructure](#3-check-the-working-infrastructure)
- [4. Implement the numerical exercises](#4-implement-the-numerical-exercises)
- [5. Run after completing the exercises](#5-run-after-completing-the-exercises)
- [6. Next action](#6-next-action)

## 1. Set up the package

Run commands from the repository root. The following commands assume the existing `.venv` environment is available. Activate it once in each new shell, then install the package:

```bash
source .venv/bin/activate
pip install --no-build-isolation -e ./toy_experiment
```

Activation selects the environment's Python and installed commands. The editable install makes later source edits visible without reinstalling. [pyproject.toml](pyproject.toml) declares Python 3.9 or later, PyTorch, Click for commands, and Loguru for logs. The existing environment must also supply the setuptools build requirement because this command disables build isolation.

For exercise V02, install the optional plotting dependency:

```bash
pip install --no-build-isolation -e './toy_experiment[plot]'
```

### 1.1 View the equations

In Cursor or VS Code, open a Markdown file and run **Markdown: Open Preview to the Side** from the command palette. This selects the built-in preview. The repository's [.vscode/settings.json](../.vscode/settings.json) explicitly enables `markdown.math.enabled` and leaves `markdown.math.macros` empty because these documents use standard commands. The installed Cursor math extension renders inline expressions between single dollar signs and displayed equations between double dollar signs using KaTeX; a separate preview extension can use a different parser or configuration. These workspace settings apply to the built-in preview; they do not configure Cherry or GitHub.

The documents use those dollar delimiters, named norm and brace commands, and `\cr` for matrix and aligned-equation row breaks. Code setting names remain in prose rather than inside equations. These choices avoid Markdown consuming backslashes or treating underscores in text labels as math syntax. GitHub supports dollar-delimited math, but uses MathJax and imposes its own command restrictions; support for a command in KaTeX alone does not establish support on GitHub. See the [GitHub math syntax](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions) and [KaTeX row separators](https://katex.org/docs/supported#environments).

Cherry has separate `inlineMath` and `mathBlock` hooks. In a Cherry application using MathJax, load the renderer before creating the editor and supply it through `externals.MathJax`; enable both hooks. If MathJax also scans dollar-delimited text, its `tex.inlineMath` configuration must include single-dollar delimiters. These are application initialization options, not settings exposed by the installed Cherry IDE extension. The built-in Cursor preview provides a direct way to read these files while diagnosing a Cherry preview failure. See [Cherry configuration](https://github.com/Tencent/cherry-markdown/blob/main/packages/cherry-markdown/src/Cherry.config.js) and [MathJax inline delimiters](https://docs.mathjax.org/en/latest/input/tex/delimiters.html).

Native equation rendering depends on the viewer and its configuration. For a distribution that must look the same in viewers without math support, use a PDF or pre-rendered equation images. Keep these Markdown files as the editable mathematical source.

## 2. Inspect the planned comparison

For example, a clean-data run at observed dimension $D=8$ embeds two-coordinate samples into eight-coordinate vectors. The suite repeats that setting for noise and velocity prediction, then repeats the three targets at the other dimensions. This command prints all 12 configurations:

```bash
jit-toy plan --config toy_experiment/configs/baseline.json --suite
```

The JSON array goes to standard output; the validation log, including source path and line number, goes to standard error. The suite changes only `observed_dim` and `prediction`. It covers $D\in\lbrace 2,8,16,512\rbrace$ and command values `x`, `eps`, and `v`. A successful plan establishes that the configurations are valid; it does not execute numerical code or create a run directory.

To inspect the smaller smoke configuration, which is intended to check that the completed components connect, run:

```bash
jit-toy plan --config toy_experiment/configs/smoke.json
```

This plan resolves to ten training updates and five sampling intervals. Those counts are too small to establish sample quality.

## 3. Check the working infrastructure

```bash
python -m unittest discover -s toy_experiment/tests -v
```

The supplied tests cover configuration validation, command dispatch, refusal to overwrite outputs, reporting of exercise exceptions, and logs that name the actual calling source line. They also check that help and planning work without importing PyTorch. They do not check tensor algebra, gradients, training convergence, or generated samples.

A passing infrastructure suite is the starting condition for the exercises. Each numerical exercise needs its own check from [TODO.md](TODO.md) before the next dependent exercise begins.

## 4. Implement the numerical exercises

An unfinished function raises `ExerciseNotImplemented` with an exercise identifier such as D01 or M01. Find those functions with:

```bash
rg -n 'ExerciseNotImplemented' toy_experiment/src/jit_toy
```

Begin with D01, `sample_spiral`. With count $N$, fixed turns and radius, and an explicit random generator, it must return finite float32 points of shape $N\times2$ whose radial distance does not exceed the configured radius. Two fresh generators with the same seed must return identical points. The equations and the reason radius and angle share one random draw are in [DESIGN.md, Section 1](DESIGN.md#1-draw-the-clean-data).

Keep an exercise exception in place until the function has an implementation and its independent check passes. The checklist order follows the data path: data, network, flow conversions, training, sampling, then saved artifacts and plotting.

## 5. Run after completing the exercises

After D01 through E03 pass their checks, use these commands for the first complete run:

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

Training must save a checkpoint before sampling can begin; sampling must save its artifact before plotting can begin. Currently, the corresponding command reaches E01, E02, or E03 and exits with an exercise error when its input-path checks pass. None of these commands currently supplies the missing numerical implementation.

Choose new output paths for each run. Commands reject existing outputs. Keep the Loguru file outside the new training directory: opening the log creates its parent directory before `train` checks that the run directory is absent.

A completed smoke run must produce parseable metrics, a reloadable checkpoint, finite samples, and a plot in a new process. Passing that sequence establishes that the components connect. The controlled 12-run comparison and repeated seeds are needed to evaluate the paper's qualitative result.

## 6. Next action

Implement D01 with count, turns, radius, dtype, and seed fixed. Pass it only when shape, finiteness, radial bounds, and fresh-generator reproducibility all hold. If a check fails, inspect the shared random draw and coordinate stacking before moving to D02.
