# JiT toy experiment: guided implementation

A practice subpackage for Section 3.3 of *Back to Basics: Let Denoising
Generative Models Denoise*. Start with [DESIGN.md](DESIGN.md), then work through
[TODO.md](TODO.md) in order. The scaffold follows the existing JiT code's flow
formulation, loss reduction, optimizer settings, and update/ODE conventions;
the design explains the toy-specific adaptations.

**Implemented:** packaging, validated configuration, Click commands, Loguru
console/file logging with source paths and line numbers, and a 12-case plan.
**Left for you:** every data tensor operation, the whole model, flow math,
training, sampling, evaluation, and experiment/checkpoint integration.
Each unfinished method raises `ExerciseNotImplemented` with its TODO ID.
There are no supplied model solutions, fake training runs, or generated results.

## Use the existing `.venv`

Run commands from the **repository root**. Its `.venv` currently points to the
shared `/Users/fang/.venv` (Python 3.9); this project does not create another one.
`pyproject.toml` manages this independent distribution, installed as `jit_toy`.

```bash
# Modern pip supports editable pyproject packages. No torch upgrade requested.
.venv/bin/python -m pip install 'pip>=23,<26' 'setuptools>=64'
.venv/bin/python -m pip install --no-build-isolation -e ./toy_experiment

.venv/bin/python -m jit_toy --help
.venv/bin/jit-toy plan --config toy_experiment/configs/smoke.json
.venv/bin/jit-toy plan --config toy_experiment/configs/baseline.json --suite
.venv/bin/python -m unittest discover -s toy_experiment/tests -v
```

Use explicit `.venv/bin/...` paths to select the shared environment. Editable
installation makes source edits immediately visible. Do not run `uv sync` here:
there is no separate project environment or lockfile. Dependency lower bounds
allow the existing PyTorch installation to be reused. The smoke config is for
eventual wiring checks, not meaningful model quality; even it cannot train yet.

`plan` emits one JSON object, or a 12-element JSON array with `--suite`, to stdout.
Logs go to stderr. A suite is a plan only, not an automatic training launcher.
JSON configs may contain partial overrides; unknown keys and invalid values fail.
All paths are relative to the current working directory, not the config's parent.

```bash
# Working now: validate/inspect the design and write source-located logs.
.venv/bin/jit-toy --log-file outputs/toy/plan.log plan --suite

# After completing the exercises below, these become executable workflows.
.venv/bin/jit-toy --log-file outputs/toy/smoke.log train \
  --config toy_experiment/configs/smoke.json --output outputs/toy/smoke
.venv/bin/jit-toy sample --checkpoint outputs/toy/smoke/checkpoint.pt \
  --output outputs/toy/smoke/samples.pt --device cpu

# Install plotting only when you reach V02; this still uses the same .venv.
.venv/bin/python -m pip install --no-build-isolation -e './toy_experiment[plot]'
.venv/bin/jit-toy plot --samples outputs/toy/smoke/samples.pt \
  --output outputs/toy/smoke/comparison.png
```

Put global logging options before the command. Keep a training log outside the
new run directory, since opening a log creates its parent and `train` refuses
existing run directories. Numerical commands exit nonzero with an exercise ID
until completed. No checkpoint or sample is produced by a placeholder.

## Find your next exercise

```bash
rg -n 'ExerciseNotImplemented|[DMFTSVE][0-9]{2}' toy_experiment/src/jit_toy
```

The comments contain shapes, equations, relevant PyTorch operations, pitfalls,
acceptance criteria, and references to the corresponding full JiT functions.
Replace the raise only after implementing that exercise. Write numerical tests
yourself as you progress; the supplied tests check the working CLI/config/logging
infrastructure and intentionally make no claim about numerical correctness.

Useful API references: [Click quickstart](https://click.palletsprojects.com/en/stable/quickstart/)
and [Loguru logger API](https://loguru.readthedocs.io/en/stable/api/logger.html).
