# JiT toy experiment: guided implementation

This package provides a working command-line interface and unfinished PyTorch exercises for Section 3.3 of *Back to Basics: Let Denoising Generative Models Denoise*. You can inspect configurations now. Data generation, the model, training, sampling, and evaluation remain yours to implement; the package has no experimental results yet.

Start with [the design](DESIGN.md) for the equations and their connection to the existing JiT code. Follow [the ordered TODO list](TODO.md) to implement one exercise at a time. Each stub raises `ExerciseNotImplemented` with a matching exercise ID. Comments give tensor shapes, relevant operations, pitfalls, and checks without supplying the numerical bodies.

## Use the existing environment

Run these commands from the repository root. Activate the existing `.venv` before installing, once per shell session. The package supports Python 3.9 and later; it does not create a second environment.

```bash
source .venv/bin/activate
pip install --no-build-isolation -e ./toy_experiment

jit-toy --help
jit-toy plan --config toy_experiment/configs/smoke.json
jit-toy plan --config toy_experiment/configs/baseline.json --suite
python -m unittest discover -s toy_experiment/tests -v
```

Editable installation makes source edits visible without reinstalling. `--no-build-isolation` uses the build tools already in the environment. If those tools need updating, run `pip install 'pip>=23,<26' 'setuptools>=64'` in the activated shell before installing this package. The required dependencies are PyTorch, Click, and Loguru; plotting has an optional dependency on matplotlib. No additional dependency or environment manager is needed.

## Inspect one run before implementing it

`plan` validates partial JSON overrides and prints a resolved configuration object. `plan --suite` prints an array of 12 configurations: four observed dimensions times three prediction types. It plans the comparison; it does not train the models. Both commands write JSON to standard output and logs to standard error. All paths are relative to the current working directory.

```bash
jit-toy --log-file outputs/toy/plan.log plan --suite
```

Each log message includes its timestamp, severity, module, function, source path, and line number. Global logging options precede the command. The [Click quickstart](https://click.palletsprojects.com/en/stable/quickstart/) and [Loguru logger reference](https://loguru.readthedocs.io/en/stable/api/logger.html) describe the two libraries used at this boundary.

## Run the experiment after completing the exercises

These commands are the intended workflow. They currently exit nonzero at an exercise stub and do not create training or sampling artifacts.

```bash
jit-toy --log-file outputs/toy/smoke.log train \
  --config toy_experiment/configs/smoke.json --output outputs/toy/smoke
jit-toy sample --checkpoint outputs/toy/smoke/checkpoint.pt \
  --output outputs/toy/smoke/samples.pt --device cpu

# Install plotting when you reach V02, in the same activated environment.
pip install --no-build-isolation -e './toy_experiment[plot]'
jit-toy plot --samples outputs/toy/smoke/samples.pt \
  --output outputs/toy/smoke/comparison.png
```

Keep the training log outside the new run directory: opening a log creates its parent, while `train` rejects an existing run directory. Sampling and plotting also reject existing output files. The smoke configuration requests ten updates to check the eventual workflow; it is not a model-quality experiment.

## Complete the next exercise

Find the stubs with:

```bash
rg -n 'ExerciseNotImplemented|[DMFTSVE][0-9]{2}' toy_experiment/src/jit_toy
```

The supplied tests cover configuration, command dispatch, error handling, and logging. They do not establish numerical correctness. Write the shape, algebra, and gradient checks in [TODO.md](TODO.md) as you implement each exercise. Begin with **D01, `sample_spiral`**: return finite float32 points of shape `[N,2]` with reproducible draws from the supplied generator.
