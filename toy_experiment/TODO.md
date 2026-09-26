# Practice TODO: complete one item at a time

All numerical bodies remain unfinished exercises. Their IDs match the exceptions and comments under `src/jit_toy/`. Implement one item, write the indicated check, run it, and then tick the box. The supplied infrastructure tests do not verify these numerical contracts.

## Contents

- [1. Orientation (working infrastructure)](#1-orientation-working-infrastructure)
- [2. Data operations — `data.py`](#2-data-operations--datapy)
- [3. Neural network — `model.py`](#3-neural-network--modelpy)
- [4. Flow operations — `flow.py`](#4-flow-operations--flowpy)
- [5. Training — `training.py`](#5-training--trainingpy)
- [6. Sampling — `sampling.py`](#6-sampling--samplingpy)
- [7. Evaluation and artifacts](#7-evaluation-and-artifacts)
- [8. Run and interpret](#8-run-and-interpret)
- [9. Optional exercises after the baseline](#9-optional-exercises-after-the-baseline)

## 1. Orientation (working infrastructure)

- [ ] Read DESIGN.md, paper Section 3.3/Table 1, and the JiT reference mapping.
- [ ] Activate with `source .venv/bin/activate` from the repository root, then run the editable install, `jit-toy --help`, `jit-toy plan --suite`, and infrastructure tests from README.md.
- [ ] Draw [B,D], [B,1], and [D,2] on paper; derive every matrix-product output shape.

## 2. Data operations — `data.py`

- [ ] **D01 — `sample_spiral`**: vectorized random polar coordinates. Practice `rand`, arithmetic, `sin`, `cos`, `stack`. Check [N,2], dtype, finite values, radial bounds and repeatable seeds; plot once.
- [ ] **D02 — `make_projection`**: random reduced QR decomposition (orthonormal Q and triangular R). Practice `randn`, `linalg.qr`, transpose, `eye`, `allclose`. Check P.T@P≈I at D=2,8,16,512; same seed gives same P on the same runtime.
- [ ] **D03 — `embed_points`**: row-batch multiplication. Check [N,D] and preserved distances/norms; no D-dependent rescaling.
- [ ] **D04 — `project_points`**: derive inverse-on-subspace row multiplication. Check embed/project round-trip, including N=1.
- [ ] **D05 — `ObservedDataset.__init__/__len__/__getitem__`**: storage and indexing. Check int indexing gives [D], len=N, out-of-range behavior; no P or intrinsic coordinates in examples.
- [ ] **D06 — `build_data`**: separate random-generator streams and fixed projection. Verify data and P are identical across x/eps/v with shared config seeds, and held-out samples differ from training samples.
- [ ] **D07 — `make_loader`**: batching/collation/shuffling. Use N=10,B=4: expect [4,D],[4,D],[2,D]; verify all examples occur once per epoch. Reference `main_jit.py::main`.

## 3. Neural network — `model.py`

- [ ] **M01 — `ToyMLP.__init__`**: register four Linear hidden layers with rectified linear unit (ReLU) activations, then a fifth Linear layer for raw output. Inspect `named_parameters` and `state_dict`; compare registration patterns with `model_jit.py::JiT.blocks` / `TimestepEmbedder.mlp`.
- [ ] **M02 — `ToyMLP.forward`**: concatenate time on features and compute [B,D] outputs. Check B=1 and B=D; backward reaches all layers with finite gradients. Verify no P input or long residual bypass. Reference `FinalLayer` for unconstrained outputs.

## 4. Flow operations — `flow.py`

- [ ] **F01 — `sample_times`**: logit-normal [B,1] from explicit mean, standard deviation, margin, and generator; no full experiment configuration. Reference `Denoiser.sample_t`; inspect histogram, finite values, deterministic generator use.
- [ ] **F02 — `make_flow_batch`**: interpolation/target with full D-dimensional noise. Check t=0 and t=1 algebra on fixed inputs, and interior mixed values. Reference `Denoiser.forward`.
- [ ] **F03 — `to_velocity`**: implement all three Table 1 row-3 branches. Feed oracle clean/noise/velocity values and compare recovered v; check B=1 and B=D, reject unknown modes. Inspect gradients through raw outputs.
- [ ] **F04 — `velocity_loss`**: feature mean then batch mean, matching `Denoiser.forward`. Verify a hand-calculated scalar example and gradient; reject broadcasting shape mistakes.

## 5. Training — `training.py`

- [ ] **T01 — `build_optimizer`**: accept a parameter iterable and learning rate; use AdamW, explicit zero decay, and JiT betas (0.9,0.95). Confirm every registered trainable parameter is included without exhausting a one-shot parameter generator.
- [ ] **T02 — `train_step`**: accept a prepared `FlowBatch` and prediction type; perform forward, conversion, loss, zero-grad/backward/step, and return scalar metrics. Do not draw noise or write logs. Reference `engine_jit.py::train_one_epoch`. Check weights change and gradients do not accumulate; overfit one fixed noisy minibatch for each prediction mode.
- [ ] **T03 — `train`**: accept the optimizer, generator, and metrics callback. Implement loader restart, device transfer, time/noise draws, and an exact update budget. With an in-memory callback, check steps 1 through train_steps, short final batches, and no report for a failed update. Let callback errors propagate.
- [ ] **E01 — `run_training` in `experiment.py`**: create the optimizer and flow generator, own the metric file and logging callback, wire D/M/F/T, and save config/environment/checkpoint. Reference `main_jit.py::main`, `util.misc.save_model`. Check JSON Lines records at log_every and the final step, source-located logs, same-seed initialization, checkpoint metadata, and protection of previous runs.

## 6. Sampling — `sampling.py`

- [ ] **S01 — `ode_step`**: Euler then Heun. Reference `Denoiser._euler_step/_heun_step`; test a constant field and a known nonconstant ODE. Average velocities at the correct time/state.
- [ ] **S02 — `sample`**: CPU noise generator, inference mode, interior grid, [B,1] times, observed-space integration. Reference `Denoiser.generate` and documented endpoint differences. Check no graph, exact sample count, mode restoration, reproducibility.

## 7. Evaluation and artifacts

- [ ] **V01 — `subspace_residual` in `evaluation.py`**: project/re-embed/residual. Check embedded clean data gives ≈0 and a known perpendicular component increases error for D>2. At D=2 this metric alone says nothing about spiral quality.
- [ ] **E02 — `run_sampling` in `experiment.py`**: strict CPU checkpoint load, restore exact P, run S02, save sample tensors and V01. Reference `sample_jit.py::main`; check model output equality before/after reload.
- [ ] **V02 — `plot_comparison` in `evaluation.py`**: optional matplotlib, tensor-to-NumPy boundary, equal axes, transparent scatter, headless save/close. Ensure outliers cannot silently disappear.
- [ ] **E03 — `run_plotting` in `experiment.py`**: validate/load saved sample schema and call V02. Verify plots work in a new process and malformed artifacts fail clearly.

## 8. Run and interpret

- [ ] Complete the smoke config end-to-end. Ten updates only check wiring.
- [ ] Save `jit-toy plan --suite` output, split its 12 entries into single-run JSON configs, and train each into a distinct directory. Hold P/data/budget/initialization fixed across prediction modes at a given D.
- [ ] Assemble a Figure 2-style 4x4 panel from saved samples, with residual metrics and recorded settings.
- [ ] Compare the qualitative trend honestly; repeat seeds, training budget, delta, and solver resolution before concluding.

## 9. Optional exercises after the baseline

- [ ] Practice `arange`, `exp`, `unsqueeze`, `cat`, and padding by adding sinusoidal time embeddings following `model_jit.py::TimestepEmbedder`.
- [ ] Practice `reshape`, `permute`, `einsum`, and contiguous/view semantics in a separate shape notebook using `JiT.unpatchify` / `Attention.forward`; do not insert image axes into the toy baseline.
- [ ] Add a second intrinsic dataset and a second model implementing the same interfaces.
- [ ] Add an exponential moving average using `Denoiser.update_ema` / `engine_jit.evaluate` as references; validate updates without autograd tracking.
- [ ] Add proper training resume (optimizer, step, random-generator states and sampler position), with uninterrupted-vs-resumed comparison.
- [ ] Add an explicit x-only JiT endpoint-policy ablation; keep the primary three-mode comparison on one shared policy.

Begin with D01 and its shape, radius, dtype, and reproducibility checks. If a check fails, correct the sampling operation before moving to D02.
