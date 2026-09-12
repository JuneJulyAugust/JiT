# Section 3.3 toy experiment design

Status: approved scope is a **learning scaffold**, with numerical implementation
reserved for the learner. This document specifies the intended completed system;
only configuration, CLI, and logging currently execute. No reproduction result
or convergence claim is implied by the scaffold tests.

## Scientific question and source

Can a fixed-width network generate low-dimensional clean data embedded in a
high-dimensional observation space more easily when it directly predicts clean
data than when it predicts noise or velocity?

Primary source: [local paper](../docs/Back%20to%20Basics-%20Let%20Denoising%20Generative%20Models%20Denoise.pdf),
arXiv:2511.13720v2, January 7, 2026. Read Section 3.1 (page 3), Table 1,
Figure 2 and Section 3.3 (pages 4-5). Figure 2 visually shows a spiral. The PDF
is an existing local reference, not a file added by this scaffold's commit.

| Item | Paper requirement or stated observation |
| --- | --- |
| Intrinsic space | d=2 |
| Observed space | D in {2,8,16,512}; D=2 is the no-expansion control |
| Embedding | Random fixed P of shape [D,d], with P-transpose P = I |
| Network access | Only observed noisy data and time; P is unknown to the model |
| Architecture | Five-layer ReLU MLP, hidden width 256 |
| Direct outputs | x, epsilon, v, compared in separate runs |
| Common loss | Velocity-space loss for all three output types |
| Display | Generated D-dimensional points projected back with P |
| Reported trend | x-prediction remains effective as D grows; eps/v degrade |

The exact spiral sampling density/scale, data counts, training budget, optimizer,
toy time-distribution parameters, time-conditioning design, and layer-count
convention are not specified in Section 3.3. Section 3.1 gives logit-normal time
sampling and a default 50-step Heun solver. The choices below make a reproducible
practice baseline, not a claim of the authors' exact Figure 2 implementation.

## Use the existing JiT implementation as the reference

Read these functions beside the exercises. Reuse their concepts and operation
order; keep the subpackage independent of image training imports (which pull in
DDP, torchvision, CUDA paths, and FID tooling).

| Existing code | Toy counterpart | Preserved behavior and adaptation |
| --- | --- | --- |
| [Denoiser.sample_t](../denoiser.py) | F01 | Normal draw, affine mean/std, sigmoid; [B,1] time column |
| Denoiser.forward | F02-F04 / T02 | Linear interpolation, full-dimensional Gaussian noise, external x-to-v conversion, dimension mean then batch mean |
| Denoiser._forward_sample | F03 | x-to-v conversion; add Table 1 eps/v branches, omit class guidance |
| Denoiser._euler_step / _heun_step | S01 | Same velocity-based Euler predictor and Heun correction algebra |
| Denoiser.generate | S02 | Noise initialization, steps+1 grid, disabled gradients; endpoint policy differs below |
| [TimestepEmbedder, JiT.blocks, FinalLayer](../model_jit.py) | M01-M02 | Registered layers, time conditioning, unconstrained output; toy uses scalar concat/ReLU MLP |
| [train_one_epoch](../engine_jit.py) | T02-T03 | train mode, move data, finite loss, zero gradients, backward, optimizer step |
| [main](../main_jit.py) | D07 / T01 / E01 | Seed/device/loader/model/optimizer lifecycle; AdamW betas=(0.9,0.95), zero weight decay |
| [save_model](../util/misc.py) | E01 | Tensor state_dict checkpointing; separate inference schema for the toy |
| [sampling demo main](../sample_jit.py) | E02 / S02 | Device checks, CPU checkpoint load, weights_only=True, strict state load, inference_mode |

Important deliberate differences:

- JiT handles image tensors [B,C,H,W] and reshapes time for broadcasting; the toy
  uses [B,D] and [B,1]. No image normalization, patchification, labels, or CFG.
- The MLP uses four ReLU hidden layers and one unconstrained Linear output layer,
  totaling five Linear layers. Scalar time concatenation makes input width D+1.
  No full input-output skip or analytic noise passthrough bypasses the bottleneck.
  Default Linear initialization is a practice choice; Transformer-specific
  adaLN/output zero initialization is not transplanted to a plain ReLU MLP.
- The toy borrows P_mean=-0.8 and P_std=0.8 from `main_jit.py`. Learning rate
  0.001 is absolute, not scaled with batch size; the toy uses a fixed update
  budget, no warmup, EMA, DDP, autocast, or unconditional CUDA synchronization.
- `main_jit.py` drops incomplete training batches; this scaffold retains them
  intentionally to practice correct dynamic batch handling.
- JiT's actual target is `(x-z)/(1-t).clamp_min(t_eps)`. Our interior time policy
  guarantees `1-t >= time_eps`, so `x-eps` is equivalent apart from rounding.
  If changing back to JiT's unrestricted sigmoid times plus denominator clamp,
  do not assume that equivalence near t=1; update target and prediction together.

## Data and tensor contracts

All creation defaults to CPU float32 with explicit `torch.Generator` streams.
The trainer moves observed batches to the selected device. CPU is the debugging
default; MPS/CUDA are explicit options once implemented. Cross-device floating
point equality is not promised; record the runtime and save actual tensors.

| Quantity | Shape | Ownership |
| --- | --- | --- |
| Intrinsic rows, x_hat | [N,2] | Data factory / evaluation |
| P | [D,2] | Data metadata / checkpoint, fixed per dimension and projection seed |
| Clean observed rows, x | [B,D] | Dataset / trainer |
| Gaussian epsilon | [B,D] | Flow batch builder |
| Time t | [B,1] | Flow / sampler |
| Noisy rows z, raw output, velocity | [B,D] | Model / flow / sampler |
| Loss | scalar tensor | Trainer, differentiable until backward |

Chosen intrinsic recipe: u uniform in [0,1), angle=2*pi*turns*u, radius=R*u,
Cartesian coordinates `(radius*cos(angle), radius*sin(angle))`. Defaults: two
turns, R=2, no added data jitter. Uniform u is not uniform arc length. Preserve
this density across comparisons. Use a random thin QR factor for P. With row
batches, embedding is `X_hat @ P.T`, visualization is `X @ P`; no ambient-axis
standardization or sqrt(D) rescaling is applied. Noise stays standard Gaussian
in all D dimensions, as in JiT's noise_scale=1 case.

RNG streams: independent CPU generators for training points (`seed`), held-out
points (`seed+1`), data shuffling (`seed+2`), flow draws (`seed+3`), and projection
(`projection_seed`). Seed model initialization separately with `torch.manual_seed`.
Sampling gets a fresh evaluation generator (`seed+1`); generating plots must not
advance training RNG. Share seeds, P, intrinsic train data, and initial model
state across prediction types at a fixed D. Reinitialize each run independently.

## Flow math and numerical policy

From Eqs. 1-2: `z_t = t*x + (1-t)*eps`, `v = x-eps`. Time increases from noise
toward data. The model's raw output is exactly one of x/eps/v. Table 1 row 3 gives:

| Prediction | Velocity used for both optimization and sampling |
| --- | --- |
| x | `(raw-z)/(1-t)` |
| eps | `(z-raw)/t` |
| v | `raw` |

All modes optimize `mean_B(mean_D((v_pred-v_target)^2))`, following
`Denoiser.forward`. This is the paper's expected squared norm divided by D, a
constant per case. Record it as `loss`, optionally also `loss_sum=loss*D`.
Do not compare raw losses from different prediction spaces or silently change
the reduction. No additional time weighting is applied after converting to v.

The first baseline uses clamped logit-normal training times on
`[delta,1-delta]`, delta=`time_eps`=0.001. Integration uses the same interval
for all three modes. Start from a standard Gaussian approximation at delta,
advance 50 uniform intervals (51 grid points), and return the state at 1-delta.
Euler uses one velocity evaluation; Heun uses a provisional Euler state and
the average of the start/end velocities, updating the original state.

This avoids evaluating eps conversion at zero or x conversion at one. It is
an approximation: the true distribution at delta is already slightly mixed with
data, and the final state is near-data. It is **not** JiT's exact default endpoint
policy: `Denoiser.generate` integrates 0->1, clamps the x denominator at t_eps
(default 0.05), and finishes with Euler. That implementation only supports x
prediction. A later endpoint-policy ablation can reproduce its x-only behavior;
do not mix policies across the three prediction modes in the main comparison.
Check sensitivity to delta and solver resolution before interpreting quality.

## Module boundaries and extension points

```text
Click CLI -> validated ExperimentConfig -> experiment orchestration
  data.py: intrinsic -> fixed P -> observed Dataset -> DataLoader
  model.py: ToyMLP(z,t) -> raw output
  flow.py: noisy batch -> output-to-velocity -> scalar loss
  training.py: optimizer -> update -> metrics loop
  sampling.py: Gaussian -> velocity ODE -> observed samples
  evaluation.py: project -> scatter + off-subspace metric
```

`data.py` holds tensor operations including sampling/QR/indexing as exercises.
`ToyMLP` is an `nn.Module`; a future model need only keep its forward contract.
Prediction semantics live in `flow.py`, so the model is identical for each mode.
Adding a dataset means implementing the [N,2] sampler contract and extending
config validation plus the data factory. Adding a solver changes config choices
and `ode_step`. New modes require explicit Table 1 conversion and algebra tests.
Avoid introducing registries/plugin loading until more implementations need them.

`experiment.py` is also unfinished so the learner practices connecting the parts
and saving tensors. The Click adapter passes validated arguments and turns only
intentional exercise exceptions into concise nonzero errors; real bugs retain
their traceback. `plan --suite` is safe to run before any tensor exercises.

## Configuration, artifacts, and logging

The independent `jit-toy` distribution uses `src/jit_toy`, a `pyproject.toml`,
and the root's existing `.venv`. Run `source .venv/bin/activate` from the repository
root before `pip install --no-build-isolation -e ./toy_experiment`; subsequent
commands use `python`, `pip`, and `jit-toy` directly. Activate once per shell
session. Required dependencies: torch, Click, Loguru;
matplotlib is optional for plotting. JSON configs avoid requiring a TOML reader
on Python 3.9. The baseline JSON lists every field; the smoke JSON overrides
only a few settings. No environment is created inside this subdirectory.

The eventual E01 creates a new run directory with:

- `config.json`: fully resolved settings.
- `environment.json`: Python, package and torch versions, device information.
- `metrics.jsonl`: step, scalar loss, loss_sum at configured intervals.
- `checkpoint.pt`: schema_version=1, config (primitive dict), model_state
  (CPU tensors), projection, reference_intrinsic.

E02 creates a samples artifact with schema_version=1, config, observed,
generated_intrinsic, reference_intrinsic, projection, and residual. E03 renders
those saved arrays without retraining. Use strict model-state loading and
weights_only=True. This schema is distinct from JiT image checkpoints and supports
inference only. True resume would additionally require optimizer, completed-step,
RNG and loader-position state; that is a later exercise, not a current feature.

Loguru sinks include timestamp, severity, module, function, full source path and
source line. Call `logger.info` directly from exercise code to retain the correct
call site. Console logs use stderr; machine-readable plans use stdout. Optional
file logging uses the same format. Keep messages single-line. Library imports do
not configure sinks; the CLI owns configuration. CLI rejects existing training
directories and existing artifact outputs; integration code must enforce that too.

## Validation and comparison plan

1. Complete local shape/algebra/gradient checks in TODO.md before training.
2. Overfit one fixed noisy minibatch to expose disconnected graphs or wrong axes.
3. Run the tiny smoke config end-to-end, including checkpoint reload and plotting.
4. Train the 12 cases with matched data, architecture, budget and RNG streams.
   Save `plan --suite` as the run specification and create one JSON config per
   array entry; CLI does not yet run a sweep automatically.
5. Assemble four rows (D), four columns (reference/x/eps/v), using shared axes and
   clearly reported outliers. Also report mean squared distance to span(P):
   `mean_N(sum_D((X - (X @ P) @ P.T)^2))`. At D=2 it is always near zero.
6. Inspect spiral coverage and off-subspace energy together. Projection alone
   can hide garbage in the other D-2 dimensions. This metric measures distance
   to the linear subspace, not to the curved spiral; it is not a complete quality
   measure. No image FID/IS is appropriate here.
7. Repeat with multiple seeds and longer training; test endpoint/step sensitivity.
   Treat Figure 2's qualitative trend as a hypothesis, not a test to hard-code.

Current infrastructure tests cover configuration/CLI/logging behavior only.
Future numerical tests belong beside them once the learner implements each unit.
They should compare independent known answers and invariants, not copy the same
implementation into assertions. No expected x/eps/v quality outcome is guaranteed.
