# Design for the Section 3.3 toy experiment

The implemented scaffold establishes configuration and command behavior only. It does not yet test the paper's claim. The numerical implementation must compare three direct prediction targets under the same data, projection, network capacity, loss space, random streams, and training budget. A difference observed after those controls are fixed can be attributed to prediction target more credibly; a difference from unmatched runs cannot.

## Contents

- [1. Experiment contract and evidence boundary](#1-experiment-contract-and-evidence-boundary)
- [2. One sample through the data path](#2-one-sample-through-the-data-path)
- [3. One sample through the flow equations](#3-one-sample-through-the-flow-equations)
- [4. One training update in execution order](#4-one-training-update-in-execution-order)
- [5. Sampling and evaluation](#5-sampling-and-evaluation)
- [6. Relationship to the full JiT code](#6-relationship-to-the-full-jit-code)
- [7. Configuration and reproducibility](#7-configuration-and-reproducibility)
- [8. Verification order and next action](#8-verification-order-and-next-action)

## 1. Experiment contract and evidence boundary

Section 3.3 and Figure 2 of the [paper](../docs/Back%20to%20Basics-%20Let%20Denoising%20Generative%20Models%20Denoise.pdf) define the experiment at a high level. Two-dimensional data is embedded into an observed space by a fixed random matrix with orthonormal columns. A five-layer rectified linear unit network with hidden width 256 receives only noisy observed data and time. Separate runs directly predict clean data, noise, or velocity. All three runs are optimized in velocity space. The paper evaluates observed dimensions 2, 8, 16, and 512 and projects generated samples back to two dimensions for Figure 2.

The paper reports that clean-data prediction remains effective as the observed dimension increases while noise and velocity prediction deteriorate. That statement is the paper's observation. This package has not trained the models, generated samples, or measured that behavior.

Section 3.3 does not specify the exact spiral distribution, training-set size, optimizer, learning rate, number of updates, scalar-time encoding, or endpoint treatment for all three prediction targets. This design must choose those details to make the exercise executable. They are implementation policy, not recovered facts about the authors' run.

The controlled comparison is:

| Factor | Fixed or varied |
| --- | --- |
| Intrinsic dimension | Fixed at `d=2` |
| Observed dimension | Varied over `D={2,8,16,512}` |
| Direct prediction | Varied over clean data `x`, noise `epsilon`, and velocity `v` |
| Projection, intrinsic samples, initial weights, random streams | Fixed across prediction targets at a given `D` |
| Network | Five Linear layers, four hidden rectified linear unit activations, width 256 |
| Loss | Fixed velocity-space mean squared error |
| Training budget and sampler | Fixed across all prediction targets |

Figure 2 supplies a qualitative target for comparison, not a test oracle. No test should require clean-data prediction to win. Such a test would encode the expected conclusion instead of checking the implementation.

## 2. One sample through the data path

Take the intrinsic row vector `x_hat=[1,0]` and let `D=3`. For this example, use the column-orthonormal projection

```text
P = [[1, 0],
     [0, 1],
     [0, 0]]
```

The paper writes column vectors as `x=P*x_hat`. The package stores samples as rows, so the corresponding operation is `x_hat @ P.T`. It produces the observed row `[1,0,0]`. Projecting that row back with `x @ P` recovers `[1,0]`. Because `P.T @ P=I`, embedding preserves lengths and pairwise distances for points in the intrinsic plane.

The real experiment draws a Gaussian matrix of shape `[D,2]` and takes the reduced QR decomposition's orthonormal factor as `P`. D02 creates this matrix once for a dimension and projection seed. Redrawing it per batch, prediction target, or plot would change the data distribution and invalidate the controlled comparison.

D01 defines the local spiral policy. Draw one scalar `u` per sample uniformly from `[0,1)`, set `angle=2*pi*turns*u` and `radius=spiral_radius*u`, then stack `radius*cos(angle)` and `radius*sin(angle)` along the feature axis. Using independent random values for radius and angle would produce a disk-like distribution rather than this spiral. Uniform `u` also means the samples are not uniform in arc length; that density must remain unchanged across runs.

The ownership rule follows from the experiment: the data layer owns intrinsic samples and `P`; the model receives neither. The model sees only `z [B,D]` and `t [B,1]`. Noise has shape `[B,D]` and contains an independent standard Gaussian coordinate in every observed dimension. Embedding two-dimensional noise with `P` would remove the off-manifold prediction problem the experiment is intended to expose.

| Value | Shape | Owner |
| --- | --- | --- |
| Intrinsic samples | `[N,2]` | Data construction and evaluation |
| Projection `P` | `[D,2]` | Data construction, checkpoint metadata, and evaluation |
| Clean observed batch | `[B,D]` | Dataset and training loop |
| Time | `[B,1]` | Flow functions and sampler |
| Gaussian noise, noisy sample, prediction, velocity | `[B,D]` | Training and sampling |
| Loss | Scalar tensor | Training update until backward completes |

Create random tensors on the central processing unit in float32 with explicit `torch.Generator` instances. The training loop moves a complete batch to the model device. This makes ownership of random state visible and keeps a data draw from silently changing another stream.

## 3. One sample through the flow equations

Take one coordinate with clean value `x=2`, noise `epsilon=-1`, and time `t=0.25`. Equation 1 gives `z=t*x+(1-t)*epsilon=-0.25`. Equation 2 gives target velocity `v=x-epsilon=3`.

If the network directly predicts the correct clean value, the conversion `(x_pred-z)/(1-t)` returns `3`. If it predicts the correct noise, `(z-epsilon_pred)/t` also returns `3`. A direct velocity prediction already equals `3`. This concrete case shows why all three network outputs can be transformed into one loss space.

[flow.py](src/jit_toy/flow.py) owns these conversions:

| Direct network output | Predicted velocity |
| --- | --- |
| Clean data | `(raw-z)/(1-t)` |
| Noise | `(z-raw)/t` |
| Velocity | `raw` |

F02 constructs `z` and the target velocity from explicit clean data, time, and noise. F03 converts the raw network output without detaching it, so gradients still reach the model. F04 rejects unequal shapes before reduction and computes `mean_B(mean_D((predicted-target)^2))`. This matches the reduction in the repository's `Denoiser.forward`. It equals the paper's squared Euclidean norm divided by `D`; the constant changes gradient scale, so the same reduction must be used in every run.

Time sampling follows `Denoiser.sample_t`: draw a normal value with mean `-0.8` and standard deviation `0.8`, then apply the sigmoid. Those parameter values come from `main_jit.py`, not Section 3.3. The toy policy clamps time to `[time_eps,1-time_eps]`, with `time_eps=0.001`, because noise conversion is singular at zero and clean-data conversion is singular at one. The clamp changes the ideal logit-normal distribution near both endpoints.

The full JiT denoiser instead clamps only the denominator in its clean-data-to-velocity conversion and integrates from zero to one. Its target `(x-z)/(1-t).clamp_min(t_eps)` equals `x-epsilon` only where the clamp is inactive. The toy must use one shared interior-time policy for all three prediction targets; combining the full JiT endpoint rule with the toy noise-prediction branch would make the comparison asymmetric.

## 4. One training update in execution order

The numerical core stays testable because random draws and file writes sit outside the parameter update:

```text
experiment.py creates model, optimizer, random generator, and metrics writer
    -> training.py fetches clean data and draws time and full-dimensional noise
        -> flow.py builds a FlowBatch containing z, t, and target velocity
            -> model.py maps z and t to one raw prediction
            -> flow.py converts the prediction and reduces the loss
        -> training.py clears gradients, runs backward, checks gradients, and steps
    -> experiment.py records returned scalar metrics
```

T01 receives only model parameters and a learning rate. It creates AdamW with the full JiT code's beta values `(0.9,0.95)` and explicit zero weight decay. The caller constructs it after moving the model to its device. The parameter iterable may be one-shot, so T01 must not consume it for validation before passing it to PyTorch.

T02 receives a model, optimizer, immutable-by-contract `FlowBatch`, and prediction name. It clears old gradients, runs the forward and loss functions, rejects a nonfinite loss, calls backward, rejects nonfinite gradients, and then steps the optimizer. It returns detached Python scalars only after the step. Because it performs no random draw and opens no file, a test can pass the same fixed batch repeatedly and determine whether the model can overfit it.

T03 owns iteration over clean batches and calls T02 exactly `train_steps` times. It draws time and noise from the caller's generator, uses the actual batch length when the last batch is short, and restarts a re-iterable loader after exhaustion. After a completed update it calls `on_metrics(step, metrics)` with one-based step numbers. Callback failures propagate, so a failed metrics write cannot be reported as a completed run.

E01 owns effects at the experiment boundary. It validates device availability, creates the run directory, records the resolved configuration and environment, builds the data/model/optimizer/generators, opens `metrics.jsonl` exclusively, and passes a writer callback to T03. After training completes, it writes the checkpoint through a temporary path and rename. It logs completion only after that rename succeeds.

The baseline deliberately omits batch-size learning-rate scaling, warmup, moving averages, mixed precision, distributed training, and gradient clipping. These mechanisms are not required to answer the first implementation question and would make failures harder to localize. The configured absolute learning rate is `0.001`.

## 5. Sampling and evaluation

Sampling starts with standard Gaussian points in all `D` observed dimensions and solves `dz/dt=v_pred(z,t)`. Euler uses the velocity at the current state. Heun first makes an Euler proposal, evaluates velocity at that proposed state and next time, averages the two velocities, and advances the original state. It must average velocities after F03 conversion, not raw clean-data or noise predictions.

The toy integrates over `[time_eps,1-time_eps]` with 50 intervals and 51 grid points. Starting from a pure Gaussian at `time_eps` approximates the true distribution at that time, and stopping at `1-time_eps` returns a near-data sample. The sampler therefore does not reproduce the exact zero-to-one endpoint behavior of `Denoiser.generate`. This limitation applies equally to all three prediction targets and must be recorded with results.

The sampler runs with inference mode enabled, returns detached CPU float32 points, and restores the model's previous training flag even when an error occurs. It never applies `P`; projection belongs to evaluation.

Projecting a generated point with `X @ P` can hide error in the other `D-2` dimensions. V01 therefore also computes

```text
mean_N(sum_D((X - (X @ P) @ P.T)^2))
```

This value measures squared distance from the linear span of `P`. It cannot measure distance along that plane to the spiral, and it is approximately zero for every point when `D=2`. A plotted spiral can look plausible while carrying large perpendicular error, and a small residual can coexist with a poor spiral. Interpret the scatter plot and residual together.

E01 writes an inference checkpoint containing schema version, resolved configuration, CPU model state, projection, and held-out intrinsic reference points. E02 loads it on the CPU with `weights_only=True`, validates the schema, restores the model strictly, samples in the selected device, and saves both observed and projected results. E03 reads that saved artifact and plots without retraining. The checkpoint is not resumable because it excludes optimizer state, completed step, loader position, and random-generator state.

## 6. Relationship to the full JiT code

The toy package reuses the existing implementation as a behavioral reference without importing its image, distributed-training, or evaluation stack.

| Existing implementation | Toy exercise | Relationship |
| --- | --- | --- |
| [Denoiser.sample_t and forward](../denoiser.py) | F01–F04 | Preserve logit-normal time sampling, interpolation, clean-data conversion, and loss reduction; adapt image tensors to `[B,D]` and add Table 1's noise and velocity branches |
| Denoiser._euler_step and _heun_step | S01 | Preserve the update equations |
| Denoiser.generate | S02 | Preserve Gaussian initialization, time grid, and disabled gradients; use the shared interior endpoint policy described above |
| [TimestepEmbedder, JiT.blocks, and FinalLayer](../model_jit.py) | M01–M02 | Preserve registered modules, time conditioning, and an unconstrained output; use the five-layer network required by Section 3.3 |
| [train_one_epoch](../engine_jit.py) | T02–T03 | Preserve training mode, device transfer, finite-loss check, gradient reset, backward, and optimizer step; omit image normalization and unconditional CUDA operations |
| [main](../main_jit.py) | D07, T01, E01 | Preserve seed, loader, model, optimizer, and checkpoint lifetimes; retain AdamW beta values and zero weight decay |
| [sampling demo](../sample_jit.py) | E02 | Preserve CPU checkpoint loading, `weights_only=True`, strict state loading, device validation, and inference mode |

The paper is the contract for the comparison. The repository code is the reference implementation for shared mechanics. The toy-specific spiral, interior endpoint policy, scalar time concatenation, learning rate, update budget, and artifact schema are local choices. Experimental results must name those choices rather than presenting them as paper settings.

## 7. Configuration and reproducibility

The baseline configuration names every setting. The smoke configuration overrides only counts needed for a short wiring run. JSON parsing rejects unknown fields so a misspelled option cannot silently fall back to a default.

Use separate CPU random generators for training points (`seed`), held-out points (`seed+1`), data shuffling (`seed+2`), flow draws (`seed+3`), and projection (`projection_seed`). Seed model initialization separately. Reinitialize each prediction-target run from the same model seed; do not continue training one target's weights under another target.

Saving seeds is necessary but insufficient for exact reproduction across hardware and software versions. Save the actual projection, resolved configuration, Python and PyTorch versions, selected device, and model state. Report comparisons as results from that recorded setup unless repeated runs establish broader behavior.

The intended experiment artifacts are:

| Artifact | Evidence it preserves |
| --- | --- |
| `config.json` | Resolved controls for the run |
| `environment.json` | Runtime versions and selected device |
| `metrics.jsonl` | One-based step and scalar loss records selected by the E01 callback |
| `checkpoint.pt` | Model state, projection, reference points, and schema version |
| Sample artifact | Observed samples, projected samples, projection, reference points, and residual |

Reject existing output paths. A partially created training directory may remain after failure, but it must not contain a completion marker or final checkpoint presented as successful.

## 8. Verification order and next action

Verification follows data flow. First check spiral shape and reproducibility. Then check `P.T @ P`, embedding norm preservation, and projection round trips. Next use the concrete values from Section 3 to verify that oracle clean-data, noise, and velocity predictions all convert to velocity `3`. Check the loss against a hand calculation and confirm gradients reach every model layer. Only then test a complete update, fixed-batch overfitting, loader restart, Euler and Heun on known fields, checkpoint reload equality, and the smoke workflow.

The 12-run comparison comes after those invariant checks. At each `D`, keep data, projection, initialization, random streams, update count, and sampler fixed while changing only the direct prediction target. Report individual seeds and variation before treating the paper's qualitative ordering as reproduced.

The next bounded action is D01, `sample_spiral`. Hold count, turns, radius, dtype, and seed fixed. Pass D01 only if the output has shape `[count,2]`, contains finite float32 values, stays within the configured radius, and two newly created generators with the same seed return identical points. If any check fails, inspect the shared `u` draw and the axis passed to `torch.stack`; do not proceed to D02.
