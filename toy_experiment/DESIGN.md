# Section 3.3 toy experiment design

The scaffold can validate and print the 12 configurations needed for the toy comparison. Its numerical methods are unfinished exercises, so it cannot yet establish whether clean-data prediction outperforms noise or velocity prediction. This design preserves the existing JiT flow equations and update conventions while making the toy-specific assumptions explicit.

## Contents

- [1. What runs today](#1-what-runs-today)
- [2. One observed sample](#2-one-observed-sample)
- [3. Data ownership and tensor shapes](#3-data-ownership-and-tensor-shapes)
- [4. From a direct prediction to a velocity field](#4-from-a-direct-prediction-to-a-velocity-field)
- [5. Training boundaries](#5-training-boundaries)
- [6. Sampling and saved artifacts](#6-sampling-and-saved-artifacts)
- [7. Relationship to the full JiT implementation](#7-relationship-to-the-full-jit-implementation)
- [8. Verification and the next exercise](#8-verification-and-the-next-exercise)

## 1. What runs today

After activating the root's `.venv` and installing the package as described in [README.md](README.md), run:

```bash
jit-toy plan --config toy_experiment/configs/smoke.json
```

Click reads the configuration path, `load_config` validates its JSON overrides, and `plan` writes the resolved settings to standard output. Loguru writes the validation message to standard error with its source location. This path does not import PyTorch or create experiment artifacts. `--log-file` optionally creates a log file; that is a separate command-line effect.

A training request reaches a different boundary. The relevant execution slice in [cli.py](src/jit_toy/cli.py) is:

```python
resolved = read_config(config)
if output.exists():
    raise click.ClickException(f"Output already exists: {output}; choose a new run directory")
from .experiment import run_training
logger.info("Requested training D={} prediction={} output={}", resolved.observed_dim, resolved.prediction, output)
run_training(resolved, output)
```

`run_training` currently raises `ExerciseNotImplemented` for E01. The command adapter converts that specific exception into a nonzero exit and a TODO reference. It does not report completion or create a checkpoint. Unexpected programming errors propagate for debugging. This distinction must remain visible while exercises are completed.

The remaining sections specify the intended numerical implementation. They describe contracts and planned checks, not observed model behavior.

## 2. One observed sample

Consider an intrinsic point `[1,0]` and an observed dimension `D=8`. For this example alone, let the first two rows of the embedding matrix `P` form the identity and the other six rows be zero. Then embedding the point gives `[1,0,0,0,0,0,0,0]`. Multiplying that observed row by `P` recovers `[1,0]`. The experiment will use a random orthonormal basis instead of these coordinate axes, but the shape and inverse-on-subspace relationships are the same.

Noise is different: it has eight independently sampled Gaussian coordinates. Restricting noise to the two-dimensional embedded plane would remove the high-dimensional prediction problem. The model receives the noisy eight-dimensional point and time, but never receives `P` or the intrinsic coordinates.

The [paper](../docs/Back%20to%20Basics-%20Let%20Denoising%20Generative%20Models%20Denoise.pdf), arXiv:2511.13720v2, specifies the comparison in Section 3.3 and Figure 2 on pages 4–5. Section 3.1 on page 3 and Table 1 on page 4 give the flow formulation. Each planned run below uses the same type of embedding as this example.

| Quantity | Paper setup |
| --- | --- |
| Intrinsic dimension | `d=2` |
| Observed dimension | `D` in `{2,8,16,512}`; `D=2` is the no-expansion control |
| Embedding | Fixed random `P [D,2]`, with `P.T @ P = I` |
| Model | Five-layer multilayer perceptron (MLP), width 256, rectified linear unit (ReLU) hidden activations |
| Direct network prediction | Clean data `x`, noise `eps`, or velocity `v`, in separate runs |
| Loss | Velocity-space loss for all three prediction types |
| Visualization | Generated observed points projected back to two dimensions |

Figure 2 reports that clean-data prediction remains effective as the observed dimension grows, while noise and velocity prediction deteriorate. That is the result to investigate, not a result established by this package.

The figure depicts a spiral. Section 3.3 does not specify its exact sampling density and scale, dataset sizes, training budget, optimizer, toy time-conditioning method, or layer-count convention. The configuration and comments label these as practice choices. They are not an exact reconstruction of the authors' Figure 2 implementation.

## 3. Data ownership and tensor shapes

Each row is one sample. For a batch `X_hat [N,2]`, embedding is `X_hat @ P.T`, producing `X [N,D]`. Visualization uses `X @ P`, producing `[N,2]`. Because the columns of `P` are orthonormal, the first transformation preserves distances. Do not standardize individual observed coordinates or multiply by a dimension-dependent scale afterward.

The chosen spiral recipe draws `u` uniformly on `[0,1)`, sets angle to `2*pi*turns*u` and radius to `spiral_radius*u`, and converts polar coordinates to Cartesian coordinates. Defaults are two turns and radius two, with no added data jitter. Uniform `u` is not uniform arc length; hold this density fixed across comparisons.

D02 constructs `P` from the thin QR decomposition of a random `[D,2]` Gaussian matrix: the columns of the returned `Q` are orthonormal. Generate `P` once per observed dimension and projection seed, and save the actual tensor. Regenerating it per batch would change the learning problem.

| Value | Shape | Owner and allowed use |
| --- | --- | --- |
| Intrinsic points | `[N,2]` | Data construction and evaluation |
| Projection `P` | `[D,2]` | Data construction, saved metadata, and evaluation |
| Clean observed batch | `[B,D]` | Dataset and training loop |
| Gaussian noise | `[B,D]` | Training loop; standard Gaussian in all D dimensions |
| Time | `[B,1]` | Flow functions and sampler |
| Noisy batch, raw prediction, velocity | `[B,D]` | Model, loss, and sampler |
| Loss | Scalar tensor | Training update until backward completes |

Create data tensors on the CPU in float32. Move observed batches, noise, and time to the selected model device in the training loop. `FlowBatch` groups noisy points, time, and target velocity. Its frozen dataclass prevents field reassignment; it does not freeze the tensors' storage. Callers and the training update must treat those input tensors as read-only.

Separate random-generator streams prevent an extra data draw from changing the projection or training noise. Training points use `seed`, held-out points use `seed+1`, shuffling uses `seed+2`, flow draws use `seed+3`, and the projection uses `projection_seed`. Seed model initialization separately with `torch.manual_seed`. Sampling starts a fresh generator at `seed+1`; it must not advance the training generator.

Across the three prediction types at one `D`, hold the projection, intrinsic dataset, initial network weights, random seeds, and training budget fixed. Separate seeded generators isolate mutable state; they do not promise bitwise equality across devices or software versions. Save the actual projection and record the runtime.

## 4. From a direct prediction to a velocity field

For one coordinate, take clean data `x=2`, noise `eps=-1`, and time `t=0.25`. Linear interpolation gives `z=-0.25`; the velocity target is `v=3`. A perfect clean-data output recovers that velocity as `(2-(-0.25))/(1-0.25)=3`. A perfect noise output recovers it as `(-0.25-(-1))/0.25=3`. This illustrates why all three direct prediction types can be compared in one loss space.

For every coordinate, the paper's equations are `z_t=t*x+(1-t)*eps` and `v=x-eps`. Time increases from noise toward data. [flow.py](src/jit_toy/flow.py) owns the following conversions, outside the model:

| Direct prediction | Velocity used for training and sampling |
| --- | --- |
| Clean data, `x` | `(raw-z)/(1-t)` |
| Noise, `eps` | `(z-raw)/t` |
| Velocity, `v` | `raw` |

The common loss is `mean_B(mean_D((v_pred-v_target)^2))`, matching the feature-mean then batch-mean reduction in `Denoiser.forward`. This equals the paper's expected squared Euclidean norm divided by `D`. The factor changes gradient scale, so keep the reduction fixed across prediction types. Report the optimized value as `loss` and `loss*D` as `loss_sum`; add no further time weighting after conversion.

F01 receives only a batch size, normal mean, normal standard deviation, endpoint margin, and explicit generator. The training loop passes `time_mean`, `time_std`, and `time_eps` from configuration. The normal draw is mapped through a sigmoid, then clamped to `[margin,1-margin]`. Clamping changes the ideal logit-normal distribution near the boundaries and is a deliberate numerical policy.

Defaults `time_mean=-0.8` and `time_std=0.8` come from `main_jit.py`; the toy-specific values are not given in Section 3.3. The margin is `time_eps=0.001`. Conversion callers use this interior interval, so neither `eps` conversion at zero nor `x` conversion at one is evaluated.

JiT computes its target as `(x-z)/(1-t).clamp_min(t_eps)`. In the toy's interior interval, the clamp is inactive in exact arithmetic and this target equals `x-eps`. If changing to JiT's unrestricted sigmoid times, that equivalence no longer holds where its denominator is clamped. Change target and prediction policy together; do not silently combine the two formulations.

## 5. Training boundaries

The five-layer convention is four hidden Linear/ReLU layers followed by one unconstrained Linear output. Concatenating scalar time to the observed vector makes the first input width `D+1`. Hidden width stays 256. This is a toy conditioning choice; full JiT uses sinusoidal time features and adaptive layer normalization.

The model's contract is `forward(z [B,D], t [B,1]) -> raw [B,D]`. It knows neither the projection nor the prediction type. No analytic noise passthrough or full input-output residual may bypass the bottleneck, because that would change the capacity comparison. Begin with PyTorch's Linear initialization; the image Transformer's specialized zero initialization is not required here.

The flow of one eventual update answers where randomness, mutation, and file effects belong:

```text
experiment.py: create model, optimizer, generator, and metric-file callback
    -> training.py: fetch clean batch; draw time/noise; construct FlowBatch
        -> model.py: predict raw output
        -> flow.py: convert to velocity and reduce loss
        -> training.py: backward; optimizer step; return scalar metrics
    -> experiment.py callback: write selected metrics and log their values
```

The projection stays in data/evaluation code. File ownership stays in `experiment.py`; the training update consumes an explicit batch and mutates only the supplied model and optimizer. This makes a fixed-batch overfitting check possible without controlling hidden random draws or creating files.

The exercise interfaces in [training.py](src/jit_toy/training.py) follow those boundaries:

- T01 receives the parameter iterable and learning rate. The experiment passes `model.parameters()` after moving the model to its device. Use AdamW with JiT's betas `(0.9,0.95)` and explicit zero weight decay.
- T02 receives the model, optimizer, `FlowBatch`, and prediction type. It returns fresh Python scalar metrics after a complete update. It does not sample noise, read configuration files, or log.
- T03 receives the re-iterable batch source, optimizer, configuration, flow generator, and `on_metrics(step, metrics)` callback. It performs exactly `train_steps` updates, reports every completed step starting at one, and returns the final metrics. Configuration is appropriate here because this loop uses the device, update budget, prediction type, and time settings together.
- E01 owns the optimizer and generator lifetimes, the metric file, and the callback. Its callback records every `log_every` steps and the final step. A callback error propagates; the run must not report completion after a failed metric write.

The baseline uses a fixed absolute learning rate of 0.001, without batch-size scaling, warmup, exponential moving averages, mixed precision, or distributed training. The short final batch is retained to practice dynamic shapes. Use the actual batch size when drawing time and noise. The data is already float32; image normalization by 255 would be incorrect.

These interfaces are exercises, not implemented training code. Keep the required PyTorch inheritance for `ToyMLP` and `ObservedDataset`, but do not add new base classes, model registries, or configuration hierarchies before another implementation requires them.

## 6. Sampling and saved artifacts

Sampling solves the ordinary differential equation `dz/dt=v_pred(z,t)` in observed space. S01 implements Euler and Heun updates. Heun evaluates velocity at the current state, makes a provisional Euler state, evaluates velocity there at the next time, and uses the average velocity to advance the original state. Average velocities, not raw clean-data or noise predictions.

S02 initializes a full-dimensional standard Gaussian and integrates 50 uniform intervals over `[time_eps,1-time_eps]`, using 51 grid points. It runs without gradient tracking, returns detached CPU float32 samples, and restores the model's previous training flag. The integrator never projects into intrinsic coordinates.

This endpoint policy is approximate: a Gaussian at `time_eps` is not the exact interpolated distribution there, and the final state is near-data rather than the exact `t=1` endpoint. Full JiT instead integrates from zero to one, clamps the clean-data denominator at its default 0.05, and finishes with Euler. That implementation supports direct clean-data prediction only. Keep one shared policy for the toy's three prediction types before considering a separate endpoint comparison.

E01 creates a new run directory. The planned artifacts have the following responsibilities:

| Artifact | Contents and lifetime |
| --- | --- |
| `config.json` | Fully resolved configuration |
| `environment.json` | Python, package and PyTorch versions; selected device |
| `metrics.jsonl` | One JSON object per recorded step: `step`, `loss`, `loss_sum`; opened exclusively and closed by E01 |
| `checkpoint.pt` | `schema_version=1`, primitive configuration dict, CPU `model_state` tensors, `projection`, `reference_intrinsic`; written after training completes |

E02 loads the checkpoint on the CPU with `weights_only=True`, validates its schema, restores the exact projection, and loads the model state strictly. It may override the device for portability. It writes a sample artifact containing `schema_version=1`, `config`, `observed`, `generated_intrinsic`, `reference_intrinsic`, `projection`, and scalar `residual`. E03 plots those saved arrays without retraining.

Checkpoints support inference only. A future resume feature would also need optimizer state, completed-step count, random-generator state, and loader position. Do not label the current schema resumable. Reject existing output paths rather than overwriting another run, and report completion only after the checkpoint write succeeds.

Projection can hide generated noise perpendicular to the data plane. Alongside the scatter plot, V01 measures `mean_N(sum_D((X-(X@P)@P.T)^2))`, the mean squared distance to the linear span of `P`. At `D=2` this is approximately zero even for poor samples. It also cannot detect samples that lie in the plane but miss the spiral. Inspect coverage and off-subspace energy together, and show outliers rather than silently clipping them from plot limits.

## 7. Relationship to the full JiT implementation

The reference is the existing repository code, not a new image-model dependency. Read the functions below beside the corresponding exercises. The package stays independent of image training imports, distributed execution, and image-quality tooling.

| Existing code | Toy exercises | Preserved behavior and explicit adaptation |
| --- | --- | --- |
| [Denoiser.sample_t / forward](../denoiser.py) | F01–F04 | Logit-normal times, interpolation, velocity conversion, loss reduction; use `[B,D]` tensors and the documented interior interval |
| Denoiser._forward_sample | F03 | Clean-data conversion; add noise/velocity branches from Table 1; omit class guidance |
| Denoiser._euler_step / _heun_step / generate | S01–S02 | Velocity-based updates and disabled gradients; endpoint policy differs as described above |
| [TimestepEmbedder / JiT.blocks / FinalLayer](../model_jit.py) | M01–M02 | Registered modules, time conditioning, unconstrained output; replace the Transformer with the specified ReLU network |
| [train_one_epoch](../engine_jit.py) | T02–T03 | Training mode, device transfer, finite loss, gradient reset, backward, optimizer step; omit image scaling and unconditional CUDA calls |
| [main](../main_jit.py) | D07 / T01 / E01 | Seed, loader, model and optimizer lifetimes; retain AdamW settings but keep short final batches |
| [save_model](../util/misc.py) | E01 | State-dictionary checkpointing; use the toy's separate inference schema |
| [sampling demo](../sample_jit.py) | E02 / S02 | Device checks, CPU loading, `weights_only=True`, strict state load, inference mode |

The configuration field names, command names, default values, and 12-case matrix remain unchanged by this refactor. Only unfinished exercise interfaces changed: F01 takes time settings directly; T01 takes parameters and learning rate; T02 takes a prepared batch; T03 receives its optimizer, generator, and metrics callback instead of opening a file.

## 8. Verification and the next exercise

The baseline scaffold tests cover configuration rejection, the 12-case matrix, command output, exercise-error translation, output-path protection, and source-located logging. They cannot establish the correctness of unfinished tensor operations. Run them with `python -m unittest discover -s toy_experiment/tests -v` in the activated environment; add independent numerical checks as the exercises become implemented.

Before a quality comparison, verify projection orthogonality and round trips, oracle prediction-to-velocity conversion, a hand-calculated loss, gradients through the model, fixed-batch overfitting, a known velocity field, and checkpoint reload equality. Then run the smoke configuration through training, sampling, and plotting. Ten updates test the workflow, not convergence.

For the eventual 12-case comparison, hold data, projection, architecture, initialization, update budget, and random streams fixed across prediction types at each dimension. Save one configuration per case and use separate output directories. A four-by-four panel can show dimensions as rows and reference/clean/noise/velocity as columns. Repeat seeds and check training-budget and endpoint sensitivity before attributing a difference to prediction type. No expected quality ranking belongs in a hard-coded pass condition.

The next implementation step is D01, `sample_spiral`. Hold `N`, turns, radius, dtype, and seed fixed. Accept the exercise when it returns finite `[N,2]` float32 points within the configured radius and two fresh generators with the same seed reproduce the same points. If those checks fail, inspect the shared radial/angular draw and feature-axis stacking before proceeding to the projection exercise.
