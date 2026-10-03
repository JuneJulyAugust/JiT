# Practice checklist: implement and verify one exercise at a time

All numerical functions remain unfinished. The exercise identifiers below match exceptions in [src/jit_toy](src/jit_toy). Implement one function, run its stated check, and tick its box only after that check passes. The supplied infrastructure tests establish command and configuration behavior; they do not establish these numerical contracts.

## Contents

- [1. Establish the starting conditions](#1-establish-the-starting-conditions)
- [2. Construct and batch the data](#2-construct-and-batch-the-data)
- [3. Register and run the network](#3-register-and-run-the-network)
- [4. Construct flow targets and the loss](#4-construct-flow-targets-and-the-loss)
- [5. Train and save a run](#5-train-and-save-a-run)
- [6. Integrate the velocity field](#6-integrate-the-velocity-field)
- [7. Evaluate and reload artifacts](#7-evaluate-and-reload-artifacts)
- [8. Run the controlled comparison](#8-run-the-controlled-comparison)
- [9. Optional exercises after the baseline](#9-optional-exercises-after-the-baseline)
- [10. Next action](#10-next-action)

## 1. Establish the starting conditions

- [ ] Read [DESIGN.md](DESIGN.md), including its paper contract, mathematical examples, and [source mapping](DESIGN.md#6-relationship-to-the-full-jit-code).
- [ ] Follow [README.md](README.md) to activate the environment, install the package, inspect help and the 12-run plan, and pass the infrastructure tests.
- [ ] For a batch of $B$ samples in $D$ observed coordinates, derive the shapes of the noisy input ($B\times D$), time column ($B\times1$), and projection ($D\times2$). Check each matrix product before writing it.

## 2. Construct and batch the data

These exercises are in [data.py](src/jit_toy/data.py). The projection belongs to data construction and evaluation; each dataset item exposes only an observed vector.

- [ ] **D01 — `sample_spiral`**: use one uniform draw per sample for both radius and angle, then stack the Cartesian coordinates. Check shape $N\times2$, float32 dtype, finite values, radial bounds, and identical output from fresh generators with the same seed. Plot the intrinsic points once to inspect the spiral.
- [ ] **D02 — `make_projection`**: use the reduced QR decomposition of a Gaussian matrix; its orthonormal factor supplies $P$. Check $P^{\mathsf T}P\approx I_2$ at $D\in\lbrace 2,8,16,512\rbrace$. On the same runtime, fresh generators with the same seed must produce the same matrix.
- [ ] **D03 — `embed_points`**: multiply intrinsic rows by $P^{\mathsf T}$. Check shape $N\times D$ and preserved norms and pairwise distances. Do not rescale with $D$.
- [ ] **D04 — `project_points`**: multiply observed rows by $P$. Check that projection after embedding recovers the original rows within float32 tolerance, including $N=1$.
- [ ] **D05 — `ObservedDataset`**: store observed data and implement its length and integer indexing. Check that length is $N$, one item has $D$ coordinates, and an out-of-range index fails. Neither $P$ nor intrinsic coordinates may appear in an item.
- [ ] **D06 — `build_data`**: construct training points, held-out points, and one fixed projection from separate generators. Check that shared configuration seeds give identical data and $P$ across the three prediction modes, while held-out points differ from training points.
- [ ] **D07 — `make_loader`**: shuffle with the loader's generator and retain the final short batch. With $N=10$ and batch size $4$, check successive batch shapes $4\times D$, $4\times D$, and $2\times D$, with every example visited once per epoch.

## 3. Register and run the network

These exercises are in [model.py](src/jit_toy/model.py). The multilayer perceptron (MLP) has four hidden Linear layers followed by a fifth Linear output layer. Each hidden layer uses a rectified linear unit (ReLU); the output can take either sign.

- [ ] **M01 — `ToyMLP.__init__`**: register the five Linear layers and four hidden activations. Inspect `named_parameters` and `state_dict`, then check that all registered parameters move with the model's device transfer.
- [ ] **M02 — `ToyMLP.forward`**: concatenate the time column along the feature axis and return shape $B\times D$. Check $B=1$ and $B=D$, and verify that backward reaches every layer with finite gradients. The network must receive neither $P$ nor intrinsic points and must have no direct input-to-output bypass.

## 4. Construct flow targets and the loss

These exercises are in [flow.py](src/jit_toy/flow.py). Use the fixed scalar example in [DESIGN.md, Section 3](DESIGN.md#3-one-sample-through-the-flow-equations) before using random batches.

- [ ] **F01 — `sample_times`**: draw normal values, apply the sigmoid, and clamp to the shared interior interval. Accept explicit mean, standard deviation, margin, and generator. Check shape $B\times1$, finite bounded times, fresh-generator reproducibility, and the histogram against the documented distribution.
- [ ] **F02 — `make_flow_batch`**: form $z_t=tx+(1-t)\epsilon$ and $v=x-\epsilon$ from supplied inputs with full $D$-coordinate noise. Check the interpolation at $t=0$, $t=1$, and the interior scalar example. Endpoint checks here concern interpolation only; F03 requires interior times.
- [ ] **F03 — `to_velocity`**: implement the three conversions from Table 1, row 3. Supply the true clean value, noise value, or velocity as an oracle output and check that each returns the same $v$. Check $B=1$ and $B=D$, reject unknown prediction names, and verify gradients through the raw output.
- [ ] **F04 — `velocity_loss`**: average squared error over features, then over samples. Check a hand-calculated scalar loss and its gradient, and reject mismatched shapes before broadcasting can hide a mistake.

## 5. Train and save a run

Training functions are in [training.py](src/jit_toy/training.py); file and checkpoint ownership belongs to [experiment.py](src/jit_toy/experiment.py).

- [ ] **T01 — `build_optimizer`**: pass the parameter iterable and learning rate to AdamW with zero weight decay and $\beta_1=0.9$, $\beta_2=0.95$. Check that every registered trainable parameter is included, including when the input iterable can be consumed only once.
- [ ] **T02 — `train_step`**: clear gradients, enable training mode, run the model, convert to velocity, reduce the loss, check finite loss, run backward, check finite gradients, and step. Return Python scalars after the step. Check that weights change and old gradients do not accumulate; overfit one fixed noisy minibatch for each prediction mode. Random draws and log writes stay outside this function.
- [ ] **T03 — `train`**: restart the loader after exhaustion, transfer actual batches to the selected device, draw time and noise, and run exactly `train_steps` updates. An in-memory callback must receive ordered one-based step numbers, including updates from short batches. A failed update must emit no metrics; callback errors must propagate.
- [ ] **E01 — `run_training`**: create data, model, optimizer, and generators, then own the metrics callback and checkpoint save. Check resolved configuration and environment files, parseable JSON Lines at `log_every` and the final step, logs naming the calling source line, repeatable initialization, and checkpoint metadata. Check refusal to overwrite an earlier run and that save failures cannot produce a completion report.

## 6. Integrate the velocity field

These exercises are in [sampling.py](src/jit_toy/sampling.py). An ordinary differential equation describes how the observed sample moves as time increases; the learned velocity is its right-hand side.

- [ ] **S01 — `ode_step`**: implement Euler and Heun using converted velocities. Check a constant field first. Then use $dz/dt=z$, $z(0.25)=1$, and step size $h=0.1$: Euler must return $1.1$ and Heun $1.105$, compared with the exact endpoint $e^{0.1}$. This distinguishes an update from the original state from an erroneous second update of the proposal.
- [ ] **S02 — `sample`**: draw full-dimensional initial Gaussian noise with the supplied CPU generator and integrate over the shared interior grid. Check the exact sample count, $B\times1$ time columns, finite detached CPU float32 output, no gradient graph, reproducibility, and restoration of the model's prior training flag even on failure.

## 7. Evaluate and reload artifacts

Evaluation functions are in [evaluation.py](src/jit_toy/evaluation.py); loading and saving belong to [experiment.py](src/jit_toy/experiment.py).

- [ ] **V01 — `subspace_residual`**: project, re-embed, subtract, sum squared errors per sample, and average over samples. Embedded clean data must give approximately zero. At $D=3$ with the example projection, the row $(1,0,2)$ must give residual $4$. At $D=2$, this quantity cannot assess spiral quality.
- [ ] **E02 — `run_sampling`**: load the checkpoint on CPU with `weights_only=True`, validate its schema, restore weights strictly and the exact saved $P$, then sample and save observed points, projected points, and the residual. Check identical raw model outputs before and after reload on fixed inputs.
- [ ] **V02 — `plot_comparison`**: convert tensors to NumPy at the plotting boundary and use optional matplotlib for a headless save. Use equal aspect ratio, matching limits, and translucent points. Check that all finite samples are visible or that clipped points are counted explicitly; close the figure after saving.
- [ ] **E03 — `run_plotting`**: validate the saved sample schema and call V02. Check plotting in a new process, refusal to overwrite an output, and clear failure for malformed artifacts.

## 8. Run the controlled comparison

- [ ] Complete the [smoke workflow](README.md#5-run-after-completing-the-exercises). Ten updates establish that components connect, not that training converges.
- [ ] Save the baseline suite plan, split its 12 entries into individual JSON configurations, and train each into a new directory. At each $D$, hold projection, data, initialization, random streams, training budget, and sampler fixed across prediction modes.
- [ ] Assemble a $4\times4$ comparison from saved samples: dimensions as rows, reference and the three prediction modes as columns. Include residuals and recorded controls. Each panel caption must explain what a point represents and what projection can hide; any numerical description must come from that saved artifact.
- [ ] Report each seed's results and variation before judging the qualitative trend. Repeat seeds with the baseline controls fixed. Any later change to training budget, time margin, or solver resolution is a separate experiment that changes one factor at a time.

## 9. Optional exercises after the baseline

Complete and record the baseline before choosing one extension. These exercises are optional changes to the experiment, not requirements for the first comparison.

- [ ] Add sinusoidal time features following `TimestepEmbedder`, recording the changed architecture.
- [ ] Practice reshaping, permutation, `einsum`, and tensor storage layout in a separate notebook using `JiT.unpatchify` and `Attention.forward` as references.
- [ ] Add a second intrinsic dataset and a second model with the same input/output contracts.
- [ ] Add an exponential moving average of model weights using `Denoiser.update_ema` and `engine_jit.evaluate` as references; check updates without a gradient graph.
- [ ] Add training resume with optimizer state, completed step, generator states, and loader position. Compare an uninterrupted run with a resumed run under identical controls.
- [ ] Run a clean-data-only comparison of the full JiT endpoint rule against the shared interior rule, recording the changed endpoint policy.

## 10. Next action

Begin with D01. Hold count, turns, radius, dtype, and seed fixed, and pass only when shape, finiteness, radial bounds, and reproducibility checks all hold. If one fails, correct the shared random draw or coordinate stacking before moving to D02.
