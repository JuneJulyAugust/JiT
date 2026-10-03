# Design for the Section 3.3 toy experiment

The package implements configuration validation and command dispatch; its numerical functions remain exercises. No local result yet establishes the paper's claim that direct clean-data prediction works better as the observed dimension grows. This design defines the controlled comparison and the checks needed before interpreting generated samples.

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

Section 3.3 and Figure 2 of the [paper](../docs/Back%20to%20Basics-%20Let%20Denoising%20Generative%20Models%20Denoise.pdf) define the comparison. For example, at observed dimension $D=8$, each two-coordinate data point becomes an eight-coordinate vector through a fixed random matrix. The matrix has orthonormal columns: each column has length one and distinct columns are perpendicular. The model receives the noisy eight-coordinate vector and time, while data construction and evaluation retain the matrix.

The paper uses a five-layer multilayer perceptron, a network of Linear layers and rectified linear unit activations, with hidden width 256. Separate runs directly predict clean data, noise, or velocity. Each output is converted to velocity before computing the training loss. Figure 2 compares observed dimensions 2, 8, 16, and 512 after projecting generated samples back to two coordinates.

The paper reports that clean-data prediction remains effective as the observed dimension increases while noise and velocity prediction deteriorate. That is the paper's observation, not a measurement from this package.

Section 3.3 does not specify the exact spiral distribution, training-set size, optimizer, learning rate, number of updates, scalar-time encoding, or endpoint treatment for all three prediction targets. The choices in this package make the exercise executable; they are not recovered settings from the authors' run. Five layers is interpreted here as five Linear layers, including the output layer.

The controlled comparison is:

| Factor | Fixed or varied |
| --- | --- |
| Intrinsic dimension | Fixed at $d=2$ |
| Observed dimension | Varied over $D\in\{2,8,16,512\}$ |
| Direct prediction | Varied over clean data $x$, noise $\epsilon$, and velocity $v$ |
| Projection, intrinsic samples, initial weights, random streams | Fixed across prediction targets at a given $D$ |
| Network | Five Linear layers, four hidden rectified linear unit activations, width 256 |
| Loss | Fixed velocity-space mean squared error |
| Training budget and sampler | Fixed across all prediction targets |

Figure 2 supplies a qualitative target for comparison, not a test oracle. No test should require clean-data prediction to win. Such a test would encode the expected conclusion instead of checking the implementation.

## 2. One sample through the data path

Take a point $\hat{x}=(1,0)$ in the underlying two-coordinate space and let $D=3$. Here, intrinsic coordinates mean these two source coordinates; observed coordinates mean the $D$ coordinates supplied to the model. The local noiseless spiral is itself a one-parameter curve within the two-coordinate space. For this hand calculation, use the simple column-orthonormal projection

$$
P=\begin{bmatrix}1&0\\0&1\\0&0\end{bmatrix},\qquad P^{\mathsf T}P=I_2.
$$

The paper writes column vectors as $x=P\hat{x}$. The package stores samples as rows, so the corresponding operation is $x=\hat{x}P^{\mathsf T}=(1,0,0)$. Projecting that row back gives $xP=(1,0)$. This simple matrix makes the arithmetic visible; the experiment must use the random matrix described below.

For any two intrinsic rows $a$ and $b$, embedding preserves squared distance:

$$
\|(a-b)P^{\mathsf T}\|_2^2=(a-b)P^{\mathsf T}P(a-b)^{\mathsf T}=\|a-b\|_2^2.
$$

The identity follows from $P^{\mathsf T}P=I_2$. Setting $b=0$ also proves preservation of vector lengths.

The package must draw a Gaussian matrix of shape $D\times2$ and factor it as $A=PR$, where $P$ has orthonormal columns and $R$ is triangular. This reduced QR decomposition is a local way to construct the fixed random matrix specified by the paper. D02 must create it once for a dimension and projection seed. Redrawing it per batch, prediction target, or plot would change the data distribution and invalidate the controlled comparison.

D01 defines the local spiral policy. For each sample draw $u\sim\operatorname{Uniform}[0,1)$ and set

$$
\phi=2\pi\,\text{turns}\,u,\qquad r=\text{spiral\_radius}\,u,\qquad \hat{x}=\bigl(r\cos\phi,\;r\sin\phi\bigr).
$$

The same $u$ controls radius and angle. For example, with two turns, radius limit 2, and $u=\tfrac14$, the point has radius $\tfrac12$ and angle $\pi$, so $\hat{x}=(-\tfrac12,0)$ up to rounding. Independent draws would fill a region rather than trace the spiral. Uniform $u$ also means points are not equally spaced along the curve; that density must remain unchanged across runs.

The data layer must own intrinsic samples and $P$; the model receives neither. With $B$ samples in a batch, it sees only $z_t\in\mathbb{R}^{B\times D}$ and $t\in\mathbb{R}^{B\times1}$. The time column broadcasts over features, giving each sample one time value. Noise has shape $B\times D$, with independent normal coordinates of mean zero and variance one. Noise limited to the intrinsic plane would remove the off-plane prediction problem the experiment is intended to expose.

| Value | Shape | Owner |
| --- | --- | --- |
| Intrinsic samples | $N\times2$ | Data construction and evaluation |
| Projection $P$ | $D\times2$ | Data construction, checkpoint metadata, and evaluation |
| Clean observed batch | $B\times D$ | Dataset and training loop |
| Time | $B\times1$ | Flow functions and sampler |
| Gaussian noise, noisy sample, prediction, velocity | $B\times D$ | Training and sampling |
| Loss | Scalar tensor | Training update until backward completes |

Create random tensors on the central processing unit (CPU) in float32 with explicit `torch.Generator` instances. The training loop must move a complete batch to the model device. This makes ownership of random state visible and keeps a data draw from silently changing another stream.

## 3. One sample through the flow equations

### 3.1 Convert each network output to velocity

Take one coordinate with clean value $x=2$, noise $\epsilon=-1$, and time $t=\tfrac14$. The paper's interpolation and velocity equations give

$$
z_t=tx+(1-t)\epsilon=\tfrac14(2)+\tfrac34(-1)=-\tfrac14,\qquad v=\frac{d z_t}{dt}=x-\epsilon=3.
$$

The two useful rearrangements are $z_t=\epsilon+tv$ and $z_t=x-(1-t)v$. Solving each for $v$, then replacing the chosen target by the network output, gives the three velocity conversions from Table 1:

$$
\begin{aligned}
\hat v_x&=\frac{\hat x-z_t}{1-t},
&\hat v_\epsilon&=\frac{z_t-\hat\epsilon}{t},
&\hat v_v&=\hat v.\\
\hat v_x\big|_{\hat x=2}&=\frac{2+1/4}{3/4}=3,
&\hat v_\epsilon\big|_{\hat\epsilon=-1}&=\frac{-1/4+1}{1/4}=3,
&\hat v_v\big|_{\hat v=3}&=3.
\end{aligned}
$$

This concrete case shows why all three outputs can be optimized in one velocity space. It does not imply that their direct prediction tasks are equally easy for a finite-width network.

[flow.py, F03](src/jit_toy/flow.py#L67) specifies these conversions. The network's direct output is named by `prediction`: `x` for clean data, `eps` for noise, and `v` for velocity.

| Direct network output | Predicted velocity |
| --- | --- |
| Clean data $\hat x$ | $\hat v_x=(\hat x-z_t)/(1-t)$ |
| Noise $\hat\epsilon$ | $\hat v_\epsilon=(z_t-\hat\epsilon)/t$ |
| Velocity $\hat v$ | $\hat v_v=\hat v$ |

### 3.2 Derive the common loss and its time weights

F02 must construct $z_t$ and the target velocity from explicit clean data, time, and noise. F03 must convert the raw output while preserving its gradient graph. F04 must reject unequal shapes before reduction and compute

$$
L_v=\frac{1}{B}\sum_{i=1}^{B}\frac{1}{D}\left\|\hat v_i-v_i\right\|_2^2.
$$

This matches the reduction in [Denoiser.forward](../denoiser.py#L49). The squared Euclidean norm is the sum of squared coordinate errors. Dividing by $D$ makes it a per-coordinate mean; this changes gradient scale relative to the paper's norm, so every prediction mode must use the same reduction. Report both this optimized mean and $DL_v$ when comparing dimensions.

For one sample at a fixed $z_t$ and $t$, substitution into the common loss gives

$$
\hat v_x-v=\frac{\hat x-x}{1-t},\qquad
\hat v_\epsilon-v=-\frac{\hat\epsilon-\epsilon}{t},\qquad
\hat v_v-v=\hat v-v.
$$

For the scalar example, a direct-output error of magnitude $0.2$ gives the following loss. Each row uses the same input and time; only the direct target changes.

| Direct output | Per-sample velocity loss in $D$ coordinates | Scalar loss at $t=0.25$, $D=1$ |
| --- | --- | --- |
| Clean data | $\lVert\hat x-x\rVert_2^2/[D(1-t)^2]$ | $0.04/(0.75)^2\approx0.0711$ |
| Noise | $\lVert\hat\epsilon-\epsilon\rVert_2^2/(Dt^2)$ | $0.04/(0.25)^2=0.64$ |
| Velocity | $\lVert\hat v-v\rVert_2^2/D$ | $0.04$ |

The common velocity loss therefore gives different time weights to errors in the direct outputs. Algebraic conversion does not make the three tasks equally tractable for a network with fixed width.

### 3.3 Explain what grows with observed dimension

For the $D=3$ projection in Section 2, choose column vectors $x=(1,0,0)^{\mathsf T}$ and $\epsilon=(0,0,1)^{\mathsf T}$. At $t=\tfrac14$, the input is $z_t=(\tfrac14,0,\tfrac34)^{\mathsf T}$ and velocity is $v=(1,0,-1)^{\mathsf T}$. The clean target's third coordinate is zero, while the noise and velocity targets have third coordinates $1$ and $-1$. Clean prediction recovers the third velocity coordinate through $(0-\tfrac34)/(1-\tfrac14)=-1$ without directly outputting it.

To generalize this example, define $Q=PP^{\mathsf T}$. This matrix projects a column vector onto the plane spanned by the columns of $P$; $I-Q$ retains the perpendicular component. Split noise as $\epsilon=Q\epsilon+(I-Q)\epsilon$. Since $x=Qx$, the perpendicular components satisfy

$$
(I-Q)x=0,\qquad (I-Q)z_t=(1-t)(I-Q)\epsilon,\qquad (I-Q)v=-(I-Q)\epsilon=-\frac{(I-Q)z_t}{1-t}.
$$

For $\epsilon\sim\mathcal N(0,I_D)$, the mean squared perpendicular noise length is $\mathbb E\|(I-Q)\epsilon\|_2^2=\operatorname{tr}(I-Q)=D-d$. Here the trace, the sum of diagonal entries, equals the number of perpendicular directions. At $D=512$ and $d=2$, that mean is 510, although any individual draw differs.

Clean-data prediction can output zero in these directions while the velocity conversion carries the observed $z_t$ through its explicit formula. Direct noise and velocity prediction require the network output to represent the perpendicular components. The width-256 network is narrower than its 512-coordinate input and has no direct input-to-output bypass. These facts motivate the capacity comparison; they do not establish how a trained model will perform.

### 3.4 Keep one time policy for all targets

Time sampling follows [Denoiser.sample_t](../denoiser.py#L45): draw $s\sim\mathcal N(-0.8,0.8^2)$ and set $t=\sigma(s)=1/(1+e^{-s})$. This is a logit-normal time distribution: a normal random variable mapped into $(0,1)$ by the sigmoid. The mean and standard deviation of $s$ come from `main_jit.py`, not Section 3.3. They are not the mean and standard deviation of $t$.

The toy must clamp time to $[\delta,1-\delta]$, with $\delta=0.001$ set by `time_eps`. Noise conversion divides by zero at $t=0$; clean-data conversion divides by zero at $t=1$. Clamping changes the time distribution near both endpoints and bounds the squared conversion factors by $\delta^{-2}=10^6$. This precaution keeps denominators finite; it does not guarantee small gradients or stable training.

The full JiT denoiser instead clamps only the denominator in its clean-data-to-velocity conversion and integrates from zero to one. Its target $(x-z_t)/\max(1-t,\delta)$ equals $x-\epsilon$ only where the clamp is inactive. The toy must use one shared interior-time policy for all three prediction targets; combining the full JiT endpoint rule with the toy noise-prediction branch would make the comparison asymmetric.

## 4. One training update in execution order

This section specifies the required execution order; the functions are still exercises. For a batch of four three-coordinate samples, the loader must return a $4\times3$ clean tensor. The training loop draws a $4\times1$ time column and $4\times3$ noise tensor. The model receives their noisy interpolation, never the clean tensor or projection.

### 4.1 Register a model with one output contract

[ToyMLP](src/jit_toy/model.py#L8) must concatenate time with the noisy input along the feature axis. The baseline layer widths are

$$
D+1\;\longrightarrow\;256\;\longrightarrow\;256\;\longrightarrow\;256\;\longrightarrow\;256\;\longrightarrow\;D.
$$

The four hidden layers use a rectified linear unit, which replaces negative values with zero. The final layer has no activation because clean, noise, and velocity targets can all have negative coordinates. This same architecture and initialization must be used for all three modes; F03, outside the model, gives the raw output its interpretation.

### 4.2 Keep effects outside the fixed-batch update

The following table answers which function owns each step in one update. The order is required by the toy contract, not observed from a local training run.

| Order | Actor and operation | Value or effect |
| --- | --- | --- |
| Before updates | E01 creates data, model, optimizer, generators, and metrics writer | The run owns their lifetimes |
| 1 | T03 fetches a clean batch and draws time and full-dimensional noise | Batch length is taken from the returned tensor |
| 2 | F02 constructs `FlowBatch` | Noisy input, time column, and target velocity |
| 3 | T02 clears gradients and enables training mode | Previous gradients cannot accumulate |
| 4 | M02 predicts; F03 converts; F04 reduces | One scalar loss with its gradient graph intact |
| 5 | T02 checks finite loss, runs backward, checks finite gradients, and steps | One completed parameter update |
| 6 | T02 returns Python scalars; T03 calls the E01 metrics callback | Logging receives no retained gradient graph |

[T01](src/jit_toy/training.py#L14) must create AdamW from the supplied parameters and learning rate, with $\beta_1=0.9$, $\beta_2=0.95$, and zero weight decay, matching the full JiT defaults. The caller must move the model to its device first. The parameter iterable may be consumed only once, so counting it before passing it to the optimizer can leave the optimizer empty.

[T02](src/jit_toy/training.py#L30) must treat `FlowBatch` tensors as read-only. The frozen dataclass prevents field reassignment but does not prevent tensor mutation. This function must draw no random numbers and open no files. A test can therefore reuse one prepared batch to check whether each prediction mode can learn that fixed case.

[T03](src/jit_toy/training.py#L59) must call T02 exactly `train_steps` times. It must use the actual batch length, including a short final batch, and restart the loader after exhaustion. For ten samples batched in fours, the first three updates use batch lengths 4, 4, and 2; a fourth update begins another pass through the loader. The metrics callback receives one-based step numbers only after completed updates. Callback failures must propagate.

[E01](src/jit_toy/experiment.py#L9) must validate device availability, create a new run directory, record configuration and environment, and own the metrics file. After training, it must save the checkpoint through a temporary path and rename, then report completion. A failed update, metrics write, or checkpoint save must not produce a completed-run report.

The baseline uses learning rate $0.001$ with no warmup or batch-size scaling. Moving averages, mixed precision, distributed training, and gradient clipping are later extensions. They would change the first comparison and make a failed numerical check harder to isolate.

## 5. Sampling and evaluation

### 5.1 Predict an average velocity, then integrate it

The target $v=x-\epsilon$ belongs to one training pair, but the model sees only $z_t$ and $t$. Different pairs can give the same noisy input. For squared loss, the ideal field is the average target velocity among pairs consistent with that input:

$$
v^*(z,t)=\mathbb E[x-\epsilon\mid z_t=z,t].
$$

This is a conditional mean: it averages over possible clean points and noise draws while holding the observed input and time fixed. The following local derivation explains the minimizer. For any proposed velocity $a$, at fixed $z$ and $t$,

$$
\mathbb E[\|a-v\|_2^2\mid z,t]=\|a-v^*(z,t)\|_2^2+\mathbb E[\|v-v^*(z,t)\|_2^2\mid z,t].
$$

The second term is independent of $a$, so the first is minimized at $a=v^*$. The clean-data conditional mean stays in the intrinsic plane, because every clean point lies there, but an average of spiral points need not lie on the spiral curve. Intermediate clean predictions should therefore not be judged as though they were final generated samples.

Sampling starts with standard Gaussian points in all $D$ observed dimensions and solves $dz_t/dt=\hat v(z_t,t)$. For $h=t_{k+1}-t_k$, Euler and Heun use

$$
\begin{aligned}
z_{k+1}^{\mathrm{Euler}}&=z_k+h\hat v(z_k,t_k),\\
z_{k+1}^{\mathrm{Heun}}&=z_k+\frac{h}{2}\left[\hat v(z_k,t_k)+\hat v\bigl(z_k+h\hat v(z_k,t_k),t_{k+1}\bigr)\right].
\end{aligned}
$$

Heun evaluates the second velocity at the Euler proposal and next time, then advances the original state using the average. It must average velocities after F03 conversion, not raw clean-data or noise predictions.

The baseline must integrate over $[\delta,1-\delta]$ with 50 intervals and 51 grid points. Starting from pure Gaussian noise at $\delta$ approximates the actual interpolation $z_\delta=\delta x+(1-\delta)\epsilon$. Stopping at $1-\delta$ also leaves an endpoint approximation. Both approximations apply to all three modes and must be recorded with results. Unlike [Denoiser.generate](../denoiser.py#L68), which integrates from zero to one and uses Euler for its final step, the toy permits Heun on every interior interval.

The sampler must run with inference mode enabled, return detached CPU float32 points, and restore the model's previous training flag even on failure. It must never apply $P$; projection belongs to evaluation.

### 5.2 Measure what projection can hide

For the simple projection in Section 2, the generated row $(1,0,2)$ projects to $(1,0)$. Its hidden third-coordinate error is 2, giving squared residual 4. A two-coordinate scatter plot would show exactly the same point for $(1,0,0)$ and $(1,0,2)$.

Projecting a generated row $X_i$ with $X_iP$ can hide error in the other $D-2$ dimensions. V01 therefore also computes

$$
R=\frac{1}{N}\sum_{i=1}^{N}\left\|X_i-X_iPP^{\mathsf T}\right\|_2^2.
$$

Because $PP^{\mathsf T}$ projects onto the plane spanned by $P$, $R$ measures mean squared distance from that plane. Compute it with projection and re-embedding, avoiding a dense $D\times D$ matrix. It is not a distance to the spiral curve, and it is zero in exact arithmetic for every point when $D=2$. Float32 arithmetic may leave a small residual.

Inspect the scatter plot and residual together: a plausible spiral can carry large perpendicular error, and a small residual can coexist with a poor spiral. A plot must include all finite generated points in its limits or count clipped points explicitly. When reporting results, explain what one plotted point represents, give any cited value from the saved artifact, and state that projection removes perpendicular components.

### 5.3 Preserve enough state for inference

E01 must save an inference checkpoint with schema version, resolved configuration, CPU model state, projection, and held-out intrinsic reference points. E02 must load it on CPU with `weights_only=True`, validate the schema, restore weights strictly, sample on the selected device, and save observed and projected points. E03 must plot that saved artifact in a new process. Resume requires additional optimizer state, completed step, loader position, and random-generator state; this checkpoint does not supply them.

## 6. Relationship to the full JiT code

The existing JiT implementation is a reference for shared mechanics. The toy package must implement its own small functions rather than import the image, distributed-training, or evaluation stack. The links below identify the relevant source entry points; they do not indicate that the toy exercises are complete.

| Existing implementation | Toy exercise | Relationship |
| --- | --- | --- |
| [Denoiser.sample_t](../denoiser.py#L45) and [forward](../denoiser.py#L49) | F01–F04 | Time distribution, interpolation, clean-data conversion, and loss reduction; adapt to row batches and add Table 1's noise and velocity branches |
| [Denoiser._euler_step](../denoiser.py#L108) and [_heun_step](../denoiser.py#L114) | S01 | The update equations derived in Section 5 |
| [Denoiser.generate](../denoiser.py#L68) | S02 | Gaussian initialization, time grid, and disabled gradients; adapt to the shared interior endpoint policy |
| [TimestepEmbedder](../model_jit.py#L40), [JiT](../model_jit.py#L205), and [FinalLayer](../model_jit.py#L162) | M01–M02 | Registered modules, time conditioning, and an unconstrained output; the toy uses the five-layer network |
| [train_one_epoch](../engine_jit.py#L16) | T02–T03 | Training mode, device transfer, finite-loss check, gradient reset, backward, and optimizer step; image normalization and CUDA synchronization are image-training details |
| [main](../main_jit.py#L115) | D07, T01, E01 | Seed, loader, model, optimizer, and checkpoint lifetimes; AdamW beta values and zero weight decay |
| [sampling demo](../sample_jit.py#L25) | E02 | CPU checkpoint loading, `weights_only=True`, strict state loading, device validation, and inference mode |

The paper is the contract for the comparison. The repository code is the reference implementation for shared mechanics. The toy-specific spiral, interior endpoint policy, scalar time concatenation, learning rate, update budget, and artifact schema are local choices. Experimental results must name those choices rather than presenting them as paper settings.

The derived requirements below connect the mathematics to an implementation check. They are design rules, not measured findings.

| Reason established above | Required design behavior | Independent check |
| --- | --- | --- |
| $P^{\mathsf T}P=I_2$ preserves distances | Fix and save one projection; do not rescale embedded data | Norm preservation and projection round trip |
| Noise has perpendicular components | Draw independent noise in every observed coordinate | F02 with a known perpendicular noise vector |
| Table 1 converts each target to velocity | Keep conversion outside one shared network architecture | Oracle outputs recover the same velocity |
| Conversion divides by $t$ or $1-t$ | Use one interior interval for all modes; this is an endpoint precaution | Finite conversions at both chosen margins |
| Plotting removes perpendicular error | Save observed samples and their residual as well as projected points | The row $(1,0,2)$ gives residual 4 in the example plane |
| Random draws and writes obscure a fixed update | Keep them outside T02 | Repeated updates on one prepared batch |

## 7. Configuration and reproducibility

### 7.1 Resolve the baseline and smaller smoke run

[baseline.json](configs/baseline.json) names every setting. [smoke.json](configs/smoke.json) supplies partial overrides; omitted values come from [ExperimentConfig](src/jit_toy/config.py#L15), whose defaults currently match the baseline. JSON parsing rejects unknown fields so a misspelled option cannot silently select a default. Save the resolved configuration rather than just the override file.

The table explains the controls that determine data quantity, training work, and sampling work. A setting changes only the operation named in its row; none of these counts guarantees convergence or sample quality.

| Setting and operation it controls | Baseline | Smoke | Concrete effect |
| --- | --- | --- | --- |
| `train_size`: number of clean training points | 8,192 | 128 | The smoke dataset contains 128 reusable points, not 128 optimizer updates |
| `eval_size`: held-out reference count and generated sample count | 2,048 | 64 | Sampling must return 64 points in the smoke run |
| `batch_size`: points per update, except a short final batch | 256 | 32 | The smoke loader has four full batches per pass |
| `train_steps`: optimizer updates | 10,000 | 10 | Ten smoke updates require two full passes and two batches of a third pass |
| `learning_rate`: optimizer step scale | 0.001 | 0.001 | Smaller counts do not change the configured step scale |
| `log_every`: interval between saved metric records | 100 | 1 | E01 must record each smoke update, and always the final update |
| `sampling_steps`: integration intervals | 50 | 5 | The smoke grid has six time points for five intervals |

### 7.2 Separate data, time, and solver settings

`spiral_turns` and `spiral_radius` set the clean curve. At their baseline values of 2 and 2, the shared draw $u=\tfrac14$ gives the point $(-\tfrac12,0)$ from Section 2. Changing either changes the clean distribution; increasing observed dimension must not change them.

`hidden_dim` and `linear_layers` set network width and depth. The baseline width is 256 and depth is five Linear layers. `observed_dim` sets input and output width, while `prediction` assigns the direct output its meaning. Across the three prediction modes at a fixed dimension, the architecture and starting weights must be identical.

`time_mean` and `time_std` set the normal variable before the sigmoid, and `time_eps` sets the endpoint margin. For a draw $s=-0.8$, the sigmoid gives $t\approx0.310$, inside the baseline interval $[0.001,0.999]$, so clamping leaves it unchanged. These settings affect which interpolation times are trained, not the clean curve or network width. Their common policy is derived in Section 3.4.

`solver` selects Euler or Heun. `sampling_steps` selects the number of equal intervals, with step size $h=(1-2\delta)/\text{sampling\_steps}$. At the baseline margin and 50 intervals, $h=0.01996$. More intervals refine the integration of the learned field; they cannot correct a poorly learned field. Record solver changes separately from training changes.

### 7.3 Give each random stream one owner

The proposed seed allocation is shown below for baseline `seed` 0 and `projection_seed` 123. Separate generator instances isolate consumption: drawing more reference points must not change training noise or data order. Model initialization has its own seeding step, even though it uses the same numeric seed as training-point generation.

| Random operation | Seed source | Baseline seed | Owner |
| --- | --- | --- | --- |
| Training points | `seed` | 0 | D06 |
| Held-out points | $\texttt{seed}+1$ | 1 | D06 |
| Data shuffling | $\texttt{seed}+2$ | 2 | E01 and D07 |
| Training time and noise | $\texttt{seed}+3$ | 3 | E01 and T03 |
| Projection | `projection_seed` | 123 | D06 |
| Model initialization | `seed` | 0 | E01 |
| Sampling noise | $\texttt{seed}+1$ | 1 | E02 and S02 |

The held-out and sampling generators reuse the numeric seed but must be separate instances. For each prediction-mode run, recreate the model and generators from the recorded seeds. Continuing another mode's trained weights would change the comparison.

Seeds alone cannot guarantee identical results across hardware and software versions. Save the actual projection, resolved configuration, Python, PyTorch and package versions, selected device, and model state. `device` selects CPU, MPS, or CUDA execution; E01 and E02 must reject an unavailable requested device. Report results within that recorded setup unless repeated experiments establish broader behavior.

### 7.4 Preserve artifacts and failure evidence

The intended experiment artifacts are:

| Artifact | Evidence it preserves |
| --- | --- |
| `config.json` | Resolved controls for the run |
| `environment.json` | Runtime versions and selected device |
| `metrics.jsonl` | One JSON object per selected update, including one-based step, `loss`, and `loss_sum`; the latter is $DL_v$, the batch mean of squared coordinate-error sums |
| `checkpoint.pt` | Model state, projection, reference points, and schema version |
| Sample artifact | Observed samples, projected samples, projection, reference points, and residual |

Reject existing output paths. A partially created training directory may remain after failure, but it must not contain a completion marker or final checkpoint presented as successful.

## 8. Verification order and next action

Verification follows data flow. First check spiral shape and reproducibility. Then check $P^{\mathsf T}P=I_2$, embedding norm preservation, and projection round trips. Next use the concrete values from Section 3 to verify that oracle clean-data, noise, and velocity predictions all convert to velocity $3$. Check the loss against a hand calculation and confirm gradients reach every model layer. Only then test a complete update, fixed-batch overfitting, loader restart, Euler and Heun on known fields, checkpoint reload equality, and the smoke workflow.

The 12-run comparison comes after those invariant checks. At each $D$, keep data, projection, initialization, random streams, update count, and sampler fixed while changing only the direct prediction target. Report individual seeds and variation before treating the paper's qualitative ordering as reproduced.

The next bounded action is D01, `sample_spiral`. Hold count, turns, radius, dtype, and seed fixed. Pass D01 only if the output has shape $\text{count}\times2$, contains finite float32 values, stays within the configured radius, and two newly created generators with the same seed return identical points. If any check fails, inspect the shared $u$ draw and the axis passed to `torch.stack`; do not proceed to D02.
