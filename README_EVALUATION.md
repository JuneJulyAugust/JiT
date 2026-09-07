# Evaluate pretrained JiT-B

## Local setup status (September 3, 2026)

- Branch: `codex/jit-b-evaluation-demo`.
- Reusing the user-provided `.venv -> /Users/fang/.venv`; no packages were installed
  into that shared environment. Its Python 3.9.6 currently has only pip/setuptools,
  and `import torch` fails. Point `.venv` at the intended PyTorch environment
  before running the demo; no additional environment is necessary.
- Official B/16 checkpoint downloaded to `checkpoints/jit-b-16/checkpoint-last.pth`.
  Its size matches the server, and ZIP metadata contains `model`, `model_ema1`,
  and `model_ema2`. Locally calculated SHA-256 (not an author-published checksum):
  `4ebcf24698748548d13bef1b4c3b26c72c6ec2bc633002b3f558697920cb2695`.
- Syntax and four CLI tests passed. Two tensor tests were skipped because PyTorch
  is missing. After activating `.venv` as shown below, run all tests with
  `python -B -m unittest discover -s tests -v`.
- The demo was attempted and stopped at `ModuleNotFoundError: No module named
  'torch'`. **No sample images or FID/IS scores have been produced yet.**

## Do I need ImageNet?

**No, not for sampling or FID/IS against the supplied reference statistics.**
JiT generates images from noise and integer class labels; it does not need input
photographs. This repository already includes `fid_stats/jit_in256_stats.npz`
and `fid_stats/jit_in512_stats.npz` (about 32 MiB each).

The original `main_jit.py` constructed an ImageNet training loader even with
`--evaluate_gen`. This branch skips that loader during generation evaluation.
ImageNet is only needed for **training** or rebuilding the reference image set.
This is generative evaluation, not ImageNet classification accuracy.

## 1. Reuse your Python environment

Run all commands from the repository root. An existing `.venv` may be a symlink;
do not run `python -m venv .venv` over it. Activate it before installing packages
or running Python commands, then check its actual interpreter:

```bash
ls -ld .venv
source .venv/bin/activate
python -c "import sys, torch; print(sys.executable); print(sys.version); print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print('MPS:', torch.backends.mps.is_available())"
```

Repeat `source .venv/bin/activate` once per new shell session. The commands below
use `python`, `pip`, `torchrun`, and `tensorboard` from that activated environment.

The small demo only needs PyTorch, NumPy, einops, and Pillow. If dependencies are
missing, first confirm this is the environment you want to modify, then:

```bash
pip install -r requirements-demo.txt
```

This can install PyTorch if it is absent; it does not create another environment.
Do not run it just to check dependencies in a shared environment. The original
project uses PyTorch 2.5.1 and torchvision 0.20.1; its complete reference setup is
in `environment.yaml`. The portable demo disables compilation and uses float32
on CPU/MPS, rather than trying to run the CUDA training stack on macOS.

`.venv`, downloaded checkpoints, datasets, and generated outputs are ignored by Git.

## 2. Download only JiT-B/16 (256 × 256)

Use the [authors' official checkpoint folder](https://www.dropbox.com/scl/fo/3ken1avtsd81ip67b9qpi/AK218ZNvXKSv74igVvht4PQ?rlkey=14gjrblmljewpl6ygxzlr3njm&dl=0).
Choose `jit-b-16/checkpoint-last.pth`, **not Download on the root folder**:
the whole collection is about 34.5 GiB. JiT-B/16 alone is 1,576,057,392 bytes
(about 1.47 GiB). These sizes were checked on September 3, 2026 (local time).

The following individual-file URL was obtained from that folder:

```bash
mkdir -p checkpoints/jit-b-16
curl --fail --location --retry 2 --max-filesize 1700000000 \
  --output checkpoints/jit-b-16/checkpoint-last.pth.part \
  'https://www.dropbox.com/scl/fo/3ken1avtsd81ip67b9qpi/AGlp4FoN0cIF8nMbS4DN7Ns/jit-b-16/checkpoint-last.pth?rlkey=14gjrblmljewpl6ygxzlr3njm&dl=1' && \
mv -n checkpoints/jit-b-16/checkpoint-last.pth.part checkpoints/jit-b-16/checkpoint-last.pth
```

Skip the download if you already have the checkpoint. You can pass its existing
path directly to the demo, avoiding a second copy. If the link changes, open the
official folder and download that individual file manually. Keep the `.part`
suffix until a download succeeds; never try to load an incomplete download.

For 512 × 512, download the separate `jit-b-32/checkpoint-last.pth` from the same
folder and use `--model JiT-B/32`. Do not reuse B/16 weights for B/32.

## 3. Run a small local sampling demo

Start with one image, using the pretrained model's first EMA weights and the
published JiT-B guidance settings (CFG 3.0, interval 0.1–1.0, 50 Heun steps):

```bash
python sample_jit.py \
  --checkpoint checkpoints/jit-b-16/checkpoint-last.pth \
  --model JiT-B/16 --device auto \
  --class-ids 207 --steps 50 --seed 0 \
  --output-dir outputs/jit-b-demo
```

`auto` selects CUDA, then Apple MPS, then CPU. CPU can be very slow. Batch size
is one to limit memory use. For several classes, use e.g. `--class-ids 207 281 388`;
indices must be between 0 and 999. `--steps 2` is a quick plumbing smoke test,
**not** a quality evaluation. Use a new/empty output directory on each run.

Outputs are PNG images and `run.json` containing the settings, device, PyTorch
version, and elapsed time. The demo checks for missing/mismatched checkpoints
and non-finite pixels; it never silently samples random untrained weights.
It uses `torch.load(..., weights_only=True)` and requires `model_ema1`.

**A few sample images are not a FID/IS benchmark.** CPU/MPS float32 results are
not claimed to numerically reproduce CUDA bfloat16 results.

## 4. Full FID / Inception Score evaluation (NVIDIA CUDA)

The original evaluation path still requires CUDA, NCCL, and `torchrun` (even for
one GPU). It is not a macOS/MPS command. Run it on an existing NVIDIA GPU machine;
no cloud machine or paid compute is provisioned by this setup.

The full evaluator also imports torchvision, OpenCV, TensorBoard, and the authors'
[custom torch-fidelity fork](https://github.com/LTH14/torch-fidelity). The ordinary
PyPI torch-fidelity package is not a substitute for its `fid_statistics_file`
extension. Use `environment.yaml` for the upstream reference environment. If
setting up an existing Python 3.10–3.12 virtual environment manually, a compatible
installation recipe is below (it changes packages in that environment):

```bash
# On the Linux/NVIDIA machine, activate its existing environment first.
source .venv/bin/activate
# Skip if a suitable torch/torchvision pair is already installed.
pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu124
pip install numpy==1.26.4 scipy==1.11.4 \
  opencv-python==4.11.0.86 timm==0.9.12 tensorboard==2.18.0 einops==0.8.1
pip install 'git+https://github.com/LTH14/torch-fidelity.git@master'
```

The non-PyTorch pins above accommodate newer Python versions; they are not an
exact reproduction of the original conda lock. See the [official PyTorch version
matrix](https://pytorch.org/get-started/previous-versions/) for platform-specific
installation. FID/IS also downloads Inception feature-extractor weights on first
use; these are separate from the JiT checkpoint and are cached by PyTorch.

One-GPU command (start with a conservative generation batch size):

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  main_jit.py \
  --model JiT-B/16 --img_size 256 --noise_scale 1.0 \
  --gen_bsz 8 --num_images 50000 \
  --sampling_method heun --num_sampling_steps 50 \
  --cfg 3.0 --interval_min 0.1 --interval_max 1.0 \
  --output_dir outputs/jit-b-fid50k \
  --resume checkpoints/jit-b-16 --evaluate_gen
```

No `--data_path` is needed on this branch. `--resume` is a **directory** containing
`checkpoint-last.pth`, not the checkpoint filename. For B/32 use `--model JiT-B/32
--img_size 512 --noise_scale 2.0` and its own checkpoint directory.

For eight GPUs, change `--nproc_per_node=1` to `8` and increase `--gen_bsz` only as
memory permits (the upstream example uses 256 per GPU). There are 50 samples per
class at 50,000 images. `--num_images 1000` exercises the metric path with one
sample per class but produces a noisy score, not the reported 50k benchmark.
The count must be divisible by 1000; do not change `--class_num` to work around it.

The evaluator prints FID and Inception Score, writes TensorBoard events, and
**deletes its generated image subfolder after successful metric computation**.
It needs temporary disk space for 50,000 PNGs; raw RGB at 256 × 256 is about
9.2 GiB before compression, and 512 × 512 requires four times as much.
The small demo above keeps its images. To view metric logs:

```bash
tensorboard --logdir outputs/jit-b-fid50k
```

## 5. Download ImageNet only if training or rebuilding references

Use the [official ImageNet download page](https://www.image-net.org/download.php),
which links to the ILSVRC dataset on Kaggle. Complete any account/access requests
and accept the applicable terms yourself. Select the **ILSVRC 2012 1,000-class
classification dataset**. The training set contains 1,281,167 images; validation
contains 50,000. JiT's `prepare_ref.py` uses the **training** split, not validation.
Do not download either split just for the evaluation above.

For the original `ILSVRC2012_img_train.tar` layout, after downloading the archive
through an authorized source, unpack into class directories:

```bash
mkdir -p data/imagenet/train
tar -xf /path/to/ILSVRC2012_img_train.tar -C data/imagenet/train
for class_archive in data/imagenet/train/*.tar; do
  class_dir="${class_archive%.tar}"
  mkdir -p "$class_dir"
  tar -xf "$class_archive" -C "$class_dir"
done
```

The archive is large and extraction needs substantial additional disk space.
This deliberately retains the archives; remove them yourself only after checking
the extracted data. A Kaggle bundle may already have the class-folder layout.
The resulting structure expected by `ImageFolder` is:

```text
data/imagenet/
  train/
    n01440764/
      *.JPEG
    n01443537/
      *.JPEG
    ... (1,000 class directories)
```

If you really need a resized reference image folder:

```bash
python prepare_ref.py \
  --data_path data/imagenet --img_size 256 \
  --output_path data/imagenet-train-256
```

This writes another full training-image set; it does **not** itself calculate an
NPZ statistics file. The bundled FID statistics avoid both this storage cost and
the ImageNet download for pretrained evaluation.

## Troubleshooting

- `No module named torch`: run `source .venv/bin/activate` from the repository
  root, then check `python -c "import sys; print(sys.executable)"` and the `.venv`
  symlink target. Another project's environment may select a different Python.
- CUDA/NCCL errors on Mac: use `sample_jit.py --device mps` (or `cpu`), not the
  original `torchrun main_jit.py` evaluator.
- Out of memory: use the batch-one demo, or lower `--gen_bsz` for CUDA metrics.
- Missing or incompatible `model_ema1`: use the matching official checkpoint;
  do not disable strict loading or fall back to random weights.
- `fid_statistics_file` errors: verify that the custom torch-fidelity fork is
  installed and run the evaluator from the repository root.
- Existing demo output directory: choose another `--output-dir`; samples are
  protected from accidental overwriting.
