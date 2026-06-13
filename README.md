# BTA Main Experiment Code

This repository contains the main experiment code for Boundary-aware Token Assignment (BTA) on fixed-mask autoregressive image editing. The release is intentionally minimal: a Python entry point, focused runner modules, one launch script, and dependency metadata.

## Project Structure

```text
.
|-- run.py                 # main PIE-Bench 512x512 SWITTI experiment
|-- bta/                   # data, policy, metric, generation, and runner modules
|-- scripts/
|   `-- run_main.sh        # shell wrapper for the full 700-sample run
|-- requirements.txt
`-- README.md
```

The repository does not include model weights, datasets, generated images, HuggingFace caches, or third-party model repositories.

## Environment

Create a Python environment with CUDA-enabled PyTorch and the required packages:

```bash
pip install -r requirements.txt
```

The reported profiling used a single NVIDIA 4090.

## Data Preparation

PIE-Bench can be requested from the official PnP Inversion resources:

- Project page: https://cure-lab.github.io/PnPInversion/
- Official code: https://github.com/cure-lab/PnPInversion
- PIE-Bench access form: https://forms.gle/hVMkTABb4uvZVjme9

Prepare PIE-Bench with the original directory layout:

```text
data/PIE-Bench_v1/
|-- mapping_file.json
`-- annotation_images/
```

If the dataset is stored elsewhere, pass its location through `--data-root` or the `DATA_ROOT` environment variable.

## Model Preparation

The runner expects an external SWITTI code checkout and local model checkpoints:

```text
external/switti/                         # official SWITTI source tree
checkpoints/yresearch_Switti/            # SWITTI checkpoint
checkpoints/yresearch_VQVAE-Switti/      # VQ-VAE checkpoint
checkpoints/openai_clip-vit-large-patch14/
checkpoints/laion_CLIP-ViT-bigG-14-laion2B-39B-b160k/
```

The DINO ViT-B/8 checkpoint is used for structure distance. If it is absent, `run.py` downloads it to `checkpoints/dino_vitbase8_pretrain.pth` when the metric is first used.

## Running the Main Experiment

Run the full 700-sample main experiment with:

```bash
bash scripts/run_main.sh
```

Custom paths can be supplied without editing the script:

```bash
DATA_ROOT=/path/to/PIE-Bench_v1 \
SWITTI_CODE_ROOT=/path/to/switti \
SWITTI_PATH=/path/to/yresearch_Switti \
VAE_PATH=/path/to/yresearch_VQVAE-Switti \
CLIP_L_PATH=/path/to/openai_clip-vit-large-patch14 \
CLIP_G_PATH=/path/to/laion_CLIP-ViT-bigG-14-laion2B-39B-b160k \
OUT_DIR=outputs/switti512_main \
bash scripts/run_main.sh
```

The wrapper evaluates the coverage-only reference and BTA under the same prompts, masks, sampler, tokenizer, and source-token cache:

```text
scale_fcmp_hard_fix3_erode8
budget_auth_psnr_guard_fix2_erode12
```

For a quick smoke test, call `run.py` directly with fewer samples:

```bash
python run.py \
  --data-root /path/to/PIE-Bench_v1 \
  --switti-code-root /path/to/switti \
  --switti-path /path/to/yresearch_Switti \
  --vae-path /path/to/yresearch_VQVAE-Switti \
  --clip-l-path /path/to/openai_clip-vit-large-patch14 \
  --clip-g-path /path/to/laion_CLIP-ViT-bigG-14-laion2B-39B-b160k \
  --out-dir outputs/smoke \
  --num-images 2 \
  --save-images 1 \
  --methods scale_fcmp_hard_fix3_erode8 budget_auth_psnr_guard_fix2_erode12
```

## Outputs

The output directory contains:

- `results.json`: full configuration, per-sample rows, and aggregate summary.
- `metrics.csv`: tabular per-sample metrics.
- `summary.md`: compact table of aggregate metrics.
- `rows_checkpoint.jsonl`: resume checkpoint when `--resume` is enabled.
- `images/`: optional source, mask overlay, and edited images when `--save-images` is greater than zero.

## Notes

BTA is implemented as a deterministic decode-time token commitment rule. The code does not train a new model, optimize prompts, modify the tokenizer, or perform pixel-space post-copying for the reported BTA row.
