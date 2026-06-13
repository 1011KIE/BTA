#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-data/PIE-Bench_v1}
SWITTI_CODE_ROOT=${SWITTI_CODE_ROOT:-external/switti}
SWITTI_PATH=${SWITTI_PATH:-checkpoints/yresearch_Switti}
VAE_PATH=${VAE_PATH:-checkpoints/yresearch_VQVAE-Switti}
CLIP_L_PATH=${CLIP_L_PATH:-checkpoints/openai_clip-vit-large-patch14}
CLIP_G_PATH=${CLIP_G_PATH:-checkpoints/laion_CLIP-ViT-bigG-14-laion2B-39B-b160k}
DINO_CKPT=${DINO_CKPT:-checkpoints/dino_vitbase8_pretrain.pth}
OUT_DIR=${OUT_DIR:-outputs/switti512_main}
PYTHON=${PYTHON:-python}

"$PYTHON" run.py \
  --data-root "$DATA_ROOT" \
  --out-dir "$OUT_DIR" \
  --switti-code-root "$SWITTI_CODE_ROOT" \
  --switti-path "$SWITTI_PATH" \
  --vae-path "$VAE_PATH" \
  --clip-l-path "$CLIP_L_PATH" \
  --clip-g-path "$CLIP_G_PATH" \
  --dino-ckpt "$DINO_CKPT" \
  --num-images 700 \
  --seed 123 \
  --save-images 0 \
  --balanced 1 \
  --resume \
  --enable-lpips \
  --enable-clip \
  --cfg 5.5 \
  --top-k 200 \
  --top-p 0.9 \
  --turn-off-cfg-start-si 10 \
  --last-scale-temp 0.1 \
  --methods \
    scale_fcmp_hard_fix3_erode8 \
    budget_auth_psnr_guard_fix2_erode12
