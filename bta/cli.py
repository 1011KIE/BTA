import argparse
import json

from .policies import parse_method


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/PIE-Bench_v1")
    ap.add_argument("--out-dir", default="outputs/switti512_main")
    ap.add_argument("--switti-code-root", default="external/switti")
    ap.add_argument("--switti-path", default="checkpoints/yresearch_Switti")
    ap.add_argument("--vae-path", default="checkpoints/yresearch_VQVAE-Switti")
    ap.add_argument("--clip-l-path", default="checkpoints/openai_clip-vit-large-patch14")
    ap.add_argument("--clip-g-path", default="checkpoints/laion_CLIP-ViT-bigG-14-laion2B-39B-b160k")
    ap.add_argument("--dino-ckpt", default="checkpoints/dino_vitbase8_pretrain.pth")
    ap.add_argument("--num-images", type=int, default=10)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument("--top-k", type=int, default=400)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--turn-off-cfg-start-si", type=int, default=9)
    ap.add_argument("--last-scale-temp", type=float, default=0.1)
    ap.add_argument("--save-images", type=int, default=4)
    ap.add_argument("--balanced", type=int, default=1)
    ap.add_argument("--enable-lpips", action="store_true")
    ap.add_argument("--enable-clip", action="store_true")
    ap.add_argument("--resume", action="store_true", help="Resume from rows_checkpoint.jsonl in out-dir.")
    ap.add_argument("--dry-run", action="store_true", help="Validate configuration without loading SWITTI weights.")
    ap.add_argument(
        "--methods",
        nargs="+",
        default=[
            "source_recon",
            "txt2img",
            "mask_nearest_fix4",
            "mask_maxpool_fix4",
            "scale_fcmp_soft_fix4",
            "scale_fcmp_hard_fix4",
        ],
    )
    return ap


def dry_run(args):
    methods = [parse_method(name) for name in args.methods]
    paths = {
        "data_root": args.data_root,
        "switti_code_root": args.switti_code_root,
        "switti": args.switti_path,
        "vae": args.vae_path,
        "clip_l": args.clip_l_path,
        "clip_g": args.clip_g_path,
        "dino_ckpt": args.dino_ckpt,
        "out_dir": args.out_dir,
    }
    print(json.dumps({"dry_run": True, "methods": [m.name for m in methods], "paths": paths}, indent=2))
