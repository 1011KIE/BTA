import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torchvision.utils import save_image

from . import backend
from .config import METRIC_PROTOCOL_VERSION
from .data import clean_prompt, decode_rle_mask, load_image, load_mapping, save_overlay, select_samples, tensor01
from .generation import generate_masked
from .methods import composite_postlock
from .policies import parse_method, resolve_routed_method
from .metrics import OptionalMetrics, core_metrics
from .reporting import append_checkpoint_row, compact_checkpoint, fmt, load_checkpoint_rows, summarize, write_summary_md

def run_experiment(args):
    backend.import_switti(args.switti_code_root)
    paths = {
        "switti": args.switti_path,
        "vae": args.vae_path,
        "clip_l": args.clip_l_path,
        "clip_g": args.clip_g_path,
    }
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    out_dir = Path(args.out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "rows_checkpoint.jsonl"
    rows, completed, incompatible_rows = load_checkpoint_rows(checkpoint_path) if args.resume else ([], set(), 0)
    if rows:
        allowed_methods = set(args.methods)
        rows = [r for r in rows if r.get("method") in allowed_methods]
        completed = {(r.get("sample_id"), r.get("method")) for r in rows}
    if args.resume and incompatible_rows:
        compact_checkpoint(checkpoint_path, rows, incompatible_rows)
    if rows:
        print(f"[resume] loaded {len(rows)} completed rows from {checkpoint_path}", flush=True)
    pipe = backend.load_pipeline(device, dtype, paths)
    opt = OptionalMetrics(device, paths["clip_l"], args.dino_ckpt)
    if args.enable_lpips:
        opt.load_lpips()
    if args.enable_clip:
        opt.load_clip()

    data_root = Path(args.data_root)
    mapping = load_mapping(data_root)
    samples, skipped = select_samples(mapping, data_root, args.num_images, args.seed, bool(args.balanced))
    t_all = time.time()

    method_cfgs = [parse_method(m) for m in args.methods]
    for sample_i, (sid, rec, img_path) in enumerate(samples):
        if all((sid, m) in completed for m in args.methods):
            print(f"[resume] skip complete sample {sample_i + 1}/{len(samples)} {sid}", flush=True)
            continue
        prompt = clean_prompt(rec.get("editing_prompt", ""))
        style_all = str(rec.get("editing_type_id", "")) == "9"
        print(f"[sample {sample_i + 1}/{len(samples)}] {sid} type={rec.get('editing_type_id')} prompt={prompt}", flush=True)
        pil, src01, xpm1 = load_image(img_path, 512, device, dtype)
        mask_np = decode_rle_mask(rec.get("mask", []), 512).astype(np.float32)
        mask_t = torch.from_numpy(mask_np)[None, None].to(device=device, dtype=torch.float32)
        if style_all:
            mask_t = torch.ones_like(mask_t)
            mask_np = np.ones((512, 512), dtype=np.float32)

        sample_dir = out_dir / "images" / sid
        if sample_i < args.save_images:
            sample_dir.mkdir(parents=True, exist_ok=True)
            pil.save(sample_dir / "source.png")
            save_overlay(pil, mask_np > 0.5, sample_dir / "mask_overlay.png")

        inv_t0 = time.time()
        # SWITTI generation is stable in bf16, but the VQVAE nearest-neighbor
        # encoder compares float activations with the codebook. Encode source
        # tokens in fp32, then return VQVAE to the generation dtype.
        pipe.vae.to(dtype=torch.float32)
        with torch.inference_mode():
            source_tokens = pipe.vae.img_to_idxBl(xpm1.float())
            src_rec01 = tensor01(pipe.vae.idxBl_to_img(source_tokens, same_shape=True, last_one=True))
        pipe.vae.to(dtype=dtype)
        inv_sec = time.time() - inv_t0
        if sample_i < args.save_images:
            save_image(src_rec01, sample_dir / "source_recon.png")
        src_self_sim = opt.dino_self_sim(src01)

        for cfg_m in method_cfgs:
            key = (sid, cfg_m.name)
            if key in completed:
                print(f"  [resume] skip {cfg_m.name}", flush=True)
                continue
            actual_cfg = resolve_routed_method(cfg_m, rec.get("editing_type_id", ""), float(mask_t.mean().item()))
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_alloc = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            if cfg_m.name == "source_recon":
                out01 = src_rec01
            else:
                with torch.inference_mode(), torch.autocast("cuda", enabled=torch.cuda.is_available(), dtype=dtype):
                    out01, _ = generate_masked(
                        pipe,
                        prompt=prompt,
                        source_tokens=source_tokens,
                        mask_t=mask_t,
                        method_cfg=actual_cfg,
                        style_all=style_all,
                        seed=args.seed,
                        cfg_scale=args.cfg,
                        top_k=args.top_k,
                        top_p=args.top_p,
                        turn_off_cfg_start_si=args.turn_off_cfg_start_si,
                        last_scale_temp=args.last_scale_temp,
                    )
                out01 = out01.float()
                if actual_cfg.post_lock:
                    out01 = composite_postlock(src01, out01, mask_t, actual_cfg.inner_feather).clamp(0, 1)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            elapsed = time.time() - t0
            peak_alloc = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            peak_reserved = torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0
            row = {
                "sample_id": sid,
                "method": cfg_m.name,
                "resolved_method": (
                    f"{actual_cfg.projection}_fix{actual_cfg.fix_si}_{actual_cfg.tau_family}_"
                    f"floor{actual_cfg.floor_mult:g}_erode{actual_cfg.mask_erode_px}"
                ),
                "image_path": str(img_path),
                "editing_type_id": rec.get("editing_type_id", ""),
                "original_prompt": rec.get("original_prompt", ""),
                "editing_prompt": prompt,
                "mask_frac": float(mask_t.mean().item()),
                "metric_protocol_version": METRIC_PROTOCOL_VERSION,
                "inverse_sec": inv_sec,
                "elapsed_sec": elapsed,
                "gen_peak_alloc_mib": float(peak_alloc / (1024**2)),
                "gen_peak_reserved_mib": float(peak_reserved / (1024**2)),
                "gen_peak_extra_alloc_mib": float(max(0, peak_alloc - start_alloc) / (1024**2)),
            }
            row.update(core_metrics(src01, out01, mask_t, opt, prompt, src_self_sim))
            rows.append(row)
            completed.add(key)
            append_checkpoint_row(checkpoint_path, row)
            if sample_i < args.save_images:
                save_image(out01, sample_dir / f"{cfg_m.name}.png")
            print(
                f"  {cfg_m.name}: PSNR={fmt(row.get('bg_psnr'))} MSE*1e4={fmt(row.get('bg_mse_x1e4'))} "
                f"SSIM*1e2={fmt(row.get('bg_ssim_x1e2'))} CLIP={row['clip_whole']} sec={elapsed:.2f} "
                f"peak_alloc={row['gen_peak_alloc_mib']:.0f}MiB",
                flush=True,
            )

    summary = summarize(rows, args.methods)
    result = {
        "metric_protocol_version": METRIC_PROTOCOL_VERSION,
        "config": vars(args),
        "protocol_note": {
            "target_external_protocol": "MLN PIE-512 Table 1",
            "actual_backbone": "SWITTI-512",
            "actual_resolution": 512,
            "strict_protocol_match": False,
            "mismatch": "Uses PIE ground-truth masks for token projection rather than MLN cross-attention-derived masks; MLN official code is unavailable here.",
            "metric_protocol_version": METRIC_PROTOCOL_VERSION,
            "background_metrics": "PIE-Bench official-style masked full-image metrics: compare source*background_mask vs output*background_mask; full-mask samples have undefined background metrics and are excluded from means.",
            "lpips": "LPIPS uses squeeze backbone, matching the common PIE-Bench evaluation implementation.",
            "ssim": "SSIM uses a Gaussian 11x11 sigma=1.5 window on masked full-image tensors.",
            "structure_distance": "PnPInversion-style DINO ViT-B/8 layer-11 key self-similarity MSE on the whole image; lower is better.",
            "clip_edited": "PIE-Bench official-style edited CLIP: compute CLIP on output*edit_mask, not a bbox crop.",
        },
        "device": device,
        "torch": torch.__version__,
        "elapsed_sec": time.time() - t_all,
        "skipped_samples": skipped[:100],
        "summary": summary,
        "rows": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with (out_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "sample_id",
            "method",
            "resolved_method",
            "editing_type_id",
            "mask_frac",
            "metric_protocol_version",
            "bg_valid",
            "bg_psnr",
            "bg_mse_x1e4",
            "bg_ssim_x1e2",
            "bg_lpips_x1e3",
            "outer5_l1",
            "outer10_l1",
            "edit_l1",
            "structure_distance",
            "clip_whole",
            "clip_edited",
            "edit_clip_image_h",
            "edit_clip_image_w",
            "edit_clip_mask_frac",
            "edit_clip_view",
            "inverse_sec",
            "elapsed_sec",
            "gen_peak_alloc_mib",
            "gen_peak_reserved_mib",
            "gen_peak_extra_alloc_mib",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in fieldnames})
    write_summary_md(out_dir / "summary.md", result)
    print(json.dumps({"out_dir": str(out_dir), "summary": summary, "elapsed_sec": result["elapsed_sec"]}, indent=2), flush=True)
