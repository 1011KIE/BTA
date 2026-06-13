import json
import math
import shutil
import time

import numpy as np

from .config import METRIC_PROTOCOL_VERSION

def summarize(rows, methods):
    metrics = [
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
        "bg_valid",
        "edit_clip_image_h",
        "edit_clip_image_w",
        "edit_clip_mask_frac",
        "elapsed_sec",
        "gen_peak_alloc_mib",
        "gen_peak_reserved_mib",
        "gen_peak_extra_alloc_mib",
    ]
    out = {}
    for method in methods:
        sub = [r for r in rows if r["method"] == method]
        out[method] = {"n": len(sub)}
        for metric in metrics:
            vals = [r[metric] for r in sub if r.get(metric) is not None and not (isinstance(r.get(metric), float) and math.isnan(r.get(metric)))]
            out[method][metric + "_n"] = len(vals)
            out[method][metric + "_mean"] = float(np.mean(vals)) if vals else None
            out[method][metric + "_median"] = float(np.median(vals)) if vals else None
    return out

def write_summary_md(path, result):
    lines = []
    lines.append("# R039 SWITTI-512 Masked Editing Results")
    lines.append("")
    lines.append(f"- Samples: {len(result['rows']) // max(1, len(result['config']['methods']))}")
    lines.append(f"- Methods: {', '.join(result['config']['methods'])}")
    lines.append(f"- Strict protocol match: {result['protocol_note']['strict_protocol_match']}")
    lines.append("")
    lines.append("| Method | PSNR up | LPIPS x1e3 down | MSE x1e4 down | SSIM x1e2 up | Distance down | CLIP whole up | CLIP edited up | sec down | Peak alloc MiB | Peak reserved MiB | Extra alloc MiB |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for method, vals in result["summary"].items():
        lines.append(
            "| {m} | {psnr} | {lpips} | {mse} | {ssim} | {dist} | {cw} | {ce} | {sec} | {peak_alloc} | {peak_reserved} | {extra_alloc} |".format(
                m=method,
                psnr=fmt(vals.get("bg_psnr_mean")),
                lpips=fmt(vals.get("bg_lpips_x1e3_mean")),
                mse=fmt(vals.get("bg_mse_x1e4_mean")),
                ssim=fmt(vals.get("bg_ssim_x1e2_mean")),
                dist=fmt(vals.get("structure_distance_mean")),
                cw=fmt(vals.get("clip_whole_mean")),
                ce=fmt(vals.get("clip_edited_mean")),
                sec=fmt(vals.get("elapsed_sec_mean")),
                peak_alloc=fmt(vals.get("gen_peak_alloc_mib_mean")),
                peak_reserved=fmt(vals.get("gen_peak_reserved_mib_mean")),
                extra_alloc=fmt(vals.get("gen_peak_extra_alloc_mib_mean")),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def fmt(x):
    return "NA" if x is None else f"{x:.3f}"

def load_checkpoint_rows(path):
    rows = []
    completed = set()
    if not path.exists():
        return rows, completed, 0
    incompatible = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("metric_protocol_version") != METRIC_PROTOCOL_VERSION:
                incompatible += 1
                continue
            key = (row.get("sample_id"), row.get("method"))
            if key[0] is None or key[1] is None or key in completed:
                continue
            rows.append(row)
            completed.add(key)
    return rows, completed, incompatible

def compact_checkpoint(path, rows, incompatible):
    if incompatible <= 0:
        return
    if path.exists():
        stale = path.with_name(f"{path.name}.stale_{int(time.time())}")
        shutil.copy2(path, stale)
        print(f"[resume] backed up stale checkpoint rows to {stale}", flush=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        f"[resume] compacted checkpoint to {len(rows)} current-protocol rows; ignored {incompatible} stale rows",
        flush=True,
    )

def append_checkpoint_row(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
