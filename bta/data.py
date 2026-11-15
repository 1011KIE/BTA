import json
import random

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

def decode_rle_mask(rle, size=512):
    arr = np.zeros(size * size, dtype=np.uint8)
    vals = list(map(int, rle or []))
    for start, length in zip(vals[0::2], vals[1::2]):
        if length > 0 and start < arr.size:
            arr[start : min(start + length, arr.size)] = 1
    mask = arr.reshape(size, size)
    # PIE-Bench evaluation masks force the image boundary into the edit mask
    # to avoid annotation errors at the edges.
    mask[0, :] = 1
    mask[-1, :] = 1
    mask[:, 0] = 1
    mask[:, -1] = 1
    return mask

def clean_prompt(text):
    return (text or "").replace("[", "").replace("]", "")

def load_mapping(data_root):
    with (data_root / "mapping_file.json").open("r", encoding="utf-8") as f:
        return json.load(f)

def mask_fraction(rec):
    return float(decode_rle_mask(rec.get("mask", []), 512).mean())

def select_samples(mapping, data_root, limit, seed, balanced=True, min_mask_frac=0.0, max_mask_frac=1.0):
    items = []
    skipped = []
    for sid in sorted(mapping.keys()):
        rec = mapping[sid]
        img_path = data_root / "annotation_images" / rec["image_path"]
        if not img_path.exists():
            skipped.append({"sample_id": sid, "reason": "missing_image"})
            continue
        is_style = str(rec.get("editing_type_id", "")) == "9"
        frac = mask_fraction(rec)
        if not is_style and (frac < min_mask_frac or frac > max_mask_frac):
            skipped.append({"sample_id": sid, "reason": "mask_frac", "mask_frac": frac})
            continue
        items.append((sid, rec, img_path, rec.get("editing_type_id", "unknown")))
    if not balanced:
        return [(sid, rec, path) for sid, rec, path, _ in items[:limit]], skipped

    rng = random.Random(seed)
    by_cat = {}
    for item in items:
        by_cat.setdefault(item[3], []).append(item)
    for bucket in by_cat.values():
        rng.shuffle(bucket)
    cats = sorted(by_cat.keys())
    out = []
    while len(out) < limit and any(by_cat.values()):
        for cat in cats:
            if by_cat.get(cat):
                out.append(by_cat[cat].pop(0))
                if len(out) >= limit:
                    break
    out.sort(key=lambda x: x[0])
    return [(sid, rec, path) for sid, rec, path, _ in out], skipped

def load_image(path, size, device, dtype):
    pil = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    x01 = transforms.ToTensor()(pil).unsqueeze(0).to(device=device)
    xpm1 = x01.mul(2).sub(1).to(dtype=dtype)
    return pil, x01, xpm1

def tensor01(x):
    return x.detach().float().clamp(-1, 1).add(1).mul(0.5)

def save_overlay(img, mask_np, path):
    base = img.convert("RGBA").resize((512, 512), Image.BICUBIC)
    mask = Image.fromarray((mask_np.astype(np.uint8) * 255)).resize((512, 512), Image.NEAREST)
    red = Image.new("RGBA", base.size, (255, 0, 0, 88))
    blank = Image.new("RGBA", base.size, (255, 0, 0, 0))
    Image.alpha_composite(base, Image.composite(red, blank, mask)).convert("RGB").save(path)
