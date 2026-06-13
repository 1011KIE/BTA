import math
from .policies import MethodCfg
import torch
import torch.nn.functional as F

def coverage_map(mask_t, pn, mode):
    if mode == "nearest":
        return F.interpolate(mask_t, size=(pn, pn), mode="nearest")[0, 0]
    if mode == "area":
        return F.interpolate(mask_t, size=(pn, pn), mode="area")[0, 0]
    if mode == "max":
        # adaptive max-pooling behaves like a token-cell any-overlap test.
        return F.adaptive_max_pool2d(mask_t.float(), (pn, pn))[0, 0]
    raise ValueError(mode)

def tau_for_scale(si, n_scales, family):
    if family == "flat":
        return 0.50
    ratio = si / max(1, n_scales - 1)
    if family == "soft":
        return 0.72 - 0.45 * ratio
    if family == "hard":
        return 0.90 - 0.55 * ratio
    if family == "coarse_strict":
        return 0.96 - 0.70 * ratio
    raise ValueError(family)

def token_authority_edit_mask(cov, si, n_scales, mode):
    ratio = si / max(1, n_scales - 1)
    if mode.startswith("budget_"):
        # Budgeted Token Authority keeps preservation and edit pressure separate.
        # The output is still a clean token-level edit mask: no pixel copy/postlock.
        boundary = cov * (1.0 - cov) * 4.0
        coarse_guard = 1.0 - ratio
        preserve = (1.0 - cov) * (0.70 + 0.55 * coarse_guard) + 0.35 * boundary
        edit = cov * (0.65 + 0.75 * ratio)
        min_core = 0.14 + 0.18 * coarse_guard
        shared_core = 0.36 - 0.16 * ratio

        if mode == "budget_clip_guard":
            # Start close to source_fix4, then release prompt authority only for
            # high-edit-value fine/core tokens to recover CLIP.
            edit = edit * 1.12
            preserve = preserve * 0.96
            gate = (edit >= preserve) & (cov >= min_core)
            return gate | ((ratio >= 0.82) & (cov >= shared_core))
        if mode == "budget_core_release":
            # More conservative than clip_guard: release only eroded edit cores.
            gate = (edit >= preserve * 1.08) & (cov >= (min_core + 0.06))
            return gate | ((ratio >= 0.88) & (cov >= (shared_core + 0.08)))
        if mode == "budget_bg_fix4":
            # Background-fidelity first, with just enough late prompt budget to
            # avoid the CLIP drop seen in source_fix4.
            preserve = preserve * 1.08
            edit = edit * 1.04
            gate = (edit >= preserve) & (cov >= (min_core + 0.03))
            return gate | ((ratio >= 0.90) & (cov >= (shared_core + 0.10)))
        if mode == "budget_bg_sem":
            # Add a narrow semantic release band to bg_fix4. This keeps coarse
            # source authority, but gives fine high-coverage tokens prompt
            # ownership where oos pilots showed CLIP was under-recovered.
            preserve = preserve * 1.03
            edit = edit * 1.16
            gate = (edit >= preserve * 0.99) & (cov >= (min_core + 0.01))
            semantic_tail = (ratio >= 0.82) & (cov >= (shared_core + 0.02))
            fine_core = (ratio >= 0.92) & (cov >= 0.24)
            return gate | semantic_tail | fine_core
        if mode == "budget_bg_sem_core":
            # More conservative than bg_sem: release only fine-scale edit cores.
            preserve = preserve * 1.05
            edit = edit * 1.13
            gate = (edit >= preserve * 1.02) & (cov >= (min_core + 0.03))
            fine_core = (ratio >= 0.88) & (cov >= (shared_core + 0.05))
            return gate | fine_core
        if mode == "budget_scale_relax":
            # Strong source authority in coarse scales, relaxed prompt authority
            # at fine scales where semantic details are decided.
            relax = 0.85 + 0.35 * ratio
            gate = (edit * relax >= preserve) & (cov >= min_core)
            return gate | ((ratio >= 0.78) & (cov >= shared_core))
        if mode == "budget_shared_boundary":
            # Shared/boundary tokens get a small edit budget, but coarse boundary
            # and low-coverage tokens remain source-owned.
            gate = (edit >= preserve * 0.98) & (cov >= min_core)
            boundary_gate = (ratio >= 0.76) & (boundary >= 0.30) & (cov >= 0.22)
            return gate | boundary_gate
        if mode == "budget_scale_guard":
            # PSNR-oriented authority: keep coarse/shared tokens source-owned,
            # then release fine high-coverage edit cores for prompt semantics.
            preserve = preserve * (1.18 - 0.10 * ratio)
            edit = edit * (1.00 + 0.24 * ratio)
            gate = (edit >= preserve * (1.08 - 0.14 * ratio)) & (cov >= (min_core + 0.04))
            semantic_tail = (ratio >= 0.88) & (cov >= (shared_core + 0.04))
            fine_core = (ratio >= 0.94) & (cov >= 0.22)
            return gate | semantic_tail | fine_core
        if mode == "budget_scale_guard_sem":
            # Slightly softer guard than scale_guard; intended to recover CLIP
            # while retaining more source authority than bg_fix4/fix2.
            preserve = preserve * (1.12 - 0.08 * ratio)
            edit = edit * (1.04 + 0.30 * ratio)
            gate = (edit >= preserve * (1.03 - 0.10 * ratio)) & (cov >= (min_core + 0.02))
            semantic_tail = (ratio >= 0.80) & (cov >= (shared_core + 0.02))
            fine_core = (ratio >= 0.90) & (cov >= 0.22)
            return gate | semantic_tail | fine_core
        if mode == "budget_psnr_guard":
            # Stronger source guard used as a diagnostic upper-fidelity clean
            # variant. It should only promote if CLIP remains acceptable.
            preserve = preserve * (1.25 - 0.10 * ratio)
            edit = edit * (0.96 + 0.20 * ratio)
            gate = (edit >= preserve * (1.12 - 0.12 * ratio)) & (cov >= (min_core + 0.06))
            fine_core = (ratio >= 0.95) & (cov >= 0.26)
            return gate | fine_core
        raise ValueError(mode)
    if mode == "source":
        # Source-owned by default; only high edit coverage receives prompt authority.
        prompt_tau = 0.92 - 0.38 * ratio
        return cov >= prompt_tau
    if mode == "source_lite":
        # Slightly loosen source ownership to recover prompt semantics while
        # keeping shared/background tokens source-owned until late scales.
        prompt_tau = 0.88 - 0.42 * ratio
        shared_tau = 0.50 - 0.20 * ratio
        edit = cov >= prompt_tau
        return edit | (cov >= shared_tau) if ratio >= 0.85 else edit
    if mode == "source_sem":
        # More semantic authority than source_lite, still stricter than balanced.
        prompt_tau = 0.86 - 0.44 * ratio
        shared_tau = 0.46 - 0.20 * ratio
        edit = cov >= prompt_tau
        return edit | (cov >= shared_tau) if ratio >= 0.75 else edit
    if mode == "balanced":
        # Shared boundary tokens open gradually at later scales.
        prompt_tau = 0.82 - 0.46 * ratio
        shared_tau = 0.54 - 0.24 * ratio
        edit = cov >= prompt_tau
        return edit | (cov >= shared_tau) if ratio >= 0.55 else edit
    if mode == "prompt":
        # More prompt-owned tokens, used to test edit-strength recovery.
        prompt_tau = 0.70 - 0.38 * ratio
        shared_tau = 0.40 - 0.18 * ratio
        edit = cov >= prompt_tau
        return edit | (cov >= shared_tau) if ratio >= 0.45 else edit
    if mode == "boundary":
        # Boundary/shared tokens stay source-owned until fine scales, then open.
        prompt_tau = 0.88 - 0.44 * ratio
        shared_tau = 0.50 - 0.20 * ratio
        edit = cov >= prompt_tau
        return edit | (cov >= shared_tau) if ratio >= 0.70 else edit
    raise ValueError(mode)


def eroded_edit_mask(mask_t, radius):
    if radius <= 0:
        return mask_t
    k = int(radius) * 2 + 1
    inv = 1.0 - mask_t.float()
    eroded = (1.0 - F.max_pool2d(inv, kernel_size=k, stride=1, padding=radius)).clamp(0, 1)
    return mask_t if float(eroded.sum().item()) <= 0 else eroded.to(dtype=mask_t.dtype)

def edit_mask_for_scale(mask_t, pn, si, n_scales, cfg, style_all):
    if style_all and cfg.force_all_style:
        return torch.ones((1, pn * pn), dtype=torch.bool, device=mask_t.device)
    if cfg.projection == "free":
        return torch.ones((1, pn * pn), dtype=torch.bool, device=mask_t.device)
    proj_mask_t = eroded_edit_mask(mask_t, cfg.mask_erode_px) if not style_all else mask_t
    if cfg.projection == "nearest":
        edit = coverage_map(proj_mask_t, pn, "nearest") > 0.5
    elif cfg.projection == "max":
        edit = coverage_map(proj_mask_t, pn, "max") > 0.0
    elif cfg.projection == "area_tau":
        cov = coverage_map(proj_mask_t, pn, "area")
        edit = cov >= tau_for_scale(si, n_scales, cfg.tau_family)
    elif cfg.projection == "token_authority":
        cov = coverage_map(proj_mask_t, pn, "area")
        edit = token_authority_edit_mask(cov, si, n_scales, cfg.tau_family)
    else:
        raise ValueError(cfg.projection)
    if cfg.floor_mult > 0 and not style_all:
        score = coverage_map(proj_mask_t, pn, "area").reshape(-1)
        mask_frac = float(proj_mask_t.mean().item())
        k = int(math.ceil(mask_frac * pn * pn * cfg.floor_mult))
        if mask_frac > 0:
            k = max(1, min(k, score.numel()))
        if k > int(edit.sum().item()):
            idx = torch.topk(score, k=k, largest=True, sorted=False).indices
            flat = edit.reshape(-1)
            flat[idx] = True
            edit = flat.reshape(pn, pn)
    return edit.reshape(1, pn * pn)

def composite_postlock(src01, out01, mask_t, feather=0):
    if feather <= 0:
        return out01 * mask_t + src01 * (1.0 - mask_t)
    # Conservative feather: blur only the binary mask edge, keep exact far background.
    k = int(feather) * 2 + 1
    soft = F.avg_pool2d(mask_t.float(), kernel_size=k, stride=1, padding=feather).clamp(0, 1)
    return out01 * soft + src01 * (1.0 - soft)



