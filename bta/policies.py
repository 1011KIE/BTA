import re
from dataclasses import dataclass, replace


@dataclass
class MethodCfg:
    name: str
    projection: str
    fix_si: int
    tau_family: str
    post_lock: bool = False
    inner_feather: int = 0
    force_all_style: bool = True
    floor_mult: float = 0.0
    route: str = ""
    mask_erode_px: int = 0


def parse_method(name):
    if name == "source_recon":
        return MethodCfg(name, "source_recon", fix_si=99, tau_family="flat")
    if name == "txt2img":
        return MethodCfg(name, "free", fix_si=-1, tau_family="flat")
    if name == "txt2img_postlock":
        return MethodCfg(name, "free", fix_si=-1, tau_family="flat", post_lock=True)
    if name == "txt2img_postlock_f5":
        return MethodCfg(name, "free", fix_si=-1, tau_family="flat", post_lock=True, inner_feather=5)
    match = re.fullmatch(r"txt2img_postlock_f(\d+)", name)
    if match:
        return MethodCfg(name, "free", fix_si=-1, tau_family="flat", post_lock=True, inner_feather=int(match.group(1)))
    match = re.fullmatch(r"(.+)_erode(\d+)", name)
    if match:
        base = parse_method(match.group(1))
        return replace(base, name=name, mask_erode_px=int(match.group(2)))
    match = re.fullmatch(r"mask_(nearest|maxpool)_fix(\d+)", name)
    if match:
        projection = "nearest" if match.group(1) == "nearest" else "max"
        return MethodCfg(name, projection, int(match.group(2)), "flat")
    match = re.fullmatch(r"scale_fcmp_(soft|hard)_fix(\d+)", name)
    if match:
        return MethodCfg(name, "area_tau", int(match.group(2)), match.group(1))
    match = re.fullmatch(r"scale_fcmp_strict_fix(\d+)", name)
    if match:
        return MethodCfg(name, "area_tau", int(match.group(1)), "coarse_strict")
    match = re.fullmatch(r"token_authority_(source|source_lite|source_sem|balanced|prompt|boundary)_fix(\d+)", name)
    if match:
        return MethodCfg(name, "token_authority", int(match.group(2)), match.group(1))
    match = re.fullmatch(
        r"budget_auth_(clip_guard|core_release|bg_fix4|bg_sem|bg_sem_core|scale_relax|shared_boundary|scale_guard|scale_guard_sem|psnr_guard)_fix(\d+)",
        name,
    )
    if match:
        return MethodCfg(name, "token_authority", int(match.group(2)), f"budget_{match.group(1)}")
    table = {
        "mask_nearest_nofix": MethodCfg(name, "nearest", -1, "flat"),
        "mask_maxpool_nofix": MethodCfg(name, "max", -1, "flat"),
        "mask_nearest_fix4": MethodCfg(name, "nearest", 4, "flat"),
        "mask_nearest_fix6": MethodCfg(name, "nearest", 6, "flat"),
        "mask_maxpool_fix4": MethodCfg(name, "max", 4, "flat"),
        "mask_maxpool_fix6": MethodCfg(name, "max", 6, "flat"),
        "scale_fcmp_soft_nofix": MethodCfg(name, "area_tau", -1, "soft"),
        "scale_fcmp_hard_nofix": MethodCfg(name, "area_tau", -1, "hard"),
        "scale_fcmp_soft_nofix_postlock": MethodCfg(name, "area_tau", -1, "soft", post_lock=True),
        "scale_fcmp_hard_nofix_postlock": MethodCfg(name, "area_tau", -1, "hard", post_lock=True),
        "scale_fcmp_hard_nofix_postlock_f5": MethodCfg(name, "area_tau", -1, "hard", post_lock=True, inner_feather=5),
        "scale_fcmp_floor_nofix": MethodCfg(name, "area_tau", -1, "soft", floor_mult=1.25),
        "scale_fcmp_floor2_nofix": MethodCfg(name, "area_tau", -1, "soft", floor_mult=2.0),
        "scale_fcmp_floor_fix4": MethodCfg(name, "area_tau", 4, "soft", floor_mult=1.25),
        "scale_fcmp_floor_nofix_postlock": MethodCfg(name, "area_tau", -1, "soft", post_lock=True, floor_mult=1.25),
        "scale_fcmp_floor2_nofix_postlock": MethodCfg(name, "area_tau", -1, "soft", post_lock=True, floor_mult=2.0),
        "scale_fcmp_floor2_nofix_postlock_f5": MethodCfg(
            name, "area_tau", -1, "soft", post_lock=True, inner_feather=5, floor_mult=2.0
        ),
        "scale_fcmp_soft_fix4": MethodCfg(name, "area_tau", 4, "soft"),
        "scale_fcmp_soft_fix6": MethodCfg(name, "area_tau", 6, "soft"),
        "scale_fcmp_hard_fix4": MethodCfg(name, "area_tau", 4, "hard"),
        "scale_fcmp_hard_fix6": MethodCfg(name, "area_tau", 6, "hard"),
        "scale_fcmp_strict_fix4": MethodCfg(name, "area_tau", 4, "coarse_strict"),
        "scale_fcmp_soft_fix4_postlock": MethodCfg(name, "area_tau", 4, "soft", post_lock=True),
        "mask_maxpool_fix4_postlock": MethodCfg(name, "max", 4, "flat", post_lock=True),
    }
    if name not in table:
        raise ValueError(f"Unknown method {name}")
    return table[name]


def resolve_routed_method(cfg, edit_type, mask_frac):
    if not cfg.route:
        return cfg

    def routed(projection, fix_si, tau_family, floor_mult=0.0):
        return MethodCfg(
            cfg.name,
            projection,
            fix_si,
            tau_family,
            post_lock=cfg.post_lock,
            inner_feather=cfg.inner_feather,
            force_all_style=cfg.force_all_style,
            floor_mult=floor_mult,
            mask_erode_px=cfg.mask_erode_px,
        )

    et = str(edit_type)
    if et == "9":
        return routed("max", -1, "flat")
    if cfg.route == "budget_mask_adapt_v1":
        if mask_frac < 0.08:
            return routed("token_authority", 4, "budget_scale_guard")
        if mask_frac < 0.20:
            return routed("token_authority", 3, "budget_scale_guard_sem")
        if mask_frac > 0.45 or et in {"2", "8"}:
            return routed("token_authority", 2, "budget_bg_sem")
        return routed("token_authority", 2, "budget_bg_fix4")
    if cfg.route == "budget_mask_adapt_v2":
        if mask_frac < 0.10:
            return routed("token_authority", 5, "budget_scale_guard")
        if mask_frac < 0.25:
            return routed("token_authority", 3, "budget_scale_guard")
        if mask_frac > 0.50 or et in {"2", "8"}:
            return routed("token_authority", 2, "budget_bg_sem_core")
        return routed("token_authority", 2, "budget_bg_fix4")
    if cfg.route == "budget_mask_adapt_v3":
        if mask_frac < 0.08:
            return routed("token_authority", 4, "budget_psnr_guard")
        if mask_frac < 0.20:
            return routed("token_authority", 3, "budget_scale_guard")
        if mask_frac > 0.45 or et in {"2", "8"}:
            return routed("token_authority", 2, "budget_bg_sem")
        return routed("token_authority", 2, "budget_scale_guard_sem")
    raise ValueError(cfg.route)
