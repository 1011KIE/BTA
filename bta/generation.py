import torch

from . import backend
from .methods import edit_mask_for_scale

def generate_masked(
    pipe,
    prompt,
    source_tokens,
    mask_t,
    method_cfg,
    style_all,
    seed,
    cfg_scale,
    top_k,
    top_p,
    turn_on_cfg_start_si=0,
    turn_off_cfg_start_si=9,
    last_scale_temp=0.1,
):
    switti = pipe.switti
    vae = pipe.vae
    vae_quant = vae.quantize
    if seed is None:
        rng = None
    else:
        switti.rng.manual_seed(seed)
        rng = switti.rng

    context, cond_vector, context_attn_bias = pipe.encode_prompt([prompt], "")
    B = context.shape[0] // 2
    cond_vector = switti.text_pooler(cond_vector)
    if switti.use_crop_cond:
        crop_coords = backend.get_crop_condition(
            2 * B * [backend.TRAIN_IMAGE_SIZE[0]],
            2 * B * [backend.TRAIN_IMAGE_SIZE[1]],
        ).to(
            cond_vector.device
        )
        crop_embed = switti.crop_embed(crop_coords.view(-1)).reshape(2 * B, switti.D)
        crop_cond = switti.crop_proj(crop_embed)
    else:
        crop_cond = None

    sos = cond_BD = cond_vector
    lvl_pos = switti.lvl_embed(switti.lvl_1L)
    if not switti.rope:
        lvl_pos += switti.pos_1LC
    next_token_map = sos.unsqueeze(1) + switti.pos_start.expand(2 * B, switti.first_l, -1) + lvl_pos[:, : switti.first_l]
    cur_L = 0
    f_hat = sos.new_zeros(B, switti.Cvae, switti.patch_nums[-1], switti.patch_nums[-1])
    sampled_tokens = []

    for b in switti.blocks:
        b.attn.kv_caching(switti.use_ar)
        b.cross_attn.kv_caching(True)
    try:
        for si, pn in enumerate(switti.patch_nums):
            ratio = si / switti.num_stages_minus_1
            x_BLC = next_token_map
            if switti.rope:
                freqs_cis = switti.freqs_cis[:, cur_L : cur_L + pn * pn]
            else:
                freqs_cis = switti.freqs_cis

            if si >= turn_off_cfg_start_si:
                x_BLC = x_BLC[:B]
                context = context[:B]
                context_attn_bias = context_attn_bias[:B]
                freqs_cis = freqs_cis[:B]
                cond_BD = cond_BD[:B]
                if crop_cond is not None:
                    crop_cond = crop_cond[:B]
                for b in switti.blocks:
                    if b.attn.caching and b.attn.cached_k is not None:
                        b.attn.cached_k = b.attn.cached_k[:B]
                        b.attn.cached_v = b.attn.cached_v[:B]
                    if b.cross_attn.caching and b.cross_attn.cached_k is not None:
                        b.cross_attn.cached_k = b.cross_attn.cached_k[:B]
                        b.cross_attn.cached_v = b.cross_attn.cached_v[:B]

            for block in switti.blocks:
                x_BLC = block(
                    x=x_BLC,
                    cond_BD=cond_BD,
                    attn_bias=None,
                    context=context,
                    context_attn_bias=context_attn_bias,
                    freqs_cis=freqs_cis,
                    crop_cond=crop_cond,
                )
            cur_L += pn * pn
            logits_BlV = switti.get_logits(x_BLC, cond_BD)
            if si < turn_on_cfg_start_si:
                logits_BlV = logits_BlV[:B]
            elif si < turn_off_cfg_start_si:
                t = cfg_scale * ratio
                logits_BlV = (1 + t) * logits_BlV[:B] - t * logits_BlV[B:]
            elif last_scale_temp is not None:
                logits_BlV = logits_BlV / last_scale_temp

            idx_Bl = backend.sample_with_top_k_top_p_(
                logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1
            )[:, :, 0]
            if method_cfg.projection != "free":
                src_idx = source_tokens[si].to(idx_Bl.device)
                if si <= method_cfg.fix_si and not style_all:
                    idx_Bl = src_idx
                else:
                    edit_mask = edit_mask_for_scale(mask_t, pn, si, len(switti.patch_nums), method_cfg, style_all)
                    idx_Bl = torch.where(edit_mask, idx_Bl, src_idx)
            sampled_tokens.append(idx_Bl.detach().clone())

            h_BChw = vae_quant.embedding(idx_Bl).transpose_(1, 2).reshape(B, switti.Cvae, pn, pn)
            f_hat, next_token_map = vae_quant.get_next_autoregressive_input(si, len(switti.patch_nums), f_hat, h_BChw)
            if si != switti.num_stages_minus_1:
                next_token_map = next_token_map.view(B, switti.Cvae, -1).transpose(1, 2)
                next_token_map = switti.word_embed(next_token_map) + lvl_pos[:, cur_L : cur_L + switti.patch_nums[si + 1] ** 2]
                next_token_map = next_token_map.repeat(2, 1, 1)
    finally:
        for b in switti.blocks:
            b.attn.kv_caching(False)
            b.cross_attn.kv_caching(False)

    img01 = vae.fhat_to_img(f_hat).add(1).mul(0.5).clamp(0, 1)
    return img01, sampled_tokens
