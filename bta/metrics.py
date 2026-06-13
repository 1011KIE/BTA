import urllib.request
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms

from .config import DINO_VITB8_URL, METRIC_PROTOCOL_VERSION

class OptionalMetrics:
    def __init__(self, device, clip_path, dino_ckpt):
        self.device = device
        self.lpips_model = None
        self.clip_model = None
        self.clip_processor = None
        self.dino_model = None
        self.dino_hook = None
        self.dino_qkv = None
        self.clip_path = clip_path
        self.dino_ckpt = Path(dino_ckpt).expanduser()

    def load_lpips(self):
        if self.lpips_model is not None:
            return True
        try:
            import lpips

            self.lpips_model = lpips.LPIPS(net="squeeze").to(self.device).eval()
            for p in self.lpips_model.parameters():
                p.requires_grad_(False)
            return True
        except Exception as exc:
            print(f"[warn] LPIPS unavailable: {exc}", flush=True)
            return False

    def load_clip(self):
        if self.clip_model is not None:
            return True
        try:
            from transformers import CLIPModel, CLIPProcessor

            self.clip_processor = CLIPProcessor.from_pretrained(self.clip_path, local_files_only=True)
            self.clip_model = CLIPModel.from_pretrained(self.clip_path, local_files_only=True).to(self.device).eval()
            return True
        except Exception as exc:
            print(f"[warn] CLIP metric unavailable: {exc}", flush=True)
            return False

    @torch.inference_mode()
    def lpips_bg(self, src01, out01, mask_t):
        if not self.load_lpips():
            return None
        bg = (1.0 - mask_t).expand_as(src01)
        src_bg = src01 * bg
        out_bg = out01 * bg
        return float(self.lpips_model(src_bg.mul(2).sub(1), out_bg.mul(2).sub(1)).item())

    @torch.inference_mode()
    def clip_score(self, img01, prompt):
        if not prompt or not self.load_clip():
            return None
        pil = transforms.ToPILImage()(img01[0].detach().cpu().clamp(0, 1))
        inputs = self.clip_processor(text=[prompt], images=[pil], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        image_features = self.clip_model.get_image_features(pixel_values=inputs["pixel_values"])
        text_features = self.clip_model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return float((100.0 * image_features @ text_features.T).item())

    def load_dino(self):
        if self.dino_model is not None:
            return True
        try:
            import timm

            self.dino_ckpt.parent.mkdir(parents=True, exist_ok=True)
            if not self.dino_ckpt.exists():
                urllib.request.urlretrieve(DINO_VITB8_URL, self.dino_ckpt)
            model = timm.create_model("vit_base_patch8_224", pretrained=False)
            state = torch.load(str(self.dino_ckpt), map_location="cpu")
            model.load_state_dict(state, strict=False)
            model.to(self.device).eval()
            for p in model.parameters():
                p.requires_grad_(False)
            self.dino_model = model
            return True
        except Exception as exc:
            print(f"[warn] DINO structure distance unavailable: {exc}", flush=True)
            return False

    def _dino_transform(self, img01):
        x = img01[0].detach().float().clamp(0, 1).mul(255.0)
        x = transforms.Resize(224, max_size=480)(x)
        x = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(x)
        return x.unsqueeze(0).to(self.device)

    @torch.inference_mode()
    def dino_self_sim(self, img01):
        if not self.load_dino():
            return None
        x = self._dino_transform(img01)
        self.dino_qkv = None

        def hook(_module, _inp, output):
            self.dino_qkv = output.detach()

        handle = self.dino_model.blocks[11].attn.qkv.register_forward_hook(hook)
        try:
            self.dino_model(x)
        finally:
            handle.remove()
        if self.dino_qkv is None:
            return None
        qkv = self.dino_qkv[0]
        patch_num = qkv.shape[0]
        head_num = self.dino_model.blocks[11].attn.num_heads
        embedding_dim = qkv.shape[-1] // 3
        keys = qkv.reshape(patch_num, 3, head_num, embedding_dim // head_num).permute(1, 2, 0, 3)[1]
        h, tokens, dim = keys.shape
        flat = keys.transpose(0, 1).reshape(tokens, h * dim)
        flat = flat[None, :, :]
        norm = flat.norm(dim=2, keepdim=True)
        denom = torch.clamp(norm @ norm.permute(0, 2, 1), min=1e-8)
        return (flat @ flat.permute(0, 2, 1)) / denom

    @torch.inference_mode()
    def structure_distance(self, src01, out01, src_self_sim=None):
        src_sim = src_self_sim if src_self_sim is not None else self.dino_self_sim(src01)
        out_sim = self.dino_self_sim(out01)
        if src_sim is None or out_sim is None:
            return None
        return float(F.mse_loss(out_sim, src_sim).item())

def has_valid_background(mask_t):
    return float((1.0 - mask_t).sum().item()) >= 1.0

def masked_mse(src01, out01, mask_t):
    if not has_valid_background(mask_t):
        return None
    bg = (1.0 - mask_t).expand_as(src01)
    return ((src01 - out01).square() * bg).mean()

def torch_ssim_global(x, y, kernel_size=11, sigma=1.5):
    c1 = 0.01**2
    c2 = 0.03**2
    radius = kernel_size // 2
    coords = torch.arange(kernel_size, device=x.device, dtype=x.dtype) - radius
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel = torch.outer(g, g).view(1, 1, kernel_size, kernel_size)
    kernel = kernel.repeat(x.shape[1], 1, 1, 1)

    def filt(z):
        z = F.pad(z, (radius, radius, radius, radius), mode="reflect")
        return F.conv2d(z, kernel, groups=z.shape[1])

    mu_x = filt(x)
    mu_y = filt(y)
    sigma_x = filt(x * x) - mu_x * mu_x
    sigma_y = filt(y * y) - mu_y * mu_y
    sigma_xy = filt(x * y) - mu_x * mu_y
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2) + 1e-12
    )
    return ssim.mean()

def edit_clip_image(img01, mask_t):
    edit = mask_t.expand_as(img01)
    h, w = img01.shape[-2:]
    return img01 * edit, {
        "edit_clip_image_h": h,
        "edit_clip_image_w": w,
        "edit_clip_mask_frac": float(mask_t.mean().item()),
        "edit_clip_view": "official_masked_full_image",
    }

def boundary_ring(mask_t, width, outer_only=True):
    k = 2 * width + 1
    dil = F.max_pool2d(mask_t.float(), k, stride=1, padding=width)
    if outer_only:
        return (dil - mask_t.float()).clamp(0, 1)
    inv = 1.0 - mask_t.float()
    ero_inv = F.max_pool2d(inv, k, stride=1, padding=width)
    inner = (ero_inv - inv).clamp(0, 1)
    return torch.maximum(dil - mask_t.float(), inner)

def masked_l1(src01, out01, region):
    reg = region.expand_as(src01)
    denom = reg.sum()
    if float(denom.item()) < 1.0:
        return None
    return float((((src01 - out01).abs() * reg).sum() / denom).item())

def core_metrics(src01, out01, mask_t, opt, prompt, src_self_sim=None):
    bg_valid = has_valid_background(mask_t)
    bg_mse = masked_mse(src01, out01, mask_t)
    bg_psnr = None if bg_mse is None else -10.0 * torch.log10(bg_mse.clamp_min(1e-12))
    bg_ssim = None
    if bg_valid:
        bg = (1.0 - mask_t).expand_as(src01)
        bg_ssim = torch_ssim_global(src01 * bg, out01 * bg)
    edit_reg = mask_t.expand_as(src01)
    edit_l1 = ((src01 - out01).abs() * edit_reg).sum() / edit_reg.sum().clamp_min(1.0)
    ring5 = boundary_ring(mask_t, 5, outer_only=True)
    ring10 = boundary_ring(mask_t, 10, outer_only=True)
    edit_img, edit_clip_info = edit_clip_image(out01, mask_t)
    lp = opt.lpips_bg(src01, out01, mask_t) if bg_valid else None
    distance = opt.structure_distance(src01, out01, src_self_sim)
    return {
        "metric_protocol_version": METRIC_PROTOCOL_VERSION,
        "bg_valid": bg_valid,
        "bg_psnr": None if bg_psnr is None else float(bg_psnr.item()),
        "bg_mse": None if bg_mse is None else float(bg_mse.item()),
        "bg_mse_x1e4": None if bg_mse is None else float(bg_mse.item() * 1e4),
        "bg_ssim": None if bg_ssim is None else float(bg_ssim.item()),
        "bg_ssim_x1e2": None if bg_ssim is None else float(bg_ssim.item() * 1e2),
        "bg_lpips": lp,
        "bg_lpips_x1e3": None if lp is None else lp * 1e3,
        "outer5_l1": masked_l1(src01, out01, ring5),
        "outer10_l1": masked_l1(src01, out01, ring10),
        "edit_l1": float(edit_l1.item()),
        "structure_distance": distance,
        "clip_whole": opt.clip_score(out01, prompt),
        "clip_edited": opt.clip_score(edit_img, prompt),
        **edit_clip_info,
    }
