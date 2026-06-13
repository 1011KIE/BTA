from pathlib import Path
import os
import sys

FrozenCLIPEmbedder = None
SwittiPipeline = None
TRAIN_IMAGE_SIZE = None
SwittiHF = None
VQVAEHF = None
sample_with_top_k_top_p_ = None
get_crop_condition = None

def import_switti(switti_code_root):
    root = Path(switti_code_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"SWITTI code root does not exist: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    global FrozenCLIPEmbedder, SwittiPipeline, TRAIN_IMAGE_SIZE, SwittiHF, VQVAEHF
    global sample_with_top_k_top_p_, get_crop_condition
    old_cwd = os.getcwd()
    try:
        os.chdir(root)
        from models.clip import FrozenCLIPEmbedder as _FrozenCLIPEmbedder
        from models.helpers import sample_with_top_k_top_p_ as _sample_with_top_k_top_p_
        from models.pipeline import TRAIN_IMAGE_SIZE as _TRAIN_IMAGE_SIZE
        from models.pipeline import SwittiPipeline as _SwittiPipeline
        from models.switti import SwittiHF as _SwittiHF
        from models.switti import get_crop_condition as _get_crop_condition
        from models.vqvae import VQVAEHF as _VQVAEHF
    finally:
        os.chdir(old_cwd)

    FrozenCLIPEmbedder = _FrozenCLIPEmbedder
    sample_with_top_k_top_p_ = _sample_with_top_k_top_p_
    TRAIN_IMAGE_SIZE = _TRAIN_IMAGE_SIZE
    SwittiPipeline = _SwittiPipeline
    SwittiHF = _SwittiHF
    get_crop_condition = _get_crop_condition
    VQVAEHF = _VQVAEHF


def load_pipeline(device, dtype, paths):
    print("[load] SWITTI", flush=True)
    switti = SwittiHF.from_pretrained(paths["switti"]).to(dtype=dtype).to(device).eval()
    print("[load] VQVAE", flush=True)
    vae = VQVAEHF.from_pretrained(paths["vae"], reso=512).to(dtype=dtype).to(device).eval()
    print("[load] CLIP text encoders", flush=True)
    text_encoder = FrozenCLIPEmbedder(paths["clip_l"], device=device, dtype=dtype)
    text_encoder_2 = FrozenCLIPEmbedder(paths["clip_g"], device=device, dtype=dtype)
    pipe = SwittiPipeline(switti, vae, text_encoder, text_encoder_2, device=device, dtype=dtype)
    for module in [pipe.switti, pipe.vae, pipe.text_encoder, pipe.text_encoder_2]:
        module.eval()
        for p in module.parameters():
            p.requires_grad_(False)
    return pipe
