"""AEROBLADE-style reconstruction distance (Ricker et al., CVPR 2024).

An image that a latent-diffusion model decoded sits almost exactly on its VAE's output, so encoding and
decoding it again changes it less than it changes a camera photo. The score is the smallest LPIPS distance
over the VAEs tried: lower = more likely generated through one of them. It covers open-source generators
built on these VAEs, not face swaps or closed tools such as Grok Imagine.

Shared by training (ml/data.py makes its VAE-decoded fakes with vae_roundtrip) and serving (analyze.py).
"""
from functools import lru_cache

import numpy as np
import torch

VAES = {  # name -> (Hugging Face repo, subfolder); FLUX's needs its license accepted on Hugging Face
    "sd15": ("stable-diffusion-v1-5/stable-diffusion-v1-5", "vae"),
    "sdxl": ("stabilityai/sdxl-vae", None),
    "flux": ("black-forest-labs/FLUX.1-schnell", "vae"),
}
# Scoring uses SD1.5's VAE only. On 40 DeepFakeFace images per class at 256px (AUC, lower distance =
# generated): SD1.5 0.92-0.97, SDXL 0.66-0.71, and FLUX's VAE rebuilds camera photos so well that adding it
# to the minimum drags it to 0.71-0.81. One VAE at 256px takes ~4 s on 2 CPU threads (128px: ~1 s, AUC 0.86-0.88).
RECON_VAES = ("sd15",)
CROP = 256  # center crop at native resolution: resizing would move a generated image off the VAE's output


@torch.no_grad()
def vae_roundtrip(img: np.ndarray, vae) -> np.ndarray:
    """Encode and decode a BGR uint8 image through a VAE, on whatever device the VAE is on."""
    h, w = img.shape[0] // 8 * 8, img.shape[1] // 8 * 8  # VAEs need sides divisible by 8
    x = torch.from_numpy(img[:h, :w, ::-1].copy()).permute(2, 0, 1)[None].to(vae.device).float() / 127.5 - 1
    y = vae.decode(vae.encode(x).latent_dist.mode()).sample[0].clamp(-1, 1)
    return ((y + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()[:, :, ::-1].copy()


@lru_cache(maxsize=1)
def models():
    import lpips
    from diffusers import AutoencoderKL

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vaes = {n: AutoencoderKL.from_pretrained(VAES[n][0], subfolder=VAES[n][1]).to(device).eval() for n in RECON_VAES}
    return vaes, lpips.LPIPS(net="vgg", verbose=False).to(device).eval()


@torch.no_grad()
def score(img: np.ndarray) -> float:
    """Smallest LPIPS distance between the image's center crop and its reconstruction by each VAE."""
    h, w = img.shape[:2]
    s = min(CROP, h, w) // 8 * 8
    y, x = (h - s) // 2, (w - s) // 2
    crop = img[y:y + s, x:x + s]
    vaes, lpips_vgg = models()
    device = next(lpips_vgg.parameters()).device

    def tensor(bgr):  # LPIPS takes RGB in [-1, 1]
        return torch.from_numpy(bgr[:, :, ::-1].copy()).permute(2, 0, 1)[None].float().to(device) / 127.5 - 1

    return min(float(lpips_vgg(tensor(crop), tensor(vae_roundtrip(crop, vae)))) for vae in vaes.values())
