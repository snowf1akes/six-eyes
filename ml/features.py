"""Task A3, step 1: frozen DINOv3 and CLIP features for every manifest row, extracted once.

Saves data/features/<model>.npy with one row per manifest row, in manifest order. Re-run it
whenever the manifest changes (for example after merging C's test set).
Run from the repo root after ml/data.py:
    python ml/features.py 256    # smoke test: first 256 manifest rows
    python ml/features.py        # every row
"""
import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel, CLIPVisionModelWithProjection

MANIFEST = Path("data/manifest.csv")
OUT = Path("data/features")
BATCH = 128
MODELS = {  # name -> (Hugging Face repo, model class, output field)
    "dinov3": ("facebook/dinov3-vitl16-pretrain-lvd1689m", AutoModel, "pooler_output"),  # 1024-d CLS token
    "clip": ("openai/clip-vit-large-patch14", CLIPVisionModelWithProjection, "image_embeds"),  # 768-d, as in UnivFD
}


def load(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


@torch.no_grad()
def extract(paths, repo, cls, field, pool):
    model = cls.from_pretrained(repo, dtype=torch.bfloat16).cuda().eval()
    proc = AutoImageProcessor.from_pretrained(repo)
    mean = torch.tensor(proc.image_mean, device="cuda").view(1, 3, 1, 1)
    std = torch.tensor(proc.image_std, device="cuda").view(1, 3, 1, 1)
    feats, start = [], time.time()
    for i in range(0, len(paths), BATCH):
        x = torch.from_numpy(np.stack(list(pool.map(load, paths[i:i + BATCH])))).cuda()
        x = (x.permute(0, 3, 1, 2).float() / 255 - mean) / std  # data.py already made every image 224x224
        feats.append(getattr(model(pixel_values=x.to(torch.bfloat16)), field).float().cpu().numpy())
        done = min(i + BATCH, len(paths))
        if done % (BATCH * 40) == 0 or done == len(paths):
            print(f"  {done}/{len(paths)} images, {done / (time.time() - start):.0f}/s", flush=True)
    del model
    torch.cuda.empty_cache()
    return np.concatenate(feats)


def main(limit):
    with MANIFEST.open() as f:
        paths = [r["path"] for r in csv.DictReader(f)][:limit]
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(8) as pool:  # PNG decoding in threads keeps the GPU busy
        for name, (repo, cls, field) in MODELS.items():
            print(f"{name}: {repo}")
            feats = extract(paths, repo, cls, field, pool)
            assert len(feats) == len(paths) and np.isfinite(feats).all(), name
            np.save(OUT / f"{name}.npy", feats)
            print(f"  saved {OUT / name}.npy {feats.shape}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
