"""Task A2: build the training set from DeepFakeFace and write data/manifest.csv.

1. Real/fake pairs from DeepFakeFace, fakes balanced across its three generators.
2. Aligned fakes: some of those real photos encoded and decoded by the SD1.5, SDXL or FLUX VAE,
   so the only difference from their real twin is the decoder's fingerprint.
3. Every image, real or fake, gets the same random resize + JPEG step, then becomes a 224px face
   crop (InsightFace detect + align) and a 224px full view.

Run from the repo root after downloading and unzipping DeepFakeFace into data/deepfakeface:
    python ml/data.py 50      # smoke test: 50 pairs + 15 VAE fakes
    python ml/data.py         # full run: 10k real + 10k DeepFakeFace fakes + 3k VAE fakes
"""
import collections
import csv
import random
import sys
import zlib
from pathlib import Path

import cv2
import torch
from diffusers import AutoencoderKL
from insightface.app import FaceAnalysis
from insightface.utils.face_align import norm_crop

SRC = Path("data/deepfakeface")
OUT = Path("data/processed")
MANIFEST = Path("data/manifest.csv")
SIZE = 224
VAL_PCT = 15
VAE_SHARE = 0.3  # VAE fakes per pair: 3k for the full 10k run
MAX_VAE_SIDE = 1024  # bigger photos are skipped for the VAE step so a 12 GB GPU never runs out of memory
# DeepFakeFace folder -> generator class. The names match analyze()'s generator_guess.
FAKES = {"text2img": "txt2img", "inpainting": "inpaint", "insight": "swap"}
VAES = {  # name -> (Hugging Face repo, subfolder)
    "sd15": ("stable-diffusion-v1-5/stable-diffusion-v1-5", "vae"),
    "sdxl": ("stabilityai/sdxl-vae", None),
    "flux": ("black-forest-labs/FLUX.1-schnell", "vae"),
}


def degrade(img, rng):
    # The same random downscale + JPEG for every image, so neither size nor compression gives the label away.
    k = rng.uniform(0.5, 1.0)
    img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(60, 95)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def face_crop(img, model):
    # The detector misses faces that fill the frame; a border fixes that (same trick as api/main.py).
    h, w = img.shape[:2]
    img = cv2.copyMakeBorder(img, h // 4, h // 4, w // 4, w // 4, cv2.BORDER_CONSTANT)
    faces = model.get(img)
    if not faces:
        return None
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return norm_crop(img, face.kps, image_size=SIZE)


def full_view(img):
    # Center square crop, not letterbox: every fake is 512x512 but reals come in hundreds of shapes,
    # so letterbox bars would appear only on reals and the model would learn "bars = real".
    h, w = img.shape[:2]
    s = min(h, w)
    y, x = (h - s) // 2, (w - s) // 2
    return cv2.resize(img[y:y + s, x:x + s], (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def views(img, model, rng):
    img = degrade(img, rng)
    face = face_crop(img, model)
    return None if face is None else {"face": face, "full": full_view(img)}


@torch.no_grad()
def vae_roundtrip(img, vae):
    h, w = img.shape[0] // 8 * 8, img.shape[1] // 8 * 8  # VAEs need sides divisible by 8
    x = torch.from_numpy(img[:h, :w, ::-1].copy()).permute(2, 0, 1)[None].cuda().float() / 127.5 - 1
    y = vae.decode(vae.encode(x).latent_dist.mode()).sample[0].clamp(-1, 1)
    return ((y + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()[:, :, ::-1].copy()


def save(rows, out, photo, label, gen, split, name):
    for view, img in out.items():
        path = OUT / view / gen / f"{name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), img)
        rows.append({"id": photo, "path": path.as_posix(), "label": label,
                     "generator": gen, "view": view, "split": split})


def detector():
    model = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"], providers=["CPUExecutionProvider"])
    model.prepare(ctx_id=-1, det_size=(320, 320))  # finds ~98% of these faces, ~10x faster than 640
    return model


def main(n_pairs):
    rng = random.Random(0)
    model = detector()

    ids = sorted(p.relative_to(SRC / "wiki").as_posix() for p in (SRC / "wiki").rglob("*.jpg"))
    rng.shuffle(ids)
    rows, kept, skipped = [], [], 0

    # Stage 1: DeepFakeFace pairs.
    for i in ids:
        if len(kept) == n_pairs:
            break
        folder = list(FAKES)[len(kept) % 3]  # rotate so fakes stay balanced across the three generators
        real, fake = cv2.imread(str(SRC / "wiki" / i)), cv2.imread(str(SRC / folder / i))
        if real is None or fake is None:
            skipped += 1
            continue
        if folder == "inpainting":
            # Inpainting fakes are the real photo stretched to 512x512. Undo the stretch (downscale only)
            # so a squashed face can't give them away.
            h, w = real.shape[:2]
            k = 512 / max(h, w)
            fake = cv2.resize(fake, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)
        real_out, fake_out = views(real, model, rng), views(fake, model, rng)
        if real_out is None or fake_out is None:
            skipped += 1
            continue
        # One split per source photo, so a fake made from a real photo never sits across the split from it.
        split = "val" if zlib.crc32(i.encode()) % 100 < VAL_PCT else "train"
        photo = i.removesuffix(".jpg").replace("/", "_")
        save(rows, real_out, photo, "real", "real", split, photo)
        save(rows, fake_out, photo, "fake", FAKES[folder], split, photo)
        kept.append((i, photo, split))
        if len(kept) % 500 == 0:
            print(f"pairs: {len(kept)}/{n_pairs}, {skipped} skipped (no face found)", flush=True)

    # Stage 2: aligned VAE fakes from real photos already in the set; each keeps its real twin's split.
    vaes = {name: AutoencoderKL.from_pretrained(repo, subfolder=sub).cuda().eval()
            for name, (repo, sub) in VAES.items()}
    n_vae, per_vae = round(n_pairs * VAE_SHARE), collections.Counter()
    for i, photo, split in kept:
        if sum(per_vae.values()) == n_vae:
            break
        name = list(VAES)[sum(per_vae.values()) % len(VAES)]
        real = cv2.imread(str(SRC / "wiki" / i))
        if max(real.shape[:2]) > MAX_VAE_SIDE:
            continue
        out = views(vae_roundtrip(real, vaes[name]), model, rng)
        if out is None:
            continue
        save(rows, out, photo, "fake", "vae-decoded", split, f"{photo}_{name}")
        per_vae[name] += 1
        if sum(per_vae.values()) % 500 == 0:
            print(f"VAE fakes: {sum(per_vae.values())}/{n_vae}", flush=True)

    with MANIFEST.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    faces = [r for r in rows if r["view"] == "face"]
    gens = collections.Counter(r["generator"] for r in faces if r["generator"] not in ("real", "vae-decoded"))
    assert sum(r["label"] == "real" for r in faces) == sum(gens.values()) == len(kept), "pairs not 1 real : 1 fake"
    assert max(gens.values()) - min(gens.values()) <= 1, f"generators unbalanced: {gens}"
    assert sum(per_vae.values()) == n_vae and max(per_vae.values()) - min(per_vae.values()) <= 1, per_vae
    print(f"Done: {len(rows)} manifest rows -> {MANIFEST} ({skipped} pairs skipped, no face found)")
    print("images per class:", dict(collections.Counter(r["generator"] for r in faces)))
    print("VAE fakes per VAE:", dict(per_vae))
    print("images per split:", dict(collections.Counter(r["split"] for r in faces)))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10_000)
