"""Contract between Builder A (detection) and Builder B (product).

B calls analyze(); A replaces the placeholder body with the real pipeline (task A6).
Inside a Daytona sandbox it runs as:
    python analyze.py <media_url> '<embedding json>'   -> prints AnalyzeResult as JSON
"""
import io
import json
import logging
import os
import sys
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pdqhash
from pydantic import BaseModel

import crops
import face
import recon

PDQ_MIN_QUALITY = 50  # PDQ's own cutoff: below it (flat or blank images) hashes match too easily
DINO = "facebook/dinov3-vitl16-pretrain-lvd1689m"
NSFW = "Falconsai/nsfw_image_detection"
FAKE_THRESHOLD = 0.5  # ponytail: placeholder cutoff; the A4 eval picks the one with 95% precision on val
# Below this face_sim the face isn't the user's ("Unlikely to be you" on the dashboard), so its deepfake
# and NSFW scores can't change the case; skipping them keeps a scan inside Modal's 150 s web request limit.
SCORE_MIN_FACE_SIM = 0.3
MODELS = Path(os.environ.get("MODELS_DIR", Path(__file__).resolve().parents[1] / "data" / "models"))
log = logging.getLogger("six_eyes")


class AnalyzeResult(BaseModel):
    face_sim: float            # cosine vs. the user's enrolled ArcFace embedding
    frame_drift: float | None  # video only: variance of per-frame face_sim
    fake_prob: float           # detector: 1 - p(real)
    generator_guess: str       # real | swap | inpaint | txt2img | vae-decoded | no-face | not-scored
    recon_score: float | None  # SD1.5 VAE reconstruction distance; lower = more likely generated; None = not scored
    c2pa: bool | None          # True valid / False invalid Content Credentials; None = none found
    nsfw: float
    pdq: str                   # 64-hex PDQ hash, used for clone clusters


def analyze(media_url: str, user_embedding: list[float]) -> AnalyzeResult:
    frames, is_video, data, mime = face.download(media_url)
    face_sim, frame_drift = face.match(frames, is_video, user_embedding)
    middle = frames[len(frames) // 2]  # a video's middle sampled frame stands for the clip
    recon_score = None
    if face_sim >= SCORE_MIN_FACE_SIM:
        (fake_prob, generator_guess), nsfw_score = detect(frames), nsfw(frames)
        if probe():  # the VAE ships alongside the detector; a checkout without data/models skips it
            recon_score = round(recon.score(middle), 4)
    else:
        fake_prob, generator_guess, nsfw_score = 0.0, "not-scored", 0.0
    return AnalyzeResult(
        face_sim=round(face_sim, 4), frame_drift=frame_drift and round(frame_drift, 6),
        fake_prob=round(fake_prob, 4), generator_guess=generator_guess, recon_score=recon_score,
        c2pa=c2pa(data, mime), nsfw=round(nsfw_score, 4), pdq=pdq(middle),
    )


def c2pa(data: bytes, mime: str) -> bool | None:
    """Content Credentials in the file itself: True when the manifest validates, False when it's there but
    invalid (the content changed after signing), None when there is none. X may strip them on upload."""
    import c2pa as c2pa_lib

    try:
        manifest = json.loads(c2pa_lib.Reader(mime, io.BytesIO(data)).json())
    except c2pa_lib.C2paError as e:
        if "ManifestNotFound" not in type(e).__name__:
            log.warning("unreadable Content Credentials (%s): %s", mime, e)
        return None
    return manifest.get("validation_state") in ("Valid", "Trusted")


@lru_cache(maxsize=1)
def probe():
    """The A3 detector: frozen DINOv3, its input normalization, the two logistic-regression heads and
    the face detector used in training. None when MODELS has no heads (a checkout without data/)."""
    if not (MODELS / "dinov3_binary.joblib").exists():
        log.warning("No detector heads in %s, so fake_prob is a placeholder. Run ml/heads.py.", MODELS)
        return None
    import joblib
    import torch
    from transformers import AutoImageProcessor, AutoModel

    gpu = torch.cuda.is_available()
    model = AutoModel.from_pretrained(DINO, dtype=torch.bfloat16 if gpu else torch.float32)
    model = model.to("cuda" if gpu else "cpu").eval()
    proc = AutoImageProcessor.from_pretrained(DINO)
    mean, std = (torch.tensor(v).view(1, 3, 1, 1) for v in (proc.image_mean, proc.image_std))
    return (model, mean, std, joblib.load(MODELS / "dinov3_binary.joblib"),
            joblib.load(MODELS / "dinov3_multi.joblib"), crops.detector())


def detect(frames: list[np.ndarray]) -> tuple[float, str]:
    """(fake_prob, generator_guess) from the DINOv3 probe, averaged over a video's sampled frames.

    Each frame becomes the same two views as in training (ml/data.py, minus its random resize + JPEG)
    and one feature row concat(face, full), as in ml/heads.py. Frames without a face are skipped.
    """
    p = probe()
    if p is None:
        return 0.91, "swap"
    import torch

    model, mean, std, binary, multi, detector = p
    views = []
    for frame in frames:
        if (crop := crops.face_crop(frame, detector)) is not None:
            views += [crop, crops.full_view(frame)]
    if not views:
        return 0.0, "no-face"
    x = torch.from_numpy(np.stack([cv2.cvtColor(v, cv2.COLOR_BGR2RGB) for v in views])).permute(0, 3, 1, 2)
    x = ((x.float() / 255 - mean) / std).to(model.device, model.dtype)
    with torch.no_grad():
        feats = model(pixel_values=x).pooler_output.float().cpu().numpy()
    rows = feats.reshape(len(views) // 2, -1)  # [face, full] side by side per frame
    fake_prob = float(binary.predict_proba(rows)[:, 1].mean())
    if fake_prob < FAKE_THRESHOLD:
        return fake_prob, "real"  # the generator head alone often says vae-decoded for sharp real photos
    probs = multi.predict_proba(rows).mean(axis=0)
    probs[list(multi.classes_).index("real")] = 0  # past the cutoff, name the likeliest generator
    return fake_prob, str(multi.classes_[probs.argmax()])


@lru_cache(maxsize=1)
def nsfw_model():
    import torch
    from transformers import pipeline

    return pipeline("image-classification", model=NSFW, device=0 if torch.cuda.is_available() else -1)


def nsfw(frames: list[np.ndarray]) -> float:
    """Probability of Falconsai's nsfw label, the highest over a video's sampled frames."""
    from PIL import Image

    images = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
    return max(next(s["score"] for s in r if s["label"] == "nsfw") for r in nsfw_model()(images, top_k=None))


def pdq(img: np.ndarray) -> str:
    """64-hex PDQ hash of a BGR image, or "" when it's too flat to hash reliably (it then joins no
    clone cluster). For a video, analyze() hashes the middle sampled frame."""
    bits, quality = pdqhash.compute(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # ponytail: our own bit order; convert to ThreatExchange's hex layout before sharing hashes outside Six Eyes.
    return np.packbits(bits.astype(np.uint8)).tobytes().hex() if quality >= PDQ_MIN_QUALITY else ""


if __name__ == "__main__":
    print(analyze(sys.argv[1], json.loads(sys.argv[2])).model_dump_json())
