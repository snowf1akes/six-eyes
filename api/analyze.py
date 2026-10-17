"""Contract between Builder A (detection) and Builder B (product).

B calls analyze(); A replaces the placeholder body with the real pipeline (task A6).
Inside a Daytona sandbox it runs as:
    python analyze.py <media_url> '<embedding json>'   -> prints AnalyzeResult as JSON
"""
import json
import sys

from pydantic import BaseModel

import face


class AnalyzeResult(BaseModel):
    face_sim: float            # cosine vs. the user's enrolled ArcFace embedding
    frame_drift: float | None  # video only: variance of per-frame face_sim
    fake_prob: float           # detector: 1 - p(real)
    generator_guess: str       # real | swap | inpaint | txt2img | vae-decoded
    recon_score: float | None  # min VAE reconstruction distance; lower = more likely generated
    c2pa: bool | None          # None = no Content Credentials found
    nsfw: float
    pdq: str                   # 64-hex PDQ hash, used for clone clusters


def analyze(media_url: str, user_embedding: list[float]) -> AnalyzeResult:
    face_sim, frame_drift = face.match(media_url, user_embedding)
    # ponytail: face match is real; the detector fields are fixed placeholders until A6 lands.
    return AnalyzeResult(
        face_sim=round(face_sim, 4), frame_drift=frame_drift and round(frame_drift, 6),
        fake_prob=0.91, generator_guess="swap", recon_score=0.031, c2pa=None, nsfw=0.05, pdq="0" * 64,
    )


if __name__ == "__main__":
    print(analyze(sys.argv[1], json.loads(sys.argv[2])).model_dump_json())
