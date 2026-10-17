"""Task A2, step 3: make the Grok part of the team test set. C runs this; its output is never used for training.

Each consented photo in data/team/real gets one Grok Imagine edit (new outfit, background or lighting)
that keeps the person's face, saved to data/team/grok with the same subfolders.
Needs XAI_API_KEY in .env. Safe to re-run: photos that already have their edit are skipped.
    python ml/gen_testset.py
"""
import base64
import os
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import httpx
from dotenv import load_dotenv

SRC, DST = Path("data/team/real"), Path("data/team/grok")
MODEL = "grok-imagine-image-2.0"
MAX_SIDE = 1024  # phone photos are ~4000px; a smaller upload is faster and cheaper
KEEP = ("Keep the person's face, identity, expression and pose exactly the same. "
        "Photorealistic, fully clothed, safe for work.")
PROMPTS = {
    "outfit": f"Change the person's outfit to a different everyday outfit. {KEEP}",
    "background": f"Replace the background with a different everyday location. {KEEP}",
    "lighting": f"Change the lighting to warm golden-hour sunlight. {KEEP}",
}


def edit(src, client):
    key = list(PROMPTS)[zlib.crc32(src.name.encode()) % len(PROMPTS)]  # stable per photo, balanced overall
    out = DST / src.relative_to(SRC).with_name(f"{src.stem}_{key}.jpg")
    if out.exists():
        return
    img = cv2.imread(str(src))  # applies the phone's EXIF rotation
    if img is None:
        return print(f"SKIPPED {src}: not an image OpenCV can read")
    k = MAX_SIDE / max(img.shape[:2])
    if k < 1:
        img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])[1]
    body = {"model": MODEL, "prompt": PROMPTS[key],
            "image": {"type": "image_url", "url": "data:image/jpeg;base64," + base64.b64encode(jpg).decode()}}
    try:
        r = client.post("/images/edits", json=body)
        r.raise_for_status()
        item = r.json()["data"][0]
        result = base64.b64decode(item["b64_json"]) if "b64_json" in item else client.get(item["url"]).content
    except httpx.HTTPStatusError as e:  # bad key, rate limit, moderation refusal: print it and move on
        return print(f"FAILED {src} ({key}): {e.response.status_code} {e.response.text[:300]}")
    except (httpx.HTTPError, KeyError, IndexError) as e:
        return print(f"FAILED {src} ({key}): {e!r}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(result)  # Grok's bytes as-is; re-encoding would add our own compression
    print(f"ok {out}")


def main():
    load_dotenv()
    api_key = os.environ.get("XAI_API_KEY") or sys.exit("Set XAI_API_KEY in .env first.")
    photos = sorted(p for p in SRC.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    if not photos:
        sys.exit(f"No photos in {SRC}. Add the team's consented photos as JPG or PNG (export iPhone HEIC as JPG).")
    with httpx.Client(base_url="https://api.x.ai/v1", headers={"Authorization": f"Bearer {api_key}"},
                      timeout=180, follow_redirects=True) as client, ThreadPoolExecutor(4) as pool:
        list(pool.map(lambda p: edit(p, client), photos))
    done = sum(1 for _ in DST.rglob("*.jpg"))
    print(f"{done} Grok edits in {DST} for {len(photos)} photos. Re-run to retry any that FAILED.")


if __name__ == "__main__":
    main()
