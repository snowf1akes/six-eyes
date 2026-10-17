"""Face matching with InsightFace buffalo_l (ArcFace): is the user's face in a found image or video?

Media is downloaded into memory (a video into a temp file, deleted right after) and never kept.
"""
import os
import tempfile
from functools import lru_cache
from urllib.parse import urlsplit

import cv2
import httpx
import numpy as np

MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_VIDEO_BYTES = 100 * 1024 * 1024
VIDEO_FRAMES = 16  # frames sampled evenly across a video


@lru_cache(maxsize=1)
def face_model():
    from insightface.app import FaceAnalysis

    model = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"],
                         providers=["CPUExecutionProvider"])
    model.prepare(ctx_id=-1, det_size=(640, 640))
    return model


def faces(img: np.ndarray, pad: bool = False) -> list:
    """Every face InsightFace finds, largest first.

    The detector misses faces that fill most of the frame (close-up selfies); pad=True adds a
    border to fix that, at the cost of shrinking small faces in wide shots.
    """
    if pad:
        h, w = img.shape[:2]
        img = cv2.copyMakeBorder(img, h // 4, h // 4, w // 4, w // 4, cv2.BORDER_CONSTANT)
    found = face_model().get(img)
    return sorted(found, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)


def similarity(img: np.ndarray, user: np.ndarray) -> float | None:
    """Cosine between the user's embedding and the closest face in the image; None if no face."""
    found = faces(img) or faces(img, pad=True)
    return max(float(f.normed_embedding @ user) for f in found) if found else None


def fetch(media_url: str) -> tuple[list[np.ndarray], bool]:
    """(frames, is_video) for an image or video URL, downloaded once and shared by every check in
    analyze(). An image is one frame; a video is up to VIDEO_FRAMES frames sampled evenly.
    """
    with httpx.stream("GET", media_url, timeout=30, follow_redirects=True) as r:
        r.raise_for_status()
        is_video = (r.headers.get("content-type", "").startswith("video/")
                    or urlsplit(media_url).path.endswith(".mp4"))
        if is_video:
            return _video_frames(r), True
        data = _read(r, MAX_IMAGE_BYTES)
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Not a readable image: {media_url}")
    return [img], False


def match(frames: list[np.ndarray], is_video: bool, user_embedding: list[float]) -> tuple[float, float | None]:
    """(face_sim, frame_drift) for frames from fetch().

    face_sim is the best match across all faces (and, for video, all sampled frames); 0.0 when
    no face is found. frame_drift is the variance of per-frame face_sim, for video only.
    """
    user = np.asarray(user_embedding, dtype=np.float32)
    user /= np.linalg.norm(user)
    if not is_video:
        return similarity(frames[0], user) or 0.0, None
    sims = [s for frame in frames if (s := similarity(frame, user)) is not None]
    return (max(sims), float(np.var(sims))) if sims else (0.0, None)


def _video_frames(r: httpx.Response) -> list[np.ndarray]:
    # OpenCV only reads video from a file path, so the video briefly touches disk.
    fd, path = tempfile.mkstemp(suffix=".mp4")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(_read(r, MAX_VIDEO_BYTES))
        cap = cv2.VideoCapture(path)
        try:
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frames = []
            for i in sorted({int(i) for i in np.linspace(0, n - 1, VIDEO_FRAMES)} if n > 0 else ()):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ok, frame = cap.read()
                if ok:
                    frames.append(frame)
        finally:
            cap.release()
    finally:
        os.remove(path)
    if not frames:
        raise ValueError("Not a readable video.")
    return frames


def _read(r: httpx.Response, limit: int) -> bytes:
    if int(r.headers.get("content-length", 0)) > limit:
        raise ValueError(f"Media is over {limit // (1024 * 1024)} MB.")
    data = bytearray()
    for chunk in r.iter_bytes():
        data += chunk
        if len(data) > limit:
            raise ValueError(f"Media is over {limit // (1024 * 1024)} MB.")
    return bytes(data)
