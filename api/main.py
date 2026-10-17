"""Six Eyes API (task B1): SQLite tables + POST /enroll.

Run from the repo root:
    .venv/Scripts/uvicorn api.main:app --reload --port 8000
"""
import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from functools import lru_cache

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

DB_PATH = os.environ.get("DB_PATH", "six_eyes.db")
MAX_PHOTO_BYTES = 10 * 1024 * 1024
SAME_PERSON_MIN = 0.3  # cosine between one selfie and the rest; calibration knob, tune on the team set

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
  id        INTEGER PRIMARY KEY,
  name      TEXT NOT NULL,
  handles   TEXT NOT NULL DEFAULT '[]',   -- JSON list of X handles, without '@'
  phone     TEXT,
  embedding TEXT NOT NULL,                -- JSON list of 512 floats, L2-normalized
  signed_at TEXT NOT NULL                 -- ISO time of consent + e-signature
);
CREATE TABLE IF NOT EXISTS cases (
  id         INTEGER PRIMARY KEY,
  user_id    INTEGER NOT NULL REFERENCES users(id),
  post_id    TEXT,
  author_id  TEXT,
  url        TEXT,
  pdq        TEXT,
  cluster_id INTEGER,
  scores     TEXT NOT NULL DEFAULT '{}',  -- JSON: AnalyzeResult + signals
  state      TEXT NOT NULL DEFAULT 'found'
             CHECK (state IN ('found', 'confirmed', 'noticed', 'resolved', 'escalated')),
  sent_at    TEXT,
  due_at     TEXT
);
CREATE TABLE IF NOT EXISTS events (
  case_id INTEGER NOT NULL REFERENCES cases(id),
  at      TEXT NOT NULL,
  kind    TEXT NOT NULL,
  detail  TEXT
);
"""


def db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA foreign_keys = ON")
    return con


with closing(db()) as con:
    con.executescript(SCHEMA)

app = FastAPI(title="Six Eyes API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o for o in os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(",") if o],
    # ponytail: any *.vercel.app origin; pin to the production domain before real users
    allow_origin_regex=r"https://[a-z0-9-]+\.vercel\.app",
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def face_model():
    from insightface.app import FaceAnalysis

    model = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"],
                         providers=["CPUExecutionProvider"])
    model.prepare(ctx_id=-1, det_size=(640, 640))
    return model


def embed(photo: bytes, n: int) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(photo, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(422, f"Photo {n} isn't an image we can read. Use JPG, PNG or WebP.")
    # The detector misses faces that fill most of the frame (close-up selfies); a border fixes that.
    h, w = img.shape[:2]
    img = cv2.copyMakeBorder(img, h // 4, h // 4, w // 4, w // 4, cv2.BORDER_CONSTANT)
    faces = face_model().get(img)
    if not faces:
        raise HTTPException(422, f"No face found in photo {n}. Use a clear, front-facing selfie.")
    # The largest face in a selfie is the person taking it.
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return face.normed_embedding


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/enroll")
def enroll(
    name: str = Form(...),
    signature: str = Form(...),
    consent: bool = Form(...),
    handles: str = Form(""),
    phone: str = Form(""),
    selfies: list[UploadFile] = File(...),
):
    name = name.strip()
    if not name:
        raise HTTPException(422, "Enter your full name.")
    if not consent:
        raise HTTPException(422, "You must agree to the consent terms to enroll.")
    if signature.strip().casefold() != name.casefold():
        raise HTTPException(422, "To sign, type your full name exactly as you entered it above.")
    if not 3 <= len(selfies) <= 5:
        raise HTTPException(422, f"Upload 3 to 5 selfies. You uploaded {len(selfies)}.")

    vecs = []
    for n, upload in enumerate(selfies, 1):
        try:
            photo = upload.file.read(MAX_PHOTO_BYTES + 1)
        finally:
            upload.file.close()  # photos never touch our storage; this also drops any spooled temp file
        if len(photo) > MAX_PHOTO_BYTES:
            raise HTTPException(413, f"Photo {n} is over 10 MB.")
        vecs.append(embed(photo, n))
    vecs = np.stack(vecs)

    # Every selfie must match the others, so one enrollment can't mix two people.
    for n, v in enumerate(vecs, 1):
        rest = np.delete(vecs, n - 1, axis=0).mean(axis=0)
        if float(v @ rest / np.linalg.norm(rest)) < SAME_PERSON_MIN:
            raise HTTPException(422, f"Photo {n} doesn't look like the same person as the others.")

    mean = vecs.mean(axis=0)
    mean /= np.linalg.norm(mean)
    handle_list = [h.lstrip("@") for h in handles.replace(",", " ").split() if h.lstrip("@")]
    signed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with closing(db()) as con, con:
        user_id = con.execute(
            "INSERT INTO users (name, handles, phone, embedding, signed_at) VALUES (?, ?, ?, ?, ?)",
            (name, json.dumps(handle_list), phone.strip() or None,
             json.dumps(mean.round(6).tolist()), signed_at),
        ).lastrowid
    return {"user_id": user_id, "photos_used": len(vecs), "handles": handle_list, "signed_at": signed_at}
