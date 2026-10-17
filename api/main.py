"""Six Eyes API (tasks B1, B4): SQLite tables, POST /enroll and POST /scan/{user_id}.

Run from the repo root (api/ must be the import root so main can import x, signals and analyze):
    .venv/Scripts/uvicorn main:app --app-dir api --reload --port 8000
"""
import json
import logging
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone

import cv2
import httpx
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import analyze
import signals
import x
from face import face_model, faces  # noqa: F401  (face_model: modal_app and test_enroll preload it)

DB_PATH = os.environ.get("DB_PATH", "six_eyes.db")
MAX_PHOTO_BYTES = 10 * 1024 * 1024
SAME_PERSON_MIN = 0.3  # cosine between one selfie and the rest; calibration knob, tune on the team set
FACE_MATCH_MIN = 0.3   # face_sim a sweep hit needs to become a case; calibration knob, tune with A
SWEEP_LIMIT = 100      # posts per keyword sweep; each is billed by X and every image is downloaded
log = logging.getLogger("six_eyes")

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
CREATE TABLE IF NOT EXISTS sweeps (
  user_id  INTEGER PRIMARY KEY REFERENCES users(id),
  since_id TEXT NOT NULL                -- newest post the keyword sweep has checked for this user
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


def embed(photo: bytes, n: int) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(photo, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(422, f"Photo {n} isn't an image we can read. Use JPG, PNG or WebP.")
    found = faces(img, pad=True)
    if not found:
        raise HTTPException(422, f"No face found in photo {n}. Use a clear, front-facing selfie.")
    # The largest face in a selfie is the person taking it.
    return found[0].normed_embedding


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


@app.post("/scan/{user_id}")
def scan(user_id: int, sweep: bool = False):
    """Search X for the user, score every new image and video, and save each one as a found case.

    sweep=true also checks recent posts that use deepfake terms without naming the user; those
    become cases only when a face in them matches the user's enrolled face.
    """
    with closing(db()) as con:
        user = con.execute("SELECT name, handles, embedding FROM users WHERE id = ?", (user_id,)).fetchone()
        if not user:
            raise HTTPException(404, f"No enrolled user with id {user_id}.")
        seen = {r[0] for r in con.execute(
            "SELECT DISTINCT post_id FROM cases WHERE user_id = ? AND post_id IS NOT NULL", (user_id,))}
        # Only name-search cases bound the next name search; sweep cases come from a different query.
        since_id = con.execute(
            "SELECT MAX(CAST(post_id AS INTEGER)) FROM cases WHERE user_id = ? AND json_extract(scores, '$.source') = 'search'",
            (user_id,)).fetchone()[0]
        sweep_since = con.execute("SELECT since_id FROM sweeps WHERE user_id = ?", (user_id,)).fetchone()
    name, handles, embedding = user[0], json.loads(user[1]), json.loads(user[2])

    try:
        posts = [(p, "search") for p in x.search(name, handles, since_id=since_id and str(since_id))]
        swept = x.sweep(signals.search_terms(), SWEEP_LIMIT, sweep_since and sweep_since[0]) if sweep else []
        posts += [(p, "sweep") for p in swept]
    except ValueError as e:
        raise HTTPException(422, str(e))
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            raise HTTPException(429, "X's rate limit was reached. Try the scan again in 15 minutes.")
        raise HTTPException(502, f"X search failed with status {e.response.status_code}.")
    except httpx.HTTPError:
        raise HTTPException(502, "Can't reach X right now. Try the scan again shortly.")

    now = datetime.now(timezone.utc)
    found, checked, failed = [], 0, 0
    for post, source in posts:
        if post.id in seen:
            continue
        seen.add(post.id)
        checked += 1
        post_signals = signals.signals(post, now).model_dump()
        for media in post.media:
            try:
                result = analyze.analyze(media.url, embedding)
            except Exception:  # one unreadable image or video shouldn't sink the whole scan
                log.exception("analyze failed for post %s media %s", post.id, media.key)
                failed += 1
                continue
            if source == "sweep" and result.face_sim < FACE_MATCH_MIN:
                continue
            found.append((post, result.pdq, {
                **result.model_dump(), "signals": post_signals, "source": source, "media": media.model_dump(),
                "possibly_sensitive": post.possibly_sensitive, "text": post.text,
            }))

    at = now.isoformat(timespec="seconds")
    with closing(db()) as con, con:
        for post, pdq, scores in found:
            case_id = con.execute(
                "INSERT INTO cases (user_id, post_id, author_id, url, pdq, scores) VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, post.id, post.author.id if post.author else None, post.url, pdq, json.dumps(scores)),
            ).lastrowid
            con.execute("INSERT INTO events (case_id, at, kind, detail) VALUES (?, ?, 'found', ?)",
                        (case_id, at, scores["source"]))
        # Oldest first, so a cluster keeps the id of its first case as new clones join it.
        cases = con.execute("SELECT id, pdq, cluster_id FROM cases WHERE user_id = ? ORDER BY id",
                            (user_id,)).fetchall()
        roots = signals.clusters([pdq for _, pdq, _ in cases])
        con.executemany("UPDATE cases SET cluster_id = ? WHERE id = ?", [
            (cases[root][0], case_id) for (case_id, _, old), root in zip(cases, roots)
            if root is not None and cases[root][0] != old
        ])
        if swept:
            con.execute("INSERT INTO sweeps (user_id, since_id) VALUES (?, ?) "
                        "ON CONFLICT (user_id) DO UPDATE SET since_id = excluded.since_id",
                        (user_id, max((p.id for p in swept), key=int)))
    return {"user_id": user_id, "posts_checked": checked, "cases_found": len(found), "media_failed": failed}
