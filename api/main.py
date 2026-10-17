"""Six Eyes API: SQLite tables, POST /enroll, POST /scan/{user_id} and GET /cases/{user_id}.

Run from the repo root (api/ must be the import root so main can import x, signals and analyze):
    .venv/Scripts/uvicorn main:app --app-dir api --reload --port 8000
"""
import json
import logging
import math
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone

import cv2
import httpx
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import analyze
import signals
import x
from face import face_model, faces  # noqa: F401  (face_model: modal_app and test_enroll preload it)

DB_PATH = os.environ.get("DB_PATH", "six_eyes.db")
MAX_PHOTO_BYTES = 10 * 1024 * 1024
SAME_PERSON_MIN = 0.3  # cosine between one selfie and the rest; calibration knob, tune on the team set
FACE_MATCH_MIN = 0.3   # face_sim a sweep hit needs to become a case; calibration knob, tune with A
SWEEP_LIMIT = 100      # posts per keyword sweep; each is billed by X and every image is downloaded
# How much each kind of harm counts toward a case's dashboard priority; calibration knobs.
PRIORITY_WEIGHTS = {"fake": 0.4, "nsfw": 0.3, "spread": 0.3}
CASE_STATES = ("found", "confirmed", "noticed", "resolved", "escalated")
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
        recluster(con, user_id)
        if swept:
            con.execute("INSERT INTO sweeps (user_id, since_id) VALUES (?, ?) "
                        "ON CONFLICT (user_id) DO UPDATE SET since_id = excluded.since_id",
                        (user_id, max((p.id for p in swept), key=int)))
    return {"user_id": user_id, "posts_checked": checked, "cases_found": len(found), "media_failed": failed}


def recluster(con: sqlite3.Connection, user_id: int) -> None:
    """Group the user's cases into clone clusters by PDQ; each cluster takes the id of its oldest case."""
    cases = con.execute("SELECT id, pdq, cluster_id FROM cases WHERE user_id = ? ORDER BY id", (user_id,)).fetchall()
    updates = []
    for (case_id, _, old), root in zip(cases, signals.clusters([pdq for _, pdq, _ in cases])):
        new = cases[root][0] if root is not None else None
        if new != old:
            updates.append((new, case_id))
    con.executemany("UPDATE cases SET cluster_id = ? WHERE id = ?", updates)


def priority(scores: dict, copies: int) -> float:
    """0 to 1: how surely the case shows the user, times how harmful it looks."""
    identity = min(max((scores["face_sim"] - 0.2) / 0.4, 0.0), 1.0)  # 0 at a stranger's ~0.2, 1 at a clear 0.6
    sig = scores.get("signals", {})
    e = sig.get("engagement", {})
    shares = e.get("like", 0) + e.get("reply", 0) + 2 * (e.get("retweet", 0) + e.get("quote", 0))
    reach = min(math.log10(1 + shares) / 4, 1.0)  # 10,000 weighted shares or more counts as full reach
    age, ratio = sig.get("author_age_days"), sig.get("follower_ratio")
    burner = (age is not None and age < 30) or (ratio is not None and ratio < 0.1)
    flagged = bool(sig.get("keywords") or sig.get("hashtags") or sig.get("domains"))
    spread = 0.4 * reach + 0.2 * min((copies - 1) / 4, 1.0) + 0.2 * burner + 0.2 * flagged
    w = PRIORITY_WEIGHTS
    return round(identity * (w["fake"] * scores["fake_prob"] + w["nsfw"] * scores["nsfw"] + w["spread"] * spread), 4)


# ponytail: no auth yet, so anyone with a user id can list that user's cases. Add sessions before real users.
@app.get("/cases/{user_id}")
def list_cases(user_id: int, state: str = "found"):
    """The user's cases in one state, highest priority first. Media stays on X; only its URL is sent."""
    if state not in CASE_STATES:
        raise HTTPException(422, f"state must be one of: {', '.join(CASE_STATES)}.")
    with closing(db()) as con:
        if not con.execute("SELECT 1 FROM users WHERE id = ?", (user_id,)).fetchone():
            raise HTTPException(404, f"No enrolled user with id {user_id}.")
        rows = con.execute("""
            SELECT c.id, c.post_id, c.author_id, c.url, c.state, c.scores,
                   (SELECT COUNT(DISTINCT k.post_id) FROM cases k
                     WHERE k.user_id = c.user_id AND k.cluster_id = c.cluster_id),
                   (SELECT MIN(e.at) FROM events e WHERE e.case_id = c.id AND e.kind = 'found')
            FROM cases c WHERE c.user_id = ? AND c.state = ?""", (user_id, state)).fetchall()
    cases = []
    for case_id, post_id, author_id, url, case_state, scores, copies, found_at in rows:
        scores, copies = json.loads(scores), max(copies, 1)
        cases.append({"id": case_id, "post_id": post_id, "author_id": author_id, "url": url, "state": case_state,
                      "found_at": found_at, "copies": copies, "priority": priority(scores, copies), "scores": scores})
    cases.sort(key=lambda c: c["priority"], reverse=True)
    return {"user_id": user_id, "cases": cases}


class Confirmation(BaseModel):
    is_me: bool           # "This is me": the person in it is the user
    not_consensual: bool  # "Not consensual": the user didn't agree to it being shared


# ponytail: no auth yet, so anyone with a case id can confirm it. Add sessions before real users.
@app.post("/cases/{case_id}/confirm")
def confirm(case_id: int, body: Confirmation):
    """Move a found case and every found clone of its image to confirmed, recording both attestations."""
    if not (body.is_me and body.not_consensual):
        raise HTTPException(422, "Confirm both that this shows you and that it was shared without your consent.")
    at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with closing(db()) as con, con:
        case = con.execute("SELECT user_id, cluster_id FROM cases WHERE id = ?", (case_id,)).fetchone()
        if not case:
            raise HTTPException(404, f"No case with id {case_id}.")
        ids = [r[0] for r in con.execute(
            "SELECT id FROM cases WHERE user_id = ? AND state = 'found' AND (id = ? OR cluster_id = ?) ORDER BY id",
            (case[0], case_id, case[1]))]
        con.executemany("UPDATE cases SET state = 'confirmed' WHERE id = ?", [(i,) for i in ids])
        detail = json.dumps({"via_case": case_id, "is_me": True, "not_consensual": True})
        con.executemany("INSERT INTO events (case_id, at, kind, detail) VALUES (?, ?, 'confirmed', ?)",
                        [(i, at, detail) for i in ids])
    return {"confirmed": ids}
