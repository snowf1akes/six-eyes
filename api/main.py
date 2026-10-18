"""Six Eyes API: SQLite tables, POST /enroll, POST /scan/{user_id}, GET /cases/{user_id},
POST /cases/{case_id}/confirm, POST /cases/{case_id}/notice, the deadline recheck (POST /recheck,
POST /timewarp/{case_id}), and the case worker (caseworker.py) which owns the 48-hour wait:
POST /cases/{case_id}/work and GET /messages/{user_id}.

Run from the repo root (api/ must be the import root so main can import x, signals and analyze):
    .venv/Scripts/uvicorn main:app --app-dir api --reload --port 8000
"""
import json
import logging
import math
import os
import smtplib
import sqlite3
import threading
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit


def _load_local_env() -> None:
    """Uvicorn does not read .env. Skip when DB_PATH is already set (tests, Modal)."""
    if "DB_PATH" in os.environ:
        return
    path = Path(__file__).resolve().parents[1] / ".env"
    if not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'").strip('"')
        # Fill missing or blank vars from .env; a shell `export X_BEARER_TOKEN=` would otherwise win.
        if key and not os.environ.get(key):
            os.environ[key] = value


_load_local_env()

import cv2
import httpx
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError

import analyze
import caseworker
import channels
import photon
import signals
import x
from face import face_model, faces  # noqa: F401  (face_model: modal_app and test_enroll preload it)

DB_PATH = os.environ.get("DB_PATH", "six_eyes.db")
MAX_PHOTO_BYTES = 10 * 1024 * 1024
SAME_PERSON_MIN = 0.3  # cosine between one selfie and the rest; calibration knob, tune on the team set
FACE_MATCH_MIN = 0.3   # face_sim a sweep hit needs to become a case; calibration knob, tune with A
SWEEP_LIMIT = 100      # posts per keyword sweep; each is billed by X and every image is downloaded
# How much each kind of harm counts toward a case's dashboard priority; calibration knobs.
PRIORITY_WEIGHTS = {"keywords": 0.4, "fake": 0.3, "nsfw": 0.15, "spread": 0.15}
CASE_STATES = ("found", "confirmed", "noticed", "resolved", "escalated")
def notice_deadline() -> timedelta:
    """How long we wait after a notice before rechecking X. NOTICE_DEADLINE_SECONDS (default 48 hours)."""
    raw = os.environ.get("NOTICE_DEADLINE_SECONDS")
    return timedelta(seconds=int(raw)) if raw else timedelta(hours=48)


def deadline_words() -> str:
    """'48 hours', '30 seconds', … — used in the user text, not in the legal notice copy."""
    secs = int(notice_deadline().total_seconds())
    if secs % 3600 == 0:
        n = secs // 3600
        return "1 hour" if n == 1 else f"{n} hours"
    if secs % 60 == 0:
        n = secs // 60
        return "1 minute" if n == 1 else f"{n} minutes"
    return "1 second" if secs == 1 else f"{secs} seconds"
INTIMATE_MIN = 0.5  # NSFW score a live notice needs; the Act only covers intimate depictions (NOTES.md flag 3)
FORGERY_MIN = 0.5   # fake_prob from which the notice calls the image a digital forgery
log = logging.getLogger("six_eyes")
# user_id -> progress of the latest background scan. In memory: Modal runs one API container.
SCANS: dict[int, dict] = {}
SCAN_LOCK = threading.Lock()
RECHECK_LOCK = threading.Lock()

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
CREATE TABLE IF NOT EXISTS messages (   -- the case worker's updates to the user
  id      INTEGER PRIMARY KEY,
  user_id INTEGER NOT NULL REFERENCES users(id),
  case_id INTEGER REFERENCES cases(id),
  at      TEXT NOT NULL,
  text    TEXT NOT NULL,
  link    TEXT                          -- optional action, e.g. the FTC form
);
"""


def db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA foreign_keys = ON")
    return con


with closing(db()) as con:
    con.executescript(SCHEMA)
    if "link" not in {r[1] for r in con.execute("PRAGMA table_info(messages)")}:
        con.execute("ALTER TABLE messages ADD COLUMN link TEXT")
        con.commit()

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
def scan(user_id: int, sweep: bool = False, background: bool = False):
    """Search X for the user, score every new image and video, and save each one as a found case.

    sweep=true also checks recent posts that use deepfake terms without naming the user; those
    become cases only when a face in them matches the user's enrolled face.
    background=true starts the scan in a thread and returns its progress at once; poll GET /scan/{user_id}.
    The dashboard uses it because Modal cuts web requests at 150 s and a first scan can run longer.
    """
    progress = {"user_id": user_id, "posts_checked": 0, "cases_found": 0, "media_failed": 0}
    if not background:
        run_scan(user_id, sweep, progress)
        return progress
    with closing(db()) as con:
        if not con.execute("SELECT 1 FROM users WHERE id = ?", (user_id,)).fetchone():
            raise HTTPException(404, f"No enrolled user with id {user_id}.")
    with SCAN_LOCK:
        if not SCANS.get(user_id, {}).get("running"):
            SCANS[user_id] = {**progress, "running": True, "error": None}
            threading.Thread(target=background_scan, args=(user_id, sweep), daemon=True).start()
        return SCANS[user_id]


@app.get("/scan/{user_id}")
def scan_status(user_id: int):
    """Progress of the user's latest background scan."""
    return SCANS.get(user_id, {"user_id": user_id, "posts_checked": 0, "cases_found": 0, "media_failed": 0,
                               "running": False, "error": None})


def background_scan(user_id: int, sweep: bool) -> None:
    status = SCANS[user_id]
    try:
        run_scan(user_id, sweep, status)
    except HTTPException as e:
        status["error"] = e.detail
    except Exception:
        log.exception("background scan failed for user %s", user_id)
        status["error"] = "The scan stopped early. Found posts so far are saved; scan again to continue."
    finally:
        status["running"] = False


def run_scan(user_id: int, sweep: bool, progress: dict) -> None:
    """The scan itself. Each case is saved as soon as it's found, so a scan cut off partway keeps its
    finds; progress counts go up as it runs."""
    with closing(db()) as con:
        user = con.execute("SELECT name, handles, embedding FROM users WHERE id = ?", (user_id,)).fetchone()
        if not user:
            raise HTTPException(404, f"No enrolled user with id {user_id}.")
        seen = {r[0] for r in con.execute(
            "SELECT DISTINCT post_id FROM cases WHERE user_id = ? AND post_id IS NOT NULL", (user_id,))}
        # Each query resumes after the newest post its own results produced. Keyword hits are a
        # subset of the name search, so they bound it too; sweep cases come from a different query.
        newest = dict(con.execute(
            "SELECT json_extract(scores, '$.source'), MAX(CAST(post_id AS INTEGER)) FROM cases WHERE user_id = ? GROUP BY 1",
            (user_id,)).fetchall())
        sweep_since = con.execute("SELECT since_id FROM sweeps WHERE user_id = ?", (user_id,)).fetchone()
    name, handles, embedding = user[0], json.loads(user[1]), json.loads(user[2])
    terms = signals.search_terms()
    keyword_since = newest.get("keyword")
    search_since = max(newest.get("search") or 0, keyword_since or 0) or None

    try:
        # Name plus a deepfake term first, so those posts are labelled before the plain name search sees them.
        posts = [(p, "keyword") for p in x.search(name, handles, since_id=keyword_since and str(keyword_since),
                                                  terms=terms)] if terms else []
        posts += [(p, "search") for p in x.search(name, handles, since_id=search_since and str(search_since))]
        swept = x.sweep(terms, SWEEP_LIMIT, sweep_since and sweep_since[0]) if sweep else []
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
    at = now.isoformat(timespec="seconds")
    for post, source in posts:
        if post.id in seen:
            continue
        seen.add(post.id)
        progress["posts_checked"] += 1
        post_signals = signals.signals(post, now).model_dump()
        for media in post.media:
            try:
                result = analyze.analyze(media.url, embedding)
            except Exception:  # one unreadable image or video shouldn't sink the whole scan
                log.exception("analyze failed for post %s media %s", post.id, media.key)
                progress["media_failed"] += 1
                continue
            if source == "sweep" and result.face_sim < FACE_MATCH_MIN:
                continue
            scores = {**result.model_dump(), "signals": post_signals, "source": source, "media": media.model_dump(),
                      "possibly_sensitive": post.possibly_sensitive, "text": post.text}
            with closing(db()) as con, con:
                case_id = con.execute(
                    "INSERT INTO cases (user_id, post_id, author_id, url, pdq, scores) VALUES (?, ?, ?, ?, ?, ?)",
                    (user_id, post.id, post.author.id if post.author else None, post.url, result.pdq,
                     json.dumps(scores)),
                ).lastrowid
                con.execute("INSERT INTO events (case_id, at, kind, detail) VALUES (?, ?, 'found', ?)",
                            (case_id, at, source))
            progress["cases_found"] += 1

    # ponytail: a scan cut off partway can leave older name-search posts behind the next since_id;
    # they come back only if they are reposted. Fine for the demo's handful of posts.
    with closing(db()) as con, con:
        recluster(con, user_id)
        if swept:
            con.execute("INSERT INTO sweeps (user_id, since_id) VALUES (?, ?) "
                        "ON CONFLICT (user_id) DO UPDATE SET since_id = excluded.since_id",
                        (user_id, max((p.id for p in swept), key=int)))


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
    flagged = bool(sig.get("keywords") or sig.get("hashtags"))  # says deepfake, AI nude, ...
    spread = 0.4 * reach + 0.2 * min((copies - 1) / 4, 1.0) + 0.2 * burner + 0.2 * bool(sig.get("domains"))
    w = PRIORITY_WEIGHTS
    return round(identity * (w["keywords"] * flagged + w["fake"] * scores["fake_prob"] + w["nsfw"] * scores["nsfw"]
                             + w["spread"] * spread), 4)


# ponytail: no auth yet, so anyone with a user id can list that user's cases. Add sessions before real users.
@app.get("/cases/{user_id}")
def list_cases(user_id: int, state: str = "found"):
    """The user's cases in one state, highest priority first. Media stays on X; only its URL is sent.

    Loading the dashboard first wakes any overdue Factory deadline sessions (or settles
    them itself if the case worker is off)."""
    if state not in CASE_STATES:
        raise HTTPException(422, f"state must be one of: {', '.join(CASE_STATES)}.")
    with closing(db()) as con:
        if not con.execute("SELECT 1 FROM users WHERE id = ?", (user_id,)).fetchone():
            raise HTTPException(404, f"No enrolled user with id {user_id}.")
    settled = recheck(user_id)
    with closing(db()) as con:
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
    return {"user_id": user_id, "cases": cases, "settled": settled, "escalations": open_escalations(user_id)}


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
    # File right away, so the dashboard can show the reports; if that fails the cases stay confirmed,
    # and the case worker or POST /cases/{id}/notice retries.
    filed, error, working = None, None, False
    if ids:
        try:
            filed = notice(ids[0])
        except HTTPException as e:
            error = e.detail
        working = caseworker.start(ids[0], "confirmed")
        # Always text from here. On Modal the droid CLI is in the image, so the case worker looks
        # "on" even when Factory never actually notifies; local has no droid, so this was the only path.
        if filed:
            n = len(filed.get("case_ids") or ids)
            text = (f"Six Eyes: we prepared {n} {'report' if n == 1 else 'reports'} for X. "
                    f"They must take the posts down within {deadline_words()} of receiving them. "
                    f"We'll text you if they're still up after that.")
            tell_user(case[0], ids[0], ("Test: " + text) if dry_run() else text)
    return {"confirmed": ids, "notice": filed, "notice_error": error, "caseworker": working}


def dry_run() -> bool:
    return os.environ.get("DRY_RUN", "1") != "0"  # only an explicit 0 goes live


# ponytail: no auth yet, so anyone with a case id can file its notice. Add sessions before real users.
@app.post("/cases/{case_id}/notice")
def notice(case_id: int):
    """Turn a confirmed case into TAKE IT DOWN notices for its whole clone cluster.

    One packet per posting account, since X's form reports one username at a time; each has the
    rendered notice and the form's answers. Nothing is ever submitted to a platform: X only takes a
    web form, which a person fills in. With DRY_RUN (the default) the packets are emailed to
    NOTICE_TEST_TO if set. Either way the cluster moves to noticed, due 48 hours later.
    """
    with closing(db()) as con:
        case = con.execute("SELECT user_id, cluster_id, state, url FROM cases WHERE id = ?", (case_id,)).fetchone()
        if not case:
            raise HTTPException(404, f"No case with id {case_id}.")
        user_id, cluster_id, state, url = case
        if state != "confirmed":
            raise HTTPException(409, f"Case {case_id} is {state}; only confirmed cases get a notice.")
        rows = con.execute(
            "SELECT id, url, scores FROM cases WHERE user_id = ? AND state = 'confirmed' AND (id = ? OR cluster_id = ?) ORDER BY id",
            (user_id, case_id, cluster_id)).fetchall()
        name, signed_at = con.execute("SELECT name, signed_at FROM users WHERE id = ?", (user_id,)).fetchone()
    if not signed_at:
        raise HTTPException(422, "This user never signed the authorization, so Six Eyes can't file for them.")
    try:
        channel = channels.channel_for(url or "")
    except KeyError:
        raise HTTPException(422, f"Six Eyes doesn't know where to send removal requests for {url}.")
    dry = dry_run()
    scores = [json.loads(s) for _, _, s in rows]
    intimate = max(s.get("nsfw", 0) for s in scores) >= INTIMATE_MIN
    if not (dry or intimate):
        raise HTTPException(422, "This doesn't look like intimate content (NSFW score under "
                                 f"{INTIMATE_MIN}), so the TAKE IT DOWN Act doesn't cover it.")

    by_account: dict[str, list[tuple[int, str, dict]]] = {}
    for (cid, post_url, _), s in zip(rows, scores):
        username = urlsplit(post_url or "").path.split("/")[1:2]
        if not username or username[0] == "i":  # x.py writes /i/ when X didn't return the author
            raise HTTPException(422, f"Couldn't tell which account posted {post_url}; X's form needs it.")
        by_account.setdefault(username[0], []).append((cid, post_url, s))

    now = datetime.now(timezone.utc)
    packets = []
    for username, items in by_account.items():
        forgery = max(s.get("fake_prob", 0) for _, _, s in items) >= FORGERY_MIN
        try:
            packet = channels.TakedownNotice(
                case_id=case_id, platform_name=channel.name, user_name=name, authorization_date=signed_at,
                reported_username=username, content_urls=[u for _, u, _ in items],
                depiction_kind=channels.FORGERY if forgery else channels.INTIMATE, signer_name=name,
                signer_relationship="The identifiable individual depicted", contact_email="",
                non_consent_confirmed=True)
        except ValidationError as e:
            raise HTTPException(422, f"The notice is incomplete: {e.errors()[0]['loc'][0]} {e.errors()[0]['msg']}.")
        packets.append({"reported_username": username, "case_ids": [i for i, _, _ in items],
                        "content_urls": packet.content_urls, "notice": channels.render(packet, now),
                        "form": channels.form_answers(channel, packet, now),
                        "checkboxes": [b.model_dump() for b in channel.checkboxes]})

    to = os.environ.get("NOTICE_TEST_TO") if dry else None
    if to:
        n = len(packets)
        body = ("TEST ONLY. Our test posts were shared with consent, so never submit these for real.\n\n"
                + "\n\n----------\n\n".join(channels.instructions(channel, p) for p in packets))
        try:
            channels.send_email(to, f"[Six Eyes test] {n} TAKE IT DOWN {'report' if n == 1 else 'reports'} "
                                    f"for {channel.name}", body)
        except RuntimeError as e:
            raise HTTPException(503, str(e))
        except (smtplib.SMTPException, OSError):
            log.exception("test notice email for case %s failed", case_id)
            raise HTTPException(502, "Couldn't send the test email. Check the SMTP settings, then try again.")

    # ponytail: the 48 hours really start when someone submits X's form, not when the packet is made.
    at, due = now.isoformat(timespec="seconds"), (now + notice_deadline()).isoformat(timespec="seconds")
    ids = [i for i, _, _ in rows]
    detail = json.dumps({"via_case": case_id, "platform": channel.name, "dry_run": dry, "delivered_to": to,
                         "packets": packets})
    with closing(db()) as con, con:
        con.executemany("UPDATE cases SET state = 'noticed', sent_at = ?, due_at = ? WHERE id = ? AND state = 'confirmed'",
                        [(at, due, i) for i in ids])
        con.executemany("INSERT INTO events (case_id, at, kind, detail) VALUES (?, ?, 'notice', ?)",
                        [(i, at, detail) for i in ids])
    return {"case_ids": ids, "dry_run": dry, "intimate": intimate, "platform": channel.name,
            "form_url": channel.url, "delivered_to": to, "sent_at": at, "due_at": due, "packets": packets}


# ponytail: no auth yet, so anyone can start the case worker (a Factory session, billed X checks). Add sessions.
@app.post("/cases/{case_id}/work")
def work_case(case_id: int):
    """Have the case worker check on one case now, deadline or not; it runs in the background."""
    with closing(db()) as con:
        if not con.execute("SELECT 1 FROM cases WHERE id = ?", (case_id,)).fetchone():
            raise HTTPException(404, f"No case with id {case_id}.")
    if not caseworker.enabled():
        raise HTTPException(503, "The case worker is off: install the droid CLI and set FACTORY_API_KEY.")
    return {"case_id": case_id, "started": caseworker.start(case_id, "check")}


def recheck(user_id: int | None = None) -> list[dict]:
    """Settle every noticed clone cluster past its deadline (the user's, or everyone's).

    Factory owns the clock: wait_until sleeps or arms a timer, and a deadline session checks X.
    This path is the backup when that timer died (dashboard load, POST /recheck, Modal's hourly
    sweep). When the case worker is on it only starts those sessions. When it's off we settle
    here and text the user ourselves. A cluster X can't be checked for stays noticed.
    """
    with RECHECK_LOCK:  # one recheck at a time, so no post is checked (and billed) twice
        if caseworker.enabled():
            caseworker.wake_due(user_id)
            return []
        with closing(db()) as con:
            due = con.execute(
                "SELECT user_id, COALESCE(cluster_id, id) AS k, MIN(id) FROM cases WHERE state = 'noticed' AND due_at <= ?"
                + (" AND user_id = ?" if user_id is not None else "") + " GROUP BY user_id, k ORDER BY 3",
                (_now(),) if user_id is None else (_now(), user_id)).fetchall()
            clusters = [(first, uid, con.execute(
                "SELECT id, post_id, url FROM cases WHERE user_id = ? AND COALESCE(cluster_id, id) = ? "
                "AND state = 'noticed' ORDER BY id", (uid, cluster)).fetchall()) for uid, cluster, first in due]
        settled = []
        for first, uid, rows in clusters:
            try:
                live = post_status([(post_id, url) for _, post_id, url in rows])
            except (RuntimeError, httpx.HTTPError) as e:
                log.warning("recheck of case %s skipped: %s", first, e)
                continue
            resolved = resolve([i for i, _, url in rows if not live[url]], "recheck")
            escalated, packet, form = escalate([i for i, _, url in rows if live[url]], "recheck")
            settled.append({"case_id": first, "resolved": resolved, "escalated": escalated, "ftc_packet": packet,
                            "ftc_url": channels.escalation()["ftc"] if escalated else None,
                            "ftc_form": form or None, "dry_run": dry_run()})
            tell_user(uid, first, *_settled_notice(uid, resolved, escalated))
        return settled


def post_status(posts: list[tuple[str, str]]) -> dict[str, bool]:
    """(post_id, url) pairs -> url: still up on X. One billed X call per distinct post."""
    return {url: x.is_live(post_id) for post_id, url in dict(posts).items()}


def resolve(case_ids: list[int], via: str) -> list[int]:
    """Noticed cases whose post is gone move to resolved; returns the ones that moved."""
    at = _now()
    with closing(db()) as con, con:
        moved = [i for i in case_ids
                 if con.execute("UPDATE cases SET state = 'resolved' WHERE id = ? AND state = 'noticed'", (i,)).rowcount]
        con.executemany("INSERT INTO events (case_id, at, kind, detail) VALUES (?, ?, 'resolved', ?)",
                        [(i, at, json.dumps({"via": via})) for i in moved])
    return moved


def escalate(case_ids: list[int], via: str) -> tuple[list[int], str | None, list[dict]]:
    """Noticed cases still up after the deadline move to escalated, each with an events row holding the
    FTC complaint packet and form answers. Nothing is filed: TakeItDown.ftc.gov is a web form a person submits."""
    at = _now()
    with closing(db()) as con, con:
        moved = [i for i in case_ids
                 if con.execute("UPDATE cases SET state = 'escalated' WHERE id = ? AND state = 'noticed'", (i,)).rowcount]
        if not moved:
            return [], None, []
        packet = ftc_packet(con, moved, at)
        urls = [r[0] for r in con.execute(f"SELECT DISTINCT url FROM cases WHERE id IN ({','.join('?' * len(moved))}) "
                                          "ORDER BY url", moved)]
        form = ftc_form(con, moved, packet, urls, at)
        detail = json.dumps({"via": via, "still_up": urls, "dry_run": dry_run(),
                             "ftc_url": channels.escalation()["ftc"], "ftc_packet": packet, "ftc_form": form})
        con.executemany("INSERT INTO events (case_id, at, kind, detail) VALUES (?, ?, 'escalated', ?)",
                        [(i, at, detail) for i in moved])
    return moved, packet, form


TIMELINE = {
    "found": "Found on {platform} by Six Eyes",
    "confirmed": "{user} confirmed it depicts them and was published without consent",
    "notice": "TAKE IT DOWN removal request made to {platform}",
    "timewarp": "Test only: the 48-hour deadline was moved to now",
    "resolved": "Removed from {platform}",
}
GENERATORS = {"real": "looks like a real photo", "swap": "face swap", "inpaint": "edited region",
              "txt2img": "generated from a text prompt", "vae-decoded": "passed through an AI image model"}


def ftc_packet(con: sqlite3.Connection, case_ids: list[int], checked_at: str) -> str:
    """Fill templates/ftc_packet.md for escalated cases from one clone cluster: their URLs and PDQ hashes,
    the cluster's timeline from events, the evidence scores and the notice copy."""
    marks = ",".join("?" * len(case_ids))
    up = con.execute(f"SELECT id, user_id, COALESCE(cluster_id, id), url, pdq, scores, sent_at, due_at FROM cases "
                     f"WHERE id IN ({marks}) ORDER BY id", case_ids).fetchall()
    _, user_id, cluster, first_url, *_ = up[0]
    name, signed_at = con.execute("SELECT name, signed_at FROM users WHERE id = ?", (user_id,)).fetchone()
    cluster_cases = con.execute("SELECT id, url, state FROM cases WHERE user_id = ? AND COALESCE(cluster_id, id) = ?",
                                (user_id, cluster)).fetchall()
    url_of = {i: u for i, u, _ in cluster_cases}
    channel = channels.channel_for(first_url or "")
    events = con.execute(f"SELECT case_id, at, kind, detail FROM events WHERE case_id IN "
                         f"({','.join('?' * len(url_of))}) ORDER BY at, rowid", list(url_of)).fetchall()

    timeline: dict[tuple[str, str], list[str]] = {}
    for case_id, at, kind, _ in events:
        if kind in TIMELINE:
            urls = timeline.setdefault((at, kind), [])
            if url_of[case_id] not in urls:
                urls.append(url_of[case_id])
    notice = next((json.loads(d) for _, _, kind, d in reversed(events) if kind == "notice"), {"packets": []})
    accounts = {_account(u) for _, _, _, u, *_ in up}

    def evidence(url: str, scores: dict) -> str:
        recon, c2pa = scores.get("recon_score"), scores.get("c2pa")
        return (f"- {url}: face match {_pct(scores.get('face_sim', 0))} with {name}'s enrollment selfies; "
                f"likely AI-generated or altered {_pct(scores.get('fake_prob', 0))} "
                f"({GENERATORS.get(scores.get('generator_guess'), scores.get('generator_guess', 'unknown'))}); "
                f"nudity {_pct(scores.get('nsfw', 0))}; reconstruction distance "
                f"{'not measured' if recon is None else f'{recon:.3f}'}; Content Credentials "
                f"{'none' if c2pa is None else 'valid' if c2pa else 'invalid'}.")

    removed = sorted({u for _, u, state in cluster_cases if state == "resolved"})
    return (channels.TEMPLATES / "ftc_packet.md").read_text().strip().format(
        generated_at=_when(checked_at), case_id=notice.get("via_case", up[0][0]), ftc_url=channels.escalation()["ftc"],
        platform_name=channel.name, request_url=channel.url or channel.name,
        signer_name=name, user_name=name,
        authorization_date=_when(signed_at, time=False), contact_email="",
        notice_date=_when(min(r[6] for r in up)), due_date=_when(min(r[7] for r in up)), checked_date=_when(checked_at),
        still_up="\n".join(f"- {u} (posted by @{_account(u)}), image PDQ hash {pdq or 'not computed'}"
                           for _, _, _, u, pdq, *_ in up),
        removed=", ".join(removed) or "none",
        evidence="\n".join(evidence(u, json.loads(s)) for _, _, _, u, _, s, *_ in up),
        timeline="\n".join(f"- {_when(at)}: {TIMELINE[kind].format(platform=channel.name, user=name)}: {', '.join(urls)}"
                           for (at, kind), urls in timeline.items()),
        notice_copy="\n\n----------\n\n".join(p["notice"] for p in notice["packets"]
                                              if p["reported_username"] in accounts) or "(no copy on record)")


def ftc_form(con: sqlite3.Connection, case_ids: list[int], packet: str, urls: list[str], checked_at: str) -> list[dict]:
    """Answers that follow TakeItDown.ftc.gov screen by screen (get-started → report)."""
    marks = ",".join("?" * len(case_ids))
    rows = con.execute(f"SELECT user_id, url, sent_at, due_at, scores FROM cases WHERE id IN ({marks}) ORDER BY url",
                       case_ids).fetchall()
    user_id, _, sent_at, due_at, _ = rows[0]
    name = con.execute("SELECT name FROM users WHERE id = ?", (user_id,)).fetchone()[0]
    channel = channels.channel_for(rows[0][1] or "")
    by_url = {url: (sent, json.loads(scores) if isinstance(scores, str) else scores or {})
              for _, url, sent, _, scores in rows}
    listed, extra = (urls or [r[1] for r in rows])[:5], max(0, len(urls or rows) - 5)
    n = len(listed)

    def field(screen: str, question: str, value: str, *, required: bool = True, type: str = "text") -> dict:
        return {"screen": screen, "question": question, "required": required, "type": type, "value": value}

    out = [
        field("Get started", "Choose one",
              "I asked the platform to remove an intimate photo or video. The platform didn't take it "
              "(or duplicates) down within 48 hours.", type="select"),
        field("About the platform", "What platform are you reporting?", channel.name),
    ]
    for i, url in enumerate(listed, 1):
        screen = f"Photo or video {i} of {n}" if n > 1 else "About the photo or video"
        sent, scores = by_url.get(url, (sent_at, {}))
        fake = scores.get("fake_prob")
        ai = "Yes" if isinstance(fake, (int, float)) and fake >= 0.5 else "I don't know"
        out += [
            field(screen, "When did you report this intimate photo or video to the platform?",
                  _us_date(sent or sent_at)),
            field(screen, "How did you contact the platform?",
                  "I used the platform's reporting process", type="select"),
            field(screen, "Enter the link where the intimate photo or video is posted on the platform", url),
            field(screen, "URL of image (if applicable)",
                  "Leave blank — Six Eyes does not store the image file.", required=False, type="skip"),
            field(screen, "Other location details",
                  f"Posted by @{_account(url)} on {channel.name}."),
            field(screen, "Platform report or ticket number (if applicable)",
                  "Leave blank — we don't have a ticket number from X.", required=False, type="skip"),
            field(screen, "Did the platform eventually remove the intimate photo or video?", "No", type="select"),
            field(screen, "Was the photo or video created or altered with AI or other technology?", ai, type="select"),
            field(screen, "Was the person shown in the photo or video under 18 when it was taken?",
                  "No", type="select"),
            field(screen, "Do you want to report a different intimate photo or video posted on this platform?",
                  "Yes" if i < n else "No", type="select"),
        ]
    more = (f" {extra} more URLs are in the complaint packet; file another report if this one only takes five."
            if extra else "")
    experience = (f"On {_when(sent_at)}, I used {channel.name}'s reporting process to file a TAKE IT DOWN "
                  f"request (form: {channel.url or channel.name}). The Act required removal within 48 hours, "
                  f"by {_when(due_at)}. On {_when(checked_at)} these posts were still up.{more}")
    out += [
        field("Share more about the platform",
              "Please share more about the platform’s response, whether duplicates were removed, and if you’ve "
              "filed previous reports.", experience, type="textarea"),
        field("Your information", "Who is reporting?",
              f"You — type your name ({name}). You are reporting for yourself, the person shown.", type="you"),
        field("Your information", "Contact email",
              "Type an email you check. The FTC may use it to follow up.", required=False, type="you"),
        field("If they ask for more detail", "Full complaint packet", packet, required=False, type="textarea"),
        field("Review and submit", "Submit the report?",
              "Only submit if this is a real case. Don't file a test report — our test posts were shared with consent."
              if dry_run() else "Review the answers, then submit.", type="select"),
    ]
    return out


# ponytail: no auth yet, so anyone with a case id can time-warp it. Test mode only, so nothing real escalates.
@app.post("/timewarp/{case_id}")
def timewarp(case_id: int):
    """Demo: end a noticed case's 48 hours now, then recheck it right away."""
    if not dry_run():
        raise HTTPException(403, "Time-warp only works in test mode (DRY_RUN=1).")
    with closing(db()) as con, con:
        case = con.execute("SELECT user_id, COALESCE(cluster_id, id), state, due_at FROM cases WHERE id = ?",
                           (case_id,)).fetchone()
        if not case:
            raise HTTPException(404, f"No case with id {case_id}.")
        user_id, cluster, state, due_at = case
        if state != "noticed":
            raise HTTPException(409, f"Case {case_id} is {state}; only noticed cases have a deadline to skip.")
        ids = [r[0] for r in con.execute("SELECT id FROM cases WHERE user_id = ? AND COALESCE(cluster_id, id) = ? "
                                         "AND state = 'noticed'", (user_id, cluster))]
        now = _now()
        con.executemany("UPDATE cases SET due_at = ? WHERE id = ?", [(now, i) for i in ids])
        con.executemany("INSERT INTO events (case_id, at, kind, detail) VALUES (?, ?, 'timewarp', ?)",
                        [(i, now, json.dumps({"was_due_at": due_at})) for i in ids])
    if caseworker.enabled():
        caseworker.skip_deadline(case_id)
        settled = caseworker.wait_settled(ids)
        if not settled:
            raise HTTPException(503, "Couldn't check the posts on X just now (is X_BEARER_TOKEN set?). Try again.")
        return {"case_id": case_id, **settled}
    settled = [s for s in recheck(user_id) if set(s["resolved"] + s["escalated"]) & set(ids)]
    if not settled:
        raise HTTPException(503, "Couldn't check the posts on X just now (is X_BEARER_TOKEN set?). Try again.")
    return {"case_id": case_id, **settled[0]}


# Modal calls this hourly too, so a Factory timer that died with the container still wakes.
@app.post("/recheck")
def recheck_all():
    """Recheck every user's cases past their deadline (or wake Factory to do it)."""
    return {"settled": recheck()}


@app.on_event("startup")
def restore_caseworker_timers():
    """Arm Factory deadline timers for noticed cases after a restart."""
    caseworker.restore()


def tell_user(user_id: int, case_id: int, text: str, link: str | None = None) -> None:
    """Text the user through Photon (if they enrolled a number) and keep a copy on the case."""
    with closing(db()) as con, con:
        phone = con.execute("SELECT phone FROM users WHERE id = ?", (user_id,)).fetchone()
        phone = phone[0] if phone else None
        con.execute("INSERT INTO messages (user_id, case_id, at, text, link) VALUES (?, ?, ?, ?, ?)",
                    (user_id, case_id, _now(), text, link))
    if not phone:
        log.info("user %s has no phone; case %s update was not texted", user_id, case_id)
        return
    try:
        msg_id = photon.send(phone, text)
        if msg_id:
            log.info("Photon text %s for case %s", msg_id, case_id)
        elif not photon.enabled():
            log.info("no Photon credentials; case %s update was not texted", case_id)
        else:
            log.warning("Photon did not send the text for case %s", case_id)
    except httpx.HTTPError:
        log.exception("Photon text for case %s failed", case_id)


def app_url() -> str:
    """Public site origin used in texts. Default is the Vercel deploy."""
    return (os.environ.get("APP_URL") or "https://six-eyes-henna.vercel.app").rstrip("/")


def dashboard_url(user_id: int) -> str:
    return f"{app_url()}/dashboard?user={user_id}"


def _settled_notice(user_id: int, resolved: list[int], escalated: list[int]) -> tuple[str, str | None]:
    """SMS after a deadline check: what happened, and where to escalate if anything is still up."""
    ftc = channels.escalation()["ftc"]
    gone, up = len(resolved), len(escalated)
    if up:
        still = (f"{up} {'post is' if up == 1 else 'posts are'} still up {deadline_words()} after we asked X "
                 f"to take them down")
        taken = f"{gone} {'was' if gone == 1 else 'were'} taken down. " if gone else ""
        dash = dashboard_url(user_id)
        text = (f"Six Eyes: {taken}{still}. File with the FTC: {ftc}  "
                f"Answers to paste: {dash}  "
                f"Open the form and fill in your own name and email. You don't need to describe the images.")
        if dry_run():
            text = "Test: " + text + " Don't file this — our test posts were shared with consent."
        return text, ftc
    n = gone
    text = f"Six Eyes: X took down {n} {'post' if n == 1 else 'posts'} we reported. Nothing else you need to do."
    return ("Test: " + text) if dry_run() else text, None


def open_escalations(user_id: int) -> list[dict]:
    """Still-up clusters already escalated, so the dashboard can keep showing the FTC packet and form link."""
    with closing(db()) as con:
        rows = con.execute("""
            SELECT c.id, COALESCE(c.cluster_id, c.id), e.detail FROM cases c
            JOIN events e ON e.rowid = (
                SELECT MAX(rowid) FROM events WHERE case_id = c.id AND kind = 'escalated')
            WHERE c.user_id = ? AND c.state = 'escalated' ORDER BY c.id""", (user_id,)).fetchall()
        by_cluster: dict[int, dict] = {}
        for cid, cluster, detail in rows:
            g = by_cluster.setdefault(cluster, {"ids": [], "detail": json.loads(detail)})
            g["ids"].append(cid)
        out = []
        for g in by_cluster.values():
            d = g["detail"]
            packet, urls = d.get("ftc_packet") or "", d.get("still_up") or []
            try:
                form = ftc_form(con, g["ids"], packet, urls, _now()) if packet or urls else d.get("ftc_form")
            except (IndexError, TypeError, ValueError):
                form = d.get("ftc_form")
            out.append({"case_id": g["ids"][0], "resolved": [], "escalated": g["ids"],
                        "ftc_packet": packet or None, "ftc_url": d.get("ftc_url"),
                        "ftc_form": form, "dry_run": d.get("dry_run")})
    return out


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _when(iso: str, time: bool = True) -> str:
    d = datetime.fromisoformat(iso).astimezone(timezone.utc)
    return f"{d:%B} {d.day}, {d:%Y}" + (f", {d:%H:%M} UTC" if time else "")


def _us_date(iso: str) -> str:
    """mm/dd/yyyy, the date format TakeItDown.ftc.gov asks for."""
    return datetime.fromisoformat(iso).astimezone(timezone.utc).strftime("%m/%d/%Y")


def _pct(n: float) -> str:
    return f"{round(n * 100)}%"


def _account(url: str) -> str:
    """The posting account in an x.com post URL, without "@"."""
    return (urlsplit(url or "").path.split("/") + ["", ""])[1]


# ponytail: no auth yet, so anyone with a user id can read that user's updates. Add sessions before real users.
@app.get("/messages/{user_id}")
def list_messages(user_id: int):
    """The case worker's updates to the user, newest first."""
    with closing(db()) as con:
        if not con.execute("SELECT 1 FROM users WHERE id = ?", (user_id,)).fetchone():
            raise HTTPException(404, f"No enrolled user with id {user_id}.")
        rows = con.execute("SELECT id, case_id, at, text, link FROM messages WHERE user_id = ? ORDER BY id DESC LIMIT 20",
                           (user_id,)).fetchall()
    return {"user_id": user_id,
            "messages": [{"id": i, "case_id": c, "at": at, "text": t, "link": link} for i, c, at, t, link in rows]}
