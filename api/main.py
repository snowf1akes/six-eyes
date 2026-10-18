"""Six Eyes API: SQLite tables, POST /enroll, POST /scan/{user_id}, GET /cases/{user_id},
POST /cases/{case_id}/confirm and POST /cases/{case_id}/notice.

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
from urllib.parse import urlsplit

import cv2
import httpx
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError

import analyze
import channels
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
NOTICE_DEADLINE = timedelta(hours=48)  # TAKE IT DOWN Act: platforms remove within 48 hours of a valid request
INTIMATE_MIN = 0.5  # NSFW score a live notice needs; the Act only covers intimate depictions (NOTES.md flag 3)
FORGERY_MIN = 0.5   # fake_prob from which the notice calls the image a digital forgery
log = logging.getLogger("six_eyes")
# user_id -> progress of the latest background scan. In memory: Modal runs one API container.
SCANS: dict[int, dict] = {}
SCAN_LOCK = threading.Lock()

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
    # File right away; if that fails the cases stay confirmed and POST /cases/{id}/notice retries.
    filed, error = None, None
    if ids:
        try:
            filed = notice(ids[0])
        except HTTPException as e:
            error = e.detail
    return {"confirmed": ids, "notice": filed, "notice_error": error}


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
    signer, contact = os.environ.get("SIGNER_NAME"), os.environ.get("CONTACT_EMAIL")
    if not (signer and contact):
        raise HTTPException(503, "Set SIGNER_NAME and CONTACT_EMAIL on the API: the team member who signs "
                                 "notices, and the team's reporting inbox.")
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
                depiction_kind=channels.FORGERY if forgery else channels.INTIMATE, signer_name=signer,
                signer_relationship=f"Authorized representative of {name}", contact_email=contact,
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
                                    f"for {channel.name}", body, reply_to=contact)
        except RuntimeError as e:
            raise HTTPException(503, str(e))
        except (smtplib.SMTPException, OSError):
            log.exception("test notice email for case %s failed", case_id)
            raise HTTPException(502, "Couldn't send the test email. Check the SMTP settings, then try again.")

    # ponytail: the 48 hours really start when someone submits X's form, not when the packet is made.
    at, due = now.isoformat(timespec="seconds"), (now + NOTICE_DEADLINE).isoformat(timespec="seconds")
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
