"""Self-check for POST /scan/{user_id}. Run from the repo root:
    .venv/Scripts/python api/test_scan.py
X and analyze() are faked, so no network, X token or face model is needed.
"""
import json
import os
import sqlite3
import tempfile

os.environ["DB_PATH"] = os.path.join(tempfile.mkdtemp(), "test.db")

import httpx  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import analyze  # noqa: E402
import x  # noqa: E402
from main import DB_PATH, app  # noqa: E402

ZERO, FAR = "0" * 64, "f" * 64
NEAR_ZERO = "0" * 56 + "ffffff7f"  # 31 bits from ZERO: a clone
author = x.Author(id="a1", username="burner", name="B", created_at="2026-10-15T00:00:00Z",
                  public_metrics={"followers_count": 1, "following_count": 500})


def post(pid, *keys, text="look @sunny"):
    return x.Post(id=pid, url=f"https://x.com/burner/status/{pid}", text=text, author=author,
                  public_metrics={"like_count": 3}, media=[x.Media(key=k, type="photo", url=f"https://p/{k}") for k in keys])


# media url -> (pdq, face_sim); a missing url makes analyze() fail like an unreadable image would.
fake = {"https://p/a": (ZERO, 0.9), "https://p/b": (FAR, 0.9), "https://p/d": (NEAR_ZERO, 0.2),
        "https://p/s1": (FAR, 0.1), "https://p/s2": (ZERO, 0.8)}
analyze.analyze = lambda url, emb: analyze.AnalyzeResult(
    face_sim=fake[url][1], frame_drift=None, fake_prob=0.9, generator_guess="swap",
    recon_score=0.03, c2pa=None, nsfw=0.1, pdq=fake[url][0])
search_results, sweep_results, since_ids, sweep_since_ids = [], [], [], []
x.search = lambda name, handles, since_id=None: since_ids.append(since_id) or search_results
x.sweep = lambda terms, limit=500, since_id=None: sweep_since_ids.append(since_id) or sweep_results

con = sqlite3.connect(DB_PATH)
user_id = con.execute("INSERT INTO users (name, handles, embedding, signed_at) VALUES (?, ?, ?, ?)",
                      ("Sunny Le", '["sunny"]', json.dumps([0.0] * 512), "2026-10-17T00:00:00+00:00")).lastrowid
con.commit()
client = TestClient(app)


def cases():
    return con.execute("SELECT id, post_id, author_id, pdq, cluster_id, state, scores FROM cases ORDER BY id").fetchall()


# First scan: post 10 has two images, post 11's image can't be analyzed.
search_results[:] = [post("10", "a", "b"), post("11", "missing")]
r = client.post(f"/scan/{user_id}")
assert r.status_code == 200, r.text
assert r.json() == {"user_id": user_id, "posts_checked": 2, "cases_found": 2, "media_failed": 1}, r.json()
rows = cases()
assert [(c[1], c[2], c[3], c[5]) for c in rows] == [("10", "a1", ZERO, "found"), ("10", "a1", FAR, "found")]
assert rows[0][4] == rows[0][0] and rows[1][4] == rows[1][0], "different images get their own clusters"
scores = json.loads(rows[0][6])
assert scores["source"] == "search" and scores["fake_prob"] == 0.9 and scores["media"]["url"] == "https://p/a"
assert scores["signals"]["follower_ratio"] == 0.002 and scores["signals"]["engagement"] == {"like": 3}
assert since_ids == [None]
assert con.execute("SELECT COUNT(*) FROM events WHERE kind = 'found'").fetchone()[0] == 2

# Second scan: post 10 is skipped, post 12 is a clone of image a (and saved despite a low face match,
# because the name search keeps everything), post 11 is retried.
search_results[:] = [post("10", "a", "b"), post("12", "d"), post("11", "missing")]
r = client.post(f"/scan/{user_id}")
assert r.json()["posts_checked"] == 2 and r.json()["cases_found"] == 1, r.json()
assert since_ids[-1] == "10", since_ids
clone = cases()[-1]
assert clone[1] == "12" and clone[4] == rows[0][0], "clone joins the first image's cluster"

# Sweep: only the post whose face matches becomes a case, and sweep cases don't move since_id.
search_results[:] = []
sweep_results[:] = [post("90", "s1", text="deepfake"), post("91", "s2", text="deepfake")]
r = client.post(f"/scan/{user_id}?sweep=true")
assert r.json()["cases_found"] == 1, r.json()
hit = cases()[-1]
assert hit[1] == "91" and json.loads(hit[6])["source"] == "sweep" and hit[4] == rows[0][0]
sweep_results[:] = []
client.post(f"/scan/{user_id}?sweep=true")
assert sweep_since_ids == [None, "91"], "the next sweep starts after the newest post already checked"
client.post(f"/scan/{user_id}")
assert since_ids[-1] == "12", since_ids
assert client.post(f"/scan/{user_id}").json()["cases_found"] == 0, "sweep is off by default"

# Dashboard list: highest priority first; a stranger's face (0.2) ranks at 0; clones count posts.
r = client.get(f"/cases/{user_id}")
assert r.status_code == 200, r.text
listed = r.json()["cases"]
assert len(listed) == len(cases()) and all(c["state"] == "found" for c in listed)
prios = [c["priority"] for c in listed]
assert prios == sorted(prios, reverse=True) and all(0 <= p <= 1 for p in prios), prios
by_post = {c["post_id"]: c for c in listed}
assert by_post["12"]["priority"] == 0 and by_post["10"]["priority"] > 0
assert by_post["91"]["copies"] == 3 and by_post["91"]["found_at"], "posts 10, 12 and 91 share image a"
assert by_post["91"]["scores"]["media"]["url"] == "https://p/s2"
assert client.get(f"/cases/{user_id}?state=bogus").status_code == 422
assert client.get("/cases/999").status_code == 404

# Confirm: both attestations required; the whole clone cluster (posts 10, 12, 91 share image a) moves,
# image b (post 10's second photo, its own cluster) stays found.
case_91 = by_post["91"]["id"]
for body in ({"is_me": True, "not_consensual": False}, {"is_me": False, "not_consensual": True}):
    assert client.post(f"/cases/{case_91}/confirm", json=body).status_code == 422
both = {"is_me": True, "not_consensual": True}
assert client.post("/cases/999/confirm", json=both).status_code == 404
r = client.post(f"/cases/{case_91}/confirm", json=both)
assert r.status_code == 200, r.text
cluster = {c[0] for c in cases() if c[3] in (ZERO, NEAR_ZERO)}
assert set(r.json()["confirmed"]) == cluster and len(cluster) == 3, (r.json(), cluster)
states = {c[3]: c[5] for c in cases()}
assert states[ZERO] == states[NEAR_ZERO] == "confirmed" and states[FAR] == "found", states
assert {c["post_id"] for c in client.get(f"/cases/{user_id}?state=confirmed").json()["cases"]} == {"10", "12", "91"}
events = con.execute("SELECT detail FROM events WHERE kind = 'confirmed'").fetchall()
assert len(events) == 3 and all(json.loads(d)["not_consensual"] for (d,) in events)
assert client.post(f"/cases/{case_91}/confirm", json=both).json() == {"confirmed": []}, "already confirmed"

# Errors.
assert client.post("/scan/999").status_code == 404


def raises(exc):
    def f(*a, **k):
        raise exc
    return f


req = httpx.Request("GET", "https://api.x.com")
for exc, code in [(httpx.HTTPStatusError("", request=req, response=httpx.Response(429, request=req)), 429),
                  (httpx.HTTPStatusError("", request=req, response=httpx.Response(401, request=req)), 502),
                  (httpx.ConnectError("down", request=req), 502),
                  (RuntimeError("X_BEARER_TOKEN is not set."), 503),
                  (ValueError("Need a name or at least one valid X handle to search."), 422)]:
    x.search = raises(exc)
    r = client.post(f"/scan/{user_id}")
    assert r.status_code == code, (exc, r.status_code, r.text)

print("scan self-check passed")
