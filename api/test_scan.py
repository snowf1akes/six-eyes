"""Self-check for POST /scan/{user_id}. Run from the repo root:
    .venv/Scripts/python api/test_scan.py
X and analyze() are faked, so no network, X token or face model is needed.
"""
import json
import os
import sqlite3
import tempfile
import time

from datetime import datetime, timedelta

os.environ["DB_PATH"] = os.path.join(tempfile.mkdtemp(), "test.db")
os.environ.update(DRY_RUN="1", NOTICE_TEST_TO="me@test.example", SIGNER_NAME="Raymond Wang",
                  CONTACT_EMAIL="reports@test.example", CASEWORKER="0")  # test_caseworker.py covers it

import httpx  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import analyze  # noqa: E402
import channels  # noqa: E402
import photon  # noqa: E402
import x  # noqa: E402
from main import DB_PATH, app  # noqa: E402

ZERO, FAR = "0" * 64, "f" * 64
NEAR_ZERO = "0" * 56 + "ffffff7f"  # 31 bits from ZERO: a clone
author = x.Author(id="a1", username="burner", name="B", created_at="2026-10-15T00:00:00Z",
                  public_metrics={"followers_count": 1, "following_count": 500})
copycat = x.Author(id="a2", username="copycat", name="C")


def post(pid, *keys, text="look @sunny", by=author, **kw):
    return x.Post(id=pid, url=f"https://x.com/{by.username}/status/{pid}", text=text, author=by, **kw,
                  public_metrics={"like_count": 3}, media=[x.Media(key=k, type="photo", url=f"https://p/{k}") for k in keys])


# media url -> (pdq, face_sim); a missing url makes analyze() fail like an unreadable image would.
fake = {"https://p/a": (ZERO, 0.9), "https://p/b": (FAR, 0.9), "https://p/d": (NEAR_ZERO, 0.2),
        "https://p/s1": (FAR, 0.1), "https://p/s2": (ZERO, 0.8), "https://p/k": ("0f" * 32, 0.9)}
analyze.analyze = lambda url, emb: analyze.AnalyzeResult(
    face_sim=fake[url][1], frame_drift=None, fake_prob=0.9, generator_guess="swap",
    recon_score=0.03, c2pa=None, nsfw=0.1, pdq=fake[url][0])
search_results, keyword_results, sweep_results = [], [], []
since_ids, keyword_since_ids, sweep_since_ids = [], [], []


def fake_search(name, handles, since_id=None, terms=None):
    if terms:  # name plus a deepfake term
        return keyword_since_ids.append(since_id) or keyword_results
    return since_ids.append(since_id) or search_results


def raises(exc):
    def f(*a, **k):
        raise exc
    return f


x.search = fake_search
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
search_results[:] = [post("10", "a", "b"), post("12", "d", by=copycat), post("11", "missing")]
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

# Name plus a deepfake term: labelled "keyword" even though the plain name search returns it too,
# flagged in text and hashtags, and it bounds both name queries next time.
assert set(keyword_since_ids) == {None}
keyword_results[:] = [post("13", "k", text="Sunny Le deepfake lol", entities={"hashtags": [{"tag": "AIgenerated"}]})]
search_results[:] = keyword_results[:]
assert client.post(f"/scan/{user_id}").json()["cases_found"] == 1
kw = json.loads(cases()[-1][6])
assert kw["source"] == "keyword", kw["source"]
assert kw["signals"]["keywords"] == ["deepfake"] and kw["signals"]["hashtags"] == ["aigenerated"], kw["signals"]
client.post(f"/scan/{user_id}")
assert keyword_since_ids[-1] == "13" and since_ids[-1] == "13", (keyword_since_ids, since_ids)
keyword_results[:] = search_results[:] = []

# Dashboard list: highest priority first; a stranger's face (0.2) ranks at 0; clones count posts.
r = client.get(f"/cases/{user_id}")
assert r.status_code == 200, r.text
listed = r.json()["cases"]
assert len(listed) == len(cases()) and all(c["state"] == "found" for c in listed)
prios = [c["priority"] for c in listed]
assert prios == sorted(prios, reverse=True) and all(0 <= p <= 1 for p in prios), prios
by_post = {c["post_id"]: c for c in listed}
assert by_post["12"]["priority"] == 0 and by_post["10"]["priority"] > 0
image_a = next(c for c in listed if c["id"] == rows[0][0])
assert by_post["13"]["priority"] > image_a["priority"], "same face match as post 10's image a and fewer copies, but says deepfake"
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
channels.send_email = raises(RuntimeError("Email isn't set up."))  # confirming works even if filing fails
r = client.post(f"/cases/{case_91}/confirm", json=both)
assert r.status_code == 200, r.text
cluster = {c[0] for c in cases() if c[3] in (ZERO, NEAR_ZERO)}
assert set(r.json()["confirmed"]) == cluster and len(cluster) == 3, (r.json(), cluster)
assert r.json()["notice"] is None and r.json()["notice_error"] == "Email isn't set up.", r.json()
states = {c[3]: c[5] for c in cases()}
assert states[ZERO] == states[NEAR_ZERO] == "confirmed" and states[FAR] == "found", states
assert {c["post_id"] for c in client.get(f"/cases/{user_id}?state=confirmed").json()["cases"]} == {"10", "12", "91"}
events = con.execute("SELECT detail FROM events WHERE kind = 'confirmed'").fetchall()
assert len(events) == 3 and all(json.loads(d)["not_consensual"] for (d,) in events)
assert client.post(f"/cases/{case_91}/confirm", json=both).json()["confirmed"] == [], "already confirmed"

# Notice, dry run: the retry emails the test inbox one packet per posting account (X's form reports one
# username at a time), covering every post in the cluster, and moves the cluster to noticed for 48 hours.
sent = []
channels.send_email = lambda to, subject, body, reply_to=None: sent.append((to, subject, body, reply_to))
image_b = next(c[0] for c in cases() if c[3] == FAR)
assert client.post(f"/cases/{image_b}/notice").status_code == 409, "image b is only found"
r = client.post(f"/cases/{case_91}/notice")
assert r.status_code == 200, r.text
n = r.json()
assert n["dry_run"] and not n["intimate"] and n["delivered_to"] == "me@test.example" and n["platform"] == "X"
assert set(n["case_ids"]) == cluster and n["form_url"].endswith("/take-it-down/authorized-rep")
packets = {p["reported_username"]: p for p in n["packets"]}
assert set(packets) == {"burner", "copycat"}, packets.keys()
assert packets["burner"]["content_urls"] == ["https://x.com/burner/status/10", "https://x.com/burner/status/91"]
assert packets["copycat"]["content_urls"] == ["https://x.com/copycat/status/12"]
text = packets["burner"]["notice"]
assert "Account that published the content: @burner" in text and "/s/ Raymond Wang" in text
assert "Authorized representative of Sunny Le" in text and "reports@test.example" in text
assert "digital forgery" in text and "{" not in text, text  # fake_prob 0.9; every placeholder filled
form = {f["question"]: f["value"] for f in packets["burner"]["form"]}
assert form["Username of the account you are reporting"] == "burner"
assert form["Your email address"] == "reports@test.example"
details = form["Please provide more details about what's happening"]  # the short details.md, not the notice
assert details.startswith("I am the authorized representative of Sunny Le") and "digital forgery" in details
assert details.endswith(f"Case reference: {case_91}.") and len(details) < 600 and "{" not in details, details
assert [b["check"] for b in packets["burner"]["checkboxes"]] == [False, True, True]
assert [f["type"] for f in packets["burner"]["form"] if f["preset"]] == ["select", "select"]  # the link picks both
[(to, subject, body, reply_to)] = sent
assert to == "me@test.example" and subject == "[Six Eyes test] 2 TAKE IT DOWN reports for X"
assert body.startswith("TEST ONLY") and "Report @copycat on X" in body and reply_to == "reports@test.example"
assert "authorized-rep" in body and "the form link already picks this" in body
noticed = con.execute("SELECT state, sent_at, due_at FROM cases WHERE id IN (%s)" % ",".join(map(str, cluster))).fetchall()
assert {s for s, _, _ in noticed} == {"noticed"}
assert datetime.fromisoformat(noticed[0][2]) - datetime.fromisoformat(noticed[0][1]) == timedelta(hours=48)
notice_events = con.execute("SELECT detail FROM events WHERE kind = 'notice'").fetchall()
assert len(notice_events) == 3 and json.loads(notice_events[0][0])["dry_run"] is True
assert client.post(f"/cases/{case_91}/notice").status_code == 409, "already noticed"
assert client.post("/cases/999/notice").status_code == 404

# Live (DRY_RUN=0): refused unless the content looks intimate; then the packet comes back, nothing is emailed.
os.environ["DRY_RUN"] = "0"
sent.clear()
r = client.post(f"/cases/{image_b}/confirm", json=both).json()
assert r["notice"] is None and "doesn't look like intimate content" in r["notice_error"], r
con.execute("UPDATE cases SET scores = json_set(scores, '$.nsfw', 0.9) WHERE id = ?", (image_b,))
con.commit()
r = client.post(f"/cases/{image_b}/notice")
assert r.status_code == 200 and r.json()["intimate"] and r.json()["delivered_to"] is None, r.text
assert sent == [], "nothing is emailed live; X's form is filed by a person"
os.environ["DRY_RUN"] = "1"
del os.environ["SIGNER_NAME"]  # no named signer: refused rather than signed by nobody
r = client.post(f"/cases/{by_post['13']['id']}/confirm", json=both).json()
assert r["notice"] is None and r["notice_error"].startswith("Set SIGNER_NAME"), r
os.environ["SIGNER_NAME"] = "Raymond Wang"


def state_of(case_id):
    return con.execute("SELECT state FROM cases WHERE id = ?", (case_id,)).fetchone()[0]


# Recheck: loading the dashboard settles noticed cases past their deadline, and nothing before it.
live_calls = []
x.is_live = lambda post_id: live_calls.append(post_id) or post_id != "10"  # post 10 was taken down
client.get(f"/cases/{user_id}")
assert live_calls == [], "nothing is due yet, so X isn't asked"

# Time-warp (test mode only) ends case 91's 48 hours now and rechecks its cluster: post 10's image a is gone,
# so it's resolved; posts 12 and 91 are still up, so they're escalated with the FTC packet, and the user is texted.
con.execute("UPDATE users SET phone = ? WHERE id = ?", ("617-555-0123", user_id))
con.commit()
texts = []
photon.send = lambda phone, text: texts.append((phone, text)) or "pho_msg_1"
case_10a, case_12 = rows[0][0], clone[0]
assert client.post("/timewarp/999").status_code == 404
assert client.post(f"/timewarp/{by_post['13']['id']}").status_code == 409, "confirmed, not noticed"
os.environ["DRY_RUN"] = "0"
assert client.post(f"/timewarp/{case_91}").status_code == 403
os.environ["DRY_RUN"] = "1"
r = client.post(f"/timewarp/{case_91}")
assert r.status_code == 200, r.text
w = r.json()
assert w["resolved"] == [case_10a] and w["escalated"] == [case_12, case_91], w
assert w["ftc_url"] == "https://takeitdown.ftc.gov"
form = w["ftc_form"]
assert form[0]["type"] == "select" and "didn't take it" in form[0]["value"]
assert any(f["value"] == "X" and "platform" in f["question"].lower() for f in form)
assert any(f["value"] == "https://x.com/copycat/status/12" for f in form)
assert any(f["value"] == "https://x.com/burner/status/91" for f in form)
assert "takeitdown.ftc.gov" in form[-2]["value"]  # complaint packet, before the submit warning
msg = client.get(f"/messages/{user_id}").json()["messages"][0]
assert msg["link"] == "https://takeitdown.ftc.gov"
assert texts == [("617-555-0123", msg["text"])]
assert "https://takeitdown.ftc.gov" in msg["text"]
assert f"https://six-eyes-henna.vercel.app/dashboard?user={user_id}" in msg["text"]
assert "48 hours" in msg["text"] and "FTC" in msg["text"] and "authorized representative" in msg["text"]
listed = client.get(f"/cases/{user_id}").json()
assert listed["escalations"] and listed["escalations"][0]["ftc_url"] == "https://takeitdown.ftc.gov"
assert sorted(live_calls) == ["10", "12", "91"], "one X check per post"
assert [state_of(i) for i in (case_10a, case_12, case_91, image_b)] == ["resolved", "escalated", "escalated", "noticed"]
packet = w["ftc_packet"]
assert "File at: https://takeitdown.ftc.gov" in packet and "{" not in packet, packet
assert packet.count(f"Case reference: {case_91}") == 3, "the packet and both notice copies agree"
assert f"- https://x.com/copycat/status/12 (posted by @copycat), image PDQ hash {NEAR_ZERO}" in packet
assert f"- https://x.com/burner/status/91 (posted by @burner), image PDQ hash {ZERO}" in packet
assert "Posts from the same request that X did remove: https://x.com/burner/status/10" in packet
assert "https://x.com/copycat/status/12: face match 20%" in packet and "altered 90% (face swap)" in packet
for step in ("Found on X by Six Eyes", "Sunny Le confirmed it depicts them", "TAKE IT DOWN removal request made to X",
             "Test only: the 48-hour deadline was moved to now", "Removed from X: https://x.com/burner/status/10"):
    assert step in packet, step
assert "Account that published the content: @burner" in packet and "@copycat" in packet, "both notices copied"
assert "Raymond Wang, authorized representative of Sunny Le" in packet
escalated = con.execute("SELECT case_id, detail FROM events WHERE kind = 'escalated' ORDER BY case_id").fetchall()
assert [i for i, _ in escalated] == [case_12, case_91]
assert json.loads(escalated[0][1])["ftc_packet"] == packet and json.loads(escalated[0][1])["still_up"] == [
    "https://x.com/burner/status/91", "https://x.com/copycat/status/12"]
assert client.post(f"/timewarp/{case_91}").status_code == 409, "already settled"

# Past the deadline for real (image b, the live notice): the next dashboard load settles it.
con.execute("UPDATE cases SET due_at = '2026-01-01T00:00:00+00:00' WHERE id = ?", (image_b,))
con.commit()
live_calls.clear()
r = client.get(f"/cases/{user_id}")
assert r.status_code == 200 and r.json()["settled"][0]["resolved"] == [image_b], r.text
assert live_calls == ["10"] and state_of(image_b) == "resolved"

# X can't be checked: the dashboard still loads, the case waits, and the time-warp says why.
case_13 = by_post["13"]["id"]
assert client.post(f"/cases/{case_13}/notice").status_code == 200
x.is_live = raises(RuntimeError("X_BEARER_TOKEN is not set."))
assert client.post(f"/timewarp/{case_13}").status_code == 503
assert client.get(f"/cases/{user_id}").status_code == 200 and state_of(case_13) == "noticed"
x.is_live = lambda post_id: True
assert client.post("/recheck").json()["settled"][0]["escalated"] == [case_13]
assert state_of(case_13) == "escalated"



def background_scan():
    """Start a background scan (the dashboard's mode) and poll GET /scan until it finishes."""
    r = client.post(f"/scan/{user_id}?background=true")
    assert r.status_code == 200 and r.json()["user_id"] == user_id, r.text
    for _ in range(200):
        status = client.get(f"/scan/{user_id}").json()
        if not status["running"]:
            return status
        time.sleep(0.05)
    raise AssertionError("background scan never finished")


# Background scan: returns at once, saves cases as it goes, and reports progress until done.
search_results[:] = [post("14", "a")]
assert background_scan() == {"user_id": user_id, "posts_checked": 1, "cases_found": 1, "media_failed": 0,
                             "running": False, "error": None}
assert cases()[-1][1] == "14"

# Errors.
assert client.post("/scan/999").status_code == 404
assert client.post("/scan/999?background=true").status_code == 404

req = httpx.Request("GET", "https://api.x.com")
for exc, code in [(httpx.HTTPStatusError("", request=req, response=httpx.Response(429, request=req)), 429),
                  (httpx.HTTPStatusError("", request=req, response=httpx.Response(401, request=req)), 502),
                  (httpx.ConnectError("down", request=req), 502),
                  (RuntimeError("X_BEARER_TOKEN is not set."), 503),
                  (ValueError("Need a name or at least one valid X handle to search."), 422)]:
    x.search = raises(exc)
    r = client.post(f"/scan/{user_id}")
    assert r.status_code == code, (exc, r.status_code, r.text)
assert background_scan()["error"] == "Need a name or at least one valid X handle to search.", "errors show in status"

print("scan self-check passed")
