"""Self-check for the case worker (caseworker.py). Run from the repo root:
    .venv/Scripts/python api/test_caseworker.py
X and the Droid session are faked, so no network, droid CLI or Factory key is needed. The fake session
calls the tools through their real MCP handlers, the way Droid does, following a fixed script.
"""
import json
import os
import sqlite3
import tempfile
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

os.environ["DB_PATH"] = os.path.join(tempfile.mkdtemp(), "test.db")
os.environ.update(DRY_RUN="1", SIGNER_NAME="Raymond Wang", CONTACT_EMAIL="reports@test.example", CASEWORKER="0")
for k in ("NOTICE_TEST_TO", "DEMO_DEADLINE_MINUTES", "DROID_PATH"):
    os.environ.pop(k, None)

from droid_sdk import (Autonomy, ExecuteAction, McpToolAction, PermissionOption, PermissionRequest,  # noqa: E402
                       ToolConfirmationOutcome, ToolUseBlock)
from fastapi.testclient import TestClient  # noqa: E402

import caseworker  # noqa: E402
import x  # noqa: E402
from caseworker import ToolError  # noqa: E402
from main import DB_PATH, app  # noqa: E402

con = sqlite3.connect(DB_PATH)
client = TestClient(app)
SCORES = json.dumps({"face_sim": 0.9, "fake_prob": 0.9, "nsfw": 0.1})


def add_user(name):
    uid = con.execute("INSERT INTO users (name, handles, embedding, signed_at) VALUES (?, '[]', '[]', ?)",
                      (name, "2026-10-17T00:00:00+00:00")).lastrowid
    con.commit()
    return uid


def add_case(user_id, post_id, account, cluster=None):
    cid = con.execute("INSERT INTO cases (user_id, post_id, url, cluster_id, scores) VALUES (?, ?, ?, ?, ?)",
                      (user_id, post_id, f"https://x.com/{account}/status/{post_id}", cluster, SCORES)).lastrowid
    con.execute("UPDATE cases SET cluster_id = ? WHERE id = ? AND cluster_id IS NULL", (cid, cid))
    con.commit()
    return cid


def states():
    return dict(con.execute("SELECT post_id, state FROM cases WHERE user_id = ?", (user,)).fetchall())


def fails(f, *args, match):
    try:
        f(*args)
    except ToolError as e:
        assert match in str(e), e
    else:
        raise AssertionError(f"{f.__name__}{args} should have failed with {match!r}")


user, other = add_user("Sunny Le"), add_user("Someone Else")
root = add_case(user, "10", "burner")
add_case(user, "11", "burner", root)
add_case(user, "12", "copycat", root)  # same image, another account
stranger = add_case(other, "20", "someone")
both = {"is_me": True, "not_consensual": True}

# Confirming drafts the reports at once (the dashboard shows them) and starts the case worker.
started = []
real_start, caseworker.start = caseworker.start, lambda case_id, reason: started.append((case_id, reason)) or True
r = client.post(f"/cases/{root}/confirm", json=both).json()
caseworker.start = real_start
assert r["caseworker"] is True and started == [(root, "confirmed")], (r, started)
assert r["notice"] and set(states().values()) == {"noticed"}, r

# Off (CASEWORKER=0, or no droid CLI): nothing starts, and the endpoints say so.
assert caseworker.start(root, "confirmed") is False
assert client.post(f"/cases/{root}/work").status_code == 503
assert client.post("/cases/999/work").status_code == 404
assert client.post("/recheck").json() == {"settled": []}  # noticed, but the 48 hours haven't passed

# The tools, called directly. Drafting again is safe; the case worker can't touch another user's case.
run = caseworker.load(root, "confirmed")
assert "already made" in caseworker.draft_notice(run, root)
fails(caseworker.draft_notice, run, stranger, match="isn't part of this case")
fails(caseworker.check_live, run, root, match="check after that")  # no billed X check before the deadline
fails(caseworker.escalate, run, root, match="check_live first")
fails(caseworker.notify_user, run, other, "hi", match="only they can be notified")
fails(caseworker.notify_user, run, user, "   ", match="empty")
fails(caseworker.notify_user, run, user, "x" * 801, match="under 800")
assert caseworker.notify_user(run, user, "Test:  your report\nis ready.").startswith("Sent")
assert caseworker.notify_user(run, user, "Second update.").startswith("Sent")
fails(caseworker.notify_user, run, user, "Third.", match="already has")
r = client.get(f"/messages/{user}").json()["messages"]
assert [m["text"] for m in r] == ["Second update.", "Test: your report is ready."] and r[0]["case_id"] == root, r
assert client.get("/messages/999").status_code == 404

# A check the team asked for runs before the deadline, but never escalates.
checked = []
x.is_live = lambda post_id: checked.append(post_id) or True
run = caseworker.load(root, "check")
out = caseworker.check_live(run, root)
assert "not passed yet" in out and out.count("Still up") == 3 and sorted(checked) == ["10", "11", "12"], out
fails(caseworker.escalate, run, root, match="Not yet")
assert set(states().values()) == {"noticed"}


class FakeSession:
    """Stands in for droid_sdk.Session: runs `script` through the session's MCP tools."""
    script: list = []
    seen: dict = {}

    def __init__(self, cwd, model, config, interactions, runtime):
        FakeSession.seen = {"cwd": cwd, "model": model, "config": config, "allow": interactions.on_permission}
        self.tools = {t.name: t for t in config.mcp_servers[0].tools}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def stream(self, prompt, timeout):
        FakeSession.seen["prompt"] = prompt
        return FakeStream(self)


class FakeStream:
    def __init__(self, session):
        self.session, self.result, self.replies = session, None, []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.result:
            raise StopAsyncIteration
        for name, args in FakeSession.script:
            self.replies.append(await self.session.tools[name].handler(args))
        FakeSession.seen["replies"] = self.replies
        self.result = SimpleNamespace(subtype="success", session_id="session-1", text="Escalated two posts.")
        return self.result


def wait_idle():
    for _ in range(200):
        if not caseworker.RUNNING:
            return
        time.sleep(0.02)
    raise AssertionError("the case worker never finished")


# Dashboard load (GET /cases) settles clusters past their deadline: post 10 is gone so it's resolved,
# posts 11 and 12 are still up so they're escalated with an FTC packet, then the case worker tells the user.
os.environ.update(CASEWORKER="1", DROID_PATH="/usr/local/bin/droid")
caseworker.Session = FakeSession
assert client.get(f"/cases/{user}").json()["settled"] == [], "nothing is due yet, so X isn't asked"
past = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(timespec="seconds")
con.execute("UPDATE cases SET due_at = ? WHERE state = 'noticed'", (past,))
con.commit()
checked.clear()
x.is_live = lambda post_id: checked.append(post_id) or post_id != "10"
FakeSession.script = [("notify_user", {"user_id": user,
                       "text": "Test: two posts are still up, so Six Eyes is reporting them to the FTC."})]
r = client.get(f"/cases/{user}")
assert r.status_code == 200, r.text
s = r.json()["settled"]
assert len(s) == 1 and s[0]["resolved"] and s[0]["escalated"] and "takeitdown.ftc.gov" in (s[0]["ftc_packet"] or ""), s
wait_idle()
seen = FakeSession.seen
assert all(not reply.is_error for reply in seen["replies"]), [r.content for r in seen["replies"]]
assert sorted(checked) == ["10", "11", "12"], "one billed check per post per session"
assert states() == {"10": "resolved", "11": "escalated", "12": "escalated"}, states()
assert "just checked every post" in seen["prompt"] and "https://x.com/copycat/status/12" in seen["prompt"]
config = seen["config"]
assert config.system_prompt == caseworker.SYSTEM and config.autonomy is Autonomy.OFF and seen["model"] == "auto"
assert seen["cwd"] != os.getcwd() and not os.path.exists(seen["cwd"]), "Droid works in a throwaway empty folder"
kinds = dict(con.execute("SELECT kind, COUNT(*) FROM events GROUP BY kind").fetchall())
assert kinds["resolved"] == 1 and kinds["escalated"] == 2 and kinds["caseworker"] == 1, kinds
log = json.loads(con.execute("SELECT detail FROM events WHERE kind = 'caseworker'").fetchone()[0])
assert log["reason"] == "rechecked" and log["outcome"] == "success" and log["session_id"] == "session-1", log
assert log["tools"] == [f"notify_user({user}, 'Test: two posts are still up, so Six Eyes is reporting them to the FTC.', '')"], log
escalated = json.loads(con.execute("SELECT detail FROM events WHERE kind = 'escalated'").fetchone()[0])
assert escalated["still_up"] == ["https://x.com/burner/status/11", "https://x.com/copycat/status/12"] and escalated["dry_run"]
assert "ftc_packet" in escalated and "File at: https://takeitdown.ftc.gov" in escalated["ftc_packet"]
assert client.get(f"/messages/{user}").json()["messages"][0]["text"].startswith("Test: two posts")
assert client.get(f"/cases/{user}").json()["settled"] == [], "already settled, so X isn't asked again"

# Permissions: Droid may call the case tools, and nothing else.
allow = seen["allow"]
options = [PermissionOption("Yes", ToolConfirmationOutcome.PROCEED_ONCE), PermissionOption("No", ToolConfirmationOutcome.CANCEL)]
use = ToolUseBlock(id="t1", name="x", input={})
mine = McpToolAction(tool_use=use, tool_name="six-eyes___escalate", impact_level="high", server_name="six-eyes")
shell = ExecuteAction(tool_use=use, full_command="rm -rf /", command="rm")
theirs = McpToolAction(tool_use=use, tool_name="escalate", impact_level="high", server_name="other")
assert allow(PermissionRequest(actions=[mine], options=options)).selected_option is ToolConfirmationOutcome.PROCEED_ONCE
for actions in ([shell], [theirs], [mine, shell], []):
    assert allow(PermissionRequest(actions=actions, options=options)).selected_option is ToolConfirmationOutcome.CANCEL

# A failed session is logged and frees the cluster, so the next start can try again.
caseworker.Session = lambda **kw: (_ for _ in ()).throw(RuntimeError("droid crashed"))
assert caseworker.start(root, "rechecked")
wait_idle()
last = json.loads(con.execute("SELECT detail FROM events WHERE kind = 'caseworker' ORDER BY rowid DESC").fetchone()[0])
assert last["outcome"] == "error" and "droid crashed" in last["error"], last
caseworker.Session, FakeSession.script = FakeSession, []
assert caseworker.start(root, "rechecked")
wait_idle()

print("caseworker self-check passed")
