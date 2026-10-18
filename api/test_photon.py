"""Self-check for api/photon.py. Run from the repo root:
    .venv/Scripts/python api/test_photon.py
"""
import os

os.environ.pop("PHOTON_API_KEY", None)
os.environ.pop("PHOTON_PROJECT_ID", None)
os.environ.pop("PHOTON_AGENT_USER_ID", None)

import httpx  # noqa: E402

import photon  # noqa: E402

assert photon.e164("617-555-0123") == "+16175550123"
assert photon.e164("(617) 555-0123") == "+16175550123"
assert photon.e164("16175550123") == "+16175550123"
assert photon.e164("+44 7911 123456") == "+447911123456"
assert photon.e164("not a number") is None
assert photon.e164("") is None
assert photon.send("6175550123", "hello") is None, "no Photon credentials: don't send"

os.environ.update(PHOTON_API_KEY="pho_sk_test", PHOTON_PROJECT_ID="pho_prj_test",
                  PHOTON_AGENT_USER_ID="pho_usr_agent", PHOTON_PLATFORM="sms")
calls = []


def fake_post(url, json=None, headers=None, timeout=None):
    calls.append((url, json, headers["idempotency-key"]))
    if url.endswith("/users"):
        body = {"id": "pho_usr_user"}
    elif url.endswith("/spaces"):
        body = {"id": "pho_spc_1"}
    else:
        body = {"id": "pho_msg_1"}
    return httpx.Response(202, json=body, request=httpx.Request("POST", url))


photon.httpx.post = fake_post
assert photon.send("617-555-0123", "X missed the deadline.") == "pho_msg_1"
assert [u.rsplit("/", 1)[-1] for u, *_ in calls] == ["users", "spaces", "messages"]
assert calls[0][1] == {"platform": "sms", "handle": "+16175550123"}
assert calls[1][1]["participantIds"] == ["pho_usr_agent", "pho_usr_user"]
assert calls[2][1]["content"]["text"] == "X missed the deadline."
assert calls[0][2] != calls[1][2]

# A second text to the same number reuses the thread (no new user/space).
calls.clear()
assert photon.send("+1 (617) 555-0123", "Follow-up.") == "pho_msg_1"
assert [u.rsplit("/", 1)[-1] for u, *_ in calls] == ["messages"]

os.environ["PHOTON_PLATFORM"] = "imessage"
photon._THREADS.clear()
calls.clear()
assert photon.send("617-555-0123", "via iMessage.") == "pho_msg_1"
assert calls[0][1]["platform"] == "imessage" and calls[1][1]["platform"] == "imessage"

print("photon self-check passed")
