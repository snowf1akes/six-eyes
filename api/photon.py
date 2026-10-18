"""Send the user a text through Photon (https://api.photon.codes/v1).

Needs PHOTON_API_KEY, PHOTON_PROJECT_ID (pho_prj_…) and PHOTON_AGENT_USER_ID (the agent's
User on the SMS platform). Enroll stores the number; we send E.164 SMS, never the images.
"""
import logging
import os
import re
import uuid

import httpx

API = "https://api.photon.codes/v1/projects"
log = logging.getLogger("six_eyes.photon")
# phone -> (photon user id, space id), so a follow-up text reuses the same SMS thread
_THREADS: dict[str, tuple[str, str]] = {}


def enabled() -> bool:
    return all(os.environ.get(k) for k in ("PHOTON_API_KEY", "PHOTON_PROJECT_ID", "PHOTON_AGENT_USER_ID"))


def e164(phone: str) -> str | None:
    """US-centric E.164: 10 digits → +1…, 11 digits starting with 1 → +1…, already +… kept."""
    raw = phone.strip()
    if raw.startswith("+"):
        digits = re.sub(r"\D", "", raw)
        return f"+{digits}" if 8 <= len(digits) <= 15 else None
    digits = re.sub(r"\D", "", raw)
    if len(digits) == 10:
        return "+1" + digits
    if len(digits) == 11 and digits.startswith("1"):
        return "+" + digits
    return None


def send(phone: str, text: str) -> str | None:
    """Text `phone`. Returns Photon's message id, or None if we didn't send (not set up, or a bad number)."""
    to = e164(phone)
    if not to:
        log.warning("not a phone number we can text: %r", phone)
        return None
    if not enabled():
        return None
    agent = os.environ["PHOTON_AGENT_USER_ID"]
    user_id, space_id = _THREADS.get(to) or (None, None)
    if not user_id:
        user_id = _post("/users", {"platform": "sms", "handle": to})["id"]
        space_id = _post("/spaces", {"platform": "sms", "participantIds": [agent, user_id]})["id"]
        _THREADS[to] = (user_id, space_id)
    msg = _post("/messages", {"spaceId": space_id, "senderId": agent,
                              "content": {"type": "text", "format": "plain", "text": text}})
    return msg.get("id")


def _post(path: str, body: dict) -> dict:
    project = os.environ["PHOTON_PROJECT_ID"]
    r = httpx.post(f"{API}/{project}{path}", json=body, headers={
        "Authorization": f"Bearer {os.environ['PHOTON_API_KEY']}",
        "Content-Type": "application/json",
        "idempotency-key": str(uuid.uuid4()),
    }, timeout=20)
    r.raise_for_status()
    return r.json() if r.content else {}
