"""Self-check for POST /enroll. Run from the repo root:
    .venv/Scripts/python api/test_enroll.py
Uses the group photo bundled with InsightFace, so no team photos are needed.
"""
import json
import os
import sqlite3
import tempfile

os.environ["DB_PATH"] = os.path.join(tempfile.mkdtemp(), "test.db")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from insightface.data import get_image  # noqa: E402

from main import DB_PATH, app, face_model  # noqa: E402

group = get_image("t1")
faces = sorted(face_model().get(group), key=lambda f: f.bbox[0])


def selfie(face, pad, flip=False):
    x1, y1, x2, y2 = face.bbox.astype(int)
    p = int((x2 - x1) * pad)
    img = group[max(0, y1 - p):y2 + p, max(0, x1 - p):x2 + p]
    return cv2.imencode(".jpg", cv2.flip(img, 1) if flip else img)[1].tobytes()


def post(photos, **overrides):
    form = {"name": "Sunny Le", "signature": "sunny le", "consent": "true",
            "handles": "@sunny, sixeyes", "phone": ""} | overrides
    files = [("selfies", (f"{i}.jpg", p, "image/jpeg")) for i, p in enumerate(photos)]
    return client.post("/enroll", data=form, files=files)


client = TestClient(app)
a = [selfie(faces[0], 0.3), selfie(faces[0], 0.4, flip=True), selfie(faces[0], 0.5)]
b = selfie(faces[1], 0.4)
blank = cv2.imencode(".jpg", np.full((200, 200, 3), 127, np.uint8))[1].tobytes()

r = post(a)
assert r.status_code == 200, r.text
row = sqlite3.connect(DB_PATH).execute(
    "SELECT embedding, handles, signed_at FROM users WHERE id = ?", (r.json()["user_id"],)).fetchone()
emb = np.array(json.loads(row[0]))
assert emb.shape == (512,) and abs(np.linalg.norm(emb) - 1) < 1e-3
assert json.loads(row[1]) == ["sunny", "sixeyes"] and row[2]

r = post([a[0], b, a[2]])
assert r.status_code == 422 and "same person" in r.text, r.text
r = post(a[:2])
assert r.status_code == 422 and "3 to 5" in r.text, r.text
r = post([a[0], blank, a[2]])
assert r.status_code == 422 and "No face" in r.text, r.text
r = post(a, signature="someone else")
assert r.status_code == 422 and "type your full name" in r.text, r.text
r = post(a, consent="false")
assert r.status_code == 422 and "consent" in r.text, r.text
r = post(a + [a[0], a[1], a[2]])
assert r.status_code == 422 and "3 to 5" in r.text, r.text
r = post([a[0], b"not an image", a[2]])
assert r.status_code == 422 and "isn't an image" in r.text, r.text
r = post([a[0], b"\xff" * (10 * 1024 * 1024 + 1), a[2]])
assert r.status_code == 413, r.text

# A phone-sized close-up (face fills ~80% of the frame) must enroll. It is also >1 MB, so it
# spools to a temp file, which must be gone after the request; no image bytes may reach the DB.
x1, y1, x2, y2 = faces[0].bbox.astype(int)
p = (x2 - x1) // 8
closeup = cv2.resize(group[y1 - p:y2 + p, x1 - p:x2 + p], None, fx=12, fy=12, interpolation=cv2.INTER_CUBIC)
big = cv2.imencode(".png", closeup)[1].tobytes()
assert len(big) > 1024 * 1024
before = set(os.listdir(tempfile.gettempdir()))
r = post([big, a[1], a[2]])
assert r.status_code == 200, r.text
assert set(os.listdir(tempfile.gettempdir())) - before == set(), "an upload was left in the temp dir"
db_bytes = open(DB_PATH, "rb").read()
assert b"\xff\xd8\xff" not in db_bytes and b"\x89PNG" not in db_bytes, "image bytes found in the database"

print("enroll self-check passed")
