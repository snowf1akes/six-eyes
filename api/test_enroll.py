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

print("enroll self-check passed")
