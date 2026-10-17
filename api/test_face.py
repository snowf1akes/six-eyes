"""Self-check for face matching in analyze(). Run from the repo root:
    .venv/Scripts/python api/test_face.py
Uses the group photo bundled with InsightFace and a local web server, so no team photos or X calls.
"""
import functools
import http.server
import os
import tempfile
import threading

import cv2
import numpy as np
from insightface.data import get_image

from analyze import analyze
from face import face_model

group = get_image("t1")
faces = sorted(face_model().get(group), key=lambda f: f.bbox[0])


def crop(face, pad=0.4):
    x1, y1, x2, y2 = face.bbox.astype(int)
    p = int((x2 - x1) * pad)
    return group[max(0, y1 - p):y2 + p, max(0, x1 - p):x2 + p]


# Enroll person 0 the way /enroll does: mean of normalized embeddings, normalized again.
enrolled = np.mean([faces[0].normed_embedding, face_model().get(cv2.flip(crop(faces[0]), 1))[0].normed_embedding], axis=0)
enrolled = (enrolled / np.linalg.norm(enrolled)).tolist()

root = tempfile.mkdtemp()
blank = np.full((300, 300, 3), 127, np.uint8)
cv2.imwrite(os.path.join(root, "group.jpg"), group)
cv2.imwrite(os.path.join(root, "someone_else.jpg"), cv2.resize(crop(faces[1]), None, fx=3, fy=3))
cv2.imwrite(os.path.join(root, "no_face.jpg"), blank)
with open(os.path.join(root, "not_an_image.jpg"), "wb") as f:
    f.write(b"hello")
# A video that cuts between person 0 and a blank frame: the face is found in some frames only.
h, w = group.shape[:2]
video = cv2.VideoWriter(os.path.join(root, "clip.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
for i in range(40):
    video.write(group if i % 2 == 0 else cv2.resize(blank, (w, h)))
video.release()

class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=root))
threading.Thread(target=server.serve_forever, daemon=True).start()
url = f"http://127.0.0.1:{server.server_port}/"

tempfile.tempdir = tempfile.mkdtemp()  # private, so a running API's downloads can't trip the check below
same = analyze(url + "group.jpg", enrolled)
other = analyze(url + "someone_else.jpg", enrolled)
none = analyze(url + "no_face.jpg", enrolled)
clip = analyze(url + "clip.mp4", enrolled)
print(f"group photo with them: {same.face_sim}  someone else: {other.face_sim}  no face: {none.face_sim}  "
      f"video: {clip.face_sim} (drift {clip.frame_drift})")

assert same.face_sim > 0.6 and same.frame_drift is None, same
assert other.face_sim < 0.3, other
assert none.face_sim == 0.0, none
assert clip.face_sim > 0.6 and clip.frame_drift is not None and clip.frame_drift < 0.01, clip
for bad in ("not_an_image.jpg", "missing.jpg"):
    try:
        analyze(url + bad, enrolled)
        raise AssertionError(f"{bad} should fail")
    except Exception as e:
        assert not isinstance(e, AssertionError), e
assert not os.listdir(tempfile.tempdir), "a downloaded video was left in the temp dir"
server.shutdown()
print("face self-check passed")
