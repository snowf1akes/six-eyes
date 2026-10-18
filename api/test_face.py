"""Self-check for face matching and clone hashes in analyze(). Run from the repo root:
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

from analyze import analyze, probe
from face import face_model
from signals import CLONE_MAX_DISTANCE, clusters

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
cv2.imwrite(os.path.join(root, "group_repost.jpg"), cv2.resize(group, None, fx=0.6, fy=0.6),
            [cv2.IMWRITE_JPEG_QUALITY, 60])  # a re-uploaded copy: smaller, recompressed
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
repost = analyze(url + "group_repost.jpg", enrolled)
distance = lambda a, b: (int(a.pdq, 16) ^ int(b.pdq, 16)).bit_count()  # noqa: E731
print(f"PDQ distance: repost {distance(same, repost)}  different photo {distance(same, other)}")
assert len(same.pdq) == 64 and same.pdq == analyze(url + "group.jpg", enrolled).pdq
assert distance(same, repost) <= CLONE_MAX_DISTANCE < distance(same, other)
assert len(clip.pdq) == 64
# Detector fields: only faces that match the user get scored; someone else and the blank image don't.
assert all(0 <= r.fake_prob <= 1 and 0 <= r.nsfw < 0.5 for r in (same, clip)), (same, clip)
assert all(r.generator_guess == "not-scored" and r.fake_prob == r.nsfw == 0 for r in (other, none)), (other, none)
assert other.recon_score is None and none.recon_score is None, (other, none)
assert all(r.c2pa is None for r in (same, other, none, clip)), "the bundled photos carry no Content Credentials"
if probe():  # a checkout without data/models keeps the placeholder
    assert same.generator_guess in {"real", "swap", "inpaint", "txt2img", "vae-decoded"}, same
    assert same.recon_score > 0 and clip.recon_score > 0, (same, clip)
# The blank image is too flat to hash, so it never joins a clone cluster.
assert none.pdq == "", none.pdq
assert clusters([same.pdq, repost.pdq, other.pdq, none.pdq]) == [0, 0, 2, None]
for bad in ("not_an_image.jpg", "missing.jpg"):
    try:
        analyze(url + bad, enrolled)
        raise AssertionError(f"{bad} should fail")
    except Exception as e:
        assert not isinstance(e, AssertionError), e
assert not [f for f in os.listdir(tempfile.tempdir) if f.endswith(".mp4")], "a downloaded video was left in the temp dir"
server.shutdown()
print("face self-check passed")
