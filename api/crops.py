"""The two 224px views the detector sees: an aligned face crop and a center-square full view.

Shared by training (ml/data.py, ml/testset.py) and serving (analyze.py), so both crop alike.
"""
import cv2
from insightface.app import FaceAnalysis
from insightface.utils.face_align import norm_crop

SIZE = 224


def detector():
    model = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"], providers=["CPUExecutionProvider"])
    model.prepare(ctx_id=-1, det_size=(320, 320))  # finds ~98% of DeepFakeFace's faces, ~10x faster than 640
    return model


def face_crop(img, model):
    """Aligned crop of the largest face, or None when there is no face."""
    # The detector misses faces that fill the frame; a border fixes that (same trick as face.py).
    h, w = img.shape[:2]
    img = cv2.copyMakeBorder(img, h // 4, h // 4, w // 4, w // 4, cv2.BORDER_CONSTANT)
    faces = model.get(img)
    if not faces:
        return None
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return norm_crop(img, face.kps, image_size=SIZE)


def full_view(img):
    # Center square crop, not letterbox: every DeepFakeFace fake is 512x512 but reals come in hundreds
    # of shapes, so letterbox bars would appear only on reals and the model would learn "bars = real".
    h, w = img.shape[:2]
    s = min(h, w)
    y, x = (h - s) // 2, (w - s) // 2
    return cv2.resize(img[y:y + s, x:x + s], (SIZE, SIZE), interpolation=cv2.INTER_AREA)
