"""Task A2, step 4: add the team's held-out photos to data/manifest.csv as split=test.

Reads C's folders (data/team/real/<person>/, grok/<person>/, facefusion/), makes the same 224px face
crop and full view as ml/data.py, and replaces any earlier test rows. Test images skip data.py's
random resize + JPEG: they stay as they are, the way analyze() will see them.
Re-run whenever C sends a new zip, then re-run ml/features.py and ml/heads.py.
    python ml/testset.py
"""
import collections
import csv
import re
from pathlib import Path

import cv2

from data import MANIFEST, OUT, detector, face_crop, full_view

TEAM = Path("data/team")
FOLDERS = {"real": ("real", "real"), "grok": ("fake", "grok"), "facefusion": ("fake", "facefusion")}  # -> (label, generator)
MAX_SIDE = 1792  # Grok's output size; 5712px phone originals shrink to it so real and fake share a resolution
IMAGE_TYPES = {".jpg", ".jpeg", ".png"}


def main():
    with MANIFEST.open() as f:
        rows = [r for r in csv.DictReader(f) if r["split"] != "test"]
    model, added, skipped = detector(), collections.Counter(), []
    for folder, (label, gen) in FOLDERS.items():
        for src in sorted(p for p in (TEAM / folder).rglob("*") if p.suffix.lower() in IMAGE_TYPES):
            img = cv2.imread(str(src))  # applies the phone's EXIF rotation; the GPS tags never reach the PNGs
            if img is None:
                skipped.append(src)
                continue
            k = MAX_SIDE / max(img.shape[:2])
            if k < 1:
                img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
            face = face_crop(img, model)
            if face is None:
                skipped.append(src)
                continue
            photo = re.sub(r"_(outfit|background|lighting)$", "", src.stem)  # a Grok edit shares its source photo's id
            for view, out in {"face": face, "full": full_view(img)}.items():
                path = OUT / "test" / view / gen / src.relative_to(TEAM / folder).with_suffix(".png")
                path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(path), out)
                rows.append({"id": photo, "path": path.as_posix(), "label": label,
                             "generator": gen, "view": view, "split": "test"})
            added[gen] += 1

    with MANIFEST.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    assert all(r["split"] == "test" for r in rows if r["generator"] in ("grok", "facefusion")), "test leaked"
    assert added["real"], f"no team photos found in {TEAM / 'real'}"
    print(f"test images added: {dict(added)}")
    for src in skipped:
        print(f"  skipped (unreadable or no face found): {src}")
    if (TEAM / "through_x").exists():
        print("  through_x/ isn't merged yet: it needs labels.csv to say which posted image each copy is")
    print(f"manifest now: {collections.Counter(r['split'] for r in rows if r['view'] == 'face')} images")


if __name__ == "__main__":
    main()
