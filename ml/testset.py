"""Task A2, step 4: add the team's photos to data/manifest.csv, held out as split=test.

Reads C's folders (data/team/real/<person>/, grok/<person>/, facefusion/), makes the same 224px face
crop and full view as ml/data.py, and replaces any earlier team rows. Team images skip data.py's
random resize + JPEG: they stay as they are, the way analyze() will see them.
Re-run whenever C sends a new zip, then re-run ml/features.py and ml/heads.py.
    python ml/testset.py                       # every team image is test: Grok never seen in training
    python ml/testset.py --train-person sunny  # Task 4: Sunny's real and Grok images train, Sean's test
"""
import argparse
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


def main(train_person):
    team = (OUT / "test").as_posix()  # team images live under processed/test whatever their split
    with MANIFEST.open() as f:
        rows = [r for r in csv.DictReader(f) if not r["path"].startswith(team)]
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
            person = src.relative_to(TEAM / folder).parts[0]  # real/<person>/..., grok/<person>/...
            split = "train" if person == train_person else "test"
            for view, out in {"face": face, "full": full_view(img)}.items():
                path = OUT / "test" / view / gen / src.relative_to(TEAM / folder).with_suffix(".png")
                path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(path), out)
                rows.append({"id": photo, "path": path.as_posix(), "label": label,
                             "generator": gen, "view": view, "split": split})
            added[(gen, split)] += 1

    with MANIFEST.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    trained = {Path(r["path"]).parent.name for r in rows if r["generator"] in ("grok", "facefusion") and r["split"] != "test"}
    assert trained <= {train_person}, f"held-out Grok or FaceFusion images leaked into training: {trained}"
    assert added[("real", "test")], f"no held-out team photos found in {TEAM / 'real'}"
    print(f"team images added (generator, split): {dict(added)}")
    for src in skipped:
        print(f"  skipped (unreadable or no face found): {src}")
    if (TEAM / "through_x").exists():
        print("  through_x/ isn't merged yet: it needs labels.csv to say which posted image each copy is")
    print(f"manifest now: {collections.Counter(r['split'] for r in rows if r['view'] == 'face')} images")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-person", help="this teammate's real and Grok images go to split=train")
    main(parser.parse_args().train_person)
