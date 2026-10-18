"""Task A3, step 2: logistic-regression probes on frozen features (the UnivFD baseline).

One image = its face features and full-view features side by side. Each backbone gets two heads:
binary real/fake, and multi-class [real, swap, inpaint, txt2img, vae-decoded] that names the generator.
Trains on split=train, reports every split, saves the heads to data/models/<backbone>_<head>.joblib.
Run from the repo root after ml/features.py:
    python ml/heads.py
"""
import collections
import csv
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MANIFEST = Path("data/manifest.csv")
FEATURES = Path("data/features")
MODELS = Path("data/models")
BACKBONES = ["clip", "dinov3"]
CLASSES = ["real", "swap", "inpaint", "txt2img", "vae-decoded"]
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]  # regularization strengths tried; the best on val wins


def images(feats):
    """(X, generator, split) with one row per image: face and full features concatenated."""
    with MANIFEST.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(feats), "features don't match the manifest: re-run ml/features.py"
    splits = collections.defaultdict(set)
    for r in rows:
        splits[r["id"]].add(r["split"])
    assert all(len(s) == 1 for s in splits.values()), "a source photo appears in two splits"

    views = collections.defaultdict(dict)
    for r, x in zip(rows, feats):
        image = (r["generator"], Path(r["path"]).name, r["split"])  # the same image's two views share this
        views[image][r["view"]] = x
    keys = list(views)
    X = np.stack([np.concatenate([views[k]["face"], views[k]["full"]]) for k in keys])
    return X, np.array([k[0] for k in keys]), np.array([k[2] for k in keys])


def probe(c):
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000, class_weight="balanced"))


def report(backbone, binary, multi, X, gen, split):
    for s in ["train", "val", "test"]:
        m = split == s
        if not m.any():
            continue
        fake = gen[m] != "real"
        p = binary.predict_proba(X[m])[:, 1]
        known = np.isin(gen[m], CLASSES)
        acc = (multi.predict(X[m][known]) == gen[m][known]).mean() if known.any() else float("nan")
        print(f"{backbone:<7} {s:<5} {m.sum():>6} {roc_auc_score(fake, p):>6.3f} {average_precision_score(fake, p):>6.3f}"
              f" {(p[~fake] >= 0.5).mean():>8.3f} {(p[fake] >= 0.5).mean():>9.3f} {acc:>9.3f}")
        if s != "train":
            per = {g: roc_auc_score(gen[m][(gen[m] == g) | ~fake] == g, p[(gen[m] == g) | ~fake])
                   for g in sorted(set(gen[m][fake]))}
            print("        AUC per generator: " + "  ".join(f"{g} {a:.3f}" for g, a in per.items()))


def main():
    MODELS.mkdir(parents=True, exist_ok=True)
    print(f"{'model':<7} {'split':<5} {'images':>6} {'AUC':>6} {'AP':>6} {'FPR@0.5':>8} {'recall@.5':>9} {'gen acc':>9}")
    for backbone in BACKBONES:
        X, gen, split = images(np.load(FEATURES / f"{backbone}.npy"))
        tr, va = split == "train", split == "val"
        fits = {c: probe(c).fit(X[tr], gen[tr] != "real") for c in C_GRID}
        c = max(fits, key=lambda c: roc_auc_score(gen[va] != "real", fits[c].predict_proba(X[va])[:, 1]))
        binary, multi = fits[c], probe(c).fit(X[tr], gen[tr])
        assert list(multi.classes_) == sorted(CLASSES), multi.classes_
        joblib.dump(binary, MODELS / f"{backbone}_binary.joblib")
        joblib.dump(multi, MODELS / f"{backbone}_multi.joblib")
        report(f"{backbone}", binary, multi, X, gen, split)
        print(f"        (C={c})")


if __name__ == "__main__":
    main()
