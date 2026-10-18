"""Task 5: one results table for every detector, and the threshold each would ship with.

Rows: the CLIP and DINOv3 probes with Grok never in training, DINOv3 + Grok (the train split as
ml/testset.py --train-person left it), the LoRA fine-tune (data/models/dinov3_lora_preds.csv), and the
reconstruction score (api/recon.py on the original files, cached in data/models/recon_scores.csv).
Columns: AUC and AP on val; AUC on each held-out generator (Grok, and FaceFusion once it has images);
the threshold with at least 95% precision on val; and at that threshold, held-out Grok caught, the held-out
teammate's real photos flagged, and the share of flags that are false when only 1% of posts are fake.
Run from the repo root after ml/features.py, ml/heads.py and ml/finetune.py:
    python ml/eval.py
"""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

import heads

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "api"))
import recon  # noqa: E402

LORA_PREDS = Path("data/models/dinov3_lora_preds.csv")
RECON_CACHE = Path("data/models/recon_scores.csv")
DFF, TEAM = Path("data/deepfakeface"), Path("data/team")
DFF_FOLDERS = {"real": "wiki", "txt2img": "text2img", "inpaint": "inpainting", "swap": "insight"}
TEAM_FOLDERS = {"real": "real", "grok": "grok", "facefusion": "facefusion"}
MAX_SIDE = 1792  # as ml/testset.py: team photos are scored at Grok's output size
MIN_PRECISION, BASE_RATE = 0.95, 0.01


def best_probe(X, gen, train, val):
    """The binary probe with the regularization that scores best on val, as ml/heads.py picks it."""
    fits = [heads.probe(c).fit(X[train], gen[train] != "real") for c in heads.C_GRID]
    return max(fits, key=lambda m: roc_auc_score(gen[val] != "real", m.predict_proba(X[val])[:, 1]))


def original(path):
    """The image a manifest row was made from, as analyze() would see it (no data.py degrade step)."""
    p = Path(path)
    if "/test/" in path:  # data/processed/test/face/<gen>/<person>/<name>.png
        src = next((TEAM / TEAM_FOLDERS[p.parts[-3]] / p.parts[-2]).glob(p.stem + ".*"))
        img = cv2.imread(str(src))
        k = MAX_SIDE / max(img.shape[:2])
        return cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else img
    gen, photo = p.parts[-2], p.stem  # data/processed/face/<gen>/<folder>_<file>[_<vae>].png
    if gen == "vae-decoded":
        photo, vae = photo.rsplit("_", 1)  # regenerate: the VAE round trip is deterministic
    folder, name = photo.split("_", 1)
    img = cv2.imread(str(DFF / DFF_FOLDERS.get(gen, "wiki") / folder / f"{name}.jpg"))
    return recon.vae_roundtrip(img, vae_model(vae)) if gen == "vae-decoded" else img


_vaes = {}


def vae_model(name):
    from diffusers import AutoencoderKL

    if name not in _vaes:
        repo, sub = recon.VAES[name]
        _vaes[name] = AutoencoderKL.from_pretrained(repo, subfolder=sub).to("cuda").eval()
    return _vaes[name]


def recon_scores(paths):
    """Reconstruction distance per face path, negated so higher = more likely generated like the others."""
    cache = {}
    if RECON_CACHE.exists():
        with RECON_CACHE.open() as f:
            cache = {r["path"]: float(r["recon_score"]) for r in csv.DictReader(f)}
    todo = [p for p in paths if p not in cache]
    for i, p in enumerate(todo, 1):
        cache[p] = recon.score(original(p))
        if i % 500 == 0 or i == len(todo):
            print(f"  reconstruction scores: {i}/{len(todo)}", flush=True)
    with RECON_CACHE.open("w", newline="") as f:
        csv.writer(f).writerows([("path", "recon_score"), *cache.items()])
    return np.array([-cache[p] for p in paths])


def summarize(name, p, gen, split, team_test):
    val, test = (split == "val") & ~np.isnan(p), team_test & ~np.isnan(p)
    fake = gen != "real"
    precision, recall, thresholds = precision_recall_curve(fake[val], p[val])
    ok = np.flatnonzero(precision[:-1] >= MIN_PRECISION)  # lowest such threshold catches the most
    t = thresholds[ok[0]] if len(ok) else np.inf
    row = {"model": name, "val AUC": roc_auc_score(fake[val], p[val]), "val AP": average_precision_score(fake[val], p[val])}
    reals = test & ~fake
    for g in sorted(set(gen[test & fake])):
        m = (test & (gen == g)) | reals
        row[f"held-out {g} AUC"] = roc_auc_score(gen[m] == g, p[m])
    caught = float(np.mean(p[test & (gen == "grok")] >= t)) if (test & (gen == "grok")).any() else np.nan
    flagged = float(np.mean(p[reals] >= t))
    false_flags = flagged * (1 - BASE_RATE) / (flagged * (1 - BASE_RATE) + caught * BASE_RATE) if flagged or caught else np.nan
    row.update({"threshold": -t if name.startswith("Reconstruction") else t,
                "val recall": float(np.mean(p[val & fake] >= t)), "held-out Grok caught": caught,
                "held-out real flagged": flagged, "false flags at 1% fakes": false_flags})
    return row


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # the table's em dashes, even when Windows redirects output
    feats = {b: heads.images(np.load(heads.FEATURES / f"{b}.npy")) for b in heads.BACKBONES}
    _, gen, split, paths = feats["dinov3"]
    team = np.array(["/test/" in p for p in paths])
    person = np.array([Path(p).parent.name if t else "" for p, t in zip(paths, team)])
    trained, held_out = set(person[team & (split == "train")]), set(person[team & (split == "test")])
    assert not trained & held_out, f"a teammate is in both training and the test: {trained & held_out}"
    team_test = team & (split == "test")

    scores = {}
    for name, backbone, with_team in [("CLIP probe", "clip", False), ("DINOv3 probe", "dinov3", False),
                                      ("DINOv3 + Grok", "dinov3", True)]:
        X, gen_b, split_b, paths_b = feats[backbone]
        assert (paths_b == paths).all()
        train = (split_b == "train") & (with_team | ~team)
        scores[name] = best_probe(X, gen_b, train, split_b == "val").predict_proba(X)[:, 1]
    if LORA_PREDS.exists():
        with LORA_PREDS.open() as f:
            lora = {r["path"]: float(r["fake_prob"]) for r in csv.DictReader(f)}
        scores["DINOv3 + LoRA"] = np.array([lora.get(p, np.nan) for p in paths])
    scored = (split == "val") | team_test
    scores["Reconstruction (SD1.5 VAE)"] = np.full(len(paths), np.nan)
    scores["Reconstruction (SD1.5 VAE)"][scored] = recon_scores(list(paths[scored]))

    rows = [summarize(name, p, gen, split, team_test) for name, p in scores.items()]
    cols = list(dict.fromkeys(k for r in rows for k in r))
    print(f"Held out: {', '.join(sorted(held_out))} ({team_test.sum()} images); in training: {', '.join(sorted(trained)) or 'nobody'}\n")
    print("| " + " | ".join(cols) + " |")
    print("|---" + "|---:" * (len(cols) - 1) + "|")
    for r in rows:
        print("| " + " | ".join(r["model"] if c == "model" else ("—" if np.isnan(r.get(c, np.nan)) else
              f"{r[c]:.0%}" if c in ("val recall", "held-out Grok caught", "held-out real flagged", "false flags at 1% fakes")
              else f"{r[c]:.3f}") for c in cols) + " |")


if __name__ == "__main__":
    main()
