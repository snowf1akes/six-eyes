"""Task 2: LoRA fine-tune of DINOv3 for the detector, run in the background on the local GPU.

LoRA (r=16) on every linear layer of the last 8 transformer blocks. Each image's face crop and full
view go through the shared backbone, and their two CLS tokens feed one linear head over the
generator classes in the manifest's train split. Trains on split=train with random resize + JPEG,
prints val AUC (real vs fake) after each epoch, and keeps the best epoch:
    data/models/dinov3_lora/              merged backbone (save_pretrained) + head.pt
    data/models/dinov3_lora_preds.csv     per-image val and test probabilities, for ml/eval.py
Run from the repo root once data/manifest.csv is final:
    python ml/finetune.py           # the full run: 3 epochs
    python ml/finetune.py 1 30      # smoke test: 1 epoch of 30 batches
"""
import collections
import csv
import random
import sys
import time
from pathlib import Path

import cv2
import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModel

MANIFEST = Path("data/manifest.csv")
OUT = Path("data/models/dinov3_lora")
PREDS = Path("data/models/dinov3_lora_preds.csv")
DINO = "facebook/dinov3-vitl16-pretrain-lvd1689m"
BATCH, LR, EPOCHS = 32, 2e-4, 3
LAST_8_BLOCKS = r"model\.layer\.(1[6-9]|2[0-3])\.(attention\.[qkvo]_proj|mlp\.(up|down)_proj)"


def images(split):
    """[(face path, full path, generator, id)] for one split, pairing views as ml/heads.py does."""
    with MANIFEST.open() as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == split]
    views = collections.defaultdict(dict)
    for r in rows:
        views[(r["generator"], Path(r["path"]).name)][r["view"]] = r
    return [(v["face"]["path"], v["full"]["path"], gen, v["face"]["id"]) for (gen, _), v in views.items()]


def augment(img, rng):
    # degrade() from ml/data.py at the 224px crop size: random downscale and back, then JPEG 60-95.
    k = rng.uniform(0.5, 1.0)
    img = cv2.resize(cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA), img.shape[1::-1])
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(60, 95)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


class Pairs(Dataset):
    def __init__(self, items, classes, train):
        self.items, self.classes, self.train = items, classes, train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        face, full, gen, _ = self.items[i]
        views = []
        for path in (face, full):
            img = cv2.imread(path)
            if self.train:
                img = augment(img, random.Random())
            views.append(torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1))
        return torch.stack(views), self.classes.index(gen) if gen in self.classes else -1


class Detector(torch.nn.Module):
    def __init__(self, backbone, n_classes, mean, std):
        super().__init__()
        self.backbone = backbone
        self.head = torch.nn.Linear(2 * backbone.config.hidden_size, n_classes)
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):  # x: (batch, 2 views, 3, 224, 224) uint8
        n = x.shape[0]
        x = (x.flatten(0, 1).float() / 255 - self.mean) / self.std
        cls = self.backbone(pixel_values=x).pooler_output  # (2 * batch, hidden): face, full, face, full...
        return self.head(cls.reshape(n, -1).float())


def loader(items, classes, train):
    return DataLoader(Pairs(items, classes, train), batch_size=BATCH, shuffle=train, drop_last=train,
                      num_workers=6, pin_memory=True, persistent_workers=True)


@torch.no_grad()
def predict(model, data):
    model.eval()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        probs = torch.cat([model(x.cuda(non_blocking=True)).float().softmax(-1).cpu() for x, _ in data])
    model.train()
    return probs.numpy()


def main(epochs, max_batches):
    torch.manual_seed(0)
    train, val, test = images("train"), images("val"), images("test")
    classes = sorted({gen for _, _, gen, _ in train})
    real = classes.index("real")
    counts = collections.Counter(gen for _, _, gen, _ in train)
    weight = torch.tensor([len(train) / (len(classes) * counts[c]) for c in classes]).cuda()  # balanced classes

    proc = AutoImageProcessor.from_pretrained(DINO)
    backbone = get_peft_model(AutoModel.from_pretrained(DINO, dtype=torch.bfloat16),
                              LoraConfig(r=16, lora_alpha=16, lora_dropout=0.05, target_modules=LAST_8_BLOCKS))
    backbone.print_trainable_parameters()
    model = Detector(backbone, len(classes), proc.image_mean, proc.image_std).cuda()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01)
    train_data, val_data = loader(train, classes, True), loader(val, classes, False)
    per_epoch = min(len(train_data), max_batches)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=epochs * per_epoch, pct_start=0.05)
    print(f"classes {classes}; {len(train)} train / {len(val)} val / {len(test)} test images; "
          f"{per_epoch} batches of {BATCH} per epoch", flush=True)

    best_auc, best = -1.0, None
    for epoch in range(1, epochs + 1):
        start = time.time()
        for i, (x, y) in enumerate(train_data):
            if i == per_epoch:
                break
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x.cuda(non_blocking=True))
            loss = torch.nn.functional.cross_entropy(logits.float(), y.cuda(), weight=weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if i % 50 == 0:
                print(f"epoch {epoch} batch {i}/{per_epoch} loss {loss.item():.3f} ({time.time() - start:.0f}s)", flush=True)
        auc = roc_auc_score([gen != "real" for _, _, gen, _ in val], 1 - predict(model, val_data)[:, real])
        print(f"epoch {epoch}: val AUC {auc:.4f}, {(time.time() - start) / 60:.1f} min, "
              f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
        if auc > best_auc:
            best_auc = auc
            best = ({k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(backbone).items()},
                    {k: v.detach().cpu().clone() for k, v in model.head.state_dict().items()})

    set_peft_model_state_dict(backbone, best[0])
    model.head.load_state_dict(best[1])
    rows = []
    for split, items in (("val", val), ("test", test)):
        probs = predict(model, loader(items, classes, False)) if items else []
        for (face, _, gen, photo), p in zip(items, probs):
            rows.append({"id": photo, "path": face, "generator": gen, "split": split,
                         "fake_prob": round(float(1 - p[real]), 6), "generator_guess": classes[p.argmax()]})
    with PREDS.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    backbone.merge_and_unload().save_pretrained(OUT)
    proc.save_pretrained(OUT)
    torch.save({"state_dict": model.head.state_dict(), "classes": classes, "val_auc": best_auc}, OUT / "head.pt")
    print(f"best val AUC {best_auc:.4f}; saved {OUT} and {PREDS} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else EPOCHS, int(sys.argv[2]) if len(sys.argv) > 2 else 10**9)
