# Detector results

Numbers only: no images, no team data. Reproduce from the repo root with
`python ml/data.py`, `python ml/testset.py`, `python ml/features.py`, `python ml/heads.py`.

## A3: frozen baselines (Sat Oct 17, 2026)

Logistic-regression probes on frozen backbone features (UnivFD-style), one image = face crop + full view.

**Data**
- train / val: DeepFakeFace. 10,000 real, plus 3,334 txt2img, 3,333 inpaint and 3,333 swap fakes,
  plus 3,000 vae-decoded fakes (real photos through the SD1.5, SDXL or FLUX VAE). Split 85/15 by source photo.
- test: team data only, never trained on. 99 real phone photos (Sean, Sunny) + 4 Grok Imagine edits so far.

| Model | Split | Images | AUC | AP | FPR @ 0.5 | Recall @ 0.5 | Generator acc |
|---|---|---:|---:|---:|---:|---:|---:|
| CLIP ViT-L/14 | train | 19,515 | 0.933 | 0.958 | 0.082 | 0.826 | 0.842 |
| CLIP ViT-L/14 | val | 3,485 | **0.903** | 0.940 | 0.116 | 0.796 | 0.698 |
| CLIP ViT-L/14 | test | 103 | 0.374 | 0.050 | **0.333** | 0.250 | — |
| DINOv3 ViT-L | train | 19,515 | 0.919 | 0.947 | 0.110 | 0.805 | 0.805 |
| DINOv3 ViT-L | val | 3,485 | **0.887** | 0.925 | 0.151 | 0.779 | 0.706 |
| DINOv3 ViT-L | test | 103 | 0.535 | 0.050 | **0.061** | 0.000 | — |

**Val AUC per generator** (each fake type vs. the real photos)

| Model | txt2img | inpaint | swap | vae-decoded |
|---|---:|---:|---:|---:|
| CLIP | 0.993 | 0.970 | 0.969 | 0.655 |
| DINOv3 | 0.988 | 0.954 | 0.900 | 0.693 |

**Reading it**
- Both probes catch visibly generated faces well (0.90–0.99) but struggle with vae-decoded fakes,
  which look identical to their real source and differ only in the decoder's fingerprint.
- Test: only 4 Grok fakes, so the test AUC is noise for now. The false-positive rate on 99 real phone
  photos is already informative: CLIP flags 33%, DINOv3 6%. DINOv3 generalizes better to new real photos.
- FPR and recall are at the default 0.5 cutoff; A5 picks the real threshold (at least 95% precision on val).
- Generator accuracy is over the five training classes (real, txt2img, inpaint, swap, vae-decoded), so
  it is left out for test, where the fakes are Grok.
- Regularization picked on val: CLIP C=10, DINOv3 C=0.01.
