# Detector results

Numbers only: no images, no team data. Reproduce from the repo root with
`python ml/data.py`, `python ml/testset.py`, `python ml/features.py`, `python ml/heads.py`, `python ml/finetune.py`.

## A3: frozen baselines, Grok never seen in training (Sat Oct 17, 2026)

Logistic-regression probes on frozen backbone features (UnivFD-style), one image = face crop + full view.

**Data**
- train / val: DeepFakeFace. 10,000 real, plus 3,334 txt2img, 3,333 inpaint and 3,333 swap fakes,
  plus 3,000 vae-decoded fakes (real photos through the SD1.5, SDXL or FLUX VAE). Split 85/15 by source photo.
- test: team data only, never trained on. 99 real phone photos (Sean, Sunny) + 49 Grok Imagine edits
  (Raymond's 50; one had no detectable face).

| Model | Split | Images | AUC | AP | FPR @ 0.5 | Recall @ 0.5 | Generator acc |
|---|---|---:|---:|---:|---:|---:|---:|
| CLIP ViT-L/14 | train | 19,515 | 0.933 | 0.958 | 0.082 | 0.826 | 0.842 |
| CLIP ViT-L/14 | val | 3,485 | **0.903** | 0.940 | 0.116 | 0.796 | 0.698 |
| CLIP ViT-L/14 | test (Grok unseen) | 148 | 0.408 | 0.284 | **0.333** | 0.265 | — |
| DINOv3 ViT-L | train | 19,515 | 0.919 | 0.947 | 0.110 | 0.805 | 0.805 |
| DINOv3 ViT-L | val | 3,485 | **0.887** | 0.925 | 0.151 | 0.779 | 0.706 |
| DINOv3 ViT-L | test (Grok unseen) | 148 | 0.585 | 0.472 | **0.061** | 0.163 | — |

**Val AUC per generator** (each fake type vs. the real photos)

| Model | txt2img | inpaint | swap | vae-decoded |
|---|---:|---:|---:|---:|
| CLIP | 0.993 | 0.970 | 0.969 | 0.655 |
| DINOv3 | 0.988 | 0.954 | 0.900 | 0.693 |

**Reading it**
- Both probes catch visibly generated faces well (0.90–0.99) but struggle with vae-decoded fakes,
  which look identical to their real source and differ only in the decoder's fingerprint.
- Grok, never seen in training, is close to chance for both (AUC 0.41 CLIP, 0.59 DINOv3), the lab-to-real
  gap the research handoff describes. DINOv3 flags far fewer real phone photos (6% vs 33%), so it ships.
- FPR and recall are at the default 0.5 cutoff; the eval task picks the real threshold (95% precision on val).
- Generator accuracy is over the five training classes, so it is left out for test, where the fakes are Grok.
- Regularization picked on val: CLIP C=10, DINOv3 C=0.01.

## Task 4: Grok in training, other person held out

`python ml/testset.py --train-person sunny`: Sunny's 50 real photos and 25 Grok edits join training (Grok
becomes a sixth class); Sean's 49 real photos and 24 Grok edits are the test. Sean is held out because he is
the enrolled teammate in the dashboard runs. Both rows are on the same 73 Sean images, so they compare
directly with the Grok-unseen rows above, which mix both teammates.

| Model | Grok in training | AUC | AP | FPR @ 0.5 | Recall @ 0.5 |
|---|---|---:|---:|---:|---:|
| CLIP probe | no | 0.429 | 0.295 | 0.388 | 0.417 |
| CLIP probe | Sunny's 25 | 0.548 | 0.374 | 0.347 | 0.542 |
| DINOv3 probe | no | 0.662 | 0.527 | 0.020 | 0.083 |
| DINOv3 probe | Sunny's 25 | **0.686** | 0.556 | 0.020 | 0.083 |
| DINOv3 + LoRA | no | 0.695 | — | — | — |
| DINOv3 + LoRA | Sunny's 25 | 0.514 | 0.338 | 0.918 | 0.917 |

- A linear probe on frozen features learns little about Grok from 25 images of another person: DINOv3 moves
  from 0.662 to 0.686, CLIP from 0.429 to 0.548. The same held when training on Sean and testing on Sunny
  (DINOv3 0.55; CLIP about 0.6), at every regularization strength tried, and with each generator weighted
  equally instead of each class. Grok redraws the whole image, so like
  the vae-decoded class its trace is low-level, which frozen features barely capture.
- The LoRA model trained without Grok scores 0.613 on all 49 Grok edits (Sean 0.695, Sunny 0.539).
- Retrained with Sunny's Grok edits (val AUC 0.929, its best), LoRA overfits to the team's photos: it calls
  92% of Sean's Grok edits fake but also 92% of his real photos (AUC 0.514). With one person's 25 Grok images
  and the Grok class weighted ~130x, it learned "a teammate's phone photo" rather than Grok's trace.

## Task 5: one table, and what ships (`ml/eval.py`)

Held out: Sean (49 real photos, 24 Grok edits); Sunny's team images are in training. Threshold = the lowest
fake_prob with at least 95% precision on DeepFakeFace val; the last three columns are at that threshold, on
Sean's images. "False flags at 1% fakes" = share of flagged posts that are real when 1 post in 100 is fake.
FaceFusion was skipped and the through-X copies aren't in yet, so those columns are absent.

| Model | Val AUC | Val AP | Held-out Grok AUC | Threshold | Val recall | Held-out Grok caught | Held-out real flagged | False flags at 1% fakes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CLIP probe | 0.903 | 0.940 | 0.429 | 0.698 | 74% | 12% | 20% | 99% |
| DINOv3 probe | 0.887 | 0.925 | 0.662 | 0.746 | 65% | 4% | 0% | 0% |
| **DINOv3 + Grok** (ships) | 0.887 | 0.925 | **0.686** | **0.742** | 65% | 4% | **0%** | 0% |
| DINOv3 + LoRA (with Grok) | **0.929** | **0.957** | 0.514 | 0.764 | 79% | 92% | 92% | 99% |
| Reconstruction (SD1.5 VAE) | 0.796 | 0.846 | 0.228 | 0.078 dist. | 27% | 12% | 39% | 100% |

**Decision (the plan's rule: LoRA ships only if it beats DINOv3 + Grok on val and on the held-out teammate's
Grok edits).** LoRA wins val but loses held-out Grok (0.514 vs 0.686), so the DINOv3 + Grok probe ships with
threshold 0.742 and LoRA stays a row here.

- At that threshold the shipped probe flags none of Sean's 49 real photos, but catches only 1 of his 24 Grok
  edits: it is precise and misses most Grok. Every detector here trades one for the other on Grok.
- The reconstruction score works on DeepFakeFace (val 0.80) and not on phone photos or Grok; it is evidence
  for open-source SD-family images only.
- Pitch framing: the detector is one evidence line next to identity match, C2PA, reconstruction, post text
  and clone spread, and the user's confirmation decides. Grok-made fakes of an unseen person remain the open
  problem, as the 2026 literature reports for closed generators.
- Val stays put (DINOv3 0.887, CLIP 0.903), so adding Grok costs nothing on DeepFakeFace.

## LoRA fine-tune of DINOv3 (`ml/finetune.py`)

LoRA r=16 on the last 8 blocks, both views through the backbone, 3 epochs at batch 32 (~7 min per epoch on
an RTX 4080 Laptop GPU, 6.3 GB). Val AUC by epoch: 0.909, 0.926, **0.928**, against 0.887 for the frozen
DINOv3 probe. Its Grok test row comes from the retrain with Grok in training (Task 4), so it isn't listed here.

## Reconstruction score (`api/recon.py`, AEROBLADE-style)

LPIPS distance between a 256px native-resolution center crop and its VAE reconstruction; lower = more likely
generated. AUC on 40 images per class (DeepFakeFace originals; team photos for Grok):

| VAE | txt2img vs real | inpaint vs real | swap vs real | Grok vs team real |
|---|---:|---:|---:|---:|
| SD1.5 | **0.92** | **0.97** | 0.66 | 0.32 |
| SDXL | 0.66 | 0.71 | 0.65 | 0.33 |
| FLUX | 0.71 | 0.81 | 0.83 | 0.18 |
| min of the three | 0.71 | 0.81 | 0.83 | 0.18 |

- It catches images decoded by the same VAE (DeepFakeFace's fakes come from SD1.5) and says nothing useful
  about face swaps or Grok, as expected for a closed tool with its own decoder.
- FLUX's 16-channel VAE rebuilds camera photos almost perfectly too, so adding it to the minimum hurts.
  analyze() uses the SD1.5 VAE alone: ~4 s per image on 2 CPU threads.
