# Context Handoff: AI-Generated Image & Deepfake Detection Research

**Prepared:** 2026-10-17 · **User:** Sean (Harvard FAS/SEAS)
**Purpose:** Carry the research and discussion from a prior Claude session into a new one.

---

## 1. What the user is doing

- Started from **Ojha, Li & Lee, "Towards Universal Fake Image Detectors that Generalize Across Generative Models"** (CVPR 2023, arXiv [2302.10174](https://arxiv.org/abs/2302.10174), last revised April 2024).
- Asked for a survey of advances from 2024 to October 2026 on two things: (a) how deepfakes are detected, and (b) why deepfakes are distinguishable from real media.
- Follow-up questions covered: how detectors work, what makes a deepfake a deepfake, whether detectors target those traits, and whether a CNN trained on a closed set of 5–10 known generators would work.
- A full literature-review doc of about 120 papers with links exists as a Claude Doc: **"AI-Generated Image & Deepfake Detection: Advances Since Ojha et al. (2024–Oct 2026)"**, https://claude.ai/code/artifact/7528e1d6-b526-4516-898b-cd3775ab9585. One open comment in it asks whether the reading list should include video and audio papers as well as still images. No answer yet.

---

## 2. Baseline: what UnivFD (Ojha et al.) showed

- Trained real/fake classifiers become **asymmetric**. They learn "fake = patterns of the training generator," so **"real" becomes a catch-all** that also absorbs fakes from unseen generators.
- Ojha's fix: **frozen CLIP ViT-L/14 features** with nearest-neighbour or a **linear probe**, trained only on ProGAN. Result on unseen diffusion and autoregressive models: +15.07 mAP and +25.90% accuracy.
- It left three open questions:
  1. Should the feature space be frozen or adapted?
  2. What is CLIP actually picking up?
  3. Do the gains hold in the wild?

---

## 3. Key findings, 2024 to Oct 2026

1. **Frozen probes on stronger backbones are state of the art again (2026).**
   - 2024–25 work adapted CLIP: FatFormer, RINE, C2P-CLIP, Effort (ICML 2025; SVD keeps the main components frozen and trains only an orthogonal remainder, 0.19M parameters, 99.41 mAP).
   - By 2026, a **linear probe on frozen DINOv3, Perception Encoder or MetaCLIP-2** beats specialised detectors by more than 20–30 points in the wild:
     - [Simplicity Prevails](https://arxiv.org/abs/2602.01738), arXiv Feb 2026
     - [Brought a Gun to a Knife Fight](https://arxiv.org/abs/2509.12995), Chameleon 96.1%
   - **Caveat:** possible pretraining contamination. On images posted after the models' training cutoffs (WebAIG-25), MetaCLIP-2 fell to 51.4% and PE to 68.1%. DINOv3 held at 94.5%.
2. **Much of the claimed generalisation came from dataset shortcuts.**
   - In GenImage, reals are JPEG at varied sizes and fakes are fixed-size PNG ([Fake or JPEG?](https://arxiv.org/abs/2403.17608)). Removing the bias took ResNet-50 cross-generator accuracy from 71.7% to 82.7%.
   - Fixes that align the training data:
     - [AlignedForensics](https://arxiv.org/abs/2410.11835) (Ojha's group, ICLR 2025): fakes are real images passed through the latent diffusion model's autoencoder.
     - [B-Free](https://arxiv.org/abs/2412.17671) (CVPR 2025).
     - [DDA](https://arxiv.org/abs/2505.14359) (NeurIPS 2025): also aligns frequency content; 90.7% across 11 benchmarks.
     - [Stay-Positive](https://arxiv.org/abs/2502.07778) (Ojha's group, ICML 2025): features associated with real images are not allowed to lower the fake score.
   - Data has mattered more than architecture.
3. **Scaling generator diversity helps.**
   - [Community Forensics](https://arxiv.org/abs/2411.04125) (CVPR 2025): 2.7M images from 4,803 generators. Gains continue up to about 1,000 generators. **End-to-end training beat frozen backbones** at this scale.
   - [D³](https://arxiv.org/abs/2404.04584) (CVPR 2025): multi-generator training with a discrepancy branch.
4. **Lab results don't carry over to real-world content.**
   - [Deepfake-Eval-2024](https://arxiv.org/abs/2503.02857): open-source detectors lose about 50% AUC on video, 48% on audio and 45% on images, measured on real 2024 deepfakes.
   - [Ren et al. 2026](https://arxiv.org/html/2602.07814v1): Flux Dev, Firefly v4 and Midjourney v7 were caught only 18–30% of the time.
   - [GPT Image 2 / Nano Banana 2 benchmark](https://arxiv.org/html/2608.01258v1) (Aug 2026): detectors lose 14–33 points.
   - [Chameleon / AIDE](https://arxiv.org/abs/2406.19435) (ICLR 2025): UnivFD and other detectors call nearly all realistic fakes "real."
   - Social-media re-sharing (TrueFake, WildRF, AIGIBench) breaks most detectors.
5. **What CLIP/DINO probes actually detect.**
   - [Cozzolino, Poggi & Verdoliva, Aug 2026](https://arxiv.org/abs/2608.12155): the cues are non-semantic and in the low-to-mid frequency range. Generated images show lower variance and lower effective dimensionality.
   - [Willi et al. 2026](https://arxiv.org/abs/2602.12381): the cues are dataset-specific style and "polish." There is no universal signature.
6. **Video and audio lag behind images.**
   - [AIGVDBench](https://openaccess.thecvf.com/content/CVPR2026/papers/Ma_Your_One-Stop_Solution_for_AI-Generated_Video_Detection_CVPR_2026_paper.pdf) (CVPR 2026): vision-language models are near chance; UnivFD drops from 85.7 to 65.3 AUC on closed generators; Effort is best.
   - [RobustSora](https://arxiv.org/html/2512.10248): detectors partly key on Sora's visible watermark.
   - [ML-ITW](https://arxiv.org/html/2603.05852v3) (speech): error goes from 0.15% to 49.4% in the wild.
   - Codec-based text-to-speech beats most audio detectors.
7. **Humans are near chance on still images and faces.**
   - About 48–62% accuracy: Frank et al. (IEEE S&P 2024), Cooke et al. (CACM 2025), the Microsoft 2025 study, McGuire et al. (J. Vision 2026).
   - People do better with time and motion: video detection rose from 57.7% to 74.6% at 8 s (Farid lab 2026).
   - [Dawel et al. PNAS 2026](https://www.pnas.org/doi/10.1073/pnas.2602122123): training on global impressions (symmetry, distinctiveness) works; training to spot artifacts helps little.
8. **Watermarking and provenance are now deployed, and all have been attacked.**
   - Deployed: SynthID (Text in Nature 2024; SynthID-Image covers 10B+ images), C2PA, and OpenAI adopting SynthID + C2PA in May 2026.
   - Required: the [EU AI Act Article 50](https://ec.europa.eu/commission/presscorner/detail/en/ip_26_1328) code of practice requires two or more machine-readable marks, applying from 2 Aug 2026.
   - Removal attacks: UnMarker (IEEE S&P 2025), "provably removable" (NeurIPS 2024), MarkNull (Aug 2026, claims 100% removal of SynthID-Image).
   - Passive detectors like UnivFD are now framed as a **fallback layer**.
9. **High-impact journals.** Nature, Science and PNAS published **no new detector architectures** in this period. Their coverage is human-perception studies, watermarking, and news features. Detector research appears at CVPR, ICCV, ECCV, NeurIPS, ICML, ICLR, AAAI and IEEE TIFS/TPAMI.
10. **Theory.**
    - If a generator matches the real distribution, no detector can beat chance ([Aczel et al. 2025](https://arxiv.org/html/2509.21135)).
    - There is a provable trade-off between evasion and false positives ([Saberi et al. ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/744e09d3213ee53de88f04164a9b06ab-Abstract.html)).
    - Detectability persists in practice because each new architecture leaves new traces, but each trace is short-lived.

---

## 4. How detectors work: five families

| Family | Mechanism | Examples |
| --- | --- | --- |
| End-to-end classifier | CNN trained on labelled real and fake images; learns whatever separates them; overfits to the training generator | Wang et al. CVPR 2020 (CNNSpot) |
| Foundation-model probe | Frozen pretrained encoder turns an image into a feature vector; a small linear classifier is trained on top. Lightly adapted variants also exist | UnivFD, RINE, FatFormer, C2P-CLIP, Effort, DINOv3/PE probes |
| Low-level artifact | Suppress image content, then classify the remainder: noise residuals, FFT/DCT, neighbouring-pixel differences, shuffled patches | NPR (CVPR 2024), SAFE, Breaking Semantic Artifacts (NeurIPS 2024) |
| Reconstruction / real-only | Generated images reconstruct almost perfectly through the generator's autoencoder or diffusion model, real ones don't; or model real images only and flag outliers | AEROBLADE, DIRE, LaRE², FIRE, DRCT, ZED, SPAI, Forensic Self-Descriptions |
| Semantic / physical reasoning | Extract geometry (lines, vanishing points, shadows) or motion statistics; or a multimodal LLM explains the anomaly | Sarkar et al. (CVPR 2024), NSG-VD, FakeVLM, LEGION, SIDA, AIGI-Holmes |

Hybrids such as AIDE and CO-SPY combine a semantic branch with a low-level branch.

---

## 5. What makes a deepfake a deepfake

**Definition.** Strictly, a face swap or reenactment made with deep learning. Broadly, any AI-generated or AI-altered media: full synthesis, face swap, lip-sync, inpainting and local edits, voice clones.

**The core gap is the camera pipeline versus the generator pipeline.**
- **Real photo:** lens → sensor recording one colour per pixel (Bayer filter) → demosaicing to fill in the other colours → sensor noise → JPEG. The scene obeys physics.
- **Latent diffusion image:** noise in a small compressed latent (e.g. 64×64×4 for a 512px image) → iterative denoising → the VAE decoder upsamples about 8× into pixels. A GAN generator upsamples a small tensor.

**Tells:**
1. **Upsampling correlations.** Neighbouring pixels are tied together, giving periodic spectral peaks. These are strong in older GANs and weaker in many diffusion models.
2. **Missing camera traces.** No demosaicing pattern, no physically realistic sensor noise, no lens behaviour.
3. **Decoder fingerprint and manifold.** Generated images lie exactly on the set of images the decoder can produce; real photos lie off it. This is why reconstruction-error detectors work. Lost fine detail gives generated images a "polish."
4. **Statistical narrowness.** Diffusion outputs regress toward the mean: lower variance, unusual colour distributions, low-to-mid-frequency differences. Faces come out average, symmetric and attractive ("AI hyperrealism", Miller et al. 2023). Humans misread this as realism.
5. **No world model.** Inconsistent perspective, shadows, reflections and lighting. Errors in hands, teeth, earrings, text, hybrid objects. In video: object permanence, flicker, implausible physics.
   - [LEGION](https://arxiv.org/html/2503.15264v1): about 89% of annotated artifacts are structural and about 6% are physics.
   - [Physics-IQ](https://openaccess.thecvf.com/content/WACV2026/html/Motamed_Do_Generative_Video_Models_Understand_Physical_Principles_WACV_2026_paper.html): the best video model scores 29.5%; Sora looked most real but scored 10% on physics. Realism and physics are uncorrelated.
   - [Shedding Light](https://arxiv.org/html/2609.10787): lighting errors of about 11° are now mostly below the 30–40° humans can notice.
6. **Face-swap-specific.** Blending boundary mismatches; feature jitter between frames from frame-by-frame swapping; audio–lip desynchronisation.
7. **Audio.** Vocoder and neural-codec artifacts; unnatural prosody; missing breathing or room acoustics.

**Spurious cues that are not deepfake-ness:** JPEG vs PNG, image size, scene content, Sora's visible watermark.

---

## 6. Do detectors target these traits?

| Trait | Detectors that target it | How | How long it lasts |
| --- | --- | --- | --- |
| Upsampling correlations | NPR, spectral detectors | Neighbouring-pixel differences or spectrum → classifier | Short; changes with each architecture and is destroyed by compression |
| Missing camera traces | ZED, SPAI, Forensic Self-Descriptions | Model real images only; flag outliers | Fairly long; independent of any one generator |
| Decoder fingerprint / manifold | AEROBLADE, LaRE², DIRE, FIRE, AlignedForensics | Reconstruction error; or training pairs that differ only in decoder traces | Short; broke on FLUX's 16-channel VAE |
| Statistical narrowness | Foundation-model probes (implicitly) | Linear classifier finds the separating direction in feature space | Unclear; effective but not designed for this |
| Geometry and lighting | Sarkar et al. | Extract lines, vanishing points and shadows → classifier | Longer, but the errors are shrinking |
| Video physics and motion | NSG-VD, Physics-IQ-style checks | Compare motion statistics to real video | Long for now |
| Semantic anomalies | Multimodal LLM detectors (FakeVLM, LEGION, AIGI-Holmes) | Fine-tuned to find and describe errors | Over-flag real images; ~17-point drop out of distribution (So-Fake) |
| Blending and temporal jitter | Video-level blending (CVPR 2025) | Train on synthesised artifact examples | Medium |
| Lip–audio mismatch | LipFD (NeurIPS 2024) | Audio–lip timing check | Medium |

**Points:**
- **UnivFD-style probes don't explicitly target any trait.** Ojha hypothesised that CLIP captured something broad, but the 2026 analyses show the probes use non-semantic low-to-mid-frequency statistics plus dataset-specific style cues.
- **For implicit detectors, the training data decides what they target.** If real and fake training images differ only in the decoder trace, the detector learns the decoder trace. If they also differ in JPEG, it learns JPEG. This is why the data-alignment work matters.
- **Humans and detectors use different cues.** Humans look for semantic errors; detectors mostly use low-level statistics. They fail on different items, so combining them helps ([Pehlivanoglu et al. 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC12779810/)).
- **Trade-off.** The traces easiest to target are the least durable; the durable ones (physics, camera traces) are harder to target and rarer. Hence hybrid detectors and watermarking.

---

## 7. Scenario discussed: a CNN trained on a closed set of 5–10 known generators

**Answer: yes, it can be very effective.** A closed set is the easy case. Ojha's failure mode is unseen generators.
- On generators seen in training, a ResNet-50 scores over 98.5% in GenImage; on unseen ones, about 55%.
- Community Forensics found end-to-end training beats frozen backbones when training generators are diverse.
- Generator fingerprints are distinct enough for a **multi-class head** (real + each generator), which gives source attribution and per-generator diagnostics.

**What breaks it in practice** (matters more than the architecture):
1. **A "generator" isn't one fixed thing.** Versions, samplers, steps, guidance, resolution, fine-tunes, LoRAs and upscalers all vary. Train across these settings.
2. **Processing erases the traces.** JPEG, resizing, cropping and social-media recompression destroy low-level traces: GenImage drops to about 51% under JPEG, and Facebook re-sharing costs 10–100% (TrueFake). Augment training with these operations.
3. **Shortcuts.** Match real and fake training sets on format, resolution, compression and content (AlignedForensics / B-Free style).
4. **False positives dominate when fakes are rare.** A USENIX Security 2024 SoK found 333 false alarms per true positive in a call-centre scenario. The real-image training set needs broad coverage.
5. **Partial fakes.** Inpainted regions need special handling (Stay-Positive).
6. **The closed-set assumption is fragile.** An 11th generator falls back into the "real" catch-all.

**Recommended layered system for a closed set:**
1. Check watermarks and C2PA first if the generators embed them. A detected mark is near-definitive; absence proves nothing.
2. **Main detector:** a pretrained backbone (ImageNet, CLIP or DINO) **fine-tuned** end to end or with LoRA on all 5–10 generators, with matched real images and heavy compression and resize augmentation. Expected to beat a frozen linear probe on the known set.
3. **For open-weight generators:** a reconstruction check through each generator's own autoencoder (AEROBLADE-style), training-free.
4. **Backstop for out-of-set generators:** a frozen DINOv3 linear probe or a real-only detector (ZED).

**Evaluation protocol:**
- Leave one generator out: train on 9, test on the 10th.
- Test on degraded copies of images.
- Report false-positive rate at a realistic fake-to-real base rate, not 50/50 accuracy.

Training data swings accuracy by 20–60 points ([Ren et al. 2026](https://arxiv.org/html/2602.07814v1)), more than the choice of architecture.

---

## 8. Suggested reading path (after Ojha et al.)

1. [Synthetic Image Verification in the Era of Generative AI](https://arxiv.org/pdf/2405.00196) — Tariang, Corvi, Cozzolino, Verdoliva et al., IEEE Security & Privacy 2024
2. [A Sanity Check for AI-generated Image Detection (Chameleon/AIDE)](https://arxiv.org/abs/2406.19435) — ICLR 2025
3. [Fake or JPEG?](https://arxiv.org/abs/2403.17608) — ECCVW 2024; [Aligned Datasets](https://arxiv.org/abs/2410.11835) — ICLR 2025
4. [Community Forensics](https://arxiv.org/abs/2411.04125) — CVPR 2025
5. [Simplicity Prevails](https://arxiv.org/abs/2602.01738) — arXiv 2026
6. [Understanding Why Foundation Models Work](https://arxiv.org/abs/2608.12155) — arXiv Aug 2026
7. [Shadows Don't Lie and Lines Can't Bend!](https://arxiv.org/abs/2311.17138v1) — CVPR 2024; [Physics-IQ](https://openaccess.thecvf.com/content/WACV2026/html/Motamed_Do_Generative_Video_Models_Understand_Physical_Principles_WACV_2026_paper.html) — WACV 2026
8. [Deepfake-Eval-2024](https://arxiv.org/abs/2503.02857) — arXiv 2025

**Useful surveys:**
- [A Survey of Defenses Against AI-Generated Visual Media](https://arxiv.org/html/2407.10575v3) (ACM CSUR 2025)
- [Deepfake Detection from the Reliability Perspective](https://dl.acm.org/doi/10.1145/3699710) (ACM CSUR 2024)
- [Detecting AI-Generated Video: Dual-View Survey](https://arxiv.org/html/2607.10787) (2026)
- [SoK: Watermarking for AI-Generated Content](https://arxiv.org/abs/2411.18479v1)

---

## 9. Caveats on sources

- The search was run in October 2026. Every link was opened or found during that search; four key items were also spot-checked directly (Simplicity Prevails, Cozzolino 2026, Brought a Gun…, Groh et al. Nat. Comms 2024).
- Headline numbers are as each paper reports them, on different benchmarks and protocols. They are not directly comparable across papers.
- Many 2025–2026 items are **arXiv preprints, not peer-reviewed**.
- Venues taken from curated lists rather than the venue page: AIGI-Holmes (ICCV 2025), FakeVLM (NeurIPS 2025), SAFE (KDD 2025), and the IJCNN 2025 entries (AI-GenBench, TrueFake).

## 10. Open threads / possible next steps

- The user may want to prototype the closed-set detector from Section 7, e.g. a training and evaluation plan, a dataset construction recipe, or code.
- The open question in the doc comment: extend the reading list to video and audio, or keep it on still images?
- Claude offered to add a "How it works" section (Sections 4–6 above) to the doc; not yet done.
