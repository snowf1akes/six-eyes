# Six Eyes

**Tagline:** Find the deepfakes of you, and use the law to take them down.

---

## Inspiration

AI can now turn one public photo of anyone into fake explicit content in minutes. The people it happens to usually don't know, and when they find out, they hit two walls.

**Finding it.** The images are scattered across strangers' accounts and reposted faster than one person can search. X can't even search by image, let alone by face. Victims end up hunting for their own abuse, post by post, seeing it again with every result.

**Removing it.** Since May 19, 2026, the **TAKE IT DOWN Act** (Pub. L. 119-12, § 3) has forced platforms to remove nonconsensual intimate images, including AI fakes, within **48 hours** of a valid request. But a request only counts if it contains every element the law requires, it has to be filed platform by platform, and if the platform ignores it, the next step is a complaint to the FTC. Almost nobody knows how any of that works.

The law finally gives victims leverage. We built Six Eyes so they can actually use it.

[Optional: add one personal sentence about why this mattered to your team.]

## What it does

### How it works (5 steps)

**1️⃣ Enroll, as math rather than photos**
The user uploads 3–5 selfies, signs the authorization, and adds their name and X handle. We keep a 512-number face embedding and **discard the photos**.

**2️⃣ Discover**
Six Eyes searches X for posts that name or tag the user, and sweeps for posts using deepfake language ("AI nude," "face swap"…). Every image is face-matched against the user's embedding.

**3️⃣ Verify, with three independent lines of evidence**
- A **DINOv3** vision-transformer detector looking at two views: the aligned face and the full image
- A **reconstruction test**: images made by Stable Diffusion rebuild almost perfectly through its own autoencoder
- **C2PA Content Credentials**, the provenance signature AI tools embed

Matches appear on the dashboard **blurred by default**. The user clicks *"This is me"* and *"Not consensual."* The models gather evidence, and **the user's confirmation decides**.

**4️⃣ Take it down, in one click**
A **Factory Droid** case worker drafts a TAKE IT DOWN notice for **every copy** of the image, mapped element by element to § 3(a)(1)(B), and Six Eyes files as the user's authorized representative.

**5️⃣ Follow up, and escalate**
The 48-hour clock starts. When it runs out, the agent checks every post again:
- **Gone →** case closed, and the user gets a text.
- **Still up →** Six Eyes builds a complete FTC complaint packet (URLs, image hashes, timeline, evidence, a copy of the notice) for TakeItDown.ftc.gov and texts the user through **Photon**.

## What makes us different

| | Doing it yourself | Six Eyes |
|---|---|---|
| **Finding images** | Manually scrolling, re-seeing the abuse | Face-matched search + deepfake-term sweep |
| **Is it fake?** | Guesswork | 3 evidence layers + calibrated detector |
| **Reposts** | Report each copy one by one | PDQ clone clusters: confirm once, cover every copy |
| **Legal request** | Must know the Act's 4 required elements | Template mapped to § 3(a)(1)(B), auto-filled |
| **48-hour deadline** | Track it yourself | An agent owns the clock |
| **Platform ignores it** | Figure out the FTC process | FTC packet prepared, user notified |
| **Privacy** | — | Selfies discarded; raw media never stored |

## The math (why it works)

**One person per enrollment.** Every selfie must match the average of the others:

$$\cos\left(v_i,\ \bar v_{-i}\right) \ge \tau_{\text{same}} \quad \forall i$$

so nobody can enroll with someone else's face.

**Clone clusters.** Two images are the same picture if their 256-bit PDQ perceptual hashes are close:

$$\text{Hamming}(h_a, h_b) \le 31$$

That survives compression, resizing and light crops, so reposts are caught.

**Calibrated threshold.** We don't use 0.5. The detector's cutoff is the lowest score that keeps **≥ 95% precision** on validation:

$$t^* = \min \{\, t : \text{precision}(t) \ge 0.95 \,\} = 0.742$$

In a tool that tells people an image is fake, false accusations are the failure mode we care about most.

## How we built it

**Backend:** Python + **FastAPI** with SQLite, deployed on **Modal**. Scans run in the background so they outlast Modal's 150 s request limit, and we face-match *before* running the heavy models, so only true matches pay for detection.

**Identity:** **InsightFace** (buffalo_l / ArcFace) embeddings, L2-normalized and averaged across selfies. Media is processed in memory and never written to storage.

**Discovery:** the **X API v2**: a targeted search (`"name" OR @handle OR to:handle`, media only) plus a keyword sweep, with context signals (flagged hashtags, link domains, account age, follower ratio).

**Detection:** we trained on **DeepFakeFace**: 10,000 real images plus text-to-image, inpainting and face-swap fakes, and 3,000 *aligned* fakes we made ourselves by passing real photos through the SD 1.5, SDXL and FLUX autoencoders, so the model learns the decoder's fingerprint and not the face. We compared **CLIP ViT-L/14** and **DINOv3 ViT-L** probes (UnivFD-style), plus a **LoRA** fine-tune of DINOv3 (rank 16, last 8 blocks). We added an **AEROBLADE-style** reconstruction score (LPIPS through the SD 1.5 VAE), a **C2PA** check, and an NSFW classifier.

**An honest test set:** we photographed our own team (with written consent) and made **49 Grok Imagine edits**, a generator the model never saw in training, then held out one teammate entirely.

**Clone tracking:** **PDQ** perceptual hashing (Meta's open-source algorithm).

**Legal engine:** a TAKE IT DOWN notice template mapped to every element of § 3(a)(1)(B), X's official report form mapped field by field, and an authorization the user e-signs at enrollment.

**Case worker:** a **Factory Droid** agent with exactly **five tools** (draft notice, wait, check live, escalate, notify), served over an in-process MCP server. Guardrails live in code, not in the prompt: autonomy is off, every action goes through a permission handler, it can only touch its own case, and it **can't escalate** unless it has itself checked that the post is still up after the deadline. Every tool call is logged. An hourly Modal job wakes any deadline a restart might have missed.

**Notifications:** **Photon** SMS on every state change. The images are never sent.

**Frontend:** **Next.js** + React + TypeScript: enrollment, and a review dashboard with blurred evidence cards.

## Results

**Detector, tested on a teammate it never saw and a generator (Grok) it never trained on:**

| Model | Val AUC | Held-out Grok AUC | Real photos falsely flagged |
|---|---:|---:|---:|
| CLIP probe | 0.903 | 0.429 | 20% |
| DINOv3 + LoRA | **0.929** | 0.514 | 92% |
| **DINOv3 + Grok probe (shipped)** | 0.887 | **0.686** | **0%** |

LoRA looked best on paper, but on the held-out teammate it collapsed, flagging 92% of his *real* photos. It had learned "a teammate's phone photo," not Grok's fingerprint. **We shipped the model that generalized**: zero false flags on the held-out teammate's real photos.

- **~23,000** training and validation images · **148** team test images, never trained on
- **5** agent tools, each with code-level guardrails
- **48-hour** deadline tracked automatically, with escalation in **1** step

## Challenges we ran into

- **X can't search by image.** We built two discovery paths (targeted and sweep) and made face-matching the gate.
- **Modal's 150 s request limit.** Early scans timed out. We moved scans to the background and only scored images whose face already matched the user, which got full scans through.
- **The lab-to-real gap.** Detectors scoring 0.90+ AUC on benchmarks dropped to near chance on Grok images. That's the same problem the 2026 research literature reports for closed generators. We stopped chasing the benchmark, built an honest held-out test, and designed Six Eyes so **the detector is one line of evidence and the user decides.**
- **The law is a web form.** X only accepts TAKE IT DOWN reports through a web form, and the FTC's complaint system is too. We mapped both field by field, and we deliberately keep a human on the final submit, because these are legal statements.

## Accomplishments that we're proud of

- An end-to-end pipeline, from **selfie to FTC packet**, in one weekend.
- A detection evaluation we'd defend in front of researchers, with **a held-out person and an unseen generator**, where we shipped the honest winner, not the flashy one.
- Turning a federal statute into code: every required element of the notice is enforced before anything is sent.
- **Privacy by design:** selfies discarded, media never stored, images blurred until the user chooses to look.

## What we learned

- **Benchmarks lie.** Validation AUC told us LoRA was best. A held-out person told us the truth.
- **Guardrails belong in tools, not prompts.** Our agent can make a wrong turn and fail safely, because the rules are enforced in code.
- **The law is the product.** The hardest and most valuable part wasn't the model, it was making a legal process usable by someone at the worst moment of their life.

## What's next for Six Eyes

- **More platforms:** each one is one connector plus one channel entry (Instagram, Bluesky, Reddit, TikTok, Snap are already mapped in `channels.json`).
- **A better Grok-era detector:** more generators in training, and through-X compression tests.
- **Sandboxed analysis per post** (e.g. in Daytona), so raw media never touches our servers.
- **StopNCII integration**, so blocked images can't be re-uploaded.
- **Minor safety:** routing anyone under 18 to NCMEC's Take It Down.

## Built With

`python` · `fastapi` · `sqlite` · `modal` · `pytorch` · `transformers` · `dinov3` · `clip` · `peft` · `lora` · `diffusers` · `lpips` · `insightface` · `opencv` · `scikit-learn` · `pdqhash` · `c2pa` · `x-api` · `grok` · `factory-droid` · `mcp` · `photon` · `next.js` · `react` · `typescript` · `vercel`

## Sponsor challenges

- **Digital Trust:** identity-verified takedowns, privacy by design, and a legally grounded process.
- **Factory:** the Droid case worker owns each case from notice to resolution or escalation, with five guarded tools over MCP.
- **Photon:** SMS updates to the user on every case change.
- **SpaceXAI:** we used **Grok Imagine** to build our out-of-distribution test set: 49 consented edits the model never saw.
- [**Daytona:** only tick this if A confirms Daytona is actually used. The repo currently runs analysis on Modal.]

## Try it out

- GitHub: https://github.com/snowf1akes/six-eyes
- Demo video: [YouTube link]
