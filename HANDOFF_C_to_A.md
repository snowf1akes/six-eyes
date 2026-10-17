# C → A handoff: team test data

Status as of Sat Oct 17, ~4:15 PM. From C (Raymond).

## Done

### C1: Consent, photos, accounts
- **Team photos:** 100 real photos, 50 each of **Sean** and **Sunny**, with varied angles, lighting and distance.
  - JPG, about 3213×5712 px, roughly 2.5 MB each (247 MB total).
  - Location (GPS) metadata is **still in the files**. We decided to leave it, so don't publish the originals.
  - There are no photos of Raymond, so the test set has 2 people.
- **X test accounts:** 2 accounts created with X Premium. Logins are in the password manager.
- **Booth answers:** **still pending** (Daytona, Photon, SpaceXAI). See "Open items".

### C2: Test data (in progress)
- **Grok edits:** being made **by hand in the Grok app** (Premium account), not with `ml/gen_testset.py`, because we have no `XAI_API_KEY` yet.
  - The target is **25 per person (50 total)**. A daily usage cap may stop it earlier.
  - **4 done so far:** `sean_01_outfit`, `sunny_01_background`, `sean_25_outfit`, `sunny_25_lighting`.
  - Each was checked by eye: the face is unchanged, and only the outfit, background or lighting differs. All are SFW.
  - **Your naming convention is used** (`<photo>_<outfit|background|lighting>.jpg`, same subfolders), so your code can treat them like `gen_testset.py` output.
  - Grok app output is JPEG at **1008×1792**. The bytes are saved as downloaded, with no re-encoding.
- **FaceFusion swaps:** not started.
- **`labels.csv`:** not built yet. It will be generated from the folders once the edits are finished.
- **Through-X copies:** scheduled for hour 9.

## Folder layout

```
data/team/
├── real/
│   ├── sean/   sean_01.jpg … sean_50.jpg
│   └── sunny/  sunny_01.jpg … sunny_50.jpg
├── grok/
│   ├── sean/   sean_NN_<edit>.jpg      (up to 25)
│   └── sunny/  sunny_NN_<edit>.jpg     (up to 25)
├── facefusion/   (coming) e.g. sean_on_sunny_03.jpg
├── through_x/    (hour 9) X's re-compressed copies of the posted demo images
└── labels.csv    (coming) path, person, real|fake, generator
```

Edited photos (25 per person, spread across the 50 shots):
`01, 25, 13, 37, 49, 07, 31, 19, 43, 04, 28, 16, 40, 10, 34, 22, 46, 02, 26, 14, 38, 08, 32, 20, 44`

## Notes for A
- **Don't run `ml/gen_testset.py` over these photos** unless you want *extra* edits. It picks its edit type per photo with a CRC hash, so it can produce a second, differently named edit of a photo we already edited by hand.
- The manual prompts are slightly different from yours: specific outfits (navy suit, red hoodie, etc.), specific places (beach, snowy mountain, city at night, library) and lighting (golden hour, cool blue studio). They all end with an instruction to keep the face the same.
- `generator` will be `grok` for all of these, no matter which Grok model the app uses.
- `data/` is gitignored, so none of this is in the repo. See "How you get the files" below.

## Open items / questions for A
1. **Booth answers:** still waiting on Daytona (GPU quota; can sandboxes reach `pbs.twimg.com` / `video.twimg.com`?), Photon (can 2.0 receive photos?) and SpaceXAI (Grok API credits).
2. **Is 2 people enough** for the test set, or do you want a third person photographed?
3. **xAI API key:** if one turns up, should C switch to `gen_testset.py` for the remaining edits?
4. **FaceFusion:** about 20–30 swaps between Sean and Sunny. Is that enough?

## How you get the files
`data/team/` is **not on GitHub** (it's gitignored, and the photos are faces with GPS data). C will send you **`six-eyes-team-data.zip`** directly:
- **AirDrop** if we're in the same room (fastest, never leaves our laptops), or
- a **private Google Drive link** shared only with you.

Unzip it **at the root of your `six-eyes` checkout**. It extracts to `data/team/...` with the layout above. Your scripts already use `data/team/real` and `data/team/grok` relative to the repo root, so no code changes are needed. When the Grok edits, FaceFusion swaps and `labels.csv` are done, C will send an updated zip. Unzipping over the old one is safe.

Delete the zip after unzipping, and don't upload the raw team photos anywhere public (no public Hugging Face dataset, no public bucket).
