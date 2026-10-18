# Notice template notes (C → B)

`notice.md` is the TAKE IT DOWN notice. Render it with Python `str.format(**fields)`. Every `{name}` is a placeholder, and the template has no other curly braces.

X's form takes separate fields, not one letter, so the full notice is the **record**: the case timeline, the dry-run packet, the FTC packet's "notice copy", and channels that accept a written notice. X's "Please provide more details about what's happening" field gets the short `details.md` (about 490 characters rendered, using a subset of the same placeholders). The other X fields are mapped one by one in `api/data/channels.json`.

## Placeholders

| Placeholder | What to put in it | Example |
|---|---|---|
| `notice_date` | Date the notice is rendered or sent | `October 17, 2026` |
| `platform_name` | From `channels.json` `name` | `X` |
| `case_id` | `cases.id` | `42` |
| `user_name` | Enrolled user's full name (`users.name`) | `Sean Kim` |
| `authorization_date` | Date the user e-signed the enroll-page authorization (`users.signed_at`) | `October 17, 2026` |
| `reported_username` | X handle of the post's author, **without** `@` (the template adds it). Needs `author.username` from the X API expansion. | `example_account` |
| `content_urls` | One line per URL: every post in the confirmed clone cluster, as `- https://x.com/...` | see below |
| `depiction_kind` | Pick from the two strings below | |
| `signer_name` | The authorized representative: a real, accountable person | `Raymond Wang` |
| `signer_relationship` | `Authorized representative of {user_name}`, filled in by code | `Authorized representative of Sean Kim` |
| `contact_email` | The team's new reporting inbox | |

`depiction_kind` values:
- likely fake (high `fake_prob`): `a digitally altered or AI-generated image of them (a "digital forgery" as defined in the Act)`
- otherwise: `an intimate image of them`

## How the template maps to the Act, §3(a)(1)(B)

| Required element | Where it is in `notice.md` |
|---|---|
| (i) Physical or electronic signature of the individual **or an authorized person acting on their behalf** | Section 4: `/s/ {signer_name}` + `{signer_relationship}` |
| (ii) Identification of the depiction and information reasonably sufficient to **locate** it | Section 1: post URLs + account handle |
| (iii) Brief statement of the individual's **good faith belief** that the depiction is **not consensual**, with any relevant information | Section 2 |
| (iv) Information sufficient to **contact** the individual or authorized person | Section 3: name + email |

## Legal flags, for C to review

1. **The authorization must exist before any notice.** Never render a notice for a user without `signed_at`. The enroll-page text (`web/content/consent.md`) must name who is authorized, for what, and how to revoke.
2. **The good-faith belief is the user's, not ours.** Section 2 is only true after the user clicks "This is me" and "Not consensual." Never generate a notice from detector scores alone.
3. **The Act only covers intimate visual depictions.** A notice for non-intimate content isn't a valid TAKE IT DOWN request. Gate on the NSFW score plus the user's confirmation. (The hackathon test posts are SFW, so they only ever run with `DRY_RUN=1`.)
4. **Never submit for real during the hackathon.** Our test posts were published *with* consent, so a real notice would be a false statement. Keep `DRY_RUN=1`.
5. **Minors:** if the depicted person is or was under 18, don't use this flow. Route to NCMEC Take It Down (takeitdown.ncmec.org).
6. **One notice can cover the whole clone cluster.** List every URL, and the Act's identical-copies duty (§3(a)(3)) covers the rest.
7. **Signature:** an electronic signature (`/s/ Name`) is fine under the Act. The signer must be a real person who can stand behind the statement, not "Six Eyes AI."

This is a working reading of the Act's text, not legal advice. C reviews it against Pub. L. 119-12 before the demo.
