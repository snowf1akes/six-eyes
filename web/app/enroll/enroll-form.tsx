"use client";

import Link from "next/link";
import { useEffect, useMemo, useState, type FormEvent, type ReactNode } from "react";
import { ShimmerButton } from "@/components/ui/shimmer-button";
import { Check, CircleAlert, CircleCheck, ImagePlus } from "../icons";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const MAX_SELFIES = 5;

type Enrolled = { user_id: number; photos_used: number; handles: string[]; signed_at: string };
type Errors = Partial<Record<"name" | "selfies" | "consent" | "signature", string>>;

export function EnrollForm({ consent }: { consent: ReactNode }) {
  const [files, setFiles] = useState<File[]>([]);
  const [errors, setErrors] = useState<Errors>({});
  const [serverError, setServerError] = useState("");
  const [busy, setBusy] = useState(false);
  const [enrolled, setEnrolled] = useState<Enrolled | null>(null);

  const previews = useMemo(() => files.map((f) => URL.createObjectURL(f)), [files]);
  useEffect(() => () => previews.forEach((u) => URL.revokeObjectURL(u)), [previews]);

  function addSelfies(input: HTMLInputElement) {
    // Copy before clearing: input.files is live, and clearing lets the same photo be picked again.
    const picked = Array.from(input.files ?? []);
    input.value = "";
    const next = [...files, ...picked];
    setFiles(next.slice(0, MAX_SELFIES));
    setErrors((e) => ({
      ...e,
      selfies: next.length > MAX_SELFIES
        ? `You can add up to ${MAX_SELFIES} selfies. We kept the first ${MAX_SELFIES}.`
        : undefined,
    }));
  }

  async function onSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const form = new FormData(e.currentTarget);
    const name = String(form.get("name") ?? "").trim();
    const signature = String(form.get("signature") ?? "").trim();

    const next: Errors = {};
    if (!name) next.name = "Enter your full name.";
    if (files.length < 3 || files.length > MAX_SELFIES)
      next.selfies = `Choose 3 to ${MAX_SELFIES} selfies. You chose ${files.length}.`;
    if (form.get("consent") !== "on") next.consent = "Agree to the consent and authorization to enroll.";
    if (!signature || signature.toLowerCase() !== name.toLowerCase())
      next.signature = "Type your full name exactly as you entered it above.";
    setErrors(next);
    setServerError("");
    if (Object.keys(next).length) return;

    const body = new FormData();
    body.set("name", name);
    body.set("handles", String(form.get("handles") ?? ""));
    body.set("phone", String(form.get("phone") ?? ""));
    body.set("signature", signature);
    body.set("consent", "true");
    files.forEach((f) => body.append("selfies", f));

    setBusy(true);
    try {
      const res = await fetch(`${API}/enroll`, { method: "POST", body });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        setServerError(typeof data.detail === "string" ? data.detail : "Enrollment didn't go through. Check your details and try again.");
        return;
      }
      setEnrolled(data);
      // Start the first scan now, so found posts are waiting by the time the dashboard opens.
      fetch(`${API}/scan/${data.user_id}?sweep=true&background=true`, { method: "POST" }).catch(() => {});
    } catch {
      setServerError("Can't reach the Six Eyes server. Check that the API is running, then try again.");
    } finally {
      setBusy(false);
    }
  }

  if (enrolled) {
    const handles = enrolled.handles.map((h) => `@${h}`).join(", ");
    return (
      <section className="sheet lifted enroll-sheet" aria-live="polite">
        <CircleCheck className="done-mark" />
        <h2 className="title">You’re enrolled</h2>
        <p className="lede">
          We built your face match from {enrolled.photos_used} selfies and deleted the photos.
          {handles ? ` Six Eyes will search X for posts that mention ${handles}.` : ""}
        </p>
        <div className="actions">
          <span className="chip chip-done">
            <Check />
            Enrolled
          </span>
          <span className="meta">
            Signed {new Date(enrolled.signed_at).toLocaleString()} · User {enrolled.user_id}
          </span>
          <Link className="btn btn-outline" href={`/dashboard?user=${enrolled.user_id}`}>
            See found posts
          </Link>
        </div>
      </section>
    );
  }

  return (
    <form className="sheet lifted enroll-sheet" onSubmit={onSubmit} noValidate>
      <section className="enroll-section">
        <h2 className="title">About you</h2>
        <div className="field">
          <label className="label" htmlFor="name">Full name</label>
          <input
            className="input" id="name" name="name" autoComplete="name" required
            aria-invalid={!!errors.name} aria-describedby={errors.name ? "name-error" : undefined}
          />
          {errors.name && <p className="error" id="name-error" role="alert">{errors.name}</p>}
        </div>
        <div className="two-up">
          <div className="field">
            <label className="label" htmlFor="handles">X handle</label>
            <input className="input" id="handles" name="handles" placeholder="@yourhandle" autoComplete="off" aria-describedby="handles-hint" />
            <p className="hint" id="handles-hint">Separate several with commas. We search X for posts that mention them.</p>
          </div>
          <div className="field">
            <label className="label" htmlFor="phone">
              Phone <span className="optional">(optional)</span>
            </label>
            <input className="input" id="phone" name="phone" type="tel" autoComplete="tel" aria-describedby="phone-hint" />
            <p className="hint" id="phone-hint">We'll text you when a report is filed, and if X hasn't taken the posts down in 48 hours.</p>
          </div>
        </div>
      </section>

      <section className="enroll-section perforation">
        <h2 className="title">Selfies</h2>
        <p className="hint" id="selfies-hint">
          Choose 3 to 5 clear, front-facing photos of just you. We use them once to build your face
          match, then delete them.
        </p>
        <div className={`slot${errors.selfies ? " invalid" : ""}`}>
          {previews.map((src, i) => (
            // eslint-disable-next-line @next/next/no-img-element -- local blob previews, never uploaded as-is
            <img key={src} src={src} alt={`Selfie ${i + 1}`} className="photo" />
          ))}
          <input
            className="file-input" id="selfies" type="file" multiple
            accept="image/jpeg,image/png,image/webp"
            disabled={files.length >= MAX_SELFIES}
            aria-invalid={!!errors.selfies}
            aria-describedby={errors.selfies ? "selfies-hint selfies-error" : "selfies-hint"}
            onChange={(e) => addSelfies(e.currentTarget)}
          />
          {files.length < MAX_SELFIES && (
            <label className="btn btn-outline" htmlFor="selfies">
              <ImagePlus />
              {files.length ? "Add more" : "Choose selfies"}
            </label>
          )}
          {files.length > 0 && (
            <>
              <span className="meta">{files.length} of {MAX_SELFIES}</span>
              <button className="btn btn-ghost" type="button" onClick={() => { setFiles([]); setErrors((e) => ({ ...e, selfies: undefined })); }}>
                Clear
              </button>
            </>
          )}
        </div>
        {errors.selfies && <p className="error" id="selfies-error" role="alert">{errors.selfies}</p>}
      </section>

      <section className="enroll-section perforation">
        <h2 className="title">Consent and authorization</h2>
        {consent}
        <div className="field">
          <label className="check">
            <input
              type="checkbox" name="consent" aria-invalid={!!errors.consent}
              aria-describedby={errors.consent ? "consent-error" : undefined}
            />
            I have read the consent and authorization above and I agree to it.
          </label>
          {errors.consent && <p className="error" id="consent-error" role="alert">{errors.consent}</p>}
        </div>
        <div className="field">
          <label className="label" htmlFor="signature">Signature</label>
          <input
            className="input" id="signature" name="signature" autoComplete="off" placeholder="Type your full name"
            aria-invalid={!!errors.signature}
            aria-describedby={errors.signature ? "signature-hint signature-error" : "signature-hint"}
          />
          <p className="hint" id="signature-hint">Typing your full name is your electronic signature.</p>
          {errors.signature && <p className="error" id="signature-error" role="alert">{errors.signature}</p>}
        </div>
      </section>

      {serverError && (
        <div className="note is-error" role="alert">
          <CircleAlert />
          <span className="note-title">Not enrolled yet</span>
          <p>{serverError}</p>
        </div>
      )}
      <div className="actions">
        <ShimmerButton className="btn btn-lg" type="submit" disabled={busy} borderRadius="10px" background="var(--ink)">
          {busy ? "Enrolling…" : "Get started"}
        </ShimmerButton>
      </div>
    </form>
  );
}
