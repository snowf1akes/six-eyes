"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { Check, CircleAlert, CircleCheck, ExternalLink, Eye, EyeOff } from "../icons";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type Signals = {
  keywords: string[];
  hashtags: string[];
  domains: string[];
  author_age_days: number | null;
  follower_ratio: number | null;
  engagement: Record<string, number>;
};
type Scores = {
  face_sim: number;
  frame_drift: number | null;
  fake_prob: number;
  generator_guess: string;
  recon_score: number | null;
  c2pa: boolean | null;
  nsfw: number;
  signals: Signals;
  source: "keyword" | "search" | "sweep";
  media: { key: string; type: "photo" | "video" | "animated_gif"; url: string };
  text: string;
};
type Case = { id: number; url: string; found_at: string | null; copies: number; priority: number; scores: Scores };
type Packet = {
  reported_username: string;
  content_urls: string[];
  notice: string;
  form: { question: string; required: boolean; type: string; preset: boolean; value: string }[];
  checkboxes: { label: string; check: boolean; why: string }[];
};
type Notice = {
  dry_run: boolean;
  intimate: boolean;
  platform: string;
  form_url: string | null;
  delivered_to: string | null;
  due_at: string;
  packets: Packet[];
};
type FtcField = { question: string; required: boolean; type: string; value: string; screen?: string };
type Settled = {
  case_id: number;
  resolved: number[];
  escalated: number[];
  ftc_packet: string | null;
  ftc_url: string | null;
  ftc_form?: FtcField[] | null;
  dry_run?: boolean | null;
};
type Filing = {
  caseId: number;
  ids: number[];
  notice: Notice | null;
  error: string | null;
  formOpened?: boolean;
  floating?: boolean;
  outcome?: Settled;
};
// Chrome and Edge only: a small window that stays on top of every other window, the platform's form included.
type DocumentPictureInPicture = { requestWindow(options: { width: number; height: number }): Promise<Window> };
type Helper = { float: Promise<Window | null> | null; form: Window | null };
type Floating = { win: Window; root: HTMLElement; caseId: number };
type Load = { status: "loading" } | { status: "error"; message: string } | { status: "ready"; cases: Case[] };

const GENERATORS: Record<string, string> = {
  real: "Looks like a real photo",
  swap: "Face swap",
  inpaint: "Edited region (inpainting)",
  txt2img: "Generated from a text prompt",
  "vae-decoded": "Passed through an AI image model",
  grok: "Edited with Grok Imagine",
  "no-face": "No face to check",
  "not-scored": "Not checked: the face doesn’t match you",
};
// Same cutoffs as the API: analyze.FAKE_THRESHOLD (95% precision on val) and main.INTIMATE_MIN.
const FAKE_MIN = 0.742;
const EXPLICIT_MIN = 0.5;
type Filter = { label: string; note: string; keep: (s: Scores) => boolean; rank: (c: Case) => number };
const FILTERS: Filter[] = [
  { label: "All", note: "", keep: () => true, rank: (c) => c.priority },
  {
    label: "Likely deepfake",
    note: "AI-made or AI-edited. The TAKE IT DOWN Act covers it only if it’s also intimate.",
    keep: (s) => s.fake_prob >= FAKE_MIN,
    rank: (c) => c.scores.fake_prob,
  },
  {
    label: "Explicit",
    note: "Intimate images: the TAKE IT DOWN Act covers them whether they’re real or AI-made.",
    keep: (s) => s.nsfw >= EXPLICIT_MIN,
    rank: (c) => c.scores.nsfw,
  },
  {
    label: "Deepfake and explicit",
    note: "AI-made intimate images, the Act’s “digital forgery” cases and the strongest to report.",
    keep: (s) => s.fake_prob >= FAKE_MIN && s.nsfw >= EXPLICIT_MIN,
    rank: (c) => c.scores.fake_prob * c.scores.nsfw,
  },
];
const SOURCES: Record<Scores["source"], string> = {
  keyword: "Found by your name with a deepfake term",
  search: "Found by name search",
  sweep: "Found by deepfake terms and your face",
};
const ENGAGEMENT: [string, string][] = [
  ["like", "Likes"],
  ["retweet", "Reposts"],
  ["quote", "Quotes"],
  ["reply", "Replies"],
  ["impression", "Views"],
];

const pct = (n: number) => `${Math.round(n * 100)}%`;

function identityLabel(sim: number) {
  if (sim >= 0.5) return "Strong match";
  if (sim >= 0.3) return "Possible match";
  return "Unlikely to be you";
}

export function Cases() {
  const user = useSearchParams().get("user");
  const [load, setLoad] = useState<Load>({ status: "loading" });
  const [filings, setFilings] = useState<Filing[]>([]);
  const [settled, setSettled] = useState<Settled[]>([]);
  const [floating, setFloating] = useState<Floating | null>(null);
  const [ftcFloat, setFtcFloat] = useState<{ win: Window; root: HTMLElement; s: Settled; formOpened: boolean } | null>(null);
  const [reload, setReload] = useState(0);
  const [filter, setFilter] = useState(0);
  const refresh = useCallback(() => setReload((n) => n + 1), []);

  function onConfirmed(filing: Filing, { float, form }: Helper) {
    setLoad((l) => (l.status === "ready" ? { ...l, cases: l.cases.filter((c) => !filing.ids.includes(c.id)) } : l));
    const url = filing.notice?.form_url;
    if (form && url) form.location.href = url;
    else form?.close();
    setFilings((f) => [{ ...filing, formOpened: Boolean(form && url) }, ...f]);
    float?.then((win) => {
      if (!win || win.closed) return;
      if (!filing.notice) return win.close();
      const root = win.document.createElement("div");
      root.className = "float";
      win.document.body.replaceChildren(root);
      win.addEventListener("pagehide", () => setFloating((cur) => (cur?.win === win ? null : cur)));
      setFloating({ win, root, caseId: filing.caseId });
      setFilings((f) => f.map((x) => (x.caseId === filing.caseId ? { ...x, floating: true } : x)));
    });
  }

  function onRetried(filing: Filing) {
    setFilings((f) => f.map((x) => (x.caseId === filing.caseId ? filing : x)));
  }

  function fileFtc(s: Settled) {
    const opening = openFloat("The FTC form opens beside this. Follow each screen: choose the matching option, or paste when it asks for text.");
    const form = openForm(Boolean(opening));
    const start = ftcStartUrl(s.ftc_url);
    if (form && start) form.location.href = start;
    else form?.close();
    opening?.then((win) => {
      if (!win || win.closed) return;
      const root = win.document.createElement("div");
      root.className = "float";
      win.document.body.replaceChildren(root);
      win.document.title = "Six Eyes · FTC";
      win.addEventListener("pagehide", () => setFtcFloat((cur) => (cur?.win === win ? null : cur)));
      setFtcFloat({ win, root, s, formOpened: Boolean(form && s.ftc_url) });
    });
  }

  useEffect(() => {
    if (!user) return;
    let live = true;
    fetch(`${API}/cases/${encodeURIComponent(user)}`)
      .then(async (res) => {
        const data = await res.json().catch(() => ({}));
        if (!live) return;
        if (!res.ok)
          setLoad({ status: "error", message: typeof data.detail === "string" ? data.detail : "Couldn't load your found posts." });
        else {
          setLoad({ status: "ready", cases: data.cases });
          const now = data.settled ?? [];
          const open = data.escalations ?? [];
          const seen = new Set(now.map((s: Settled) => s.case_id));
          setSettled([...now, ...open.filter((s: Settled) => !seen.has(s.case_id) && (s.ftc_packet || s.escalated?.length))]);
        }
      })
      .catch(() => live && setLoad({ status: "error", message: "Can't reach the Six Eyes server. Check that the API is running, then reload." }));
    return () => {
      live = false;
    };
  }, [user, reload]);

  if (!user)
    return (
      <Note title="No account selected">
        Open this page from your enrollment confirmation, or <Link href="/enroll">enroll first</Link>.
      </Note>
    );
  if (load.status === "loading") return <p className="meta">Loading found posts…</p>;
  if (load.status === "error") return <Note title="Couldn’t load found posts">{load.message}</Note>;

  const floated = floating && filings.find((f) => f.caseId === floating.caseId);
  const active = FILTERS[filter];
  const shown = load.cases.filter((c) => active.keep(c.scores)).sort((a, b) => active.rank(b) - active.rank(a));
  return (
    <>
      {floating && floated?.notice && createPortal(<FloatPanel notice={floated.notice} formOpened={!!floated.formOpened} />, floating.root)}
      {ftcFloat && createPortal(<FtcFloatPanel s={ftcFloat.s} formOpened={ftcFloat.formOpened} />, ftcFloat.root)}
      {filings.map((f) => (
        <FilingNote key={f.caseId} filing={f} onRetried={onRetried} onFileFtc={fileFtc} />
      ))}
      {settled.map((s) => (
        <SettledNote key={s.case_id} s={s} onFile={fileFtc} />
      ))}
      <Scan user={user} onProgress={refresh} />
      {load.cases.length ? (
        <>
          <div className="case-filters" role="group" aria-label="Show found posts">
            {FILTERS.map((f, i) => (
              <button
                key={f.label}
                type="button"
                className={`btn btn-outline btn-sm case-filter${i === filter ? " pressed" : ""}`}
                aria-pressed={i === filter}
                onClick={() => setFilter(i)}
              >
                {f.label} · {load.cases.filter((c) => f.keep(c.scores)).length}
              </button>
            ))}
          </div>
          {active.note && <p className="meta">{active.note}</p>}
          <p className="meta">
            {filter ? `${shown.length} of ` : ""}
            {load.cases.length} found {load.cases.length === 1 ? "post" : "posts"} · User {user}
          </p>
          {shown.length ? (
            <ol className="case-list">
              {shown.map((c) => (
                <li key={c.id}>
                  <CaseCard c={c} onConfirmed={onConfirmed} />
                </li>
              ))}
            </ol>
          ) : (
            <p className="lede">No found posts match this filter.</p>
          )}
        </>
      ) : (
        <p className="lede">
          {filings.length ? "Nothing left to review." : "No posts found yet."} Scan X to look for posts that show you.
        </p>
      )}
    </>
  );
}

type ScanState =
  | { status: "idle" }
  | { status: "scanning"; checked: number; found: number }
  | { status: "done"; checked: number; found: number; failed: number }
  | { status: "error"; message: string };

type ScanStatus = { running: boolean; posts_checked: number; cases_found: number; media_failed: number; error: string | null };

// Starts a background scan (start = true) or reads the latest one's progress.
async function scanStatus(user: string, start = false): Promise<ScanStatus> {
  const res = await fetch(`${API}/scan/${encodeURIComponent(user)}${start ? "?sweep=true&background=true" : ""}`, {
    method: start ? "POST" : "GET",
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : "The scan didn’t start. Try again.");
  return data;
}

// Searches X for the user's name and handles, plus deepfake keywords matched against their face. The scan runs
// in the background on the server, because a first scan can outlast a web request; this polls it every few
// seconds and refreshes the list as new posts are found.
function Scan({ user, onProgress }: { user: string; onProgress: () => void }) {
  const [scan, setScan] = useState<ScanState>({ status: "idle" });
  const live = useRef(true);

  const follow = useCallback(
    async (s: ScanStatus) => {
      let shown = -1;
      while (s.running && live.current) {
        setScan({ status: "scanning", checked: s.posts_checked, found: s.cases_found });
        if (s.cases_found !== shown) {
          shown = s.cases_found;
          onProgress();
        }
        await new Promise((r) => setTimeout(r, 3000));
        s = await scanStatus(user);
      }
      if (!live.current) return;
      onProgress();
      setScan(
        s.error
          ? { status: "error", message: s.error }
          : { status: "done", checked: s.posts_checked, found: s.cases_found, failed: s.media_failed },
      );
    },
    [user, onProgress],
  );

  // Pick up a scan that's already running, such as the one enrolling starts.
  useEffect(() => {
    live.current = true;
    scanStatus(user)
      .then((s) => (s.running ? follow(s) : undefined))
      .catch(() => {});
    return () => {
      live.current = false;
    };
  }, [user, follow]);

  async function run() {
    setScan({ status: "scanning", checked: 0, found: 0 });
    try {
      await follow(await scanStatus(user, true));
    } catch (e) {
      setScan({ status: "error", message: e instanceof TypeError ? "Can't reach the Six Eyes server. Check that the API is running." : (e as Error).message });
    }
  }

  return (
    <div className="scan">
      <div className="actions">
        <button className="btn" type="button" disabled={scan.status === "scanning"} onClick={run}>
          {scan.status === "scanning" ? "Scanning X…" : "Scan X now"}
        </button>
        <span className="meta" role="status">
          {scan.status === "scanning" &&
            `Checked ${scan.checked} ${scan.checked === 1 ? "post" : "posts"} so far, found ${scan.found} new. ` +
              "New posts appear below as they're found."}
          {scan.status === "done" &&
            `Checked ${scan.checked} ${scan.checked === 1 ? "post" : "posts"}, found ${scan.found} new.` +
              (scan.failed ? ` ${scan.failed} couldn’t be downloaded.` : "")}
        </span>
      </div>
      {scan.status === "error" && <p className="error" role="alert">{scan.message}</p>}
    </div>
  );
}

function CaseCard({ c, onConfirmed }: { c: Case; onConfirmed: (f: Filing, h: Helper) => void }) {
  const s = c.scores;
  const sig = s.signals;
  const author = new URL(c.url).pathname.split("/")[1];
  const terms = [...sig.keywords, ...sig.hashtags.map((h) => `#${h}`)];
  const flags = [...terms, ...sig.domains];

  return (
    <article className="sheet case" aria-labelledby={`case-${c.id}`}>
      <BlurredMedia media={s.media} />
      <div className="case-body">
        <header className="case-head">
          <div>
            <h2 className="title" id={`case-${c.id}`}>
              Priority {Math.round(c.priority * 100)}
            </h2>
            <p className="meta">
              @{author} · {SOURCES[s.source]}
              {c.found_at && ` · ${new Date(c.found_at).toLocaleString()}`}
            </p>
            {terms.length > 0 && <p className="meta">Uses {terms.join(", ")}</p>}
          </div>
          <a className="btn btn-outline" href={c.url} target="_blank" rel="noreferrer noopener">
            <ExternalLink />
            Open on X
          </a>
        </header>
        {s.text && <p className="case-text">{s.text}</p>}

        <div className="evidence">
          <Evidence title="Identity">
            <Row label="Face match" value={`${pct(s.face_sim)} · ${identityLabel(s.face_sim)}`} />
            {s.frame_drift !== null && (
              <Row label="Change across frames" value={s.frame_drift.toFixed(4)} hint="Higher means the face flickers, common in face swaps." />
            )}
          </Evidence>
          <Evidence title="Authenticity and content">
            <Row label="Likely fake" value={pct(s.fake_prob)} />
            <Row label="Looks like" value={GENERATORS[s.generator_guess] ?? s.generator_guess} />
            <Row
              label="Explicit"
              value={pct(s.nsfw)}
              hint="From an NSFW image classifier. 50% and up counts as intimate under the TAKE IT DOWN Act."
            />
            <Row
              label="Reconstruction distance"
              value={s.recon_score === null ? "Not measured" : s.recon_score.toFixed(3)}
              hint="Lower means an AI image model can recreate it, a sign it was generated."
            />
            <Row
              label="Content Credentials"
              value={s.c2pa === null ? "None found" : s.c2pa ? "Present" : "Present but invalid"}
            />
          </Evidence>
          <Evidence title="Distribution">
            <Row label="Copies" value={c.copies === 1 ? "Only this post" : `Seen in ${c.copies} posts`} />
            <Row
              label="Account"
              value={[
                sig.author_age_days === null ? null : `${Math.round(sig.author_age_days)} days old`,
                sig.follower_ratio === null ? null : `${sig.follower_ratio} followers per account followed`,
              ].filter(Boolean).join(" · ") || "Unknown"}
            />
            <Row
              label="Engagement"
              value={ENGAGEMENT.filter(([k]) => k in sig.engagement)
                .map(([k, label]) => `${sig.engagement[k].toLocaleString()} ${label.toLowerCase()}`)
                .join(" · ") || "None reported"}
            />
            <Row label="Flagged terms and links" value={flags.length ? flags.join(", ") : "None"} />
          </Evidence>
        </div>
        <Attest caseId={c.id} copies={c.copies} onConfirmed={onConfirmed} />
      </div>
    </article>
  );
}

const when = (iso: string) =>
  new Date(iso).toLocaleString(undefined, { weekday: "short", hour: "numeric", minute: "2-digit" });

function FilingNote({ filing, onRetried, onFileFtc }: { filing: Filing; onRetried: (f: Filing) => void; onFileFtc: (s: Settled) => void }) {
  const [busy, setBusy] = useState(false);
  const [skipError, setSkipError] = useState<string | null>(null);
  const { notice: n, error, ids, outcome } = filing;
  const posts = `${ids.length} ${ids.length === 1 ? "post" : "posts"}`;

  async function retry() {
    setBusy(true);
    try {
      const res = await fetch(`${API}/cases/${filing.caseId}/notice`, { method: "POST" });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : "The removal request wasn’t made.");
      onRetried({ ...filing, notice: data, error: null });
    } catch (e) {
      onRetried({ ...filing, error: e instanceof TypeError ? "Can't reach the Six Eyes server." : (e as Error).message });
    } finally {
      setBusy(false);
    }
  }

  async function skipWait() {
    setBusy(true);
    setSkipError(null);
    try {
      const res = await fetch(`${API}/timewarp/${filing.caseId}`, { method: "POST" });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Couldn't skip the wait.");
      onRetried({ ...filing, outcome: data });
    } catch (e) {
      setSkipError(e instanceof TypeError ? "Can't reach the Six Eyes server." : (e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  if (!n)
    return (
      <div className="note is-error" role="alert">
        <CircleAlert />
        <span className="note-title">{posts} confirmed, but the removal request wasn’t made</span>
        <p>{error}</p>
        <div className="note-more actions">
          <button className="btn btn-outline" type="button" disabled={busy} onClick={retry}>
            {busy ? "Trying…" : "Try again"}
          </button>
        </div>
      </div>
    );

  const reports = n.packets.length === 1 ? "1 report" : `${n.packets.length} reports`;
  return (
    <div className="note" role="status">
      <CircleCheck />
      <span className="note-title">
        {n.dry_run ? `Test: ${reports} for ${n.platform} prepared` : `${reports} ready to file with ${n.platform}`}
      </span>
      <p>
        {n.dry_run
          ? `Test mode: our test posts were shared with consent, so don’t submit these.${n.delivered_to ? ` A copy went to ${n.delivered_to}.` : ""}`
          : `Submit each report through ${n.platform}’s form. ${n.platform} must remove the posts within 48 hours of receiving it.`}
        {n.dry_run && !n.intimate && " Live, this would be refused: it doesn’t look intimate, so the TAKE IT DOWN Act doesn’t cover it."}
        {!n.dry_run && ` Six Eyes will check back after ${when(n.due_at)}.`}
        {filing.formOpened && ` ${n.platform}’s form is open in a new ${filing.floating ? "tab" : "window"}.`}
        {filing.floating && ` The answers are also in a floating window that stays on top of ${n.platform}’s form.`}
      </p>
      {n.packets.map((p) => (
        <FormGuide key={p.reported_username} packet={p} formUrl={n.form_url} platform={n.platform} />
      ))}
      {outcome ? (
        <div className="note-more">
          <SettledNote s={outcome} onFile={onFileFtc} />
        </div>
      ) : n.dry_run ? (
        <div className="note-more actions">
          <button className="btn btn-outline" type="button" disabled={busy} onClick={skipWait}>
            {busy ? "Checking X…" : "Skip the 48-hour wait (test)"}
          </button>
          {skipError && <p className="error" role="alert">{skipError}</p>}
        </div>
      ) : null}
    </div>
  );
}

function SettledNote({ s, onFile }: { s: Settled; onFile?: (s: Settled) => void }) {
  const gone = s.resolved.length;
  const up = s.escalated.length;
  const posts = (n: number) => `${n} ${n === 1 ? "post" : "posts"}`;
  const title = up
    ? gone
      ? `${posts(gone)} taken down, ${posts(up)} still up`
      : `${posts(up)} still up after the deadline`
    : `${posts(gone)} taken down`;
  const steps = s.ftc_form?.length ? s.ftc_form : s.ftc_packet
    ? [{ question: "Full complaint packet", required: false, type: "textarea", value: s.ftc_packet }]
    : [];
  return (
    <div className="note" role="status">
      {up ? <CircleAlert /> : <CircleCheck />}
      <span className="note-title">{title}</span>
      <p>
        {up
          ? "X did not remove the remaining posts in time. Next step: a complaint to the FTC (it enforces the TAKE IT DOWN Act). Open the form — a guide stays on top matching each FTC screen. Paste the post facts; type your own name and email. You don’t need to describe the images. Don’t file a test report for real."
          : "X removed the reported posts. Nothing else to do."}
      </p>
      {up && (s.ftc_url || steps.length > 0) && (
        <div className="note-more">
          <div className="form-guide-head">
            {onFile && s.ftc_url ? (
              <button className="btn btn-primary" type="button" onClick={() => onFile(s)}>
                <ExternalLink />
                File with the FTC
              </button>
            ) : s.ftc_url ? (
              <a className="btn btn-outline" href={s.ftc_url} target="_blank" rel="noreferrer noopener">
                <ExternalLink />
                Escalate at TakeItDown.ftc.gov
              </a>
            ) : null}
          </div>
          {steps.length > 0 && <FtcGuide steps={steps} />}
        </div>
      )}
    </div>
  );
}

function FtcGuide({ steps }: { steps: FtcField[] }) {
  return (
    <div className="form-steps-grouped">
      {groupFtc(steps).map((g) => (
        <section key={g.screen || g.fields[0].index}>
          {g.screen ? <h3 className="form-screen">{g.screen}</h3> : null}
          <ol className="form-steps">
            {g.fields.map(({ field: f, index }) => (
              <li key={index}>
                <span className="form-q">
                  {f.question}
                  {!f.required && <span className="hint"> (optional)</span>}
                </span>
                <div className="form-a">
                  <span className={f.value.includes("\n") ? "fp-value clamp" : undefined}>{ftcAnswer(f)}</span>
                  {ftcCopyable(f) && <CopyButton text={f.value} />}
                </div>
              </li>
            ))}
          </ol>
        </section>
      ))}
    </div>
  );
}

function FtcFloatPanel({ s, formOpened }: { s: Settled; formOpened: boolean }) {
  const steps = s.ftc_form?.length ? s.ftc_form : s.ftc_packet
    ? [{ question: "Full complaint packet", required: false, type: "textarea", value: s.ftc_packet }]
    : [];
  const [done, setDone] = useState<number[]>([]);
  return (
    <>
      <header className="fp-head">
        <div className="fp-brand">
          <span className="fp-kicker">Six Eyes · Report to the FTC</span>
          <span className={`chip ${s.dry_run ? "chip-test" : "chip-done"}`}>{s.dry_run ? "Test" : "Live"}</span>
        </div>
        <h1 className="title">TakeItDown.ftc.gov</h1>
        <p className={`fp-callout${s.dry_run ? " is-test" : ""}`}>
          {s.dry_run
            ? "Test mode: our test posts were shared with consent, so don’t submit this. Type your own name and email when the form asks."
            : "Follow each screen. Choose the option named here, or paste when it asks for text. Type your own name and email — Six Eyes only fills the post facts. You don’t need to upload or describe the images."}
        </p>
      </header>
      <main className="fp-body">
        {groupFtc(steps).map((g) => (
          <section key={g.screen || g.fields[0].index}>
            {g.screen ? <h2 className="fp-screen">{g.screen}</h2> : null}
            <ol className="fp-steps">
              {g.fields.map(({ field: f, index: i }) => {
                const isDone = done.includes(i);
                return (
                  <li key={i} className={`fp-step${isDone ? " done" : ""}`}>
                    <span className="fp-num" aria-label={isDone ? "Copied" : `Step ${i + 1}`}>
                      {isDone ? <Check /> : i + 1}
                    </span>
                    <span className="fp-q">
                      {f.question}
                      {!f.required && " (optional)"}
                    </span>
                    <div className="fp-a">
                      <span className={`fp-value${f.value.includes("\n") ? " clamp" : ""}`}>{ftcAnswer(f)}</span>
                      {ftcCopyable(f) && (
                        <CopyButton text={f.value} onCopied={() => setDone((d) => [...new Set([...d, i])])} />
                      )}
                    </div>
                  </li>
                );
              })}
            </ol>
          </section>
        ))}
      </main>
      <footer className="fp-foot">
        <span className="fp-progress" role="status">
          {(() => {
            const copyable = steps.filter(ftcCopyable).length;
            const n = done.filter((i) => ftcCopyable(steps[i])).length;
            if (!copyable) return "Choose each option on the matching screen";
            return n === copyable ? "All answers copied" : `${n} of ${copyable} copied`;
          })()}
        </span>
        {ftcStartUrl(s.ftc_url) && (
          <a className={`btn ${formOpened ? "btn-outline" : "btn-primary"}`} href={ftcStartUrl(s.ftc_url)!} target="_blank" rel="noreferrer noopener">
            <ExternalLink />
            {formOpened ? "Reopen the FTC form" : "Open the FTC form"}
          </a>
        )}
      </footer>
    </>
  );
}

// One report through the platform's form: each question with the answer to paste, then the checkboxes.
function FormGuide({ packet, formUrl, platform }: { packet: Packet; formUrl: string | null; platform: string }) {
  const preset = packet.form.filter((f) => f.preset);
  return (
    <section className="note-more form-guide">
      <div className="form-guide-head">
        <span className="note-title">
          Report @{packet.reported_username} · {packet.content_urls.length}{" "}
          {packet.content_urls.length === 1 ? "post" : "posts"}
        </span>
        {formUrl && (
          <a className="btn btn-outline" href={formUrl} target="_blank" rel="noreferrer noopener">
            <ExternalLink />
            Open {platform}’s form
          </a>
        )}
      </div>
      {preset.length > 0 && (
        <p className="hint">
          The form opens with {preset.map((f) => `“${f.value}”`).join(" and ")} already chosen. If it asks anyway, pick those.
        </p>
      )}
      <ol className="form-steps">
        {packet.form.filter((f) => !f.preset).map((f) => (
          <li key={f.question}>
            <span className="form-q">
              {f.question}
              {!f.required && <span className="hint"> (optional)</span>}
            </span>
            <div className="form-a">
              <span>{f.type === "select" ? `Choose “${f.value}”` : f.value}</span>
              {f.type !== "select" && f.type !== "you" && f.type !== "skip" && <CopyButton text={f.value} />}
            </div>
          </li>
        ))}
        {packet.checkboxes.map((b) => (
          <li key={b.label}>
            <span className="form-q">{b.check ? "Tick" : "Leave unticked"}: {b.label}</span>
            <span className="hint">{b.why}</span>
          </li>
        ))}
      </ol>
      <details>
        <summary>Read the notice</summary>
        <pre className="notice-text">{packet.notice}</pre>
      </details>
    </section>
  );
}

// The floating window: what to paste into the platform's form, step by step, readable at a glance beside it.
function FloatPanel({ notice: n, formOpened }: { notice: Notice; formOpened: boolean }) {
  const [tab, setTab] = useState(0);
  const [copied, setCopied] = useState<Record<string, number[]>>({});
  const packet = n.packets[Math.min(tab, n.packets.length - 1)];
  const preset = packet.form.filter((f) => f.preset);
  const steps = packet.form.filter((f) => !f.preset);
  const done = copied[packet.reported_username] ?? [];
  const copyable = steps.filter((f) => f.type !== "select" && f.type !== "you" && f.type !== "skip").length;
  const posts = (k: number) => `${k} ${k === 1 ? "post" : "posts"}`;

  return (
    <>
      <header className="fp-head">
        <div className="fp-brand">
          <span className="fp-kicker">Six Eyes · Report to {n.platform}</span>
          <span className={`chip ${n.dry_run ? "chip-test" : "chip-done"}`}>{n.dry_run ? "Test" : "Live"}</span>
        </div>
        {n.packets.length > 1 ? (
          <div className="fp-tabs" role="tablist" aria-label="Accounts to report">
            {n.packets.map((p, i) => (
              <button
                key={p.reported_username}
                className={`btn btn-outline btn-sm fp-tab${i === tab ? " pressed" : ""}`}
                type="button"
                role="tab"
                aria-selected={i === tab}
                onClick={() => setTab(i)}
              >
                @{p.reported_username} · {p.content_urls.length}
              </button>
            ))}
          </div>
        ) : (
          <h1 className="title">
            @{packet.reported_username} <span className="meta">· {posts(packet.content_urls.length)}</span>
          </h1>
        )}
        <p className={`fp-callout${n.dry_run ? " is-test" : ""}`}>
          {n.dry_run
            ? `Test mode: our test posts were shared with consent, so don’t submit this.${n.intimate ? "" : " Live, it wouldn’t qualify: it doesn’t look intimate."}`
            : `${n.platform} must remove the posts within 48 hours of receiving this. Type your own email and name; paste the post URLs Six Eyes filled in.`}
        </p>
      </header>

      <main className="fp-body">
        {preset.length > 0 && (
          <details className="hint">
            <summary>The link answers the first {preset.length} questions</summary>
            If {n.platform} asks anyway, choose {preset.map((f) => `“${f.value}”`).join(" and ")}.
          </details>
        )}
        <ol className="fp-steps">
          {steps.map((f, i) => {
            const isDone = done.includes(i);
            return (
              <li key={`${packet.reported_username} ${f.question}`} className={`fp-step${isDone ? " done" : ""}`}>
                <span className="fp-num" aria-label={isDone ? "Copied" : `Step ${i + 1}`}>
                  {isDone ? <Check /> : i + 1}
                </span>
                <span className="fp-q">
                  {f.question}
                  {!f.required && " (optional)"}
                </span>
                <div className="fp-a">
                  <span className={`fp-value${f.value.includes("\n") ? " clamp" : ""}`}>
                    {f.type === "select" ? `Choose “${f.value}”` : f.value}
                  </span>
                  {f.type !== "select" && f.type !== "you" && f.type !== "skip" && (
                    <CopyButton
                      text={f.value}
                      onCopied={() => setCopied((c) => ({ ...c, [packet.reported_username]: [...new Set([...done, i])] }))}
                    />
                  )}
                </div>
              </li>
            );
          })}
        </ol>

        <section className="fp-section" aria-labelledby="fp-boxes">
          <h2 className="fp-kicker" id="fp-boxes">Checkboxes</h2>
          <ul className="fp-boxes">
            {packet.checkboxes.map((b) => (
              <li key={b.label} className="fp-box">
                <span className={`fp-tick${b.check ? " on" : ""}`} role="img" aria-label={b.check ? "Tick" : "Leave unticked"}>
                  {b.check && <Check />}
                </span>
                <span className="fp-box-label">{b.label}</span>
                {b.why && <span className="hint">{b.why}</span>}
              </li>
            ))}
          </ul>
        </section>
      </main>

      <footer className="fp-foot">
        <span className="fp-progress" role="status">
          {done.length === copyable ? "All answers copied" : `${done.length} of ${copyable} copied`}
        </span>
        {n.form_url && (
          <a className={`btn ${formOpened ? "btn-outline" : "btn-primary"}`} href={n.form_url} target="_blank" rel="noreferrer noopener">
            <ExternalLink />
            {formOpened ? `Reopen ${n.platform}’s form` : `Open ${n.platform}’s form`}
          </a>
        )}
      </footer>
    </>
  );
}

function ftcStartUrl(url: string | null) {
  if (!url) return null;
  const base = url.replace(/\/$/, "");
  return base === "https://takeitdown.ftc.gov" ? `${base}/get-started` : url;
}

function ftcAnswer(f: FtcField) {
  return f.type === "select" ? `Choose “${f.value}”` : f.value;
}

function ftcCopyable(f: FtcField) {
  return f.type !== "select" && f.type !== "skip" && f.type !== "you";
}

function groupFtc(steps: FtcField[]) {
  const groups: { screen: string; fields: { field: FtcField; index: number }[] }[] = [];
  steps.forEach((field, index) => {
    const screen = field.screen ?? "";
    if (!groups.length || groups[groups.length - 1].screen !== screen) groups.push({ screen, fields: [{ field, index }] });
    else groups[groups.length - 1].fields.push({ field, index });
  });
  return groups;
}

function CopyButton({ text, onCopied }: { text: string; onCopied?: () => void }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      className="btn btn-ghost btn-sm"
      type="button"
      onClick={async (e) => {
        // Inside the floating window, only that window's clipboard is allowed: it has the focus.
        const nav = e.currentTarget.ownerDocument.defaultView?.navigator ?? navigator;
        await nav.clipboard.writeText(text);
        setCopied(true);
        onCopied?.();
      }}
    >
      {copied && <Check />}
      {copied ? "Copied" : "Copy"}
    </button>
  );
}

// Each new window uses up the click that opened it, so the floating window opens on the first of the two presses
// and the platform's form on the second. Both must be called during the click.
function openFloat(wait = "Press both buttons and your report appears here. This window stays on top of other windows."): Promise<Window | null> | null {
  const pip = (window as Window & { documentPictureInPicture?: DocumentPictureInPicture }).documentPictureInPicture;
  if (!pip) return null;
  return pip.requestWindow({ width: 440, height: 680 }).then(
    (win) => {
      win.document.documentElement.className = document.documentElement.className;
      win.document.body.className = document.body.className;
      for (const node of document.head.querySelectorAll('link[rel="stylesheet"], style'))
        win.document.head.append(node.cloneNode(true));
      win.document.title = "Six Eyes report";
      floatMessage(win, wait);
      return win;
    },
    () => null,
  );
}

function floatMessage(win: Window, text: string) {
  const box = win.document.createElement("div");
  const kicker = win.document.createElement("span");
  const p = win.document.createElement("p");
  box.className = "fp-wait";
  kicker.className = "fp-kicker";
  kicker.textContent = "Six Eyes";
  p.textContent = text;
  box.append(kicker, p);
  win.document.body.replaceChildren(box);
}

// A tab when the floating window has the answers; otherwise a window on the right half, beside the dashboard.
function openForm(asTab: boolean): Window | null {
  const half = Math.round(screen.availWidth / 2);
  const win = asTab
    ? window.open("", "_blank")
    : window.open("", "take-it-down", `popup,width=${half},height=${screen.availHeight},left=${half},top=0`);
  if (!win) return null;
  win.opener = null;
  win.document.title = "Preparing your report…";
  win.document.body.textContent = "Preparing your report…";
  return win;
}

// Both statements are required before a case moves on; pressing the second one confirms.
function Attest({ caseId, copies, onConfirmed }: { caseId: number; copies: number; onConfirmed: (f: Filing, h: Helper) => void }) {
  const [isMe, setIsMe] = useState(false);
  const [notConsensual, setNotConsensual] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const float = useRef<Promise<Window | null> | null>(null);

  function closeFloat() {
    float.current?.then((w) => w?.close());
    float.current = null;
  }

  async function toggle(which: "isMe" | "notConsensual") {
    const next = { isMe, notConsensual, [which]: !(which === "isMe" ? isMe : notConsensual) };
    setIsMe(next.isMe);
    setNotConsensual(next.notConsensual);
    setError("");
    const pressed = Number(next.isMe) + Number(next.notConsensual);
    if (pressed === 1 && !float.current) {
      const opening = openFloat();
      float.current = opening;
      opening?.then((w) => {
        if (!w && float.current === opening) float.current = null;
      });
    }
    if (pressed === 0) closeFloat();
    if (pressed < 2) return;
    const helper: Helper = { float: float.current, form: openForm(Boolean(float.current)) };
    float.current = null;
    helper.float?.then((w) => w && floatMessage(w, "Preparing your report…"));
    setBusy(true);
    try {
      const res = await fetch(`${API}/cases/${caseId}/confirm`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ is_me: true, not_consensual: true }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Couldn’t confirm this post.");
      // The server files from the first newly confirmed case, so retries use that one too.
      onConfirmed(
        { caseId: data.confirmed[0] ?? caseId, ids: data.confirmed, notice: data.notice, error: data.notice_error },
        helper,
      );
    } catch (e) {
      helper.float?.then((w) => w?.close());
      helper.form?.close();
      setError(e instanceof TypeError ? "Can't reach the Six Eyes server. Try again." : (e as Error).message);
      setNotConsensual(false);
      setIsMe(false);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="attest">
      <p className="hint">
        Press both to confirm.
        {copies === 2 && " The other post with this image moves too."}
        {copies > 2 && ` The ${copies - 1} other posts with this image move too.`}
      </p>
      <div className="actions">
        <AttestButton pressed={isMe} disabled={busy} onClick={() => toggle("isMe")}>
          This is me
        </AttestButton>
        <AttestButton pressed={notConsensual} disabled={busy} onClick={() => toggle("notConsensual")}>
          Not consensual
        </AttestButton>
        {busy && <span className="meta">Confirming…</span>}
      </div>
      {error && <p className="error" role="alert">{error}</p>}
    </div>
  );
}

function AttestButton({ pressed, disabled, onClick, children }: { pressed: boolean; disabled: boolean; onClick: () => void; children: ReactNode }) {
  return (
    <button className={`btn btn-outline attest-btn${pressed ? " pressed" : ""}`} type="button" aria-pressed={pressed} disabled={disabled} onClick={onClick}>
      {pressed && <Check />}
      {children}
    </button>
  );
}

// Plain <img>/<video> straight from X's CDN: next/image would proxy and cache a copy on our server.
function BlurredMedia({ media }: { media: Scores["media"] }) {
  const [shown, setShown] = useState(false);
  const isVideo = media.type !== "photo";
  return (
    <div className={`case-media${shown ? " shown" : ""}`}>
      {isVideo ? (
        <video
          src={media.url}
          controls={shown}
          muted
          loop={media.type === "animated_gif"}
          playsInline
          preload="metadata"
          aria-label={shown ? "Video from the post" : "Video from the post, blurred"}
        />
      ) : (
        // eslint-disable-next-line @next/next/no-img-element -- see above: must not pass through our server
        <img
          src={media.url}
          alt={shown ? "Image from the post" : "Image from the post, blurred"}
          referrerPolicy="no-referrer"
          loading="lazy"
        />
      )}
      <button className={`btn ${shown ? "btn-outline" : "btn-primary"} reveal`} type="button" aria-pressed={shown} onClick={() => setShown(!shown)}>
        {shown ? <EyeOff /> : <Eye />}
        {shown ? "Blur again" : "Reveal"}
      </button>
    </div>
  );
}

function Evidence({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="evidence-group">
      <h3 className="evidence-title">{title}</h3>
      <dl>{children}</dl>
    </section>
  );
}

function Row({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="evidence-row">
      <dt>{label}</dt>
      <dd>
        {value}
        {hint && <span className="hint">{hint}</span>}
      </dd>
    </div>
  );
}

function Note({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="note is-error" role="alert">
      <CircleAlert />
      <span className="note-title">{title}</span>
      <p>{children}</p>
    </div>
  );
}
