"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
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
  source: "search" | "sweep";
  media: { key: string; type: "photo" | "video" | "animated_gif"; url: string };
  text: string;
};
type Case = { id: number; url: string; found_at: string | null; copies: number; priority: number; scores: Scores };
type Load = { status: "loading" } | { status: "error"; message: string } | { status: "ready"; cases: Case[] };

const GENERATORS: Record<string, string> = {
  real: "Looks like a real photo",
  swap: "Face swap",
  inpaint: "Edited region (inpainting)",
  txt2img: "Generated from a text prompt",
  "vae-decoded": "Passed through an AI image model",
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
  const [moved, setMoved] = useState(0);
  const [reload, setReload] = useState(0);
  const refresh = useCallback(() => setReload((n) => n + 1), []);

  function onConfirmed(ids: number[]) {
    setMoved((n) => n + ids.length);
    setLoad((l) => (l.status === "ready" ? { ...l, cases: l.cases.filter((c) => !ids.includes(c.id)) } : l));
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
        else setLoad({ status: "ready", cases: data.cases });
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

  return (
    <>
      {moved > 0 && (
        <div className="note" role="status">
          <CircleCheck />
          <span className="note-title">
            {moved} {moved === 1 ? "post" : "posts"} confirmed
          </span>
          <p>Six Eyes will prepare a removal request for each one. Nothing is sent until you review it.</p>
        </div>
      )}
      <Scan user={user} onProgress={refresh} />
      {load.cases.length ? (
        <>
          <p className="meta">
            {load.cases.length} found {load.cases.length === 1 ? "post" : "posts"} · User {user}
          </p>
          <ol className="case-list">
            {load.cases.map((c) => (
              <li key={c.id}>
                <CaseCard c={c} onConfirmed={onConfirmed} />
              </li>
            ))}
          </ol>
        </>
      ) : (
        <p className="lede">
          {moved > 0 ? "Nothing left to review." : "No posts found yet."} Scan X to look for posts that show you.
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

function CaseCard({ c, onConfirmed }: { c: Case; onConfirmed: (ids: number[]) => void }) {
  const s = c.scores;
  const sig = s.signals;
  const author = new URL(c.url).pathname.split("/")[1];
  const flags = [...sig.keywords, ...sig.hashtags.map((h) => `#${h}`), ...sig.domains];

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
              @{author} · {s.source === "sweep" ? "Found by keyword search" : "Found by name search"}
              {c.found_at && ` · ${new Date(c.found_at).toLocaleString()}`}
            </p>
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
          <Evidence title="Authenticity">
            <Row label="Likely fake" value={pct(s.fake_prob)} />
            <Row label="Looks like" value={GENERATORS[s.generator_guess] ?? s.generator_guess} />
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

// Both statements are required before a case moves on; pressing the second one confirms.
function Attest({ caseId, copies, onConfirmed }: { caseId: number; copies: number; onConfirmed: (ids: number[]) => void }) {
  const [isMe, setIsMe] = useState(false);
  const [notConsensual, setNotConsensual] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function toggle(which: "isMe" | "notConsensual") {
    const next = { isMe, notConsensual, [which]: !(which === "isMe" ? isMe : notConsensual) };
    setIsMe(next.isMe);
    setNotConsensual(next.notConsensual);
    setError("");
    if (!next.isMe || !next.notConsensual) return;
    setBusy(true);
    try {
      const res = await fetch(`${API}/cases/${caseId}/confirm`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ is_me: true, not_consensual: true }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Couldn’t confirm this post.");
      onConfirmed(data.confirmed);
    } catch (e) {
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
