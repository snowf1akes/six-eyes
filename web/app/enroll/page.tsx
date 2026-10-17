import { readFileSync } from "node:fs";
import path from "node:path";
import type { Metadata } from "next";
import { EnrollForm } from "./enroll-form";

export const metadata: Metadata = { title: "Enroll" };

// C owns content/consent.md. Supports "## " headings, "- " lists and paragraphs.
function Consent() {
  const md = readFileSync(path.join(process.cwd(), "content", "consent.md"), "utf8");
  return (
    <div className="legal">
      {md.split(/\r?\n\s*\r?\n/).map((raw, i) => {
        const block = raw.trim();
        const lines = block.split(/\r?\n/).map((l) => l.trim());
        if (!block || block.startsWith("# ")) return null;
        if (block.startsWith("## ")) return <h3 key={i}>{block.slice(3)}</h3>;
        if (lines.every((l) => l.startsWith("- ")))
          return (
            <ul key={i}>
              {lines.map((l, j) => (
                <li key={j}>{l.slice(2)}</li>
              ))}
            </ul>
          );
        return <p key={i}>{lines.join(" ")}</p>;
      })}
    </div>
  );
}

export default function EnrollPage() {
  return (
    <main className="page">
      <div className="enroll">
        <div className="enroll-head">
          <h1 className="page-title">Enroll</h1>
          <p className="lede">
            Give us a few selfies once. Six Eyes searches X for posts of you, shows you anything that
            looks like a deepfake, and files the takedown when you say so.
          </p>
        </div>
        <EnrollForm consent={<Consent />} />
      </div>
    </main>
  );
}
