import type { Metadata } from "next";
import { Suspense } from "react";
import { Cases } from "./cases";

export const metadata: Metadata = { title: "Found posts" };

export default function DashboardPage() {
  return (
    <main className="page">
      <div className="dashboard">
        <div className="page-head">
          <h1 className="page-title">Found posts</h1>
          <p className="lede">
            Posts on X that may show you, most urgent first. Every image stays blurred until you choose to
            look.
          </p>
        </div>
        <Suspense fallback={<p className="meta">Loading found posts…</p>}>
          <Cases />
        </Suspense>
      </div>
    </main>
  );
}
