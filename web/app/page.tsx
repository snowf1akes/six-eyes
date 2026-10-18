"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { ShimmerButton } from "@/components/ui/shimmer-button";
import { Starfield } from "@/components/ui/starfield";
import { WarpTunnel } from "@/components/ui/warp-tunnel";

// Placeholder until the team picks the one-line analogy.
const TAGLINE =
  "An AI agent that watches X for intimate images and deepfakes of you, files the legal takedown, and follows it until they're gone.";
const WARP_MS = 900; // matches the .warp canvas fade in globals.css

export default function Home() {
  const router = useRouter();
  const [warping, setWarping] = useState(false);
  // cacheComponents keeps this page mounted but hidden after navigating away (and runs effect cleanups), so
  // drop the overlay then, or Back would show the faded-out warp instead of the page.
  useEffect(() => () => setWarping(false), []);

  // Get started jumps to /enroll through the warp tunnel; with reduced motion it's a plain link.
  function warp(e: { preventDefault(): void }) {
    if (matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    e.preventDefault();
    if (warping) return;
    setWarping(true);
    setTimeout(() => router.push("/enroll"), WARP_MS);
  }

  return (
    <main className="landing">
      <div className="landing-sky">
        <Starfield
          starCount={16000}
          waveFrequency={15}
          starEscapeWidth={420}
          starColor={{ r: 22, g: 24, b: 29 }}
          maxOpacity={255}
          rotationSpeed={0.0002}
          waveSpeed={0.005}
        />
      </div>
      <div className="landing-copy">
        <h1 className="landing-title">Six Eyes</h1>
        <p className="lede">{TAGLINE}</p>
        <ShimmerButton href="/enroll" onNavigate={warp} className="btn btn-lg" borderRadius="10px" background="var(--ink)">
          Get started
        </ShimmerButton>
      </div>
      {warping && (
        <div className="warp">
          <WarpTunnel speed={2.5} />
        </div>
      )}
    </main>
  );
}
