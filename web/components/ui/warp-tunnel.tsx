"use client";

import { useEffect, useRef } from "react";

const NEAR_Z = 5;
const FAR_Z = -45;
const CAMERA_Z = 6;

// One streak: position, head depth and length. Biased away from the center so the vanishing point stays clear.
function streak(spread: number, z = FAR_Z + Math.random() * (NEAR_Z - FAR_Z)) {
  const r = 0.4 + Math.sqrt(Math.random()) * spread;
  const a = Math.random() * Math.PI * 2;
  return { x: Math.cos(a) * r, y: Math.sin(a) * r * 0.65, z, len: 1 + Math.random() * 1.6 };
}

export interface WarpTunnelProps {
  count?: number; // star streaks
  speed?: number; // warp speed multiplier
  spread?: number; // tunnel radius
  color?: string; // "r g b" of the streak heads; tails fade out
}

// Hyperspace: streaks race from z = -45 past a camera at z = 6 with a 60° field of view. The same scene as
// the three.js original, projected by hand onto a 2D canvas so it needs no 3D library.
export function WarpTunnel({ count = 400, speed = 1, spread = 7, color = "22 24 29" }: WarpTunnelProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current!;
    const ctx = canvas.getContext("2d")!;
    const streaks = Array.from({ length: count }, () => streak(spread));
    let frame = 0, last = performance.now();

    const draw = (now: number) => {
      const dpr = devicePixelRatio, w = canvas.clientWidth, h = canvas.clientHeight;
      if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
        canvas.width = Math.round(w * dpr);
        canvas.height = Math.round(h * dpr);
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      const focal = h / 2 / Math.tan(Math.PI / 6);
      const dz = Math.min((now - last) / 1000, 0.05) * 24 * speed;
      last = now;
      for (let i = 0; i < streaks.length; i++) {
        const s = streaks[i];
        s.z += dz;
        if (s.z - s.len > NEAR_Z) {
          streaks[i] = streak(spread, FAR_Z - Math.random() * 6);
          continue;
        }
        const head = focal / (CAMERA_Z - Math.min(s.z, CAMERA_Z - 0.1)); // clip at the camera
        const tail = focal / (CAMERA_Z - (s.z - s.len));
        const hx = w / 2 + s.x * head, hy = h / 2 - s.y * head;
        const tx = w / 2 + s.x * tail, ty = h / 2 - s.y * tail;
        const gradient = ctx.createLinearGradient(tx, ty, hx, hy);
        gradient.addColorStop(0, `rgb(${color} / 0)`);
        gradient.addColorStop(1, `rgb(${color} / 0.9)`);
        ctx.strokeStyle = gradient;
        ctx.lineWidth = Math.min(3, Math.max(0.5, head * 0.012));
        ctx.beginPath();
        ctx.moveTo(tx, ty);
        ctx.lineTo(hx, hy);
        ctx.stroke();
      }
      frame = requestAnimationFrame(draw);
    };
    frame = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(frame);
  }, [count, speed, spread, color]);

  return <canvas ref={canvasRef} aria-hidden style={{ display: "block", width: "100%", height: "100%" }} />;
}
