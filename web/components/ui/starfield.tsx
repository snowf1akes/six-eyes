"use client";

import { useEffect, useRef } from "react";

type Point = { x: number; y: number };
type Star = {
  orbital: number;
  position: Point;
  realPosition: Point;
  rSpeed: number;
  waveSpeed1: number;
  waveSpeed2: number;
  wave1: number;
  wave2: number;
};

export interface StarfieldProps {
  starCount?: number;
  waveFrequency?: number;
  starEscapeWidth?: number;
  starColor?: { r: number; g: number; b: number };
  maxOpacity?: number;
  rotationSpeed?: number;
  waveSpeed?: number;
}

// Stars orbit the center between starEscapeWidth / 2 and starEscapeWidth px out, leaving an empty eye.
// The canvas is transparent between stars, so whatever is behind it shows through.
export function Starfield({
  starCount = 25000,
  waveFrequency = 20,
  starEscapeWidth = 255,
  starColor = { r: 168, g: 85, b: 247 },
  maxOpacity = 255,
  rotationSpeed = 0.0005,
  waveSpeed = 0.01,
}: StarfieldProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { r, g, b } = starColor; // primitives, so an inline color object doesn't restart the animation

  useEffect(() => {
    const canvas = canvasRef.current!;
    const context = canvas.getContext("2d")!;
    const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
    const start = Date.now();
    let stars: Star[] = [];
    let w = 0, h = 0, time = 0, frame = 0;
    let image: ImageData, data: Uint32Array;

    const rotate = ({ x, y }: Point, radians: number): Point => {
      const cos = Math.cos(radians), sin = Math.sin(radians), cx = w / 2, cy = h / 2;
      return { x: cos * (x - cx) + sin * (y - cy) + cx, y: cos * (y - cy) - sin * (x - cx) + cy };
    };
    const opacity = (orbital: number) =>
      Math.min(255, Math.max(0, Math.floor((1 - orbital / starEscapeWidth) * maxOpacity + Math.random() * 80)));
    const pixel = (s: Star) => Math.floor(s.realPosition.y + s.wave1) * w + Math.floor(s.realPosition.x + s.wave2);

    const setSize = () => {
      w = canvas.width = canvas.clientWidth;
      h = canvas.height = canvas.clientHeight;
      image = context.createImageData(w, h);
      data = new Uint32Array(image.data.buffer);
      stars = [];
    };

    const createStar = () => {
      const orbital = (Math.random() * (starEscapeWidth / 2) + 1 + Math.random() * (starEscapeWidth / 2) + starEscapeWidth) / 2;
      const position = rotate({ x: w / 2, y: h / 2 + orbital }, Math.PI * 2 * Math.random());
      const waveSpeed1 = Math.random() * waveSpeed, waveSpeed2 = Math.random() * waveSpeed;
      stars.push({
        orbital, position, realPosition: position, waveSpeed1, waveSpeed2,
        rSpeed: Math.random() * rotationSpeed + opacity(orbital) / 20000,
        wave1: Math.sin(time * waveSpeed1) * waveFrequency,
        wave2: Math.sin(time * waveSpeed2) * waveFrequency,
      });
    };

    const drawStar = (s: Star) => {
      const prev = pixel(s);
      if (prev >= 0 && prev < data.length) data[prev] = 0;
      s.wave1 = Math.sin(time * s.waveSpeed1) * waveFrequency;
      s.wave2 = Math.sin(time * s.waveSpeed2) * waveFrequency;
      s.realPosition = rotate(s.position, s.rSpeed * time);
      const i = pixel(s);
      // ImageData is RGBA bytes, so on little-endian machines one Uint32 pixel reads ABGR.
      if (i >= 0 && i < data.length) data[i] = (opacity(s.orbital) << 24) | (b << 16) | (g << 8) | r;
    };

    const render = () => {
      time = (Date.now() - start) / 10;
      // Stars fade in 100 a frame; with reduced motion they all appear at once and the ring holds still.
      for (let n = Math.min(still ? starCount : 100, starCount - stars.length); n > 0; n--) createStar();
      for (const s of stars) drawStar(s);
      context.putImageData(image, 0, 0);
      if (!still || stars.length < starCount) frame = requestAnimationFrame(render);
    };

    const resize = () => {
      cancelAnimationFrame(frame);
      setSize();
      render();
    };
    resize();
    window.addEventListener("resize", resize);
    return () => {
      window.removeEventListener("resize", resize);
      cancelAnimationFrame(frame);
    };
  }, [starCount, waveFrequency, starEscapeWidth, r, g, b, maxOpacity, rotationSpeed, waveSpeed]);

  return <canvas ref={canvasRef} aria-hidden style={{ display: "block", width: "100%", height: "100%" }} />;
}
