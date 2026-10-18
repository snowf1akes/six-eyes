import Link from "next/link";
import type { ComponentProps, CSSProperties } from "react";

export interface ShimmerButtonProps extends ComponentProps<"button"> {
  shimmerColor?: string;
  shimmerSize?: string;
  borderRadius?: string;
  shimmerDuration?: string;
  background?: string;
  href?: string; // renders a link that looks the same, for navigation
  onNavigate?: ComponentProps<typeof Link>["onNavigate"]; // with href: runs on in-app navigation only
}

// Magic UI's ShimmerButton in plain CSS (globals.css, "Shimmer button"): this app has no Tailwind.
export function ShimmerButton({
  shimmerColor = "#ffffff",
  shimmerSize = "0.05em",
  shimmerDuration = "3s",
  borderRadius = "100px",
  background = "rgba(0, 0, 0, 1)",
  href,
  onNavigate,
  className,
  style,
  children,
  ...props
}: ShimmerButtonProps) {
  const look = {
    className: className ? `shimmer-btn ${className}` : "shimmer-btn",
    style: {
      "--spread": "90deg",
      "--shimmer-color": shimmerColor,
      "--radius": borderRadius,
      "--speed": shimmerDuration,
      "--cut": shimmerSize,
      "--bg": background,
      ...style,
    } as CSSProperties,
  };
  const inner = (
    <>
      <span className="shimmer-spark" aria-hidden><span><span /></span></span>
      {children}
      <span className="shimmer-highlight" aria-hidden />
      <span className="shimmer-backdrop" aria-hidden />
    </>
  );
  return href
    ? <Link href={href} onNavigate={onNavigate} {...look}>{inner}</Link>
    : <button {...props} {...look}>{inner}</button>;
}
