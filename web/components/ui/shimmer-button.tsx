import type { ComponentProps, CSSProperties } from "react";

export interface ShimmerButtonProps extends ComponentProps<"button"> {
  shimmerColor?: string;
  shimmerSize?: string;
  borderRadius?: string;
  shimmerDuration?: string;
  background?: string;
}

// Magic UI's ShimmerButton in plain CSS (globals.css, "Shimmer button"): this app has no Tailwind.
export function ShimmerButton({
  shimmerColor = "#ffffff",
  shimmerSize = "0.05em",
  shimmerDuration = "3s",
  borderRadius = "100px",
  background = "rgba(0, 0, 0, 1)",
  className,
  style,
  children,
  ...props
}: ShimmerButtonProps) {
  return (
    <button
      {...props}
      className={className ? `shimmer-btn ${className}` : "shimmer-btn"}
      style={{
        "--spread": "90deg",
        "--shimmer-color": shimmerColor,
        "--radius": borderRadius,
        "--speed": shimmerDuration,
        "--cut": shimmerSize,
        "--bg": background,
        ...style,
      } as CSSProperties}
    >
      <span className="shimmer-spark" aria-hidden><span><span /></span></span>
      {children}
      <span className="shimmer-highlight" aria-hidden />
      <span className="shimmer-backdrop" aria-hidden />
    </button>
  );
}
