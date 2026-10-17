import type { Metadata } from "next";
import { EB_Garamond, Geist } from "next/font/google";
import Link from "next/link";
import "./globals.css";

// Everything that is read or operated.
const geist = Geist({ variable: "--font-sans", subsets: ["latin"] });

// Headlines and names, as on DapUp.
const garamond = EB_Garamond({
  variable: "--font-display",
  subsets: ["latin"],
  weight: ["400", "500", "600"],
});

export const metadata: Metadata = {
  title: { default: "Six Eyes", template: "%s | Six Eyes" },
  description: "Find deepfakes of you on X and get them taken down under the TAKE IT DOWN Act.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${geist.variable} ${garamond.variable}`}>
      <body>
        <header className="site-header">
          <div className="site-header-inner">
            <Link href="/" className="wordmark">
              Six Eyes
            </Link>
          </div>
        </header>
        {children}
        <footer className="site-footer">
          <span className="wordmark">Six Eyes</span> · Built at HackHarvard 2026
        </footer>
      </body>
    </html>
  );
}
