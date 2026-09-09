import type { Metadata } from "next";
import { IBM_Plex_Sans, Space_Grotesk } from "next/font/google";
import Link from "next/link";
import "./globals.css";

// next/font self-hosts these at build time (no runtime request to Google
// Fonts, no layout shift from a late-loading web font) — Space Grotesk
// for anything that needs personality (the wordmark, page titles),
// IBM Plex Sans for everything read at length or in a data table later.
const spaceGrotesk = Space_Grotesk({
  subsets: ["latin"],
  weight: ["500", "600", "700"],
  variable: "--font-display",
});

const plexSans = IBM_Plex_Sans({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--font-body",
});

export const metadata: Metadata = {
  title: "Padel AI Platform",
  description: "Upload a padel match and get back auto-generated highlights, stats, and a reel.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={`${spaceGrotesk.variable} ${plexSans.variable}`}>
        <div className="app-shell">
          <header className="site-header">
            <Link href="/" className="wordmark">
              <BallMark />
              padel
            </Link>
            <nav className="site-nav">
              <Link href="/">Upload</Link>
              <Link href="/matches">Matches</Link>
            </nav>
          </header>
          <main className="site-main">{children}</main>
        </div>
      </body>
    </html>
  );
}

/**
 * The one deliberate signature mark for this shell — a padel ball
 * rendered as a plain two-seam circle, small and static (no spin, no
 * pulse; restrained on purpose). Everything else in the layout stays
 * quiet so this is the one thing that's actually memorable.
 */
function BallMark() {
  return (
    <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true" className="ball-mark">
      <circle cx="9" cy="9" r="8" fill="var(--color-accent)" />
      <path
        d="M2.5 5.5C5 7 6.8 9.6 6.2 15.5M15.5 12.5C13 11 11.2 8.4 11.8 2.5"
        stroke="var(--color-ink)"
        strokeWidth="1.1"
        fill="none"
        strokeLinecap="round"
      />
    </svg>
  );
}
