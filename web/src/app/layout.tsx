import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "GuitarTabs AI - Generate Guitar Tabs from Any Song",
  description: "Transform any song into accurate guitar tablature using AI. Upload audio files or paste YouTube links to get tabs in seconds.",
  keywords: ["guitar tabs", "tab generator", "AI music", "guitar tablature", "music transcription"],
  authors: [{ name: "GuitarTabs AI" }],
  openGraph: {
    title: "GuitarTabs AI - Generate Guitar Tabs from Any Song",
    description: "Transform any song into accurate guitar tablature using AI.",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <link
          href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="antialiased">
        {children}
      </body>
    </html>
  );
}
