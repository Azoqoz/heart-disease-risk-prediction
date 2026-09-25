import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Heart Disease Detection | Guided Clinical Scan",
  description: "A guided educational heart-disease model screening experience.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
