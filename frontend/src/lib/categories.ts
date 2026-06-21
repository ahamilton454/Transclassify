// Parse the free-text category box into API categories, and bucket confidence.
import type { Category } from "../api/generated";

/** One category per line. "Name: optional hint" → {name, description}. */
export function parseCategories(text: string): Category[] {
  return text
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line !== "")
    .map((line) => {
      const idx = line.indexOf(":");
      if (idx === -1) return { name: line };
      const name = line.slice(0, idx).trim();
      const description = line.slice(idx + 1).trim();
      return description ? { name, description } : { name };
    })
    .filter((c) => c.name !== "");
}

export type ConfidenceLevel = "resolved" | "review";

/** Below this, the row is flagged for review (amber). */
export const REVIEW_THRESHOLD = 0.6;

export function confidenceLevel(confidence: number): ConfidenceLevel {
  return confidence >= REVIEW_THRESHOLD ? "resolved" : "review";
}
