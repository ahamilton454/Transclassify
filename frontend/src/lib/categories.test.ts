import { describe, expect, it } from "vitest";
import { confidenceLevel, parseCategories, REVIEW_THRESHOLD } from "./categories";

describe("parseCategories", () => {
  it("parses plain names", () => {
    expect(parseCategories("Food\nTransport")).toEqual([{ name: "Food" }, { name: "Transport" }]);
  });

  it("parses name: description", () => {
    expect(parseCategories("Food: meals out\nIncome")).toEqual([
      { name: "Food", description: "meals out" },
      { name: "Income" },
    ]);
  });

  it("ignores blank lines and whitespace", () => {
    expect(parseCategories("  Food  \n\n   \nTransport")).toEqual([
      { name: "Food" },
      { name: "Transport" },
    ]);
  });

  it("treats name with trailing colon as no description", () => {
    expect(parseCategories("Food:")).toEqual([{ name: "Food" }]);
  });

  it("returns empty for empty input", () => {
    expect(parseCategories("")).toEqual([]);
  });
});

describe("confidenceLevel", () => {
  it("resolved at/above threshold", () => {
    expect(confidenceLevel(REVIEW_THRESHOLD)).toBe("resolved");
    expect(confidenceLevel(0.95)).toBe("resolved");
  });

  it("review below threshold", () => {
    expect(confidenceLevel(0.59)).toBe("review");
    expect(confidenceLevel(0)).toBe("review");
  });
});
