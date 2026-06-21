import { describe, expect, it } from "vitest";
import { detectColumns, parseCsv, toTransactions } from "./csv";

describe("parseCsv", () => {
  it("parses headers and rows", () => {
    const { headers, rows } = parseCsv("Date,Description,Amount\n2026-05-01,Coffee,-5.75");
    expect(headers).toEqual(["Date", "Description", "Amount"]);
    expect(rows).toEqual([["2026-05-01", "Coffee", "-5.75"]]);
  });

  it("handles quoted fields with commas and escaped quotes", () => {
    const { rows } = parseCsv('a,b\n"hello, world","say ""hi"""');
    expect(rows[0]).toEqual(["hello, world", 'say "hi"']);
  });

  it("handles CRLF and trailing newline", () => {
    const { rows } = parseCsv("a,b\r\n1,2\r\n");
    expect(rows).toEqual([["1", "2"]]);
  });

  it("drops blank rows", () => {
    const { rows } = parseCsv("a,b\n1,2\n\n  ,  \n");
    expect(rows).toEqual([["1", "2"]]);
  });

  it("returns empty for empty input", () => {
    expect(parseCsv("")).toEqual({ headers: [], rows: [] });
  });
});

describe("detectColumns", () => {
  it("detects standard headers", () => {
    expect(detectColumns(["Date", "Description", "Amount"])).toEqual({
      description: 1,
      amount: 2,
      date: 0,
    });
  });

  it("detects synonyms (Payee/Debit/Posted)", () => {
    const m = detectColumns(["Posted Date", "Payee", "Debit"]);
    expect(m.date).toBe(0);
    expect(m.description).toBe(1);
    expect(m.amount).toBe(2);
  });

  it("returns -1 when a column is missing", () => {
    expect(detectColumns(["foo", "bar"])).toEqual({ description: -1, amount: -1, date: -1 });
  });
});

describe("toTransactions", () => {
  const parsed = parseCsv(
    'Date,Description,Amount\n2026-05-01,SQ *STARBUCKS,-5.75\n2026-05-02,"AMZN, Mktp","$1,234.50"',
  );
  const mapping = detectColumns(parsed.headers);

  it("maps rows to transactions with parsed amounts", () => {
    const txns = toTransactions(parsed, mapping);
    expect(txns).toHaveLength(2);
    expect(txns[0]).toEqual({ id: "0", description: "SQ *STARBUCKS", amount: -5.75, date: "2026-05-01" });
    expect(txns[1].amount).toBe(1234.5); // strips $ and comma
  });

  it("skips rows with empty description", () => {
    const p = parseCsv("Description,Amount\nCoffee,-5\n,-9");
    const txns = toTransactions(p, detectColumns(p.headers));
    expect(txns).toHaveLength(1);
  });

  it("null amount/date when unmapped or unparseable", () => {
    const p = parseCsv("Description\nCoffee");
    const txns = toTransactions(p, detectColumns(p.headers));
    expect(txns[0]).toEqual({ id: "0", description: "Coffee", amount: null, date: null });
  });
});
