// Pure CSV helpers: parsing + heuristic column detection.
// Kept framework-free and pure so they're easy to unit-test.

export interface ParsedCsv {
  headers: string[];
  rows: string[][];
}

export interface ColumnMapping {
  description: number; // required; -1 if undetected
  amount: number; // -1 if none
  date: number; // -1 if none
}

/** Minimal RFC-4180-ish parser: handles quoted fields, escaped quotes, CRLF. */
export function parseCsv(text: string): ParsedCsv {
  const rows: string[][] = [];
  let field = "";
  let row: string[] = [];
  let inQuotes = false;

  const pushField = () => {
    row.push(field);
    field = "";
  };
  const pushRow = () => {
    pushField();
    rows.push(row);
    row = [];
  };

  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (inQuotes) {
      if (ch === '"') {
        if (text[i + 1] === '"') {
          field += '"';
          i++;
        } else {
          inQuotes = false;
        }
      } else {
        field += ch;
      }
    } else if (ch === '"') {
      inQuotes = true;
    } else if (ch === ",") {
      pushField();
    } else if (ch === "\n") {
      pushRow();
    } else if (ch === "\r") {
      // swallow; \n handles the row break
    } else {
      field += ch;
    }
  }
  // flush trailing field/row if the file doesn't end in a newline
  if (field.length > 0 || row.length > 0) pushRow();

  // drop fully-empty trailing rows
  const cleaned = rows.filter((r) => r.some((c) => c.trim() !== ""));
  if (cleaned.length === 0) return { headers: [], rows: [] };

  const [headers, ...dataRows] = cleaned;
  return { headers, rows: dataRows };
}

const DESCRIPTION_HINTS = ["description", "desc", "name", "payee", "memo", "details", "merchant", "transaction"];
const AMOUNT_HINTS = ["amount", "amt", "value", "debit", "credit", "total"];
const DATE_HINTS = ["date", "posted", "time", "day"];

function bestMatch(headers: string[], hints: string[]): number {
  const lower = headers.map((h) => h.toLowerCase().trim());
  // exact-ish first, then substring
  for (const hint of hints) {
    const exact = lower.findIndex((h) => h === hint);
    if (exact !== -1) return exact;
  }
  for (const hint of hints) {
    const partial = lower.findIndex((h) => h.includes(hint));
    if (partial !== -1) return partial;
  }
  return -1;
}

/** Guess which columns hold description / amount / date from header names. */
export function detectColumns(headers: string[]): ColumnMapping {
  return {
    description: bestMatch(headers, DESCRIPTION_HINTS),
    amount: bestMatch(headers, AMOUNT_HINTS),
    date: bestMatch(headers, DATE_HINTS),
  };
}

export interface TransactionInput {
  id: string;
  description: string;
  amount: number | null;
  date: string | null;
}

function parseAmount(raw: string | undefined): number | null {
  if (raw == null) return null;
  const cleaned = raw.replace(/[$,\s]/g, "");
  if (cleaned === "") return null;
  const n = Number(cleaned);
  return Number.isFinite(n) ? n : null;
}

/** Build API transactions from parsed rows + a column mapping. */
export function toTransactions(parsed: ParsedCsv, mapping: ColumnMapping): TransactionInput[] {
  const out: TransactionInput[] = [];
  parsed.rows.forEach((r, i) => {
    const description = mapping.description >= 0 ? (r[mapping.description] ?? "").trim() : "";
    if (description === "") return; // skip rows with no description to categorize
    out.push({
      id: String(i),
      description,
      amount: mapping.amount >= 0 ? parseAmount(r[mapping.amount]) : null,
      date: mapping.date >= 0 ? (r[mapping.date] ?? "").trim() || null : null,
    });
  });
  return out;
}
