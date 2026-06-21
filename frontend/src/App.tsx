import { useMutation } from "@tanstack/react-query";
import { useMemo, useRef, useState } from "react";
import {
  categorizeV1CategorizePost,
  enrichV1EnrichPost,
  type EnrichResult,
  type Transaction,
} from "./api/generated";
import { ResultsTable } from "./components/ResultsTable";
import { parseCategories } from "./lib/categories";
import {
  type ColumnMapping,
  type ParsedCsv,
  detectColumns,
  parseCsv,
  toTransactions,
} from "./lib/csv";

type Mode = "categorize" | "enrich";

const DEFAULT_CATEGORIES = `Food & Drink: restaurants, coffee, groceries
Transport: rideshare, gas, transit
Shopping: retail and online purchases
Subscriptions: recurring software & media
Income: paychecks and deposits
Other`;

const COLUMN_FIELDS: { key: keyof ColumnMapping; label: string }[] = [
  { key: "description", label: "Description (required)" },
  { key: "amount", label: "Amount" },
  { key: "date", label: "Date" },
];

export function App() {
  const [parsed, setParsed] = useState<ParsedCsv | null>(null);
  const [mapping, setMapping] = useState<ColumnMapping | null>(null);
  const [categoriesText, setCategoriesText] = useState(DEFAULT_CATEGORIES);
  const [mode, setMode] = useState<Mode>("categorize");
  const [dragging, setDragging] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  const transactions = useMemo<Transaction[]>(
    () => (parsed && mapping ? toTransactions(parsed, mapping) : []),
    [parsed, mapping],
  );
  const categories = useMemo(() => parseCategories(categoriesText), [categoriesText]);

  const run = useMutation({
    mutationFn: async (): Promise<{ results: EnrichResult[]; totalCost: number }> => {
      const body = { transactions, categories };
      if (mode === "enrich") {
        const res = await enrichV1EnrichPost({ body, throwOnError: true });
        return { results: res.data.results, totalCost: res.data.total_cost_usd };
      }
      const res = await categorizeV1CategorizePost({ body, throwOnError: true });
      return { results: res.data.results, totalCost: res.data.total_cost_usd };
    },
  });

  function loadFile(file: File) {
    const reader = new FileReader();
    reader.onload = () => {
      const p = parseCsv(String(reader.result));
      setParsed(p);
      setMapping(detectColumns(p.headers));
      run.reset();
    };
    reader.readAsText(file);
  }

  const canRun = transactions.length > 0 && categories.length > 0 && !run.isPending;

  return (
    <div className="wrap">
      <h1>Transclassify</h1>
      <p className="tagline">Transaction categorization &amp; enrichment — bring your own categories.</p>

      {/* 1. Upload */}
      <div className="card">
        <h2><span className="step-num">1</span>Upload a CSV</h2>
        <div
          className={`dropzone${dragging ? " drag" : ""}`}
          onClick={() => fileInput.current?.click()}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            const f = e.dataTransfer.files[0];
            if (f) loadFile(f);
          }}
        >
          {parsed
            ? `${transactions.length} transactions loaded — drop another file to replace`
            : "Drop a bank/card CSV here, or click to browse"}
          <input
            ref={fileInput}
            type="file"
            accept=".csv,text/csv"
            hidden
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) loadFile(f);
            }}
          />
        </div>
        <p className="privacy">
          Files are parsed in your browser. Only the transaction text and your category list are sent
          to the API — no bank login, no account required.
        </p>

        {parsed && mapping && (
          <div className="row" style={{ marginTop: 16 }}>
            {COLUMN_FIELDS.map(({ key, label }) => (
              <div key={key}>
                <label>{label}</label>
                <select
                  value={mapping[key]}
                  onChange={(e) => setMapping({ ...mapping, [key]: Number(e.target.value) })}
                >
                  <option value={-1}>— none —</option>
                  {parsed.headers.map((h, i) => (
                    <option key={i} value={i}>
                      {h}
                    </option>
                  ))}
                </select>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* 2. Categories */}
      <div className="card">
        <h2><span className="step-num">2</span>Your categories</h2>
        <label>One per line. Optionally add a hint after a colon.</label>
        <textarea value={categoriesText} onChange={(e) => setCategoriesText(e.target.value)} />
        <p className="note">{categories.length} categories</p>
      </div>

      {/* 3. Mode + run */}
      <div className="card">
        <h2><span className="step-num">3</span>Run</h2>
        <div className="row" style={{ alignItems: "center" }}>
          <div style={{ flex: "0 0 auto" }}>
            <div className="modes">
              <button
                className={mode === "categorize" ? "active" : ""}
                onClick={() => setMode("categorize")}
              >
                Categorize
              </button>
              <button className={mode === "enrich" ? "active" : ""} onClick={() => setMode("enrich")}>
                Enrich
              </button>
            </div>
          </div>
          <div style={{ flex: "0 0 auto" }}>
            <button className="run" disabled={!canRun} onClick={() => run.mutate()}>
              {run.isPending ? "Working…" : mode === "enrich" ? "Enrich transactions" : "Categorize transactions"}
            </button>
          </div>
          <div style={{ flex: 1 }}>
            <span className="note">
              {mode === "enrich"
                ? "Adds merchant website, MCC, and recurring detection via a web lookup."
                : "Cleaned merchant + best-fit category + confidence."}
            </span>
          </div>
        </div>
        {run.isError && (
          <p className="privacy" style={{ borderColor: "var(--outflow)", color: "var(--outflow)" }}>
            Request failed: {(run.error as Error).message}. Is the backend running on :8000?
          </p>
        )}
      </div>

      {/* 4. Results */}
      {run.data && (
        <div className="card">
          <h2>
            <span className="step-num">4</span>Results
            <span className="note" style={{ float: "right", textTransform: "none", letterSpacing: 0 }}>
              {run.data.results.length} rows · total cost{" "}
              <strong style={{ color: "var(--resolved)" }}>${run.data.totalCost.toFixed(5)}</strong>
            </span>
          </h2>
          <ResultsTable results={run.data.results} transactions={transactions} mode={mode} />
        </div>
      )}
    </div>
  );
}
