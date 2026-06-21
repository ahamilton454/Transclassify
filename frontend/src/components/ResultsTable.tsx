import {
  createColumnHelper,
  flexRender,
  getCoreRowModel,
  useReactTable,
} from "@tanstack/react-table";
import { useMemo } from "react";
import type { EnrichResult, Transaction } from "../api/generated";
import { confidenceLevel } from "../lib/categories";

// EnrichResult is a superset of CategorizeResult, so it types both modes.
export interface Row extends EnrichResult {
  raw: string; // original description, for the raw→resolved column
}

const helper = createColumnHelper<Row>();

function ConfidencePill({ value }: { value: number }) {
  const level = confidenceLevel(value);
  const style =
    level === "resolved"
      ? { background: "var(--resolved-bg)", color: "var(--resolved)" }
      : { background: "var(--review-bg)", color: "var(--review)" };
  return (
    <span className="pill conf" style={style}>
      {(value * 100).toFixed(0)}%
    </span>
  );
}

export function ResultsTable({
  results,
  transactions,
  mode,
}: {
  results: EnrichResult[];
  transactions: Transaction[];
  mode: "categorize" | "enrich";
}) {
  const data = useMemo<Row[]>(() => {
    const rawById = new Map(transactions.map((t) => [t.id, t.description]));
    return results.map((r) => ({ ...r, raw: rawById.get(r.id) ?? "" }));
  }, [results, transactions]);

  const columns = useMemo(() => {
    const base = [
      helper.accessor("raw", {
        header: "Raw transaction",
        cell: (c) => <span className="raw">{c.getValue()}</span>,
      }),
      helper.display({
        id: "arrow",
        header: "",
        cell: () => <span className="arrow">→</span>,
      }),
      helper.accessor("merchant", {
        header: "Merchant",
        cell: (c) =>
          c.row.original.error ? (
            <span style={{ color: "var(--outflow)" }}>— {c.row.original.error}</span>
          ) : (
            <span className="merchant">{c.getValue()}</span>
          ),
      }),
      helper.accessor("category", {
        header: "Category",
        cell: (c) => <span className="pill" style={{ background: "var(--bg)" }}>{c.getValue()}</span>,
      }),
      helper.accessor("confidence", {
        header: "Confidence",
        cell: (c) => <ConfidencePill value={c.getValue()} />,
      }),
      helper.accessor("cost_usd", {
        header: "Cost",
        cell: (c) => (
          <span className="conf" title="LLM (+ web lookup) cost for this row">
            {c.getValue() == null ? "—" : `$${(c.getValue() as number).toFixed(5)}`}
          </span>
        ),
      }),
    ];

    if (mode === "enrich") {
      base.push(
        helper.accessor("website", {
          header: "Website",
          cell: (c) =>
            c.getValue() ? (
              <a href={c.getValue() as string} target="_blank" rel="noreferrer" className="raw">
                {(c.getValue() as string).replace(/^https?:\/\//, "")}
              </a>
            ) : (
              <span className="note">—</span>
            ),
        }) as never,
        helper.accessor("mcc", {
          header: "MCC",
          cell: (c) => <span className="raw">{c.getValue() ?? "—"}</span>,
        }) as never,
        helper.accessor("recurring", {
          header: "Recurring",
          cell: (c) => <span className="note">{c.getValue() == null ? "—" : c.getValue() ? "yes" : "no"}</span>,
        }) as never,
      );
    }
    return base;
  }, [mode]);

  const table = useReactTable({ data, columns, getCoreRowModel: getCoreRowModel() });

  return (
    <table>
      <thead>
        {table.getHeaderGroups().map((hg) => (
          <tr key={hg.id}>
            {hg.headers.map((h) => (
              <th key={h.id}>{flexRender(h.column.columnDef.header, h.getContext())}</th>
            ))}
          </tr>
        ))}
      </thead>
      <tbody>
        {table.getRowModel().rows.map((row) => (
          <tr key={row.id} className={row.original.error ? "error-row" : undefined}>
            {row.getVisibleCells().map((cell) => (
              <td key={cell.id}>{flexRender(cell.column.columnDef.cell, cell.getContext())}</td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}
