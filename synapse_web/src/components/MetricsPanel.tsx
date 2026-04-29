"use client";

import { useEffect, useState } from "react";
import { fetchMetrics } from "@/lib/api";

export default function MetricsPanel() {
  const [metrics, setMetrics] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState("");

  const load = async () => {
    try {
      const data = await fetchMetrics();
      setMetrics(data);
      setError("");
    } catch {
      setError("Server not reachable");
    }
  };

  useEffect(() => { load(); }, []);

  const counters = metrics?.counters as Record<string, number> | undefined;
  const counterEntries = counters ? Object.entries(counters) : [];

  return (
    <div style={{ padding: "24px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 24 }}>
        <div>
          <h2 style={{ fontSize: 20, fontWeight: 600, margin: 0 }}>Metrics</h2>
          <span style={{ fontSize: 13, color: "var(--text-muted)" }}>Runtime observability</span>
        </div>
        <button
          onClick={load}
          style={{
            padding: "8px 16px",
            fontSize: 13,
            background: "var(--bg-tertiary)",
            border: "1px solid var(--border)",
            borderRadius: 6,
            color: "var(--text-secondary)",
            cursor: "pointer",
          }}
        >
          Refresh
        </button>
      </div>

      {error && (
        <div style={{ padding: 16, background: "rgba(239,68,68,0.1)", border: "1px solid var(--error)", borderRadius: 8, color: "var(--error)", marginBottom: 16 }}>
          {error}
        </div>
      )}

      {metrics && (
        <div style={{ display: "grid", gap: 16 }}>
          {/* Summary Cards */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 12 }}>
            {[
              { label: "Total Runs", value: counters?.runs_total || 0 },
              { label: "LLM Calls", value: counters?.llm_calls_total || 0 },
              { label: "Tool Calls", value: counters?.tool_calls_total || 0 },
              { label: "Guardrail Blocks", value: counters?.guardrail_events_total || 0 },
              { label: "Memory Events", value: counters?.memory_events_total || 0 },
              { label: "Tool Errors", value: counters?.tool_calls_errors || 0 },
            ].map(({ label, value }) => (
              <div key={label} style={{
                padding: 16,
                background: "var(--bg-secondary)",
                border: "1px solid var(--border)",
                borderRadius: 10,
              }}>
                <div style={{ fontSize: 12, color: "var(--text-muted)", marginBottom: 8 }}>{label}</div>
                <div style={{ fontSize: 28, fontWeight: 700 }}>{value}</div>
              </div>
            ))}
          </div>

          {/* All Counters Table */}
          {counterEntries.length > 0 && (
            <div style={{
              background: "var(--bg-secondary)",
              border: "1px solid var(--border)",
              borderRadius: 10,
              padding: 16,
            }}>
              <h3 style={{ fontSize: 14, fontWeight: 600, marginTop: 0, marginBottom: 12 }}>All Counters</h3>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                <thead>
                  <tr>
                    <th style={{ textAlign: "left", padding: "8px 12px", color: "var(--text-muted)", borderBottom: "1px solid var(--border)" }}>Metric</th>
                    <th style={{ textAlign: "right", padding: "8px 12px", color: "var(--text-muted)", borderBottom: "1px solid var(--border)" }}>Value</th>
                  </tr>
                </thead>
                <tbody>
                  {counterEntries.map(([key, val]) => (
                    <tr key={key}>
                      <td style={{ padding: "6px 12px", borderBottom: "1px solid var(--border)", color: "var(--text-secondary)", fontFamily: "monospace", fontSize: 12 }}>{key}</td>
                      <td style={{ padding: "6px 12px", borderBottom: "1px solid var(--border)", textAlign: "right" }}>{val}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
