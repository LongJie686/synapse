"use client";

import { useEffect, useState } from "react";
import { fetchMetrics } from "@/lib/api";
import { Lang, t } from "@/lib/i18n";

interface Props { lang: Lang }

export default function MetricsPanel({ lang }: Props) {
  const [metrics, setMetrics] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState("");

  const load = async () => {
    try { setMetrics(await fetchMetrics()); setError(""); } catch { setError(t(lang, "serverUnreachable")); }
  };

  useEffect(() => { load(); }, []);

  const counters = (metrics?.counters || {}) as Record<string, number>;
  const entries = Object.entries(counters);

  const cardDefs: { labelKey: "totalRuns" | "llmCalls" | "toolCalls" | "guardrailBlocks" | "memoryEvents" | "toolErrors"; key: string }[] = [
    { labelKey: "totalRuns", key: "runs.total" },
    { labelKey: "llmCalls", key: "llm_calls.total" },
    { labelKey: "toolCalls", key: "tool_calls.total" },
    { labelKey: "guardrailBlocks", key: "guardrail_events.total" },
    { labelKey: "memoryEvents", key: "memory_events.total" },
    { labelKey: "toolErrors", key: "tool_calls.errors" },
  ];

  return (
    <div style={{ padding: 24 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 24 }}>
        <div>
          <h2 style={{ fontSize: 20, fontWeight: 600, margin: 0 }}>{t(lang, "metricsTitle")}</h2>
          <span style={{ fontSize: 13, color: "var(--text-muted)" }}>{t(lang, "runtime")}</span>
        </div>
        <button onClick={load} style={{
          padding: "8px 16px", fontSize: 13, background: "var(--bg-tertiary)",
          border: "1px solid var(--border)", borderRadius: 6, color: "var(--text-secondary)", cursor: "pointer",
        }}>{t(lang, "refresh")}</button>
      </div>

      {error && (
        <div style={{ padding: 16, background: "rgba(239,68,68,0.1)", border: "1px solid var(--error)", borderRadius: 8, color: "var(--error)", marginBottom: 16 }}>{error}</div>
      )}

      {metrics && (
        <div style={{ display: "grid", gap: 16 }}>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 12 }}>
            {cardDefs.map(({ labelKey, key }) => (
              <div key={key} style={{ padding: 16, background: "var(--bg-secondary)", border: "1px solid var(--border)", borderRadius: 10 }}>
                <div style={{ fontSize: 12, color: "var(--text-muted)", marginBottom: 8 }}>{t(lang, labelKey)}</div>
                <div style={{ fontSize: 28, fontWeight: 700 }}>{counters[key] || 0}</div>
              </div>
            ))}
          </div>

          {entries.length > 0 && (
            <div style={{ background: "var(--bg-secondary)", border: "1px solid var(--border)", borderRadius: 10, padding: 16 }}>
              <h3 style={{ fontSize: 14, fontWeight: 600, marginTop: 0, marginBottom: 12 }}>{t(lang, "allCounters")}</h3>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                <thead>
                  <tr>
                    <th style={{ textAlign: "left", padding: "8px 12px", color: "var(--text-muted)", borderBottom: "1px solid var(--border)" }}>{t(lang, "metric")}</th>
                    <th style={{ textAlign: "right", padding: "8px 12px", color: "var(--text-muted)", borderBottom: "1px solid var(--border)" }}>{t(lang, "value")}</th>
                  </tr>
                </thead>
                <tbody>
                  {entries.map(([key, val]) => (
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
