"use client";

import { useEffect, useState, useRef } from "react";
import { fetchAgents, fetchMultiAgentPatterns, getMultiAgentStreamUrl, AgentInfo } from "@/lib/api";
import { Lang, t, agentName } from "@/lib/i18n";
import { useToast } from "@/components/Toast";

interface Props { lang: Lang }

interface PatternOption {
  id: string;
  labelKey: "patternSupervisor" | "patternParallel" | "patternCollaboration";
}

const PATTERNS: PatternOption[] = [
  { id: "supervisor", labelKey: "patternSupervisor" },
  { id: "parallel", labelKey: "patternParallel" },
  { id: "collaboration", labelKey: "patternCollaboration" },
];

export default function MultiAgentPanel({ lang }: Props) {
  const [agents, setAgents] = useState<AgentInfo[]>([]);
  const [selectedAgents, setSelectedAgents] = useState<string[]>([]);
  const [pattern, setPattern] = useState("supervisor");
  const [task, setTask] = useState("");
  const [running, setRunning] = useState(false);
  const [output, setOutput] = useState<{ type: string; text: string }[]>([]);
  const bottomRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const { toast } = useToast();

  useEffect(() => {
    fetchAgents().then((data) => {
      setAgents(data);
      if (data.length >= 2) setSelectedAgents(data.slice(0, 2).map((a) => a.id));
    }).catch(() => {});
  }, []);

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }); }, [output]);

  const toggleAgent = (id: string) => {
    setSelectedAgents((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : prev.length < 5 ? [...prev, id] : prev
    );
  };

  const handleSend = async () => {
    if (!task.trim() || running || selectedAgents.length < 2) return;
    setRunning(true);
    setOutput([{ type: "header", text: `[${t(lang, PATTERNS.find((p) => p.id === pattern)!.labelKey)}] ${task}` }]);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const apiKey = process.env.NEXT_PUBLIC_API_KEY;
      const res = await fetch(getMultiAgentStreamUrl(), {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(apiKey ? { "X-API-Key": apiKey } : {}),
        },
        body: JSON.stringify({ message: task, pattern, agent_ids: selectedAgents }),
        signal: controller.signal,
      });

      if (!res.ok || !res.body) throw new Error("Stream failed");

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed) continue;
          try {
            const event = JSON.parse(trimmed);
            const { type, data } = event;

            if (type === "agent:message" && data.content) {
              setOutput((prev) => [...prev, { type: "agent", text: `[${data.agentId || "agent"}] ${data.content}` }]);
            } else if (type === "agent:delegate" && data.to) {
              setOutput((prev) => [...prev, { type: "delegate", text: `[${data.from || "supervisor"} -> ${data.to}] ${data.task || ""}` }]);
            } else if (type === "agent:think" && data.thought) {
              setOutput((prev) => [...prev, { type: "think", text: data.thought }]);
            } else if (type === "run:end") {
              setOutput((prev) => [...prev, { type: "end", text: "Done" }]);
            } else if (type === "run:error") {
              setOutput((prev) => [...prev, { type: "error", text: data.error || "Unknown error" }]);
            }
          } catch {
            console.warn("Multi-agent SSE parse error: invalid JSON on stream line");
          }
        }
      }
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        toast((err as Error).message, "error");
      }
    } finally {
      setRunning(false);
      abortRef.current = null;
    }
  };

  const handleStop = () => {
    abortRef.current?.abort();
    setRunning(false);
  };

  const inputStyle = {
    width: "100%", padding: "8px 12px", fontSize: 13,
    background: "var(--bg-tertiary)", border: "1px solid var(--border)",
    borderRadius: 6, color: "var(--text-primary)", outline: "none",
  };

  return (
    <div style={{ padding: 24, display: "flex", flexDirection: "column", height: "100%" }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ fontSize: 20, fontWeight: 600, margin: 0 }}>{t(lang, "multiAgentTitle")}</h2>
        <span style={{ fontSize: 13, color: "var(--text-muted)" }}>{t(lang, "multiAgentDesc")}</span>
      </div>

      {/* Pattern selector */}
      <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
        {PATTERNS.map((p) => (
          <button
            key={p.id}
            onClick={() => setPattern(p.id)}
            style={{
              padding: "8px 16px", fontSize: 13,
              background: pattern === p.id ? "var(--accent)" : "var(--bg-secondary)",
              border: `1px solid ${pattern === p.id ? "var(--accent)" : "var(--border)"}`,
              borderRadius: 6, color: pattern === p.id ? "#fff" : "var(--text-secondary)",
              cursor: "pointer",
            }}
          >
            {t(lang, p.labelKey)}
          </button>
        ))}
      </div>

      {/* Agent selector */}
      <div style={{ marginBottom: 16 }}>
        <div style={{ fontSize: 12, color: "var(--text-muted)", marginBottom: 8 }}>{t(lang, "agentParticipants")} ({selectedAgents.length})</div>
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
          {agents.map((a) => (
            <button
              key={a.id}
              onClick={() => toggleAgent(a.id)}
              style={{
                padding: "6px 12px", fontSize: 12,
                background: selectedAgents.includes(a.id) ? "var(--accent-soft)" : "var(--bg-secondary)",
                border: `1px solid ${selectedAgents.includes(a.id) ? "var(--accent)" : "var(--border)"}`,
                borderRadius: 6, color: selectedAgents.includes(a.id) ? "var(--accent)" : "var(--text-secondary)",
                cursor: "pointer",
              }}
            >
              {agentName(lang, a.id, a.name)}
            </button>
          ))}
        </div>
      </div>

      {/* Task input */}
      <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
        <input
          value={task}
          onChange={(e) => setTask(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && handleSend()}
          placeholder={t(lang, "typePlaceholder")}
          disabled={running}
          style={inputStyle}
        />
        {running ? (
          <button onClick={handleStop} style={{ padding: "8px 16px", background: "var(--error)", border: "none", borderRadius: 6, color: "#fff", cursor: "pointer", fontSize: 13 }}>
            {t(lang, "stop")}
          </button>
        ) : (
          <button
            onClick={handleSend}
            disabled={!task.trim() || selectedAgents.length < 2}
            style={{
              padding: "8px 16px", fontSize: 13,
              background: task.trim() && selectedAgents.length >= 2 ? "var(--accent)" : "var(--bg-tertiary)",
              border: "none", borderRadius: 6,
              color: task.trim() && selectedAgents.length >= 2 ? "#fff" : "var(--text-muted)",
              cursor: task.trim() && selectedAgents.length >= 2 ? "pointer" : "default",
            }}
          >
            {t(lang, "sendTask")}
          </button>
        )}
      </div>

      {/* Output */}
      <div style={{ flex: 1, overflowY: "auto", background: "var(--bg-secondary)", border: "1px solid var(--border)", borderRadius: 10, padding: 16 }}>
        {output.length === 0 && (
          <div style={{ color: "var(--text-muted)", textAlign: "center", padding: 40 }}>{t(lang, "sendToStart")}</div>
        )}
        {output.map((item, i) => (
          <div key={i} style={{
            marginBottom: 8, fontSize: 13, lineHeight: 1.6, whiteSpace: "pre-wrap", wordBreak: "break-word",
            color: item.type === "error" ? "var(--error)" : item.type === "delegate" ? "var(--accent)" : item.type === "think" ? "var(--text-muted)" : item.type === "header" ? "var(--text-primary)" : "var(--text-primary)",
            fontWeight: item.type === "header" ? 600 : item.type === "delegate" ? 500 : 400,
            fontStyle: item.type === "think" ? "italic" : "normal",
            padding: item.type === "agent" ? "8px 12px" : 0,
            background: item.type === "agent" ? "var(--bg-tertiary)" : "transparent",
            borderRadius: item.type === "agent" ? 6 : 0,
          }}>
            {item.text}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
