"use client";

import { useEffect, useState } from "react";
import { AgentInfo, fetchAgents, createAgent, deleteAgent } from "@/lib/api";
import { Lang, t, agentName, agentRole } from "@/lib/i18n";

interface Props { lang: Lang }

export default function AgentsPanel({ lang }: Props) {
  const [agents, setAgents] = useState<AgentInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [showForm, setShowForm] = useState(false);
  const [form, setForm] = useState({ id: "", name: "", role: "", goal: "", model: "glm-4-flash", tools: "calculator" });

  const load = async () => {
    try { setLoading(true); setAgents(await fetchAgents()); } catch {} finally { setLoading(false); }
  };

  useEffect(() => { load(); }, []);

  const handleCreate = async () => {
    if (!form.id || !form.name || !form.role) return;
    try {
      await createAgent({ ...form, tools: form.tools.split(",").map((s) => s.trim()).filter(Boolean) });
      setShowForm(false);
      setForm({ id: "", name: "", role: "", goal: "", model: "glm-4-flash", tools: "calculator" });
      load();
    } catch (err) { alert((err as Error).message); }
  };

  const handleDelete = async (id: string) => {
    if (!confirm(t(lang, "deleteConfirm"))) return;
    await deleteAgent(id); load();
  };

  const inputStyle = {
    width: "100%", padding: "8px 12px", fontSize: 13,
    background: "var(--bg-tertiary)", border: "1px solid var(--border)",
    borderRadius: 6, color: "var(--text-primary)", outline: "none",
  };

  return (
    <div style={{ padding: 24 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 24 }}>
        <div>
          <h2 style={{ fontSize: 20, fontWeight: 600, margin: 0 }}>{t(lang, "agentsTitle")}</h2>
          <span style={{ fontSize: 13, color: "var(--text-muted)" }}>{agents.length} {t(lang, "registered")}</span>
        </div>
        <button onClick={() => setShowForm(!showForm)} style={{
          padding: "8px 16px", fontSize: 13, background: "var(--accent)",
          border: "none", borderRadius: 6, color: "#fff", cursor: "pointer",
        }}>+ {t(lang, "newAgent")}</button>
      </div>

      {showForm && (
        <div style={{ padding: 20, marginBottom: 20, background: "var(--bg-secondary)", border: "1px solid var(--border)", borderRadius: 10 }}>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            {([
              [t(lang, "id"), "id", "my-agent"],
              [t(lang, "name"), "name", "My Agent"],
              [t(lang, "role"), "role", "Expert assistant"],
              [t(lang, "goal"), "goal", "Help users"],
              [t(lang, "model"), "model", "glm-4-flash"],
              [t(lang, "toolsField"), "tools", "calculator,web_search"],
            ] as const).map(([label, key, placeholder]) => (
              <div key={key}>
                <label style={{ fontSize: 12, color: "var(--text-secondary)", marginBottom: 4, display: "block" }}>{label}</label>
                <input value={(form as Record<string, string>)[key]} onChange={(e) => setForm({ ...form, [key]: e.target.value })} placeholder={placeholder} style={inputStyle} />
              </div>
            ))}
          </div>
          <div style={{ marginTop: 16, display: "flex", gap: 8 }}>
            <button onClick={handleCreate} style={{ padding: "8px 20px", fontSize: 13, background: "var(--accent)", border: "none", borderRadius: 6, color: "#fff", cursor: "pointer" }}>{t(lang, "create")}</button>
            <button onClick={() => setShowForm(false)} style={{ padding: "8px 20px", fontSize: 13, background: "var(--bg-tertiary)", border: "1px solid var(--border)", borderRadius: 6, color: "var(--text-secondary)", cursor: "pointer" }}>{t(lang, "cancel")}</button>
          </div>
        </div>
      )}

      {loading ? (
        <div style={{ color: "var(--text-muted)", textAlign: "center", padding: 40 }}>Loading...</div>
      ) : agents.length === 0 ? (
        <div style={{ color: "var(--text-muted)", textAlign: "center", padding: 40 }}>{t(lang, "noAgents")}</div>
      ) : (
        <div style={{ display: "grid", gap: 12 }}>
          {agents.map((agent) => (
            <div key={agent.id} style={{
              padding: 16, background: "var(--bg-secondary)", border: "1px solid var(--border)",
              borderRadius: 10, display: "flex", justifyContent: "space-between", alignItems: "center",
            }}>
              <div>
                <div style={{ fontWeight: 600, fontSize: 15 }}>{agentName(lang, agent.id, agent.name)}</div>
                <div style={{ fontSize: 13, color: "var(--text-secondary)", marginTop: 4 }}>{agentRole(lang, agent.id, agent.role)}</div>
                {agent.tools.length > 0 && (
                  <div style={{ marginTop: 8, display: "flex", gap: 6, flexWrap: "wrap" }}>
                    {agent.tools.map((tool) => (
                      <span key={tool} style={{ padding: "2px 8px", fontSize: 11, background: "var(--accent-soft)", borderRadius: 4, color: "var(--accent)" }}>{tool}</span>
                    ))}
                  </div>
                )}
              </div>
              <button onClick={() => handleDelete(agent.id)} style={{
                padding: "6px 12px", fontSize: 12, background: "transparent",
                border: "1px solid var(--error)", borderRadius: 6, color: "var(--error)", cursor: "pointer",
              }}>{t(lang, "delete")}</button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
