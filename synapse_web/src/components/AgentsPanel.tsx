"use client";

import { useEffect, useState } from "react";
import { AgentInfo, fetchAgents, createAgent, deleteAgent } from "@/lib/api";

export default function AgentsPanel() {
  const [agents, setAgents] = useState<AgentInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [showForm, setShowForm] = useState(false);
  const [form, setForm] = useState({ id: "", name: "", role: "", goal: "", model: "glm-4-flash", tools: "calculator" });

  const load = async () => {
    try {
      setLoading(true);
      const data = await fetchAgents();
      setAgents(data);
    } catch {
      // server not running
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, []);

  const handleCreate = async () => {
    if (!form.id || !form.name || !form.role) return;
    try {
      await createAgent({
        ...form,
        tools: form.tools.split(",").map((t) => t.trim()).filter(Boolean),
      });
      setShowForm(false);
      setForm({ id: "", name: "", role: "", goal: "", model: "glm-4-flash", tools: "calculator" });
      load();
    } catch (err) {
      alert("Failed to create agent: " + (err as Error).message);
    }
  };

  const handleDelete = async (id: string) => {
    if (!confirm("Delete this agent?")) return;
    await deleteAgent(id);
    load();
  };

  return (
    <div style={{ padding: "24px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 24 }}>
        <div>
          <h2 style={{ fontSize: 20, fontWeight: 600, margin: 0 }}>Agents</h2>
          <span style={{ fontSize: 13, color: "var(--text-muted)" }}>{agents.length} registered</span>
        </div>
        <button
          onClick={() => setShowForm(!showForm)}
          style={{
            padding: "8px 16px",
            fontSize: 13,
            background: "var(--accent)",
            border: "none",
            borderRadius: 6,
            color: "#fff",
            cursor: "pointer",
          }}
        >
          + New Agent
        </button>
      </div>

      {showForm && (
        <div style={{
          padding: 20,
          marginBottom: 20,
          background: "var(--bg-secondary)",
          border: "1px solid var(--border)",
          borderRadius: 10,
        }}>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            {[
              { label: "ID", key: "id", placeholder: "my-agent" },
              { label: "Name", key: "name", placeholder: "My Agent" },
              { label: "Role", key: "role", placeholder: "Expert assistant" },
              { label: "Goal", key: "goal", placeholder: "Help users" },
              { label: "Model", key: "model", placeholder: "glm-4-flash" },
              { label: "Tools (comma separated)", key: "tools", placeholder: "calculator,web_search" },
            ].map(({ label, key, placeholder }) => (
              <div key={key}>
                <label style={{ fontSize: 12, color: "var(--text-secondary)", marginBottom: 4, display: "block" }}>{label}</label>
                <input
                  value={(form as Record<string, string>)[key]}
                  onChange={(e) => setForm({ ...form, [key]: e.target.value })}
                  placeholder={placeholder}
                  style={{
                    width: "100%",
                    padding: "8px 12px",
                    fontSize: 13,
                    background: "var(--bg-tertiary)",
                    border: "1px solid var(--border)",
                    borderRadius: 6,
                    color: "var(--text-primary)",
                    outline: "none",
                  }}
                />
              </div>
            ))}
          </div>
          <div style={{ marginTop: 16, display: "flex", gap: 8 }}>
            <button onClick={handleCreate} style={{ padding: "8px 20px", fontSize: 13, background: "var(--accent)", border: "none", borderRadius: 6, color: "#fff", cursor: "pointer" }}>Create</button>
            <button onClick={() => setShowForm(false)} style={{ padding: "8px 20px", fontSize: 13, background: "var(--bg-tertiary)", border: "1px solid var(--border)", borderRadius: 6, color: "var(--text-secondary)", cursor: "pointer" }}>Cancel</button>
          </div>
        </div>
      )}

      {loading ? (
        <div style={{ color: "var(--text-muted)", textAlign: "center", padding: 40 }}>Loading...</div>
      ) : agents.length === 0 ? (
        <div style={{ color: "var(--text-muted)", textAlign: "center", padding: 40 }}>No agents. Create one above or start the server.</div>
      ) : (
        <div style={{ display: "grid", gap: 12 }}>
          {agents.map((agent) => (
            <div key={agent.id} style={{
              padding: 16,
              background: "var(--bg-secondary)",
              border: "1px solid var(--border)",
              borderRadius: 10,
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
            }}>
              <div>
                <div style={{ fontWeight: 600, fontSize: 15 }}>{agent.name}</div>
                <div style={{ fontSize: 13, color: "var(--text-secondary)", marginTop: 4 }}>{agent.role}</div>
                {agent.tools.length > 0 && (
                  <div style={{ marginTop: 8, display: "flex", gap: 6, flexWrap: "wrap" }}>
                    {agent.tools.map((t) => (
                      <span key={t} style={{
                        padding: "2px 8px",
                        fontSize: 11,
                        background: "var(--accent-soft)",
                        borderRadius: 4,
                        color: "var(--accent)",
                      }}>{t}</span>
                    ))}
                  </div>
                )}
              </div>
              <button
                onClick={() => handleDelete(agent.id)}
                style={{
                  padding: "6px 12px",
                  fontSize: 12,
                  background: "transparent",
                  border: "1px solid var(--error)",
                  borderRadius: 6,
                  color: "var(--error)",
                  cursor: "pointer",
                }}
              >
                Delete
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
