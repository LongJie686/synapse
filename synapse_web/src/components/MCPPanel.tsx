"use client";

import { useEffect, useState } from "react";
import { fetchMCPServers, addMCPServer, removeMCPServer, connectMCPServers, MCPServerInfo } from "@/lib/api";
import { Lang, t } from "@/lib/i18n";
import { useToast } from "@/components/Toast";

interface Props { lang: Lang }

const MCP_PRESETS = [
  {
    id: "filesystem",
    label_zh: "文件系统",
    label_en: "Filesystem",
    name: "filesystem",
    command: "npx",
    args: "-y @modelcontextprotocol/server-filesystem .",
    url: "",
  },
  {
    id: "memory",
    label_zh: "记忆存储",
    label_en: "Memory",
    name: "memory",
    command: "npx",
    args: "-y @modelcontextprotocol/server-memory",
    url: "",
  },
  {
    id: "fetch",
    label_zh: "网页抓取",
    label_en: "Fetch",
    name: "fetch",
    command: "npx",
    args: "-y @modelcontextprotocol/server-fetch",
    url: "",
  },
  {
    id: "brave-search",
    label_zh: "Brave 搜索",
    label_en: "Brave Search",
    name: "brave-search",
    command: "npx",
    args: "-y @modelcontextprotocol/server-brave-search",
    url: "",
  },
  {
    id: "sqlite",
    label_zh: "SQLite 数据库",
    label_en: "SQLite",
    name: "sqlite",
    command: "npx",
    args: "-y @modelcontextprotocol/server-sqlite",
    url: "",
  },
  {
    id: "custom",
    label_zh: "自定义",
    label_en: "Custom",
    name: "", command: "", args: "", url: "",
  },
];

export default function MCPPanel({ lang }: Props) {
  const [servers, setServers] = useState<MCPServerInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [showForm, setShowForm] = useState(false);
  const [editMode, setEditMode] = useState<"form" | "json">("form");
  const { toast, confirm: toastConfirm } = useToast();
  const [jsonContent, setJsonContent] = useState(`{
  "mcpServers": {
    "my-server": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    }
  }
}`);
  const [form, setForm] = useState({ name: "", command: "", args: "", url: "" });

  const load = async () => {
    try { setLoading(true); setServers(await fetchMCPServers()); } catch { toast("Failed to load MCP servers", "error"); } finally { setLoading(false); }
  };

  useEffect(() => { load(); }, []);

  const applyPreset = (presetId: string) => {
    const preset = MCP_PRESETS.find((p) => p.id === presetId);
    if (preset) {
      setForm({ name: preset.name, command: preset.command, args: preset.args, url: preset.url });
    }
  };

  const handleAdd = async () => {
    if (!form.name) return;
    try {
      await addMCPServer({
        name: form.name,
        command: form.command,
        args: form.args ? form.args.split(" ") : [],
        url: form.url,
      });
      setShowForm(false);
      setForm({ name: "", command: "", args: "", url: "" });
      load();
    } catch (err) { toast((err as Error).message, "error"); }
  };

  const handleAddJson = async () => {
    try {
      const parsed = JSON.parse(jsonContent);
      const mcpServers = parsed.mcpServers || parsed;
      for (const [name, config] of Object.entries(mcpServers)) {
        const cfg = config as Record<string, unknown>;
        await addMCPServer({
          name,
          command: String(cfg.command || ""),
          args: Array.isArray(cfg.args) ? cfg.args.map(String) : [],
          url: String(cfg.url || ""),
        });
      }
      setShowForm(false);
      load();
    } catch (err) { toast((err as Error).message, "error"); }
  };

  const handleRemove = async (name: string) => {
    if (!await toastConfirm(t(lang, "deleteMcpConfirm"))) return;
    try { await removeMCPServer(name); load(); } catch (err) { toast((err as Error).message, "error"); }
  };

  const handleConnect = async () => {
    try {
      const r = await connectMCPServers();
      toast(`${r.status}: ${r.tools_registered} tools`, "success");
      load();
    } catch (err) { toast((err as Error).message, "error"); }
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
          <h2 style={{ fontSize: 20, fontWeight: 600, margin: 0 }}>{t(lang, "mcpTitle")}</h2>
          <span style={{ fontSize: 13, color: "var(--text-muted)" }}>{t(lang, "mcpDesc")}</span>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <button onClick={handleConnect} style={{ padding: "8px 16px", fontSize: 13, background: "var(--bg-tertiary)", border: "1px solid var(--border)", borderRadius: 6, color: "var(--text-secondary)", cursor: "pointer" }}>
            {t(lang, "connectAll")}
          </button>
          <button onClick={() => { setShowForm(!showForm); setEditMode("form"); }} style={{ padding: "8px 16px", fontSize: 13, background: "var(--accent)", border: "none", borderRadius: 6, color: "#fff", cursor: "pointer" }}>
            + {t(lang, "newMcp")}
          </button>
        </div>
      </div>

      {showForm && (
        <div style={{ padding: 20, marginBottom: 20, background: "var(--bg-secondary)", border: "1px solid var(--border)", borderRadius: 10 }}>
          {/* Mode tabs */}
          <div style={{ display: "flex", gap: 0, marginBottom: 16, borderBottom: "1px solid var(--border)" }}>
            <button onClick={() => setEditMode("form")} style={{
              padding: "8px 16px", fontSize: 13, border: "none", cursor: "pointer",
              background: editMode === "form" ? "var(--accent)" : "transparent",
              color: editMode === "form" ? "#fff" : "var(--text-secondary)",
              borderRadius: "6px 6px 0 0",
            }}>{lang === "zh" ? "表单" : "Form"}</button>
            <button onClick={() => setEditMode("json")} style={{
              padding: "8px 16px", fontSize: 13, border: "none", cursor: "pointer",
              background: editMode === "json" ? "var(--accent)" : "transparent",
              color: editMode === "json" ? "#fff" : "var(--text-secondary)",
              borderRadius: "6px 6px 0 0",
            }}>JSON</button>
          </div>

          {editMode === "form" ? (
            <>
              {/* Preset picker */}
              <div style={{ marginBottom: 16 }}>
                <div style={{ fontSize: 12, color: "var(--text-muted)", marginBottom: 8 }}>
                  {lang === "zh" ? "常用预设" : "Presets"}
                </div>
                <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                  {MCP_PRESETS.map((preset) => (
                    <button key={preset.id} onClick={() => applyPreset(preset.id)} style={{
                      padding: "6px 12px", fontSize: 12, background: form.name === preset.name && preset.name ? "var(--accent)" : "var(--bg-tertiary)",
                      border: "1px solid var(--border)", borderRadius: 6,
                      color: form.name === preset.name && preset.name ? "#fff" : "var(--text-secondary)",
                      cursor: "pointer",
                    }}>
                      {lang === "zh" ? preset.label_zh : preset.label_en}
                    </button>
                  ))}
                </div>
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                <div>
                  <label style={{ fontSize: 12, color: "var(--text-secondary)", marginBottom: 4, display: "block" }}>{t(lang, "name")}</label>
                  <input value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} placeholder="filesystem" style={inputStyle} />
                </div>
                <div>
                  <label style={{ fontSize: 12, color: "var(--text-secondary)", marginBottom: 4, display: "block" }}>{t(lang, "command")}</label>
                  <input value={form.command} onChange={(e) => setForm({ ...form, command: e.target.value })} placeholder="npx" style={inputStyle} />
                </div>
                <div>
                  <label style={{ fontSize: 12, color: "var(--text-secondary)", marginBottom: 4, display: "block" }}>{t(lang, "arguments")}</label>
                  <input value={form.args} onChange={(e) => setForm({ ...form, args: e.target.value })} placeholder="-y @mcp/server-filesystem /tmp" style={inputStyle} />
                </div>
                <div>
                  <label style={{ fontSize: 12, color: "var(--text-secondary)", marginBottom: 4, display: "block" }}>URL (SSE)</label>
                  <input value={form.url} onChange={(e) => setForm({ ...form, url: e.target.value })} placeholder="http://..." style={inputStyle} />
                </div>
              </div>
            </>
          ) : (
            <div>
              <div style={{ fontSize: 12, color: "var(--text-muted)", marginBottom: 8 }}>
                {lang === "zh" ? "粘贴 Claude Desktop / Cursor 格式的 JSON 配置" : "Paste Claude Desktop / Cursor style JSON config"}
              </div>
              <textarea
                value={jsonContent}
                onChange={(e) => setJsonContent(e.target.value)}
                spellCheck={false}
                style={{
                  ...inputStyle, fontFamily: "monospace", fontSize: 12, lineHeight: 1.5,
                  resize: "vertical", minHeight: 200,
                }}
                rows={10}
              />
            </div>
          )}

          <div style={{ marginTop: 16, display: "flex", gap: 8 }}>
            <button onClick={editMode === "json" ? handleAddJson : handleAdd} style={{ padding: "8px 20px", fontSize: 13, background: "var(--accent)", border: "none", borderRadius: 6, color: "#fff", cursor: "pointer" }}>{t(lang, "create")}</button>
            <button onClick={() => setShowForm(false)} style={{ padding: "8px 20px", fontSize: 13, background: "var(--bg-tertiary)", border: "1px solid var(--border)", borderRadius: 6, color: "var(--text-secondary)", cursor: "pointer" }}>{t(lang, "cancel")}</button>
          </div>
        </div>
      )}

      {loading ? (
        <div style={{ color: "var(--text-muted)", textAlign: "center", padding: 40 }}>{t(lang, "loading")}</div>
      ) : servers.length === 0 ? (
        <div style={{ color: "var(--text-muted)", textAlign: "center", padding: 40 }}>{t(lang, "noMcpServers")}</div>
      ) : (
        <div style={{ display: "grid", gap: 12 }}>
          {servers.map((server) => (
            <div key={server.name} style={{
              padding: 16, background: "var(--bg-secondary)", border: "1px solid var(--border)",
              borderRadius: 10, display: "flex", justifyContent: "space-between", alignItems: "flex-start",
            }}>
              <div style={{ flex: 1 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <span style={{ fontWeight: 600, fontSize: 15 }}>{server.name}</span>
                  {server.connected && (
                    <span style={{ padding: "1px 6px", fontSize: 10, background: "rgba(34,197,94,0.15)", borderRadius: 3, color: "var(--success)" }}>
                      {t(lang, "connected")}
                    </span>
                  )}
                </div>
                {server.command && (
                  <div style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 4, fontFamily: "monospace" }}>
                    {server.command} {Array.isArray(server.args) ? server.args.join(" ") : server.args}
                  </div>
                )}
                {server.url && (
                  <div style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 2 }}>{server.url}</div>
                )}
                {server.tools && server.tools.length > 0 && (
                  <div style={{ marginTop: 8, display: "flex", gap: 6, flexWrap: "wrap" }}>
                    {server.tools.slice(0, 10).map((tool) => (
                      <span key={tool} style={{ padding: "2px 8px", fontSize: 11, background: "var(--bg-tertiary)", border: "1px solid var(--border)", borderRadius: 4, color: "var(--text-secondary)" }}>{tool}</span>
                    ))}
                    {server.tools.length > 10 && (
                      <span style={{ padding: "2px 8px", fontSize: 11, color: "var(--text-muted)" }}>+{server.tools.length - 10}</span>
                    )}
                  </div>
                )}
              </div>
              <button onClick={() => handleRemove(server.name)} style={{
                padding: "6px 12px", fontSize: 12, background: "transparent",
                border: "1px solid var(--error)", borderRadius: 6, color: "var(--error)", cursor: "pointer",
                flexShrink: 0,
              }}>{t(lang, "delete")}</button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
