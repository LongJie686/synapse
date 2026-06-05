"use client";

import { useEffect, useState } from "react";
import { load as parseYaml } from "js-yaml";
import { fetchSkills, createSkill, deleteSkill, reloadSkills, SkillInfo } from "@/lib/api";
import { Lang, t } from "@/lib/i18n";
import { useToast } from "@/components/Toast";

interface Props { lang: Lang }

const SKILL_TEMPLATES = [
  {
    id: "custom",
    label_zh: "空白模板",
    label_en: "Blank",
    name: "", display_name: "", description: "", tools: "calculator",
    system_prompt: "", temperature: "0.7", max_tokens: "4096",
  },
  {
    id: "researcher",
    label_zh: "网络调研",
    label_en: "Web Researcher",
    name: "web-researcher", display_name: "Web Researcher", description: "深度网络调研专家，搜索、抓取并综合互联网信息",
    tools: "web_search,web_scrape,calculator",
    system_prompt: "You are a professional web researcher. Search, scrape, and synthesize information from the internet. Always cite your sources.",
    temperature: "0.3", max_tokens: "4096",
  },
  {
    id: "coder",
    label_zh: "Python 开发",
    label_en: "Python Developer",
    name: "python-dev", display_name: "Python Developer", description: "Python 开发专家，编写、调试并运行 Python 代码",
    tools: "code_execute,calculator,web_scrape",
    system_prompt: "You are a Python development expert. Write clean, efficient, well-documented Python code. Use code_execute to run and test your code.",
    temperature: "0.2", max_tokens: "4096",
  },
  {
    id: "code-reviewer",
    label_zh: "代码审查",
    label_en: "Code Reviewer",
    name: "code-reviewer", display_name: "Code Reviewer", description: "代码质量、安全性与可维护性审查专家",
    tools: "calculator,web_scrape",
    system_prompt: "You are a senior code reviewer. Review code for correctness, quality, security, and maintainability. Provide actionable feedback with severity ratings (CRITICAL/HIGH/MEDIUM/LOW).",
    temperature: "0.2", max_tokens: "4096",
  },
  {
    id: "security-reviewer",
    label_zh: "安全审查",
    label_en: "Security Reviewer",
    name: "security-reviewer", display_name: "Security Reviewer", description: "安全漏洞检测与修复建议专家",
    tools: "web_scrape,calculator",
    system_prompt: "You are a security specialist. Detect vulnerabilities following OWASP Top 10. Rate severity, explain attack vectors, and provide concrete fixes with code examples.",
    temperature: "0.1", max_tokens: "4096",
  },
  {
    id: "architect",
    label_zh: "架构设计",
    label_en: "Architect",
    name: "architect", display_name: "Architect", description: "系统设计、可扩展性与技术决策专家",
    tools: "calculator,web_scrape",
    system_prompt: "You are a senior software architect. Design systems for scalability, evaluate trade-offs, and document decisions with rationale. Use ASCII diagrams when helpful.",
    temperature: "0.3", max_tokens: "4096",
  },
  {
    id: "tdd-guide",
    label_zh: "TDD 专家",
    label_en: "TDD Guide",
    name: "tdd-guide", display_name: "TDD Guide", description: "测试驱动开发方法论专家",
    tools: "calculator,web_scrape",
    system_prompt: "You are a TDD specialist. Enforce Red-Green-Refactor cycle. Write tests first, then minimal implementation. Target 80%+ coverage.",
    temperature: "0.2", max_tokens: "4096",
  },
];

export default function SkillsPanel({ lang }: Props) {
  const [skills, setSkills] = useState<SkillInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [showForm, setShowForm] = useState(false);
  const [editMode, setEditMode] = useState<"template" | "yaml">("template");
  const { toast, confirm: toastConfirm } = useToast();
  const [yamlContent, setYamlContent] = useState(`name: my-skill
display_name: My Skill
description: Description here
tools:
  - calculator
  - web_search
system_prompt: |
  You are a helpful assistant...
temperature: 0.7
max_tokens: 4096
`);
  const [form, setForm] = useState({
    name: "", display_name: "", description: "", tools: "calculator",
    system_prompt: "", temperature: "0.7", max_tokens: "4096",
  });

  const load = async () => {
    try { setLoading(true); setSkills(await fetchSkills()); } catch { toast("Failed to load skills", "error"); } finally { setLoading(false); }
  };

  useEffect(() => { load(); }, []);

  const applyTemplate = (templateId: string) => {
    const tmpl = SKILL_TEMPLATES.find((t) => t.id === templateId);
    if (tmpl) {
      setForm({
        name: tmpl.name, display_name: tmpl.display_name,
        description: tmpl.description, tools: tmpl.tools,
        system_prompt: tmpl.system_prompt, temperature: tmpl.temperature,
        max_tokens: tmpl.max_tokens,
      });
    }
  };

  const handleCreate = async () => {
    if (!form.name || !form.display_name) return;
    try {
      await createSkill({
        name: form.name,
        display_name: form.display_name,
        description: form.description,
        tools: form.tools.split(",").map((s) => s.trim()).filter(Boolean),
        system_prompt: form.system_prompt,
        temperature: parseFloat(form.temperature) || 0.7,
        max_tokens: parseInt(form.max_tokens) || 4096,
      });
      setShowForm(false);
      setForm({ name: "", display_name: "", description: "", tools: "calculator", system_prompt: "", temperature: "0.7", max_tokens: "4096" });
      load();
    } catch (err) { toast((err as Error).message, "error"); }
  };

  const handleCreateYaml = async () => {
    try {
      const raw = parseYaml(yamlContent) as Record<string, unknown>;
      if (!raw || typeof raw !== "object") throw new Error("Invalid YAML: expected a mapping");

      const str = (key: string, fallback = "") =>
        typeof raw[key] === "string" ? (raw[key] as string) : fallback;

      const toolsRaw = raw.tools;
      let tools: string[] = ["calculator"];
      if (Array.isArray(toolsRaw)) {
        tools = toolsRaw.map(String).filter(Boolean);
      } else if (typeof toolsRaw === "string" && toolsRaw.trim()) {
        tools = toolsRaw.split(",").map((s) => s.trim()).filter(Boolean);
      }

      await createSkill({
        name: str("name"),
        display_name: str("display_name") || str("name"),
        description: str("description"),
        tools,
        system_prompt: str("system_prompt"),
        temperature: parseFloat(str("temperature", "0.7")) || 0.7,
        max_tokens: parseInt(str("max_tokens", "4096")) || 4096,
      });
      setShowForm(false);
      load();
    } catch (err) { toast((err as Error).message, "error"); }
  };

  const handleDelete = async (name: string) => {
    if (!await toastConfirm(t(lang, "deleteSkillConfirm"))) return;
    try { await deleteSkill(name); load(); } catch (err) { toast((err as Error).message, "error"); }
  };

  const handleReload = async () => {
    try { const r = await reloadSkills(); toast(`${r.status}: ${r.count}`, "success"); load(); } catch (err) { toast((err as Error).message, "error"); }
  };

  const inputStyle = {
    width: "100%", padding: "8px 12px", fontSize: 13,
    background: "var(--bg-tertiary)", border: "1px solid var(--border)",
    borderRadius: 6, color: "var(--text-primary)", outline: "none",
  };

  const toolOptions = ["calculator", "web_search", "web_scrape", "http_request", "sql_query", "code_execute"];

  const toggleTool = (tool: string) => {
    const current = form.tools.split(",").map((s) => s.trim()).filter(Boolean);
    const next = current.includes(tool) ? current.filter((t) => t !== tool) : [...current, tool];
    setForm({ ...form, tools: next.join(",") });
  };

  return (
    <div style={{ padding: 24 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 24 }}>
        <div>
          <h2 style={{ fontSize: 20, fontWeight: 600, margin: 0 }}>{t(lang, "skillsTitle")}</h2>
          <span style={{ fontSize: 13, color: "var(--text-muted)" }}>{t(lang, "skillsDesc")}</span>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <button onClick={handleReload} style={{ padding: "8px 16px", fontSize: 13, background: "var(--bg-tertiary)", border: "1px solid var(--border)", borderRadius: 6, color: "var(--text-secondary)", cursor: "pointer" }}>
            {t(lang, "reloadSkills")}
          </button>
          <button onClick={() => { setShowForm(!showForm); setEditMode("template"); }} style={{ padding: "8px 16px", fontSize: 13, background: "var(--accent)", border: "none", borderRadius: 6, color: "#fff", cursor: "pointer" }}>
            + {t(lang, "newSkill")}
          </button>
        </div>
      </div>

      {showForm && (
        <div style={{ padding: 20, marginBottom: 20, background: "var(--bg-secondary)", border: "1px solid var(--border)", borderRadius: 10 }}>
          {/* Mode tabs */}
          <div style={{ display: "flex", gap: 0, marginBottom: 16, borderBottom: "1px solid var(--border)" }}>
            <button onClick={() => setEditMode("template")} style={{
              padding: "8px 16px", fontSize: 13, border: "none", cursor: "pointer",
              background: editMode === "template" ? "var(--accent)" : "transparent",
              color: editMode === "template" ? "#fff" : "var(--text-secondary)",
              borderRadius: "6px 6px 0 0",
            }}>{lang === "zh" ? "模板创建" : "Template"}</button>
            <button onClick={() => setEditMode("yaml")} style={{
              padding: "8px 16px", fontSize: 13, border: "none", cursor: "pointer",
              background: editMode === "yaml" ? "var(--accent)" : "transparent",
              color: editMode === "yaml" ? "#fff" : "var(--text-secondary)",
              borderRadius: "6px 6px 0 0",
            }}>YAML</button>
          </div>

          {editMode === "template" ? (
            <>
              {/* Template picker */}
              <div style={{ marginBottom: 16 }}>
                <div style={{ fontSize: 12, color: "var(--text-muted)", marginBottom: 8 }}>
                  {lang === "zh" ? "快速模板" : "Quick Template"}
                </div>
                <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                  {SKILL_TEMPLATES.map((tmpl) => (
                    <button key={tmpl.id} onClick={() => applyTemplate(tmpl.id)} style={{
                      padding: "6px 12px", fontSize: 12, background: form.name === tmpl.name && tmpl.name ? "var(--accent)" : "var(--bg-tertiary)",
                      border: "1px solid var(--border)", borderRadius: 6,
                      color: form.name === tmpl.name && tmpl.name ? "#fff" : "var(--text-secondary)",
                      cursor: "pointer",
                    }}>
                      {lang === "zh" ? tmpl.label_zh : tmpl.label_en}
                    </button>
                  ))}
                </div>
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                <div>
                  <label style={{ fontSize: 12, color: "var(--text-secondary)", marginBottom: 4, display: "block" }}>ID</label>
                  <input value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} placeholder="my-skill" style={inputStyle} />
                </div>
                <div>
                  <label style={{ fontSize: 12, color: "var(--text-secondary)", marginBottom: 4, display: "block" }}>{lang === "zh" ? "显示名称" : "Display Name"}</label>
                  <input value={form.display_name} onChange={(e) => setForm({ ...form, display_name: e.target.value })} placeholder="My Skill" style={inputStyle} />
                </div>
                <div style={{ gridColumn: "1 / -1" }}>
                  <label style={{ fontSize: 12, color: "var(--text-secondary)", marginBottom: 4, display: "block" }}>{t(lang, "taskDesc")}</label>
                  <input value={form.description} onChange={(e) => setForm({ ...form, description: e.target.value })} placeholder="..." style={inputStyle} />
                </div>
                {/* Tool chips instead of text input */}
                <div style={{ gridColumn: "1 / -1" }}>
                  <label style={{ fontSize: 12, color: "var(--text-secondary)", marginBottom: 4, display: "block" }}>{t(lang, "toolsField")}</label>
                  <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                    {toolOptions.map((tool) => {
                      const active = form.tools.split(",").map((s) => s.trim()).includes(tool);
                      return (
                        <button key={tool} onClick={() => toggleTool(tool)} style={{
                          padding: "4px 10px", fontSize: 12, borderRadius: 4, cursor: "pointer",
                          background: active ? "var(--accent)" : "var(--bg-tertiary)",
                          border: active ? "none" : "1px solid var(--border)",
                          color: active ? "#fff" : "var(--text-secondary)",
                        }}>{tool}</button>
                      );
                    })}
                  </div>
                </div>
                <div>
                  <label style={{ fontSize: 12, color: "var(--text-secondary)", marginBottom: 4, display: "block" }}>{t(lang, "temperature")}</label>
                  <input value={form.temperature} onChange={(e) => setForm({ ...form, temperature: e.target.value })} placeholder="0.7" style={inputStyle} />
                </div>
                <div>
                  <label style={{ fontSize: 12, color: "var(--text-secondary)", marginBottom: 4, display: "block" }}>Max Tokens</label>
                  <input value={form.max_tokens} onChange={(e) => setForm({ ...form, max_tokens: e.target.value })} placeholder="4096" style={inputStyle} />
                </div>
                <div style={{ gridColumn: "1 / -1" }}>
                  <label style={{ fontSize: 12, color: "var(--text-secondary)", marginBottom: 4, display: "block" }}>{t(lang, "systemPrompt")}</label>
                  <textarea value={form.system_prompt} onChange={(e) => setForm({ ...form, system_prompt: e.target.value })} placeholder="You are..." rows={3} style={{ ...inputStyle, resize: "vertical" }} />
                </div>
              </div>
            </>
          ) : (
            /* YAML editor mode */
            <div>
              <div style={{ fontSize: 12, color: "var(--text-muted)", marginBottom: 8 }}>
                {lang === "zh" ? "粘贴或编辑 YAML 配置，然后点击创建" : "Paste or edit YAML config, then click Create"}
              </div>
              <textarea
                value={yamlContent}
                onChange={(e) => setYamlContent(e.target.value)}
                spellCheck={false}
                style={{
                  ...inputStyle, fontFamily: "monospace", fontSize: 12, lineHeight: 1.5,
                  resize: "vertical", minHeight: 260,
                }}
                rows={14}
              />
            </div>
          )}

          <div style={{ marginTop: 16, display: "flex", gap: 8 }}>
            <button onClick={editMode === "yaml" ? handleCreateYaml : handleCreate} style={{ padding: "8px 20px", fontSize: 13, background: "var(--accent)", border: "none", borderRadius: 6, color: "#fff", cursor: "pointer" }}>{t(lang, "create")}</button>
            <button onClick={() => setShowForm(false)} style={{ padding: "8px 20px", fontSize: 13, background: "var(--bg-tertiary)", border: "1px solid var(--border)", borderRadius: 6, color: "var(--text-secondary)", cursor: "pointer" }}>{t(lang, "cancel")}</button>
          </div>
        </div>
      )}

      {loading ? (
        <div style={{ color: "var(--text-muted)", textAlign: "center", padding: 40 }}>{t(lang, "loading")}</div>
      ) : skills.length === 0 ? (
        <div style={{ color: "var(--text-muted)", textAlign: "center", padding: 40 }}>{t(lang, "noSkills")}</div>
      ) : (
        <div style={{ display: "grid", gap: 12 }}>
          {skills.map((skill) => (
            <div key={skill.name} style={{
              padding: 16, background: "var(--bg-secondary)", border: "1px solid var(--border)",
              borderRadius: 10, display: "flex", justifyContent: "space-between", alignItems: "flex-start",
            }}>
              <div style={{ flex: 1 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <span style={{ fontWeight: 600, fontSize: 15 }}>{skill.display_name}</span>
                  <span style={{ padding: "1px 6px", fontSize: 10, background: "var(--accent-soft)", borderRadius: 3, color: "var(--accent)" }}>skill</span>
                </div>
                <div style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 2, fontFamily: "monospace" }}>skill-{skill.name}</div>
                {skill.description && <div style={{ fontSize: 13, color: "var(--text-secondary)", marginTop: 6 }}>{skill.description}</div>}
                <div style={{ marginTop: 8, display: "flex", gap: 6, flexWrap: "wrap" }}>
                  {skill.tools.map((tool) => (
                    <span key={tool} style={{ padding: "2px 8px", fontSize: 11, background: "var(--bg-tertiary)", border: "1px solid var(--border)", borderRadius: 4, color: "var(--text-secondary)" }}>{tool}</span>
                  ))}
                  <span style={{ padding: "2px 8px", fontSize: 11, color: "var(--text-muted)" }}>T={skill.temperature}</span>
                </div>
              </div>
              <button onClick={() => handleDelete(skill.name)} style={{
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
