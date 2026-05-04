"use client";

import { useEffect, useState } from "react";
import {
  fetchScheduledTasks, createScheduledTask, deleteScheduledTask,
  updateScheduledTask, runScheduledTask, ScheduledTaskInfo,
  fetchAgents, AgentInfo,
} from "@/lib/api";
import { Lang, t } from "@/lib/i18n";
import { useToast } from "@/components/Toast";

interface Props { lang: Lang }

function formatTime(ts: number): string {
  if (!ts) return "-";
  return new Date(ts * 1000).toLocaleString();
}

export default function SchedulerPanel({ lang }: Props) {
  const [tasks, setTasks] = useState<ScheduledTaskInfo[]>([]);
  const [agents, setAgents] = useState<AgentInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [showForm, setShowForm] = useState(false);
  const { toast, confirm: toastConfirm } = useToast();
  const [form, setForm] = useState({
    name: "", message: "", cron: "0 9 * * *", agent_id: "general-assistant", description: "",
  });

  const load = async () => {
    try {
      setLoading(true);
      const [tList, aList] = await Promise.all([fetchScheduledTasks(), fetchAgents()]);
      setTasks(tList);
      setAgents(aList);
    } catch { toast("Failed to load tasks", "error"); } finally { setLoading(false); }
  };

  useEffect(() => { load(); }, []);

  const handleCreate = async () => {
    if (!form.name || !form.message || !form.cron) return;
    try {
      await createScheduledTask(form);
      setShowForm(false);
      setForm({ name: "", message: "", cron: "0 9 * * *", agent_id: "general-assistant", description: "" });
      load();
    } catch (err) { toast((err as Error).message, "error"); }
  };

  const handleDelete = async (id: string) => {
    if (!await toastConfirm(t(lang, "deleteConfirm"))) return;
    try { await deleteScheduledTask(id); load(); } catch (err) { toast((err as Error).message, "error"); }
  };

  const handleToggle = async (task: ScheduledTaskInfo) => {
    try {
      await updateScheduledTask(task.id, { enabled: !task.enabled } as Partial<ScheduledTaskInfo>);
      load();
    } catch (err) { toast((err as Error).message, "error"); }
  };

  const handleRun = async (id: string) => {
    try { await runScheduledTask(id); toast("Task triggered", "success"); load(); } catch (err) { toast((err as Error).message, "error"); }
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
          <h2 style={{ fontSize: 20, fontWeight: 600, margin: 0 }}>{t(lang, "schedulerTitle")}</h2>
          <span style={{ fontSize: 13, color: "var(--text-muted)" }}>{t(lang, "schedulerDesc")}</span>
        </div>
        <button onClick={() => setShowForm(!showForm)} style={{ padding: "8px 16px", fontSize: 13, background: "var(--accent)", border: "none", borderRadius: 6, color: "#fff", cursor: "pointer" }}>
          + {t(lang, "newTask")}
        </button>
      </div>

      {showForm && (
        <div style={{ padding: 20, marginBottom: 20, background: "var(--bg-secondary)", border: "1px solid var(--border)", borderRadius: 10 }}>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            <div>
              <label style={{ fontSize: 12, color: "var(--text-secondary)", marginBottom: 4, display: "block" }}>{t(lang, "taskName")}</label>
              <input value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} placeholder="Daily report" style={inputStyle} />
            </div>
            <div>
              <label style={{ fontSize: 12, color: "var(--text-secondary)", marginBottom: 4, display: "block" }}>{t(lang, "cronExpression")}</label>
              <input value={form.cron} onChange={(e) => setForm({ ...form, cron: e.target.value })} placeholder="0 9 * * *" style={inputStyle} />
            </div>
            <div>
              <label style={{ fontSize: 12, color: "var(--text-secondary)", marginBottom: 4, display: "block" }}>{t(lang, "agent")}</label>
              <select value={form.agent_id} onChange={(e) => setForm({ ...form, agent_id: e.target.value })} style={inputStyle}>
                {agents.map((a) => <option key={a.id} value={a.id}>{a.name}</option>)}
              </select>
            </div>
            <div>
              <label style={{ fontSize: 12, color: "var(--text-secondary)", marginBottom: 4, display: "block" }}>{t(lang, "taskDesc")}</label>
              <input value={form.description} onChange={(e) => setForm({ ...form, description: e.target.value })} placeholder="..." style={inputStyle} />
            </div>
            <div style={{ gridColumn: "1 / -1" }}>
              <label style={{ fontSize: 12, color: "var(--text-secondary)", marginBottom: 4, display: "block" }}>{t(lang, "taskMessage")}</label>
              <textarea value={form.message} onChange={(e) => setForm({ ...form, message: e.target.value })} placeholder="Summarize today's news..." rows={3} style={{ ...inputStyle, resize: "vertical" }} />
            </div>
          </div>
          <div style={{ marginTop: 16, display: "flex", gap: 8 }}>
            <button onClick={handleCreate} style={{ padding: "8px 20px", fontSize: 13, background: "var(--accent)", border: "none", borderRadius: 6, color: "#fff", cursor: "pointer" }}>{t(lang, "create")}</button>
            <button onClick={() => setShowForm(false)} style={{ padding: "8px 20px", fontSize: 13, background: "var(--bg-tertiary)", border: "1px solid var(--border)", borderRadius: 6, color: "var(--text-secondary)", cursor: "pointer" }}>{t(lang, "cancel")}</button>
          </div>
        </div>
      )}

      {loading ? (
        <div style={{ color: "var(--text-muted)", textAlign: "center", padding: 40 }}>{t(lang, "loading")}</div>
      ) : tasks.length === 0 ? (
        <div style={{ color: "var(--text-muted)", textAlign: "center", padding: 40 }}>{t(lang, "noTasks")}</div>
      ) : (
        <div style={{ display: "grid", gap: 12 }}>
          {tasks.map((task) => (
            <div key={task.id} style={{
              padding: 16, background: "var(--bg-secondary)", border: "1px solid var(--border)",
              borderRadius: 10,
            }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                <div>
                  <div style={{ fontWeight: 600, fontSize: 15 }}>
                    {task.name}
                    <span style={{
                      marginLeft: 8, padding: "2px 8px", fontSize: 11, borderRadius: 4,
                      background: task.enabled ? "var(--accent-soft)" : "var(--bg-tertiary)",
                      color: task.enabled ? "var(--accent)" : "var(--text-muted)",
                    }}>
                      {task.enabled ? t(lang, "enabled") : t(lang, "disabled")}
                    </span>
                  </div>
                  <div style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 4, fontFamily: "monospace" }}>
                    {task.cron} | {t(lang, "agent")}: {task.agent_id}
                  </div>
                  {task.description && <div style={{ fontSize: 13, color: "var(--text-secondary)", marginTop: 6 }}>{task.description}</div>}
                  <div style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 4 }}>
                    {t(lang, "lastRun")}: {task.last_run ? formatTime(task.last_run) : t(lang, "never")}
                    {" | "}{t(lang, "runCount")}: {task.run_count}
                  </div>
                </div>
                <div style={{ display: "flex", gap: 6 }}>
                  <button onClick={() => handleToggle(task)} style={{
                    padding: "6px 12px", fontSize: 12, background: "transparent",
                    border: "1px solid var(--border)", borderRadius: 6, color: "var(--text-secondary)", cursor: "pointer",
                  }}>
                    {task.enabled ? t(lang, "disabled") : t(lang, "enabled")}
                  </button>
                  <button onClick={() => handleRun(task.id)} style={{
                    padding: "6px 12px", fontSize: 12, background: "transparent",
                    border: "1px solid var(--accent)", borderRadius: 6, color: "var(--accent)", cursor: "pointer",
                  }}>{t(lang, "trigger")}</button>
                  <button onClick={() => handleDelete(task.id)} style={{
                    padding: "6px 12px", fontSize: 12, background: "transparent",
                    border: "1px solid var(--error)", borderRadius: 6, color: "var(--error)", cursor: "pointer",
                  }}>{t(lang, "delete")}</button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
