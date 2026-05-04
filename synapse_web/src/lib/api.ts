const API_BASE = "/api";

export interface AgentInfo {
  id: string;
  name: string;
  role: string;
  goal: string;
  tools: string[];
}

export interface SessionInfo {
  session_id: string;
  agent_id: string;
  title?: string;
  message_count: number;
  messages?: { role: string; content: string }[];
}

export interface RunResponse {
  run_id: string;
  session_id: string;
  agent_id: string;
  status: string;
}

export interface Metrics {
  counters: Record<string, number>;
  gauges: Record<string, number>;
  [key: string]: unknown;
}

export async function fetchAgents(): Promise<AgentInfo[]> {
  const res = await fetch(`${API_BASE}/agents`);
  if (!res.ok) throw new Error("Failed to fetch agents");
  return res.json();
}

export async function fetchAgent(id: string): Promise<AgentInfo> {
  const res = await fetch(`${API_BASE}/agents/${id}`);
  if (!res.ok) throw new Error("Failed to fetch agent");
  return res.json();
}

export async function createAgent(data: Partial<AgentInfo> & { id: string; name: string; role: string }): Promise<AgentInfo> {
  const res = await fetch(`${API_BASE}/agents`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error("Failed to create agent");
  return res.json();
}

export async function deleteAgent(id: string): Promise<void> {
  const res = await fetch(`${API_BASE}/agents/${id}`, { method: "DELETE" });
  if (!res.ok) throw new Error("Failed to delete agent");
}

export async function createSession(data: { agent_id?: string; user_id?: string; title?: string }): Promise<SessionInfo> {
  const res = await fetch(`${API_BASE}/sessions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error("Failed to create session");
  return res.json();
}

export async function fetchSessions(): Promise<SessionInfo[]> {
  const res = await fetch(`${API_BASE}/sessions`);
  if (!res.ok) throw new Error("Failed to fetch sessions");
  return res.json();
}

export async function deleteSession(id: string): Promise<void> {
  const res = await fetch(`${API_BASE}/sessions/${id}`, { method: "DELETE" });
  if (!res.ok) throw new Error("Failed to delete session");
}

export async function createRun(message: string, agentId?: string, sessionId?: string): Promise<RunResponse> {
  const res = await fetch(`${API_BASE}/runs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, agent_id: agentId, session_id: sessionId }),
  });
  if (!res.ok) throw new Error("Failed to create run");
  return res.json();
}

export async function fetchMetrics(): Promise<Metrics> {
  const res = await fetch(`${API_BASE}/metrics`);
  if (!res.ok) throw new Error("Failed to fetch metrics");
  return res.json();
}

export async function checkHealth(): Promise<{ status: string }> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error("Health check failed");
  return res.json();
}

export function getStreamUrl(): string {
  return `${API_BASE}/runs/stream`;
}

export async function fetchMultiAgentPatterns(): Promise<{ patterns: { id: string; name: string; description: string }[] }> {
  const res = await fetch(`${API_BASE}/multi-agent/patterns`);
  if (!res.ok) throw new Error("Failed to fetch patterns");
  return res.json();
}

export function getMultiAgentStreamUrl(): string {
  return `${API_BASE}/multi-agent/stream`;
}

// ── Skills ──────────────────────────────────────────────────────────────

export interface SkillInfo {
  name: string;
  display_name: string;
  description: string;
  tools: string[];
  temperature: number;
  max_tokens: number;
}

export async function fetchSkills(): Promise<SkillInfo[]> {
  const res = await fetch(`${API_BASE}/skills`);
  if (!res.ok) throw new Error("Failed to fetch skills");
  return res.json();
}

export async function createSkill(data: {
  name: string;
  display_name: string;
  description?: string;
  tools?: string[];
  system_prompt?: string;
  temperature?: number;
  max_tokens?: number;
}): Promise<{ name: string; status: string }> {
  const res = await fetch(`${API_BASE}/skills`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error("Failed to create skill");
  return res.json();
}

export async function deleteSkill(name: string): Promise<void> {
  const res = await fetch(`${API_BASE}/skills/${name}`, { method: "DELETE" });
  if (!res.ok) throw new Error("Failed to delete skill");
}

export async function reloadSkills(): Promise<{ status: string; count: number }> {
  const res = await fetch(`${API_BASE}/skills/reload`, { method: "POST" });
  if (!res.ok) throw new Error("Failed to reload skills");
  return res.json();
}

// ── MCP Servers ─────────────────────────────────────────────────────────

export interface MCPServerInfo {
  name: string;
  command: string;
  args: string[];
  url: string;
  has_env: boolean;
  connected?: boolean;
  tools?: string[];
}

export async function fetchMCPServers(): Promise<MCPServerInfo[]> {
  const res = await fetch(`${API_BASE}/mcp/servers`);
  if (!res.ok) throw new Error("Failed to fetch MCP servers");
  return res.json();
}

export async function addMCPServer(data: {
  name: string;
  command?: string;
  args?: string[];
  env?: Record<string, string>;
  url?: string;
}): Promise<{ name: string; status: string }> {
  const res = await fetch(`${API_BASE}/mcp/servers`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error("Failed to add MCP server");
  return res.json();
}

export async function removeMCPServer(name: string): Promise<void> {
  const res = await fetch(`${API_BASE}/mcp/servers/${name}`, { method: "DELETE" });
  if (!res.ok) throw new Error("Failed to remove MCP server");
}

export async function connectMCPServers(): Promise<{ status: string; tools_registered: number }> {
  const res = await fetch(`${API_BASE}/mcp/connect`, { method: "POST" });
  if (!res.ok) throw new Error("Failed to connect MCP servers");
  return res.json();
}

// ── Scheduler ───────────────────────────────────────────────────────────

export interface ScheduledTaskInfo {
  id: string;
  name: string;
  description: string;
  agent_id: string;
  message: string;
  cron: string;
  enabled: boolean;
  last_run: number;
  next_run: number;
  run_count: number;
  created_at: number;
}

export async function fetchScheduledTasks(): Promise<ScheduledTaskInfo[]> {
  const res = await fetch(`${API_BASE}/scheduler/tasks`);
  if (!res.ok) throw new Error("Failed to fetch tasks");
  return res.json();
}

export async function createScheduledTask(data: {
  name: string;
  message: string;
  cron: string;
  agent_id?: string;
  description?: string;
}): Promise<ScheduledTaskInfo> {
  const res = await fetch(`${API_BASE}/scheduler/tasks`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error("Failed to create task");
  return res.json();
}

export async function updateScheduledTask(
  id: string,
  data: Partial<ScheduledTaskInfo>,
): Promise<ScheduledTaskInfo> {
  const res = await fetch(`${API_BASE}/scheduler/tasks/${id}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error("Failed to update task");
  return res.json();
}

export async function deleteScheduledTask(id: string): Promise<void> {
  const res = await fetch(`${API_BASE}/scheduler/tasks/${id}`, { method: "DELETE" });
  if (!res.ok) throw new Error("Failed to delete task");
}

export async function runScheduledTask(id: string): Promise<{ id: string; status: string }> {
  const res = await fetch(`${API_BASE}/scheduler/tasks/${id}/run`, { method: "POST" });
  if (!res.ok) throw new Error("Failed to trigger task");
  return res.json();
}

// ── Files ───────────────────────────────────────────────────────────────

export interface UploadedFile {
  id: string;
  filename: string;
  stored_name: string;
  size: number;
  mime_type: string;
  is_image: boolean;
  url: string;
}

export async function uploadFile(file: globalThis.File): Promise<UploadedFile> {
  const formData = new FormData();
  formData.append("file", file);
  const baseUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
  const res = await fetch(`${baseUrl}/api/files/upload`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) throw new Error("Failed to upload file");
  return res.json();
}

export async function readImageAsBase64(file: globalThis.File): Promise<{ data_url: string; filename: string }> {
  const formData = new FormData();
  formData.append("file", file);
  const baseUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
  const res = await fetch(`${baseUrl}/api/files/read-image`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) throw new Error("Failed to read image");
  return res.json();
}
