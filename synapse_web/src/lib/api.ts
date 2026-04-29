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
