const BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function req<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, options);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Request failed");
  }
  if (res.status === 204) return undefined as T;
  return res.json();
}

// ── Clusters ──────────────────────────────────────────────────────────────────

export const getClusters = () => req<ClusterRow[]>("/clusters");

export const createCluster = (data: { name: string; region: string; overall_risk?: string }) =>
  req("/clusters", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });

export const updateCluster = (id: number, data: Partial<{ name: string; region: string; overall_risk: string }>) =>
  req(`/clusters/${id}`, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });

export const deleteCluster = (id: number) =>
  req<void>(`/clusters/${id}`, { method: "DELETE" });

// ── Agents ────────────────────────────────────────────────────────────────────

export const getAgents = (clusterId?: number) =>
  req<AgentRow[]>(`/agents${clusterId ? `?cluster_id=${clusterId}` : ""}`);

export const createAgent = (data: { cluster_id: number; name: string; email: string; role?: string; risk_level?: string }) =>
  req("/agents", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });

export const updateAgent = (id: number, data: Partial<{ cluster_id: number; name: string; email: string; role: string; risk_level: string }>) =>
  req(`/agents/${id}`, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });

export const deleteAgent = (id: number) =>
  req<void>(`/agents/${id}`, { method: "DELETE" });

// ── Calls ─────────────────────────────────────────────────────────────────────

export const getCalls = (params?: { cluster_id?: number; agent_id?: number }) => {
  const qs = new URLSearchParams();
  if (params?.cluster_id) qs.set("cluster_id", String(params.cluster_id));
  if (params?.agent_id) qs.set("agent_id", String(params.agent_id));
  return req<CallRow[]>(`/calls${qs.toString() ? `?${qs}` : ""}`);
};

export const updateCall = (id: number, data: Partial<{ agent_id: number; call_date: string; upload_status: string }>) =>
  req(`/calls/${id}`, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });

export const deleteCall = (id: number) =>
  req<void>(`/calls/${id}`, { method: "DELETE" });

// ── Reports ───────────────────────────────────────────────────────────────────

export const getReportSummary   = (clusterId?: number) => req<ReportSummary>(`/reports/summary${clusterId ? `?cluster_id=${clusterId}` : ""}`);
export const getEmotionDist     = (clusterId?: number) => req<EmotionDist[]>(`/reports/emotion-distribution${clusterId ? `?cluster_id=${clusterId}` : ""}`);
export const getEmotionTrend    = (clusterId?: number) => req<Record<string, number | string>[]>(`/reports/emotion-trend${clusterId ? `?cluster_id=${clusterId}` : ""}`);
export const getRiskTrend       = (clusterId?: number) => req<Record<string, number | string>[]>(`/reports/risk-trend${clusterId ? `?cluster_id=${clusterId}` : ""}`);
export const getCallVolume      = () => req<Record<string, number | string>[]>("/reports/call-volume");
export const getAgentRiskScores = (clusterId?: number) => req<AgentRiskScore[]>(`/reports/agent-risk-scores${clusterId ? `?cluster_id=${clusterId}` : ""}`);

// ── Shared types ──────────────────────────────────────────────────────────────

export interface ClusterRow {
  id: number;
  name: string;
  region: string;
  overall_risk: string;
  agent_count: number;
  risky_agents: number;
  medium_agents: number;
  safe_agents: number;
  calls_today: number;
  total_calls: number;
  created_at: string | null;
}

export interface AgentRow {
  id: number;
  name: string;
  email: string;
  role: string;
  risk_level: string;
  is_active: boolean;
  cluster_id: number;
  cluster_name: string | null;
  calls_today: number;
  total_calls: number;
  created_at: string | null;
}

export interface CallRow {
  id: number;
  uuid: string;
  filename: string;
  file_size: number | null;
  duration_sec: number | null;
  upload_status: string;
  call_date: string | null;
  created_at: string | null;
  agent: { id: number; name: string; email: string } | null;
  cluster: { id: number; name: string } | null;
  analysis: {
    predicted_emotion: string;
    confidence: number;
    risk_level: string;
    transcription_text: string | null;
    valence: string | null;
    arousal: string | null;
  } | null;
  recommendation: {
    action: string;
    urgency: string;
    reason: string | null;
    instruction: string | null;
    action_color: string | null;
    recommended_tone: string | null;
    example_phrases: string[] | null;
    do_list: string[] | null;
    dont_list: string[] | null;
  } | null;
}

export interface ReportSummary {
  total_calls: number;
  analyzed_calls: number;
  escalations: number;
  total_agents: number;
  risky_agents: number;
}

export interface EmotionDist {
  emotion: string;
  count: number;
}

export interface AgentRiskScore {
  agent_id: number;
  agent_name: string;
  cluster: string | null;
  risk_level: string;
  critical: number;
  high: number;
  medium: number;
  low: number;
  total_calls: number;
  risk_score: number;
}
