"use client";

import { useEffect, useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer,
} from "recharts";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const CLUSTER_COLORS = ["#ef4444", "#f97316", "#eab308", "#22c55e", "#3b82f6", "#8b5cf6"];

// ── Types ─────────────────────────────────────────────────────────────────────

interface ClusterRow {
  id: number;
  name: string;
  total_calls: number;
  calls_today: number;
  agent_count: number;
}

interface AgentRisk {
  agent_id: number;
  agent_name: string;
  risk_score: number;
}

interface Summary {
  total_calls: number;
  analyzed_calls: number;
  escalations: number;
  total_agents: number;
  risky_agents: number;
}

interface TrendPoint { month: string; [key: string]: string | number; }

interface RecentCall {
  id: number;
  agent: { name: string } | null;
  cluster: { name: string } | null;
  analysis: { risk_level: string } | null;
  call_date: string | null;
}

// ── Stat Card ─────────────────────────────────────────────────────────────────

function StatCard({ label, value, highlight = false }: { label: string; value: string | number; highlight?: boolean }) {
  return (
    <div className={`rounded-2xl p-5 shadow-sm border flex flex-col justify-between h-28 ${highlight ? "bg-red-500 border-red-500" : "bg-white border-gray-100"}`}>
      <p className={`text-xs font-medium ${highlight ? "text-red-100" : "text-gray-400"}`}>{label}</p>
      <p className={`text-3xl font-bold ${highlight ? "text-white" : "text-gray-800"}`}>{value}</p>
    </div>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────────

export default function Dashboard() {
  const [clusters, setClusters]     = useState<ClusterRow[]>([]);
  const [summary, setSummary]       = useState<Summary | null>(null);
  const [trend, setTrend]           = useState<TrendPoint[]>([]);
  const [agentRisk, setAgentRisk]   = useState<AgentRisk[]>([]);
  const [recentCall, setRecentCall] = useState<RecentCall | null>(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/clusters`).then((r) => r.json()).catch(() => []),
      fetch(`${API}/reports/summary`).then((r) => r.json()).catch(() => null),
      fetch(`${API}/reports/call-volume`).then((r) => r.json()).catch(() => []),
      fetch(`${API}/reports/agent-risk-scores`).then((r) => r.json()).catch(() => []),
      fetch(`${API}/calls`).then((r) => r.json()).catch(() => []),
    ]).then(([cls, sum, vol, ar, calls]) => {
      setClusters(cls);
      setSummary(sum);
      setTrend(vol);
      setAgentRisk(ar.slice(0, 5));
      // most recent call that has an analysis
      const withAnalysis = (calls as RecentCall[]).find((c) => c.analysis);
      setRecentCall(withAnalysis ?? (calls[0] ?? null));
    });
  }, []);

  const maxCalls = Math.max(...clusters.map((c) => c.total_calls), 1);
  const clusterKeys = trend.length > 0 ? Object.keys(trend[0]).filter((k) => k !== "month") : [];
  const trendColors: Record<string, string> = {};
  clusterKeys.forEach((k, i) => { trendColors[k] = CLUSTER_COLORS[i % CLUSTER_COLORS.length]; });

  const highLoadPct = summary && summary.total_agents > 0
    ? `${Math.round((summary.risky_agents / summary.total_agents) * 100)}%`
    : "—";

  return (
    <div className="space-y-6">
      {/* Top row */}
      <div className="grid grid-cols-12 gap-6">

        {/* Groups Present */}
        <div className="col-span-4 bg-white rounded-2xl p-5 shadow-sm border border-gray-100">
          <h2 className="text-base font-semibold text-gray-700 mb-4">Groups Present</h2>
          <div className="flex flex-col gap-2 mb-6">
            {clusters.map((c, i) => (
              <div key={c.id} className="flex items-center gap-2 text-sm text-gray-600">
                <span className="inline-block w-3 h-3 rounded-sm flex-shrink-0" style={{ backgroundColor: CLUSTER_COLORS[i % CLUSTER_COLORS.length] }} />
                {c.name}
              </div>
            ))}
          </div>
          <div className="flex items-end gap-3 h-32">
            {clusters.map((c, i) => (
              <div key={c.id} className="flex flex-col items-center gap-1 flex-1">
                <div className="w-full rounded-t-md transition-all duration-500"
                  style={{ height: `${(c.total_calls / maxCalls) * 100}%`, backgroundColor: CLUSTER_COLORS[i % CLUSTER_COLORS.length] }} />
                <span className="text-xs text-gray-500">{c.total_calls}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Right: stat cards + trend */}
        <div className="col-span-8 flex flex-col gap-6">
          <div className="grid grid-cols-3 gap-6">
            <StatCard label="Total Agents" value={summary?.total_agents ?? "—"} />
            <StatCard label="High Load" value={highLoadPct} highlight />
            <StatCard label="Total Calls" value={summary?.total_calls ?? "—"} />
          </div>

          <div className="flex-1 bg-white rounded-2xl p-5 shadow-sm border border-gray-100">
            <h2 className="text-base font-semibold text-gray-700 mb-3">Emotional Trend Line</h2>
            <ResponsiveContainer width="100%" height={130}>
              <LineChart data={trend}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" />
                <XAxis dataKey="month" tick={{ fontSize: 10, fill: "#9ca3af" }} axisLine={false} tickLine={false} />
                <YAxis hide />
                <Tooltip contentStyle={{ borderRadius: 8, border: "none", boxShadow: "0 4px 12px rgba(0,0,0,0.1)", fontSize: 12 }} />
                <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 11 }} />
                {clusterKeys.map((key) => (
                  <Line key={key} type="monotone" dataKey={key} stroke={trendColors[key]} strokeWidth={2} dot={false} activeDot={{ r: 4 }} />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Bottom row */}
      <div className="grid grid-cols-2 gap-6">

        {/* Top 5 Risk Agents */}
        <div className="bg-white rounded-2xl p-5 shadow-sm border border-gray-100">
          <h2 className="text-base font-semibold text-gray-700 mb-4">Top 5 Risk Agent</h2>
          {agentRisk.length === 0 ? (
            <p className="text-sm text-gray-400">No data yet</p>
          ) : (
            <ol className="space-y-3">
              {agentRisk.map((a, i) => (
                <li key={a.agent_id} className="flex items-center gap-3">
                  <span className="w-6 h-6 rounded-full bg-red-100 text-red-600 text-xs font-bold flex items-center justify-center flex-shrink-0">{i + 1}</span>
                  <span className="text-sm text-gray-700 w-36 truncate">{a.agent_name}</span>
                  <div className="flex-1 h-1.5 bg-gray-100 rounded-full">
                    <div className="h-1.5 bg-red-400 rounded-full" style={{ width: `${a.risk_score}%` }} />
                  </div>
                  <span className="text-xs text-gray-400 w-10 text-right">{a.risk_score}</span>
                </li>
              ))}
            </ol>
          )}
        </div>

        {/* Recent Tagged Call */}
        <div className="bg-white rounded-2xl p-5 shadow-sm border border-gray-100">
          <h2 className="text-base font-semibold text-gray-700 mb-4">Recent Tagged Call</h2>
          {!recentCall ? (
            <p className="text-sm text-gray-400">No calls yet</p>
          ) : (
            <>
              <div className="flex items-center gap-3 mb-5 p-3 bg-gray-50 rounded-xl">
                <div className="w-10 h-10 rounded-full bg-red-100 flex items-center justify-center flex-shrink-0">
                  <svg className="w-5 h-5 text-red-500" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M10 9a3 3 0 100-6 3 3 0 000 6zm-7 9a7 7 0 1114 0H3z" clipRule="evenodd" />
                  </svg>
                </div>
                <div>
                  <p className="text-sm font-semibold text-gray-800">{recentCall.agent?.name ?? "—"}</p>
                  <p className="text-xs text-gray-400">Agent · {recentCall.cluster?.name ?? "—"}</p>
                </div>
                {recentCall.analysis && (
                  <span className={`ml-auto px-2 py-1 text-xs font-semibold rounded-full ${
                    recentCall.analysis.risk_level === "Critical" ? "bg-red-100 text-red-700" :
                    recentCall.analysis.risk_level === "High"     ? "bg-orange-100 text-orange-700" :
                    recentCall.analysis.risk_level === "Medium"   ? "bg-yellow-100 text-yellow-700" :
                    "bg-green-100 text-green-700"
                  }`}>
                    {recentCall.analysis.risk_level} Risk
                  </span>
                )}
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-gray-50 rounded-xl p-4 text-center">
                  <p className="text-xs text-gray-400 mb-1">Call Date</p>
                  <p className="text-base font-bold text-gray-800">
                    {recentCall.call_date ? new Date(recentCall.call_date).toLocaleDateString("en-US", { month: "short", day: "numeric" }) : "—"}
                  </p>
                </div>
                <div className="bg-red-50 rounded-xl p-4 text-center">
                  <p className="text-xs text-gray-400 mb-1">Risk Level</p>
                  <p className="text-base font-bold text-red-600">{recentCall.analysis?.risk_level ?? "—"}</p>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
