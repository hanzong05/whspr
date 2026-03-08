"use client";

import {
  AreaChart, Area,
  BarChart, Bar,
  PieChart, Pie, Cell,
  RadarChart, Radar, PolarGrid, PolarAngleAxis,
  LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from "recharts";
import { useState, useEffect } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// ── Static helpers ────────────────────────────────────────────────────────────

const emotionColors: Record<string, string> = {
  angry: "#ef4444", frustrated: "#f97316", neutral: "#94a3b8", happy: "#22c55e", sad: "#3b82f6", satisfied: "#8b5cf6",
};

const riskColors: Record<string, string> = {
  Critical: "#ef4444", High: "#f97316", Medium: "#eab308", Low: "#22c55e",
};

const clusterColors = ["#ef4444", "#f97316", "#eab308", "#22c55e", "#3b82f6", "#8b5cf6"];

const riskScoreColor = (s: number) => s >= 75 ? "text-red-600" : s >= 50 ? "text-orange-500" : s >= 30 ? "text-yellow-600" : "text-green-600";
const riskScoreBar   = (s: number) => s >= 75 ? "bg-red-500"  : s >= 50 ? "bg-orange-400"   : s >= 30 ? "bg-yellow-400"   : "bg-green-500";

const DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
const maxHeat = 14;

// Static heatmap (no API equivalent)
const heatmapData = [
  { hour: "8am",  Mon: 3, Tue: 4, Wed: 2, Thu: 5, Fri: 6, Sat: 1, Sun: 0 },
  { hour: "9am",  Mon: 6, Tue: 7, Wed: 5, Thu: 8, Fri: 9, Sat: 3, Sun: 1 },
  { hour: "10am", Mon: 8, Tue: 9, Wed: 8, Thu: 10, Fri: 11, Sat: 4, Sun: 2 },
  { hour: "11am", Mon: 10, Tue: 11, Wed: 9, Thu: 12, Fri: 13, Sat: 5, Sun: 2 },
  { hour: "12pm", Mon: 7, Tue: 8, Wed: 7, Thu: 9, Fri: 10, Sat: 6, Sun: 3 },
  { hour: "1pm",  Mon: 9, Tue: 10, Wed: 8, Thu: 11, Fri: 12, Sat: 4, Sun: 2 },
  { hour: "2pm",  Mon: 11, Tue: 12, Wed: 10, Thu: 13, Fri: 14, Sat: 3, Sun: 1 },
  { hour: "3pm",  Mon: 10, Tue: 11, Wed: 9, Thu: 12, Fri: 13, Sat: 2, Sun: 1 },
  { hour: "4pm",  Mon: 8, Tue: 9, Wed: 8, Thu: 10, Fri: 11, Sat: 2, Sun: 0 },
  { hour: "5pm",  Mon: 5, Tue: 6, Wed: 5, Thu: 7, Fri: 8, Sat: 1, Sun: 0 },
];

// ── Types ─────────────────────────────────────────────────────────────────────

interface Summary { total_calls: number; analyzed_calls: number; escalations: number; total_agents: number; risky_agents: number; }
interface EmotionDist { emotion: string; count: number; }
interface AgentRisk { agent_id: number; agent_name: string; cluster: string | null; risk_level: string; critical: number; high: number; medium: number; low: number; total_calls: number; risk_score: number; }
interface ClusterOption { id: number; name: string; }

// ── Summary Card ──────────────────────────────────────────────────────────────

function SummaryCard({ label, value, sub, accent = false }: { label: string; value: string | number; sub?: string; accent?: boolean }) {
  return (
    <div className={`rounded-2xl p-5 border ${accent ? "bg-red-500 border-red-500" : "bg-white border-gray-100"} shadow-sm`}>
      <p className={`text-xs font-medium mb-1 ${accent ? "text-red-100" : "text-gray-400"}`}>{label}</p>
      <p className={`text-3xl font-bold ${accent ? "text-white" : "text-gray-800"}`}>{value}</p>
      {sub && <p className={`text-xs mt-1 ${accent ? "text-red-200" : "text-gray-400"}`}>{sub}</p>}
    </div>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────────

export default function ReportsPage() {
  const [clusterId, setClusterId] = useState<string>("all");
  const [clusters, setClusters] = useState<ClusterOption[]>([]);

  const [summary, setSummary]         = useState<Summary | null>(null);
  const [emotionDist, setEmotionDist] = useState<EmotionDist[]>([]);
  const [emotionTrend, setEmotionTrend] = useState<Record<string, number | string>[]>([]);
  const [riskTrend, setRiskTrend]     = useState<Record<string, number | string>[]>([]);
  const [callVolume, setCallVolume]   = useState<Record<string, number | string>[]>([]);
  const [agentRisk, setAgentRisk]     = useState<AgentRisk[]>([]);

  const fetchAll = async (cid: string) => {
    const q = cid !== "all" ? `?cluster_id=${cid}` : "";
    try {
      const [s, ed, et, rt, cv, ar] = await Promise.all([
        fetch(`${API}/reports/summary${q}`).then((r) => r.json()),
        fetch(`${API}/reports/emotion-distribution${q}`).then((r) => r.json()),
        fetch(`${API}/reports/emotion-trend${q}`).then((r) => r.json()),
        fetch(`${API}/reports/risk-trend${q}`).then((r) => r.json()),
        fetch(`${API}/reports/call-volume`).then((r) => r.json()),
        fetch(`${API}/reports/agent-risk-scores${q}`).then((r) => r.json()),
      ]);
      setSummary(s);
      setEmotionDist(ed);
      setEmotionTrend(et);
      setRiskTrend(rt);
      setCallVolume(cv);
      setAgentRisk(ar);
    } catch { /* keep previous data on error */ }
  };

  useEffect(() => {
    fetch(`${API}/clusters`).then((r) => r.json()).then(setClusters).catch(() => {});
  }, []);

  useEffect(() => { fetchAll(clusterId); }, [clusterId]);

  // Derive cluster keys from call volume data
  const clusterKeys = callVolume.length > 0 ? Object.keys(callVolume[0]).filter((k) => k !== "month") : [];
  const emotionKeys = emotionTrend.length > 0 ? Object.keys(emotionTrend[0]).filter((k) => k !== "month") : [];
  const riskKeys    = riskTrend.length > 0    ? Object.keys(riskTrend[0]).filter((k) => k !== "month")    : [];

  const totalEmotions = emotionDist.reduce((a, b) => a + b.count, 0) || 1;
  const emotionPieData = emotionDist.map((e) => ({ name: e.emotion, value: e.count, color: emotionColors[e.emotion] ?? "#94a3b8" }));

  return (
    <div className="space-y-6">

      {/* ── Filters ── */}
      <div className="flex flex-wrap items-center gap-3">
        <select value={clusterId} onChange={(e) => setClusterId(e.target.value)}
          className="px-4 py-2.5 text-sm bg-white border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-red-400">
          <option value="all">All Clusters</option>
          {clusters.map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}
        </select>
        <button className="flex items-center gap-2 px-4 py-2.5 bg-white border border-gray-200 hover:bg-gray-50 text-sm font-medium text-gray-600 rounded-xl transition-colors ml-auto">
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" /></svg>
          Export Report
        </button>
      </div>

      {/* ── Summary Row ── */}
      <div className="grid grid-cols-5 gap-4">
        <SummaryCard label="Total Calls Analyzed" value={summary?.total_calls ?? "—"} />
        <SummaryCard label="Analyzed Calls" value={summary?.analyzed_calls ?? "—"} accent />
        <SummaryCard label="Escalations" value={summary?.escalations ?? "—"} sub="Needs attention" />
        <SummaryCard label="Total Agents" value={summary?.total_agents ?? "—"} />
        <SummaryCard label="Risky Agents" value={summary?.risky_agents ?? "—"} sub="High load" />
      </div>

      {/* ── Row 1: Call Volume + Emotion Distribution ── */}
      <div className="grid grid-cols-12 gap-6">
        <div className="col-span-8 bg-white rounded-2xl p-5 shadow-sm border border-gray-100">
          <h3 className="text-sm font-semibold text-gray-800 mb-1">Call Volume by Cluster</h3>
          <p className="text-xs text-gray-400 mb-4">Monthly breakdown · last 12 months</p>
          <ResponsiveContainer width="100%" height={220}>
            <AreaChart data={callVolume}>
              <defs>
                {clusterKeys.map((k, i) => (
                  <linearGradient key={k} id={`grad${i}`} x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={clusterColors[i % clusterColors.length]} stopOpacity={0.15} />
                    <stop offset="95%" stopColor={clusterColors[i % clusterColors.length]} stopOpacity={0} />
                  </linearGradient>
                ))}
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" />
              <XAxis dataKey="month" tick={{ fontSize: 11, fill: "#9ca3af" }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fontSize: 11, fill: "#9ca3af" }} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={{ borderRadius: 10, border: "none", boxShadow: "0 4px 12px rgba(0,0,0,.08)", fontSize: 12 }} />
              <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 11 }} />
              {clusterKeys.map((k, i) => (
                <Area key={k} type="monotone" dataKey={k} stroke={clusterColors[i % clusterColors.length]} strokeWidth={2} fill={`url(#grad${i})`} dot={false} activeDot={{ r: 4 }} />
              ))}
            </AreaChart>
          </ResponsiveContainer>
        </div>

        <div className="col-span-4 bg-white rounded-2xl p-5 shadow-sm border border-gray-100">
          <h3 className="text-sm font-semibold text-gray-800 mb-1">Emotion Distribution</h3>
          <p className="text-xs text-gray-400 mb-3">All time · selected cluster</p>
          <ResponsiveContainer width="100%" height={140}>
            <PieChart>
              <Pie data={emotionPieData} cx="50%" cy="50%" innerRadius={40} outerRadius={65} paddingAngle={3} dataKey="value">
                {emotionPieData.map((e, i) => <Cell key={i} fill={e.color} />)}
              </Pie>
              <Tooltip contentStyle={{ borderRadius: 10, border: "none", boxShadow: "0 4px 12px rgba(0,0,0,.08)", fontSize: 12 }} formatter={(v) => [`${v} calls`, ""]} />
            </PieChart>
          </ResponsiveContainer>
          <div className="space-y-1.5 mt-1">
            {emotionPieData.map((e) => (
              <div key={e.name} className="flex items-center gap-2 text-xs">
                <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ backgroundColor: e.color }} />
                <span className="text-gray-600 flex-1 capitalize">{e.name}</span>
                <span className="font-semibold text-gray-800">{e.value}</span>
                <span className="text-gray-400">{((e.value / totalEmotions) * 100).toFixed(0)}%</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ── Row 2: Emotion Trend + Risk Trend ── */}
      <div className="grid grid-cols-2 gap-6">
        <div className="bg-white rounded-2xl p-5 shadow-sm border border-gray-100">
          <h3 className="text-sm font-semibold text-gray-800 mb-1">Emotion Trend Over Time</h3>
          <p className="text-xs text-gray-400 mb-4">Monthly emotion detection · all agents</p>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={emotionTrend}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" />
              <XAxis dataKey="month" tick={{ fontSize: 10, fill: "#9ca3af" }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fontSize: 10, fill: "#9ca3af" }} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={{ borderRadius: 10, border: "none", boxShadow: "0 4px 12px rgba(0,0,0,.08)", fontSize: 12 }} />
              <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 11 }} />
              {emotionKeys.map((k) => (
                <Line key={k} type="monotone" dataKey={k} stroke={emotionColors[k] ?? "#94a3b8"} strokeWidth={2} dot={false} activeDot={{ r: 4 }} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-white rounded-2xl p-5 shadow-sm border border-gray-100">
          <h3 className="text-sm font-semibold text-gray-800 mb-1">Risk Level Trend</h3>
          <p className="text-xs text-gray-400 mb-4">Monthly risk distribution · all clusters</p>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={riskTrend} barSize={14}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" />
              <XAxis dataKey="month" tick={{ fontSize: 10, fill: "#9ca3af" }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fontSize: 10, fill: "#9ca3af" }} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={{ borderRadius: 10, border: "none", boxShadow: "0 4px 12px rgba(0,0,0,.08)", fontSize: 12 }} />
              <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 11 }} />
              {riskKeys.map((k, i) => (
                <Bar key={k} dataKey={k} stackId="a" fill={riskColors[k] ?? clusterColors[i]} radius={i === riskKeys.length - 1 ? [4, 4, 0, 0] : [0, 0, 0, 0]} />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* ── Row 3: Heatmap ── */}
      <div className="bg-white rounded-2xl p-5 shadow-sm border border-gray-100">
        <h3 className="text-sm font-semibold text-gray-800 mb-1">Call Volume Heatmap</h3>
        <p className="text-xs text-gray-400 mb-4">Angry / high-risk calls by hour and day</p>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr>
                <th className="text-left pr-3 py-1 text-gray-400 font-medium w-10" />
                {DAYS.map((d) => <th key={d} className="text-center py-1 text-gray-400 font-medium px-1">{d}</th>)}
              </tr>
            </thead>
            <tbody>
              {heatmapData.map((row) => (
                <tr key={row.hour}>
                  <td className="pr-3 py-1 text-gray-400 font-medium text-right whitespace-nowrap">{row.hour}</td>
                  {DAYS.map((d) => {
                    const val = row[d as keyof typeof row] as number;
                    const intensity = val / maxHeat;
                    return (
                      <td key={d} className="px-1 py-1">
                        <div className="w-full h-7 rounded-md flex items-center justify-center text-xs font-semibold"
                          style={{ backgroundColor: `rgba(239,68,68,${intensity * 0.85 + 0.05})`, color: intensity > 0.5 ? "white" : "#ef4444" }}
                          title={`${val} calls`}>
                          {val}
                        </div>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
          <div className="flex items-center gap-2 mt-3 justify-end">
            <span className="text-xs text-gray-400">Low</span>
            {[0.1, 0.3, 0.5, 0.7, 0.9].map((o) => <div key={o} className="w-5 h-3 rounded" style={{ backgroundColor: `rgba(239,68,68,${o})` }} />)}
            <span className="text-xs text-gray-400">High</span>
          </div>
        </div>
      </div>

      {/* ── Row 4: Agent Risk Scores ── */}
      <div className="bg-white rounded-2xl shadow-sm border border-gray-100 overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-100">
          <h3 className="text-sm font-semibold text-gray-800">Agent Emotional Risk Scores</h3>
          <p className="text-xs text-gray-400 mt-0.5">Ranked by risk score · higher = more at-risk</p>
        </div>
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-gray-50 border-b border-gray-100">
              {["Rank", "Agent", "Cluster", "Total Calls", "Risk Level", "Risk Score"].map((h) => (
                <th key={h} className="text-left px-5 py-3 text-xs font-semibold text-gray-500 uppercase tracking-wide">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-50">
            {agentRisk.length === 0 ? (
              <tr><td colSpan={6} className="text-center py-12 text-gray-400 text-sm">No data available</td></tr>
            ) : (
              agentRisk.map((a, i) => (
                <tr key={a.agent_id} className="hover:bg-gray-50 transition-colors">
                  <td className="px-5 py-3.5">
                    <span className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${i < 3 ? "bg-red-100 text-red-600" : "bg-gray-100 text-gray-500"}`}>{i + 1}</span>
                  </td>
                  <td className="px-5 py-3.5">
                    <div className="flex items-center gap-2">
                      <div className="w-7 h-7 rounded-full bg-red-100 text-red-600 text-xs font-bold flex items-center justify-center flex-shrink-0">{a.agent_name.charAt(0)}</div>
                      <span className="font-medium text-gray-800">{a.agent_name}</span>
                    </div>
                  </td>
                  <td className="px-5 py-3.5 text-gray-500">{a.cluster ?? "—"}</td>
                  <td className="px-5 py-3.5 font-semibold text-gray-800">{a.total_calls}</td>
                  <td className="px-5 py-3.5 text-gray-600 capitalize">{a.risk_level}</td>
                  <td className="px-5 py-3.5">
                    <div className="flex items-center gap-3">
                      <div className="flex-1 max-w-24 h-2 bg-gray-100 rounded-full">
                        <div className={`h-2 rounded-full ${riskScoreBar(a.risk_score)}`} style={{ width: `${a.risk_score}%` }} />
                      </div>
                      <span className={`text-sm font-bold w-8 ${riskScoreColor(a.risk_score)}`}>{a.risk_score}</span>
                    </div>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

    </div>
  );
}
