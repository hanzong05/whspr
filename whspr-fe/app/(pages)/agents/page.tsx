"use client";

import { useState, useEffect } from "react";
import Modal from "@/components/ui/Modal";
import ActionButtons from "@/components/ui/ActionButtons";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// ── Types ─────────────────────────────────────────────────────────────────────

type RiskLevel = "Risky" | "Medium" | "Safe";

interface AgentRow {
  id: number;
  name: string;
  email: string;
  role: string;
  risk_level: RiskLevel;
  is_active: boolean;
  cluster_id: number;
  cluster_name: string | null;
  calls_today: number;
  total_calls: number;
}

interface ClusterOption { id: number; name: string; }

// ── Helpers ───────────────────────────────────────────────────────────────────

const riskConfig: Record<RiskLevel, { badge: string; dot: string }> = {
  Risky:  { badge: "bg-red-100 text-red-700",      dot: "bg-red-500" },
  Medium: { badge: "bg-yellow-100 text-yellow-700", dot: "bg-yellow-500" },
  Safe:   { badge: "bg-green-100 text-green-700",   dot: "bg-green-500" },
};

// ── Page ──────────────────────────────────────────────────────────────────────

export default function AgentsPage() {
  const [agents, setAgents] = useState<AgentRow[]>([]);
  const [clusters, setClusters] = useState<ClusterOption[]>([]);
  const [loading, setLoading] = useState(true);

  const [clusterFilter, setClusterFilter] = useState<string>("all");
  const [riskFilter, setRiskFilter] = useState("All");
  const [search, setSearch] = useState("");

  const [viewTarget, setViewTarget] = useState<AgentRow | null>(null);
  const [editTarget, setEditTarget] = useState<AgentRow | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<AgentRow | null>(null);
  const [showAdd, setShowAdd] = useState(false);

  const [formName, setFormName] = useState("");
  const [formEmail, setFormEmail] = useState("");
  const [formRole, setFormRole] = useState("");
  const [formClusterId, setFormClusterId] = useState<string>("");
  const [formRisk, setFormRisk] = useState<RiskLevel>("Safe");
  const [formError, setFormError] = useState("");

  const fetchAgents = async () => {
    setLoading(true);
    try {
      const params = clusterFilter !== "all" ? `?cluster_id=${clusterFilter}` : "";
      const data: AgentRow[] = await fetch(`${API}/agents${params}`).then((r) => r.json());
      setAgents(data);
    } catch { setAgents([]); }
    setLoading(false);
  };

  useEffect(() => {
    fetch(`${API}/clusters`).then((r) => r.json()).then((data: ClusterOption[]) => {
      setClusters(data);
      if (data.length > 0) setFormClusterId(String(data[0].id));
    }).catch(() => {});
  }, []);

  useEffect(() => { fetchAgents(); }, [clusterFilter]);

  const openEdit = (agent: AgentRow) => {
    setEditTarget(agent);
    setFormName(agent.name);
    setFormEmail(agent.email);
    setFormRole(agent.role);
    setFormClusterId(String(agent.cluster_id));
    setFormRisk(agent.risk_level);
    setFormError("");
  };

  const openAdd = () => {
    setFormName(""); setFormEmail(""); setFormRole("");
    setFormClusterId(clusters[0] ? String(clusters[0].id) : "");
    setFormRisk("Safe"); setFormError("");
    setShowAdd(true);
  };

  const saveEdit = async () => {
    if (!editTarget) return;
    setFormError("");
    try {
      const res = await fetch(`${API}/agents/${editTarget.id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: formName, email: formEmail, role: formRole, cluster_id: Number(formClusterId), risk_level: formRisk }),
      });
      if (!res.ok) { const e = await res.json(); setFormError(e.detail ?? "Failed"); return; }
      setEditTarget(null);
      fetchAgents();
    } catch { setFormError("Network error"); }
  };

  const saveAdd = async () => {
    setFormError("");
    try {
      const res = await fetch(`${API}/agents`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: formName, email: formEmail, role: formRole, cluster_id: Number(formClusterId), risk_level: formRisk }),
      });
      if (!res.ok) { const e = await res.json(); setFormError(e.detail ?? "Failed"); return; }
      setShowAdd(false);
      fetchAgents();
    } catch { setFormError("Network error"); }
  };

  const handleDelete = async () => {
    if (!deleteTarget) return;
    await fetch(`${API}/agents/${deleteTarget.id}`, { method: "DELETE" });
    setDeleteTarget(null);
    fetchAgents();
  };

  const filtered = agents.filter((a) => {
    const matchRisk   = riskFilter === "All" || a.risk_level === riskFilter;
    const matchSearch = a.name.toLowerCase().includes(search.toLowerCase()) || a.email.toLowerCase().includes(search.toLowerCase());
    return matchRisk && matchSearch;
  });

  const AgentForm = (
    <Modal.Body>
      <div>
        <label className="block text-xs font-medium text-gray-600 mb-1.5">Full Name</label>
        <input type="text" value={formName} onChange={(e) => setFormName(e.target.value)} placeholder="e.g. Maria Santos"
          className="w-full px-3 py-2.5 text-sm border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-red-400 focus:border-transparent" />
      </div>
      <div>
        <label className="block text-xs font-medium text-gray-600 mb-1.5">Email</label>
        <input type="email" value={formEmail} onChange={(e) => setFormEmail(e.target.value)} placeholder="name@whspr.com"
          className="w-full px-3 py-2.5 text-sm border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-red-400 focus:border-transparent" />
      </div>
      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="block text-xs font-medium text-gray-600 mb-1.5">Role</label>
          <input type="text" value={formRole} onChange={(e) => setFormRole(e.target.value)} placeholder="e.g. Senior CSR"
            className="w-full px-3 py-2.5 text-sm border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-red-400 focus:border-transparent" />
        </div>
        <div>
          <label className="block text-xs font-medium text-gray-600 mb-1.5">Cluster</label>
          <select value={formClusterId} onChange={(e) => setFormClusterId(e.target.value)}
            className="w-full px-3 py-2.5 text-sm border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-red-400 focus:border-transparent">
            {clusters.map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}
          </select>
        </div>
      </div>
      <div>
        <label className="block text-xs font-medium text-gray-600 mb-1.5">Risk Level</label>
        <div className="flex gap-2">
          {(["Safe", "Medium", "Risky"] as RiskLevel[]).map((r) => (
            <button key={r} onClick={() => setFormRisk(r)}
              className={`flex-1 py-2 text-sm font-medium rounded-xl border transition-colors ${formRisk === r ? riskConfig[r].badge + " border-transparent" : "text-gray-500 border-gray-200 hover:bg-gray-50"}`}>
              {r}
            </button>
          ))}
        </div>
      </div>
      {formError && <p className="text-xs text-red-500">{formError}</p>}
    </Modal.Body>
  );

  return (
    <div>

      {/* ── VIEW MODAL ── */}
      {viewTarget && (
        <Modal onClose={() => setViewTarget(null)} maxWidth="sm">
          <Modal.Header title="Agent Details" onClose={() => setViewTarget(null)} />
          <Modal.Body>
            <div className="flex items-center gap-4 p-4 bg-gray-50 rounded-xl">
              <div className="w-14 h-14 rounded-full bg-red-100 text-red-600 text-xl font-bold flex items-center justify-center flex-shrink-0">{viewTarget.name.charAt(0)}</div>
              <div>
                <p className="text-base font-semibold text-gray-800">{viewTarget.name}</p>
                <p className="text-sm text-gray-500">{viewTarget.role}</p>
                <p className="text-xs text-gray-400">{viewTarget.email}</p>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div className="bg-gray-50 rounded-xl p-4">
                <p className="text-xs text-gray-400 mb-1">Cluster</p>
                <p className="text-sm font-semibold text-gray-800">{viewTarget.cluster_name ?? "—"}</p>
              </div>
              <div className="bg-gray-50 rounded-xl p-4">
                <p className="text-xs text-gray-400 mb-2">Risk Level</p>
                <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold ${riskConfig[viewTarget.risk_level].badge}`}>
                  <span className={`w-1.5 h-1.5 rounded-full ${riskConfig[viewTarget.risk_level].dot}`} />
                  {viewTarget.risk_level}
                </span>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div className="bg-gray-50 rounded-xl p-4 text-center">
                <p className="text-xs text-gray-400 mb-1">Calls Today</p>
                <p className="text-3xl font-bold text-gray-800">{viewTarget.calls_today}</p>
              </div>
              <div className="bg-gray-50 rounded-xl p-4 text-center">
                <p className="text-xs text-gray-400 mb-1">Total Calls</p>
                <p className="text-3xl font-bold text-gray-800">{viewTarget.total_calls}</p>
              </div>
            </div>
          </Modal.Body>
          <Modal.Footer>
            <button onClick={() => setViewTarget(null)} className="flex-1 py-2.5 text-sm font-medium text-gray-600 bg-gray-100 hover:bg-gray-200 rounded-xl transition-colors">Close</button>
          </Modal.Footer>
        </Modal>
      )}

      {/* ── EDIT MODAL ── */}
      {editTarget && (
        <Modal onClose={() => setEditTarget(null)} maxWidth="sm">
          <Modal.Header title="Edit Agent" description={editTarget.name} onClose={() => setEditTarget(null)} />
          {AgentForm}
          <Modal.Footer>
            <button onClick={() => setEditTarget(null)} className="flex-1 py-2.5 text-sm font-medium text-gray-600 bg-gray-100 hover:bg-gray-200 rounded-xl transition-colors">Cancel</button>
            <button onClick={saveEdit} className="flex-1 py-2.5 text-sm font-medium text-white bg-red-500 hover:bg-red-600 rounded-xl transition-colors">Save</button>
          </Modal.Footer>
        </Modal>
      )}

      {/* ── ADD MODAL ── */}
      {showAdd && (
        <Modal onClose={() => setShowAdd(false)} maxWidth="sm">
          <Modal.Header title="Add Agent" description="Create a new CSR agent" onClose={() => setShowAdd(false)} />
          {AgentForm}
          <Modal.Footer>
            <button onClick={() => setShowAdd(false)} className="flex-1 py-2.5 text-sm font-medium text-gray-600 bg-gray-100 hover:bg-gray-200 rounded-xl transition-colors">Cancel</button>
            <button onClick={saveAdd} disabled={!formName || !formEmail} className={`flex-1 py-2.5 text-sm font-medium text-white rounded-xl transition-colors ${formName && formEmail ? "bg-red-500 hover:bg-red-600" : "bg-gray-300 cursor-not-allowed"}`}>Add Agent</button>
          </Modal.Footer>
        </Modal>
      )}

      {/* ── DELETE MODAL ── */}
      {deleteTarget && (
        <Modal onClose={() => setDeleteTarget(null)} maxWidth="sm">
          <Modal.Header title="Delete Agent" onClose={() => setDeleteTarget(null)} />
          <Modal.Body>
            <div className="w-12 h-12 bg-red-100 rounded-full flex items-center justify-center mx-auto">
              <svg className="w-6 h-6 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" /></svg>
            </div>
            <p className="text-sm text-gray-500 text-center">
              Are you sure you want to remove <span className="font-medium text-gray-700">{deleteTarget.name}</span>? This cannot be undone.
            </p>
          </Modal.Body>
          <Modal.Footer>
            <button onClick={() => setDeleteTarget(null)} className="flex-1 py-2.5 text-sm font-medium text-gray-600 bg-gray-100 hover:bg-gray-200 rounded-xl transition-colors">Cancel</button>
            <button onClick={handleDelete} className="flex-1 py-2.5 text-sm font-medium text-white bg-red-500 hover:bg-red-600 rounded-xl transition-colors">Delete</button>
          </Modal.Footer>
        </Modal>
      )}

      {/* ── TOOLBAR ── */}
      <div className="flex flex-wrap items-center gap-3 mb-6">
        <div className="relative flex-1 min-w-48">
          <svg className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" /></svg>
          <input type="text" placeholder="Search agents..." value={search} onChange={(e) => setSearch(e.target.value)}
            className="w-full pl-9 pr-4 py-2.5 text-sm bg-white border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-red-400 focus:border-transparent" />
        </div>
        <select value={clusterFilter} onChange={(e) => setClusterFilter(e.target.value)} className="px-4 py-2.5 text-sm bg-white border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-red-400">
          <option value="all">All Clusters</option>
          {clusters.map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}
        </select>
        <select value={riskFilter} onChange={(e) => setRiskFilter(e.target.value)} className="px-4 py-2.5 text-sm bg-white border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-red-400">
          {["All", "Risky", "Medium", "Safe"].map((r) => <option key={r}>{r}</option>)}
        </select>
        <span className="text-sm text-gray-400">{filtered.length} agents</span>
        <button onClick={openAdd} className="flex items-center gap-2 px-4 py-2.5 bg-red-500 hover:bg-red-600 text-white text-sm font-semibold rounded-xl transition-colors shadow-sm">
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" /></svg>
          Add Agent
        </button>
      </div>

      {/* ── TABLE ── */}
      <div className="bg-white rounded-2xl shadow-sm border border-gray-100 overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-100 bg-gray-50">
              {["Agent", "Cluster", "Role", "Calls Today", "Total Calls", "Risk", "Actions"].map((h) => (
                <th key={h} className="text-left px-5 py-3.5 text-xs font-semibold text-gray-500 uppercase tracking-wide">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-50">
            {loading ? (
              <tr><td colSpan={7} className="text-center py-12 text-gray-400 text-sm">Loading...</td></tr>
            ) : filtered.length === 0 ? (
              <tr><td colSpan={7} className="text-center py-12 text-gray-400 text-sm">No agents found</td></tr>
            ) : (
              filtered.map((agent) => (
                <tr key={agent.id} className="hover:bg-gray-50 transition-colors">
                  <td className="px-5 py-4">
                    <div className="flex items-center gap-3">
                      <div className="w-8 h-8 rounded-full bg-red-100 text-red-600 text-sm font-bold flex items-center justify-center flex-shrink-0">{agent.name.charAt(0)}</div>
                      <div>
                        <p className="font-medium text-gray-800">{agent.name}</p>
                        <p className="text-xs text-gray-400">{agent.email}</p>
                      </div>
                    </div>
                  </td>
                  <td className="px-5 py-4 text-gray-500">{agent.cluster_name ?? "—"}</td>
                  <td className="px-5 py-4 text-gray-500">{agent.role}</td>
                  <td className="px-5 py-4"><span className="font-semibold text-gray-800">{agent.calls_today}</span></td>
                  <td className="px-5 py-4">
                    <div className="flex items-center gap-2">
                      <span className="font-semibold text-gray-800">{agent.total_calls}</span>
                      <div className="flex-1 max-w-16 h-1.5 bg-gray-100 rounded-full">
                        <div className="h-1.5 bg-red-300 rounded-full" style={{ width: `${Math.min((agent.total_calls / 500) * 100, 100)}%` }} />
                      </div>
                    </div>
                  </td>
                  <td className="px-5 py-4">
                    <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold ${riskConfig[agent.risk_level].badge}`}>
                      <span className={`w-1.5 h-1.5 rounded-full ${riskConfig[agent.risk_level].dot}`} />
                      {agent.risk_level}
                    </span>
                  </td>
                  <td className="px-5 py-4">
                    <ActionButtons onView={() => setViewTarget(agent)} onEdit={() => openEdit(agent)} onDelete={() => setDeleteTarget(agent)} />
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
