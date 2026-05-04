"use client";

import { useState, useEffect } from "react";
import Modal from "@/components/ui/Modal";
import ActionButtons from "@/components/ui/ActionButtons";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// ── Types ─────────────────────────────────────────────────────────────────────

type RiskLevel = "Risky" | "Medium" | "Safe";

interface ClusterRow {
  id: number;
  name: string;
  overall_risk: RiskLevel;
  agent_count: number;
  risky_agents: number;
  medium_agents: number;
  safe_agents: number;
  calls_today: number;
  total_calls: number;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

const riskConfig: Record<
  RiskLevel,
  { badge: string; dot: string; bar: string }
> = {
  Risky: {
    badge: "bg-red-100 text-red-700",
    dot: "bg-red-500",
    bar: "bg-red-500",
  },
  Medium: {
    badge: "bg-yellow-100 text-yellow-700",
    dot: "bg-yellow-500",
    bar: "bg-yellow-500",
  },
  Safe: {
    badge: "bg-green-100 text-green-700",
    dot: "bg-green-500",
    bar: "bg-green-500",
  },
};

const getRisk = (level: string | undefined | null): RiskLevel =>
  (level as RiskLevel) in riskConfig ? (level as RiskLevel) : "Safe";

// ── Page ──────────────────────────────────────────────────────────────────────

export default function ClustersPage() {
  const [clusters, setClusters] = useState<ClusterRow[]>([]);
  const [loading, setLoading] = useState(true);

  const [riskFilter, setRiskFilter] = useState("All");
  const [search, setSearch] = useState("");

  const [viewTarget, setViewTarget] = useState<ClusterRow | null>(null);
  const [editTarget, setEditTarget] = useState<ClusterRow | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<ClusterRow | null>(null);
  const [showAdd, setShowAdd] = useState(false);

  const [formName, setFormName] = useState("");
  const [formError, setFormError] = useState("");
  const [deleteError, setDeleteError] = useState("");
  const [deletePassword, setDeletePassword] = useState("");
  const fetchClusters = async () => {
    setLoading(true);
    try {
      const data: ClusterRow[] = await fetch(`${API}/clusters`).then((r) =>
        r.json(),
      );
      setClusters(data);
    } catch {
      setClusters([]);
    }
    setLoading(false);
  };

  useEffect(() => {
    fetchClusters();
  }, []);

  const openEdit = (c: ClusterRow) => {
    setEditTarget(c);
    setFormName(c.name);
    setFormError("");
  };

  const openAdd = () => {
    setFormName("");
    setFormError("");
    setShowAdd(true);
  };

  const saveEdit = async () => {
    if (!editTarget) return;
    setFormError("");
    try {
      const res = await fetch(`${API}/clusters/${editTarget.id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: formName,
        }),
      });
      if (!res.ok) {
        const e = await res.json();
        setFormError(
          typeof e.detail === "string"
            ? e.detail
            : Array.isArray(e.detail)
              ? e.detail.map((x: any) => x.msg).join(", ")
              : "Failed",
        );
        return;
      }
      setEditTarget(null);
      fetchClusters();
    } catch {
      setFormError("Network error");
    }
  };

  const saveAdd = async () => {
    setFormError("");
    try {
      const res = await fetch(`${API}/clusters`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: formName,
        }),
      });
      if (!res.ok) {
        const e = await res.json();
        setFormError(
          typeof e.detail === "string"
            ? e.detail
            : Array.isArray(e.detail)
              ? e.detail.map((x: any) => x.msg).join(", ")
              : "Failed",
        );
        return;
      }
      setShowAdd(false);
      fetchClusters();
    } catch {
      setFormError("Network error");
    }
  };

  const handleDelete = async () => {
    if (!deleteTarget) return;

    setDeleteError("");

    try {
      const stored = localStorage.getItem("user");
      const currentUser = stored ? JSON.parse(stored) : null;

      if (!currentUser?.id) {
        setDeleteError("You must be logged in to delete.");
        return;
      }

      const res = await fetch(`${API}/clusters/${deleteTarget.id}`, {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: currentUser.id,
          password: deletePassword,
        }),
      });

      const data = await res.json().catch(() => null);

      if (!res.ok) {
        setDeleteError(data?.detail || "Failed to delete cluster.");
        return;
      }

      setDeletePassword("");
      setDeleteTarget(null);
      fetchClusters();
    } catch {
      setDeleteError("Network error. Check your connection and try again.");
    }
  };

  const filtered = clusters.filter((c) => {
    const matchRisk = riskFilter === "All" || c.overall_risk === riskFilter;
    const matchSearch = c.name.toLowerCase().includes(search.toLowerCase());
    return matchRisk && matchSearch;
  });

  const ClusterForm = (
    <Modal.Body>
      <div>
        <label className="block text-xs font-medium text-gray-600 mb-1.5">
          Cluster Name
        </label>
        <input
          type="text"
          value={formName}
          onChange={(e) => setFormName(e.target.value)}
          placeholder="e.g. West Coast"
          className="w-full px-3 py-2.5 text-sm border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-red-400 focus:border-transparent"
        />
      </div>
      {formError && <p className="text-xs text-red-500">{formError}</p>}
    </Modal.Body>
  );

  return (
    <div>
      {/* ── VIEW MODAL ── */}
      {viewTarget && (
        <Modal onClose={() => setViewTarget(null)} maxWidth="sm">
          <Modal.Header
            title="Cluster Details"
            onClose={() => setViewTarget(null)}
          />
          <Modal.Body>
            <div className="flex items-center gap-4 p-4 bg-gray-50 rounded-xl">
              <div className="w-12 h-12 rounded-xl bg-red-100 text-red-600 text-lg font-bold flex items-center justify-center flex-shrink-0">
                {viewTarget.name.charAt(0)}
              </div>
              <div>
                <p className="text-base font-semibold text-gray-800">
                  {viewTarget.name}
                </p>
              </div>
              <span
                className={`ml-auto inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold ${riskConfig[getRisk(viewTarget.overall_risk)].badge}`}
              >
                <span
                  className={`w-1.5 h-1.5 rounded-full ${riskConfig[getRisk(viewTarget.overall_risk)].dot}`}
                />
                {viewTarget.overall_risk}
              </span>
            </div>
            <div className="grid grid-cols-3 gap-3">
              {[
                ["Agents", viewTarget.agent_count],
                ["Calls Today", viewTarget.calls_today],
                ["Total Calls", viewTarget.total_calls],
              ].map(([label, val]) => (
                <div
                  key={String(label)}
                  className="bg-gray-50 rounded-xl p-3 text-center"
                >
                  <p className="text-xs text-gray-400 mb-1">{label}</p>
                  <p className="text-2xl font-bold text-gray-800">{val}</p>
                </div>
              ))}
            </div>
            <div className="bg-gray-50 rounded-xl p-4">
              <p className="text-xs font-medium text-gray-500 mb-3">
                Agent Risk Breakdown
              </p>
              <div className="space-y-2">
                {(
                  [
                    ["Risky", viewTarget.risky_agents],
                    ["Medium", viewTarget.medium_agents],
                    ["Safe", viewTarget.safe_agents],
                  ] as [RiskLevel, number][]
                ).map(([level, count]) => (
                  <div key={level} className="flex items-center gap-3">
                    <span
                      className={`w-16 text-xs font-medium ${riskConfig[level].badge.split(" ")[1]}`}
                    >
                      {level}
                    </span>
                    <div className="flex-1 h-2 bg-gray-200 rounded-full overflow-hidden">
                      <div
                        className={`h-2 rounded-full ${riskConfig[level].bar}`}
                        style={{
                          width:
                            viewTarget.agent_count > 0
                              ? `${(count / viewTarget.agent_count) * 100}%`
                              : "0%",
                        }}
                      />
                    </div>
                    <span className="text-xs font-semibold text-gray-700 w-4 text-right">
                      {count}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </Modal.Body>
          <Modal.Footer>
            <button
              onClick={() => setViewTarget(null)}
              className="flex-1 py-2.5 text-sm font-medium text-gray-600 bg-gray-100 hover:bg-gray-200 rounded-xl transition-colors"
            >
              Close
            </button>
          </Modal.Footer>
        </Modal>
      )}

      {/* ── EDIT MODAL ── */}
      {editTarget && (
        <Modal onClose={() => setEditTarget(null)} maxWidth="sm">
          <Modal.Header
            title="Edit Cluster"
            description={editTarget.name}
            onClose={() => setEditTarget(null)}
          />
          {ClusterForm}
          <Modal.Footer>
            <button
              onClick={() => setEditTarget(null)}
              className="flex-1 py-2.5 text-sm font-medium text-gray-600 bg-gray-100 hover:bg-gray-200 rounded-xl transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={saveEdit}
              className="flex-1 py-2.5 text-sm font-medium text-white bg-red-500 hover:bg-red-600 rounded-xl transition-colors"
            >
              Save
            </button>
          </Modal.Footer>
        </Modal>
      )}

      {/* ── ADD MODAL ── */}
      {showAdd && (
        <Modal onClose={() => setShowAdd(false)} maxWidth="sm">
          <Modal.Header
            title="Add Cluster"
            description="Create a new cluster group"
            onClose={() => setShowAdd(false)}
          />
          {ClusterForm}
          <Modal.Footer>
            <button
              onClick={() => setShowAdd(false)}
              className="flex-1 py-2.5 text-sm font-medium text-gray-600 bg-gray-100 hover:bg-gray-200 rounded-xl transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={saveAdd}
              disabled={!formName}
              className={`flex-1 py-2.5 text-sm font-medium text-white rounded-xl transition-colors ${formName ? "bg-red-500 hover:bg-red-600" : "bg-gray-300 cursor-not-allowed"}`}
            >
              Add Cluster
            </button>
          </Modal.Footer>
        </Modal>
      )}

      {/* ── DELETE MODAL ── */}
      {deleteTarget && (
        <Modal
          onClose={() => {
            setDeleteTarget(null);
            setDeleteError("");
            setDeletePassword("");
          }}
          maxWidth="sm"
        >
          <Modal.Header
            title="Delete Cluster"
            onClose={() => {
              setDeleteTarget(null);
              setDeleteError("");
            }}
          />
          <Modal.Body>
            <div className="mt-3">
              <label className="block text-xs font-medium text-gray-600 mb-1.5">
                Confirm your password
              </label>
              <input
                type="password"
                value={deletePassword}
                onChange={(e) => setDeletePassword(e.target.value)}
                placeholder="Enter your password"
                className="w-full px-3 py-2.5 text-sm border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-red-400"
              />
            </div>
            <div className="w-12 h-12 bg-red-100 rounded-full flex items-center justify-center mx-auto">
              <svg
                className="w-6 h-6 text-red-500"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                />
              </svg>
            </div>
            {deleteError && (
              <div className="flex items-start gap-2.5 px-3.5 py-3 bg-red-50 border border-red-200 rounded-xl">
                <svg
                  className="w-4 h-4 text-red-500 flex-shrink-0 mt-0.5"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M12 9v2m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"
                  />
                </svg>
                <p className="text-xs text-red-700">{deleteError}</p>
              </div>
            )}
            <p className="text-sm text-gray-500 text-center">
              Are you sure you want to delete{" "}
              <span className="font-medium text-gray-700">
                {deleteTarget.name}
              </span>
              ?
              <br />
              <span className="text-red-500 font-medium">
                This will delete ALL calls under this cluster and may affect
                assigned agents.
              </span>
            </p>
          </Modal.Body>
          <Modal.Footer>
            <button
              onClick={() => {
                setDeleteTarget(null);
                setDeleteError("");
              }}
              className="flex-1 py-2.5 text-sm font-medium text-gray-600 bg-gray-100 hover:bg-gray-200 rounded-xl transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={handleDelete}
              disabled={!deletePassword}
              className={`flex-1 py-2.5 text-sm font-medium text-white rounded-xl transition-colors ${
                deletePassword
                  ? "bg-red-500 hover:bg-red-600"
                  : "bg-gray-300 cursor-not-allowed"
              }`}
            >
              Delete
            </button>
          </Modal.Footer>
        </Modal>
      )}

      {/* ── TOOLBAR ── */}
      <div className="flex flex-wrap items-center gap-3 mb-6">
        <div className="relative flex-1 min-w-48">
          <svg
            className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
            />
          </svg>
          <input
            type="text"
            placeholder="Search clusters..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full pl-9 pr-4 py-2.5 text-sm bg-white border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-red-400 focus:border-transparent"
          />
        </div>
        <select
          value={riskFilter}
          onChange={(e) => setRiskFilter(e.target.value)}
          className="px-4 py-2.5 text-sm bg-white border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-red-400"
        >
          {["All", "Risky", "Medium", "Safe"].map((r) => (
            <option key={r}>{r}</option>
          ))}
        </select>
        <span className="text-sm text-gray-400">
          {filtered.length} clusters
        </span>
        <button
          onClick={openAdd}
          className="flex items-center gap-2 px-4 py-2.5 bg-red-500 hover:bg-red-600 text-white text-sm font-semibold rounded-xl transition-colors shadow-sm"
        >
          <svg
            className="w-4 h-4"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M12 4v16m8-8H4"
            />
          </svg>
          Add Cluster
        </button>
      </div>

      {/* ── TABLE ── */}
      <div className="bg-white rounded-2xl shadow-sm border border-gray-100 overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-100 bg-gray-50">
              {[
                "Cluster",
                "Agents",
                "Calls Today",
                "Total Calls",
                "Agent Risk",
                "Overall Risk",
                "Actions",
              ].map((h) => (
                <th
                  key={h}
                  className="text-left px-5 py-3.5 text-xs font-semibold text-gray-500 uppercase tracking-wide"
                >
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-50">
            {loading ? (
              <tr>
                <td
                  colSpan={8}
                  className="text-center py-12 text-gray-400 text-sm"
                >
                  Loading...
                </td>
              </tr>
            ) : filtered.length === 0 ? (
              <tr>
                <td
                  colSpan={8}
                  className="text-center py-12 text-gray-400 text-sm"
                >
                  No clusters found
                </td>
              </tr>
            ) : (
              filtered.map((cluster) => (
                <tr
                  key={cluster.id}
                  className="hover:bg-gray-50 transition-colors"
                >
                  <td className="px-5 py-4">
                    <div className="flex items-center gap-3">
                      <div className="w-8 h-8 rounded-lg bg-red-100 text-red-600 text-sm font-bold flex items-center justify-center flex-shrink-0">
                        {cluster.name.charAt(0)}
                      </div>
                      <span className="font-medium text-gray-800">
                        {cluster.name}
                      </span>
                    </div>
                  </td>
                  <td className="px-5 py-4 font-semibold text-gray-800">
                    {cluster.agent_count}
                  </td>
                  <td className="px-5 py-4 font-semibold text-gray-800">
                    {cluster.calls_today}
                  </td>
                  <td className="px-5 py-4">
                    <div className="flex items-center gap-2">
                      <span className="font-semibold text-gray-800">
                        {cluster.total_calls}
                      </span>
                      <div className="flex-1 max-w-16 h-1.5 bg-gray-100 rounded-full">
                        <div
                          className="h-1.5 bg-red-300 rounded-full"
                          style={{
                            width: `${Math.min((cluster.total_calls / 700) * 100, 100)}%`,
                          }}
                        />
                      </div>
                    </div>
                  </td>
                  <td className="px-5 py-4">
                    <div className="flex items-center gap-1.5">
                      {cluster.risky_agents > 0 && (
                        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-700">
                          <span className="w-1.5 h-1.5 rounded-full bg-red-500" />
                          {cluster.risky_agents}
                        </span>
                      )}
                      {cluster.medium_agents > 0 && (
                        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-yellow-100 text-yellow-700">
                          <span className="w-1.5 h-1.5 rounded-full bg-yellow-500" />
                          {cluster.medium_agents}
                        </span>
                      )}
                      {cluster.safe_agents > 0 && (
                        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-700">
                          <span className="w-1.5 h-1.5 rounded-full bg-green-500" />
                          {cluster.safe_agents}
                        </span>
                      )}
                    </div>
                  </td>
                  <td className="px-5 py-4">
                    <span
                      className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold ${riskConfig[getRisk(cluster.overall_risk)].badge}`}
                    >
                      <span
                        className={`w-1.5 h-1.5 rounded-full ${riskConfig[getRisk(cluster.overall_risk)].dot}`}
                      />
                      {cluster.overall_risk}
                    </span>
                  </td>
                  <td className="px-5 py-4">
                    <ActionButtons
                      onView={() => setViewTarget(cluster)}
                      onEdit={() => openEdit(cluster)}
                      onDelete={() => {
                        setDeleteError("");
                        setDeleteTarget(cluster);
                      }}
                    />
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
