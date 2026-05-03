"use client";

import { useEffect, useState } from "react";
import Modal from "@/components/ui/Modal";
import ActionButtons from "@/components/ui/ActionButtons";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type RiskLevel = "Risky" | "Medium" | "Safe";

interface AgentRow {
  id: number;
  name: string;
  email?: string;
  role?: string;
  risk_level: RiskLevel;
  is_active: boolean;
  cluster_id: number;
  cluster_name: string | null;
  calls_today: number;
  total_calls: number;
}

interface ClusterOption {
  id: number;
  name: string;
}

const riskConfig: Record<RiskLevel, { badge: string; dot: string }> = {
  Risky: { badge: "bg-red-100 text-red-700", dot: "bg-red-500" },
  Medium: { badge: "bg-yellow-100 text-yellow-700", dot: "bg-yellow-500" },
  Safe: { badge: "bg-green-100 text-green-700", dot: "bg-green-500" },
};

const getRisk = (level: string | undefined): RiskLevel =>
  (level as RiskLevel) in riskConfig ? (level as RiskLevel) : "Safe";

export default function AgentsPage() {
  const [agents, setAgents] = useState<AgentRow[]>([]);
  const [clusters, setClusters] = useState<ClusterOption[]>([]);
  const [loading, setLoading] = useState(true);
  const [userRole, setUserRole] = useState<string>("");

  useEffect(() => {
    try {
      const raw = localStorage.getItem("user");
      if (raw) setUserRole(JSON.parse(raw)?.role?.toLowerCase() ?? "");
    } catch {
      /* ignore */
    }
  }, []);

  const [clusterFilter, setClusterFilter] = useState("all");
  const [riskFilter, setRiskFilter] = useState("All");
  const [search, setSearch] = useState("");

  const [viewTarget, setViewTarget] = useState<AgentRow | null>(null);
  const [editTarget, setEditTarget] = useState<AgentRow | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<AgentRow | null>(null);
  const [showAdd, setShowAdd] = useState(false);

  const [flagTarget, setFlagTarget] = useState<AgentRow | null>(null);
  const [flagComment, setFlagComment] = useState("");
  const [flagError, setFlagError] = useState("");
  const [flagging, setFlagging] = useState(false);

  const [formName, setFormName] = useState("");
  const [formClusterId, setFormClusterId] = useState("");
  const [formRisk, setFormRisk] = useState<RiskLevel>("Safe");
  const [formError, setFormError] = useState("");

  const fetchAgents = async () => {
    setLoading(true);

    try {
      const params =
        clusterFilter !== "all" ? `?cluster_id=${clusterFilter}` : "";

      const res = await fetch(`${API}/agents${params}`);
      const data = await res.json();

      setAgents(Array.isArray(data) ? data : []);
    } catch {
      setAgents([]);
    }

    setLoading(false);
  };

  useEffect(() => {
    fetch(`${API}/clusters`)
      .then((r) => r.json())
      .then((data) => {
        const rows = Array.isArray(data) ? data : [];
        setClusters(rows);

        if (rows.length > 0) {
          setFormClusterId(String(rows[0].id));
        }
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    fetchAgents();
  }, [clusterFilter]);

  const openAdd = () => {
    setFormName("");
    setFormClusterId(clusters[0] ? String(clusters[0].id) : "");
    setFormRisk("Safe");
    setFormError("");
    setShowAdd(true);
  };

  const openEdit = (agent: AgentRow) => {
    setEditTarget(agent);
    setFormName(agent.name);
    setFormClusterId(String(agent.cluster_id));
    setFormRisk(agent.risk_level);
    setFormError("");
  };

  const openFlag = (agent: AgentRow) => {
    setFlagTarget(agent);
    setFlagComment("");
    setFlagError("");
  };

  const saveAdd = async () => {
    setFormError("");

    try {
      const res = await fetch(`${API}/agents`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: formName,
          cluster_id: Number(formClusterId),
        }),
      });

      if (!res.ok) {
        const e = await res.json();
        setFormError(e.detail ?? "Failed to add agent");
        return;
      }

      setShowAdd(false);
      fetchAgents();
    } catch {
      setFormError("Network error");
    }
  };

  const saveEdit = async () => {
    if (!editTarget) return;

    setFormError("");

    try {
      const res = await fetch(`${API}/agents/${editTarget.id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: formName,
          cluster_id: Number(formClusterId),
          risk_level: formRisk,
        }),
      });

      if (!res.ok) {
        const e = await res.json();
        setFormError(e.detail ?? "Failed to update agent");
        return;
      }

      setEditTarget(null);
      fetchAgents();
    } catch {
      setFormError("Network error");
    }
  };

  const handleDelete = async () => {
    if (!deleteTarget) return;

    await fetch(`${API}/agents/${deleteTarget.id}`, {
      method: "DELETE",
    });

    setDeleteTarget(null);
    fetchAgents();
  };

  const handleFlag = async () => {
    if (!flagTarget) return;

    setFlagError("");

    if (!flagComment.trim()) {
      setFlagError("Comment is required before flagging.");
      return;
    }

    setFlagging(true);

    try {
      const res = await fetch(`${API}/agents/${flagTarget.id}/flag`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          comment: flagComment.trim(),
        }),
      });

      if (!res.ok) {
        const e = await res.json();
        setFlagError(e.detail ?? "Failed to flag agent");
        setFlagging(false);
        return;
      }

      setFlagTarget(null);
      setFlagComment("");
      fetchAgents();
    } catch {
      setFlagError("Network error");
    }

    setFlagging(false);
  };

  const handleResetRisk = async (agent: AgentRow) => {
    await fetch(`${API}/agents/${agent.id}/reset-risk`, {
      method: "PATCH",
    });

    fetchAgents();
  };

  const filtered = agents.filter((a) => {
    const matchRisk = riskFilter === "All" || a.risk_level === riskFilter;
    const matchSearch = a.name.toLowerCase().includes(search.toLowerCase());
    return matchRisk && matchSearch;
  });

  const ClusterAndNameFields = (
    <>
      <div>
        <label className="block text-xs font-medium text-gray-600 mb-1.5">
          Full Name
        </label>
        <input
          type="text"
          value={formName}
          onChange={(e) => setFormName(e.target.value)}
          placeholder="e.g. Maria Santos"
          className="w-full px-3 py-2.5 text-sm border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-red-400"
        />
      </div>

      <div>
        <label className="block text-xs font-medium text-gray-600 mb-1.5">
          Cluster
        </label>
        <select
          value={formClusterId}
          onChange={(e) => setFormClusterId(e.target.value)}
          className="w-full px-3 py-2.5 text-sm border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-red-400"
        >
          {clusters.map((c) => (
            <option key={c.id} value={c.id}>
              {c.name}
            </option>
          ))}
        </select>
      </div>
    </>
  );

  const AddAgentForm = (
    <Modal.Body>
      {ClusterAndNameFields}
      {formError && <p className="text-xs text-red-500">{formError}</p>}
    </Modal.Body>
  );

  const EditAgentForm = (
    <Modal.Body>
      {ClusterAndNameFields}

      <div>
        <label className="block text-xs font-medium text-gray-600 mb-1.5">
          Risk Level
        </label>
        <div className="flex gap-2">
          {(["Safe", "Medium", "Risky"] as RiskLevel[]).map((r) => (
            <button
              key={r}
              type="button"
              onClick={() => setFormRisk(r)}
              className={`flex-1 py-2 text-sm font-medium rounded-xl border transition-colors ${
                formRisk === r
                  ? `${riskConfig[r].badge} border-transparent`
                  : "text-gray-500 border-gray-200 hover:bg-gray-50"
              }`}
            >
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
      {viewTarget && (
        <Modal onClose={() => setViewTarget(null)} maxWidth="sm">
          <Modal.Header
            title="Agent Details"
            onClose={() => setViewTarget(null)}
          />

          <Modal.Body>
            <div className="flex items-center gap-4 p-4 bg-gray-50 rounded-xl">
              <div className="w-14 h-14 rounded-full bg-red-100 text-red-600 text-xl font-bold flex items-center justify-center">
                {viewTarget.name.charAt(0)}
              </div>

              <div>
                <p className="text-base font-semibold text-gray-800">
                  {viewTarget.name}
                </p>
                <p className="text-xs text-gray-400">
                  Role: {viewTarget.role ?? "agent"}
                </p>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div className="bg-gray-50 rounded-xl p-4">
                <p className="text-xs text-gray-400 mb-1">Cluster</p>
                <p className="text-sm font-semibold text-gray-800">
                  {viewTarget.cluster_name ?? "—"}
                </p>
              </div>

              <div className="bg-gray-50 rounded-xl p-4">
                <p className="text-xs text-gray-400 mb-2">Risk Level</p>
                <span
                  className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold ${
                    riskConfig[getRisk(viewTarget.risk_level)].badge
                  }`}
                >
                  <span
                    className={`w-1.5 h-1.5 rounded-full ${
                      riskConfig[getRisk(viewTarget.risk_level)].dot
                    }`}
                  />
                  {viewTarget.risk_level}
                </span>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div className="bg-gray-50 rounded-xl p-4 text-center">
                <p className="text-xs text-gray-400 mb-1">Calls Today</p>
                <p className="text-3xl font-bold text-gray-800">
                  {viewTarget.calls_today}
                </p>
              </div>

              <div className="bg-gray-50 rounded-xl p-4 text-center">
                <p className="text-xs text-gray-400 mb-1">Total Calls</p>
                <p className="text-3xl font-bold text-gray-800">
                  {viewTarget.total_calls}
                </p>
              </div>
            </div>
          </Modal.Body>

          <Modal.Footer>
            <button
              onClick={() => setViewTarget(null)}
              className="flex-1 py-2.5 text-sm font-medium text-gray-600 bg-gray-100 hover:bg-gray-200 rounded-xl"
            >
              Close
            </button>
          </Modal.Footer>
        </Modal>
      )}

      {editTarget && (
        <Modal onClose={() => setEditTarget(null)} maxWidth="sm">
          <Modal.Header
            title="Edit Agent"
            description={editTarget.name}
            onClose={() => setEditTarget(null)}
          />

          {EditAgentForm}

          <Modal.Footer>
            <button
              onClick={() => setEditTarget(null)}
              className="flex-1 py-2.5 text-sm font-medium text-gray-600 bg-gray-100 hover:bg-gray-200 rounded-xl"
            >
              Cancel
            </button>

            <button
              onClick={saveEdit}
              className="flex-1 py-2.5 text-sm font-medium text-white bg-red-500 hover:bg-red-600 rounded-xl"
            >
              Save
            </button>
          </Modal.Footer>
        </Modal>
      )}

      {showAdd && (
        <Modal onClose={() => setShowAdd(false)} maxWidth="sm">
          <Modal.Header
            title="Add Agent"
            description="Create a new CSR agent"
            onClose={() => setShowAdd(false)}
          />

          {AddAgentForm}

          <Modal.Footer>
            <button
              onClick={() => setShowAdd(false)}
              className="flex-1 py-2.5 text-sm font-medium text-gray-600 bg-gray-100 hover:bg-gray-200 rounded-xl"
            >
              Cancel
            </button>

            <button
              onClick={saveAdd}
              disabled={!formName.trim()}
              className={`flex-1 py-2.5 text-sm font-medium text-white rounded-xl ${
                formName.trim()
                  ? "bg-red-500 hover:bg-red-600"
                  : "bg-gray-300 cursor-not-allowed"
              }`}
            >
              Add Agent
            </button>
          </Modal.Footer>
        </Modal>
      )}

      {deleteTarget && (
        <Modal onClose={() => setDeleteTarget(null)} maxWidth="sm">
          <Modal.Header
            title="Delete Agent"
            onClose={() => setDeleteTarget(null)}
          />

          <Modal.Body>
            <p className="text-sm text-gray-500 text-center">
              Are you sure you want to remove{" "}
              <span className="font-medium text-gray-700">
                {deleteTarget.name}
              </span>
              ? This cannot be undone.
            </p>
          </Modal.Body>

          <Modal.Footer>
            <button
              onClick={() => setDeleteTarget(null)}
              className="flex-1 py-2.5 text-sm font-medium text-gray-600 bg-gray-100 hover:bg-gray-200 rounded-xl"
            >
              Cancel
            </button>

            <button
              onClick={handleDelete}
              className="flex-1 py-2.5 text-sm font-medium text-white bg-red-500 hover:bg-red-600 rounded-xl"
            >
              Delete
            </button>
          </Modal.Footer>
        </Modal>
      )}

      {flagTarget && (
        <Modal onClose={() => setFlagTarget(null)} maxWidth="sm">
          <Modal.Header
            title="Flag Agent"
            description={flagTarget.name}
            onClose={() => setFlagTarget(null)}
          />

          <Modal.Body>
            <div className="p-4 bg-red-50 border border-red-100 rounded-xl">
              <p className="text-sm text-red-700 font-medium">
                You are about to flag this agent for admin review.
              </p>
              <p className="text-xs text-red-600 mt-1">
                A comment is required before this action.
              </p>
            </div>

            <textarea
              value={flagComment}
              onChange={(e) => setFlagComment(e.target.value)}
              rows={4}
              placeholder="Enter reason for flagging..."
              className="w-full px-3 py-2.5 text-sm border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-red-400 resize-none"
            />

            {flagError && <p className="text-xs text-red-500">{flagError}</p>}
          </Modal.Body>

          <Modal.Footer>
            <button
              onClick={() => setFlagTarget(null)}
              disabled={flagging}
              className="flex-1 py-2.5 text-sm font-medium text-gray-600 bg-gray-100 hover:bg-gray-200 rounded-xl disabled:opacity-50"
            >
              Cancel
            </button>

            <button
              onClick={handleFlag}
              disabled={flagging}
              className="flex-1 py-2.5 text-sm font-medium text-white bg-red-500 hover:bg-red-600 rounded-xl disabled:opacity-50"
            >
              {flagging ? "Flagging..." : "Confirm Flag"}
            </button>
          </Modal.Footer>
        </Modal>
      )}

      <div className="flex flex-wrap items-center gap-3 mb-6">
        <div className="relative flex-1 min-w-48">
          <input
            type="text"
            placeholder="Search agents..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full px-4 py-2.5 text-sm bg-white border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-red-400"
          />
        </div>

        <select
          value={clusterFilter}
          onChange={(e) => setClusterFilter(e.target.value)}
          className="px-4 py-2.5 text-sm bg-white border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-red-400"
        >
          <option value="all">All Clusters</option>
          {clusters.map((c) => (
            <option key={c.id} value={c.id}>
              {c.name}
            </option>
          ))}
        </select>

        <select
          value={riskFilter}
          onChange={(e) => setRiskFilter(e.target.value)}
          className="px-4 py-2.5 text-sm bg-white border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-red-400"
        >
          {["All", "Risky", "Medium", "Safe"].map((r) => (
            <option key={r}>{r}</option>
          ))}
        </select>

        <span className="text-sm text-gray-400">{filtered.length} agents</span>

        {userRole !== "admin" && (
          <button
            onClick={openAdd}
            className="flex items-center gap-2 px-4 py-2.5 bg-red-500 hover:bg-red-600 text-white text-sm font-semibold rounded-xl"
          >
            Add Agent
          </button>
        )}
      </div>

      <div className="bg-white rounded-2xl shadow-sm border border-gray-100 overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-100 bg-gray-50">
              {[
                "Agent",
                "Cluster",
                "Calls Today",
                "Total Calls",
                "Role",
                "Risk",
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
                <td colSpan={7} className="text-center py-12 text-gray-400">
                  Loading...
                </td>
              </tr>
            ) : filtered.length === 0 ? (
              <tr>
                <td colSpan={7} className="text-center py-12 text-gray-400">
                  No agents found
                </td>
              </tr>
            ) : (
              filtered.map((agent) => (
                <tr key={agent.id} className="hover:bg-gray-50">
                  <td className="px-5 py-4">
                    <div className="flex items-center gap-3">
                      <div className="w-8 h-8 rounded-full bg-red-100 text-red-600 text-sm font-bold flex items-center justify-center">
                        {agent.name.charAt(0)}
                      </div>

                      <div>
                        <p className="font-medium text-gray-800">
                          {agent.name}
                        </p>
                      </div>
                    </div>
                  </td>

                  <td className="px-5 py-4 text-gray-500">
                    {agent.cluster_name ?? "—"}
                  </td>

                  <td className="px-5 py-4 font-semibold text-gray-800">
                    {agent.calls_today}
                  </td>

                  <td className="px-5 py-4 font-semibold text-gray-800">
                    {agent.total_calls}
                  </td>

                  <td className="px-5 py-4">
                    <span className="inline-flex px-2.5 py-1 rounded-full text-xs font-semibold bg-gray-100 text-gray-700">
                      {agent.role ?? "agent"}
                    </span>
                  </td>

                  <td className="px-5 py-4">
                    <span
                      className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold ${
                        riskConfig[getRisk(agent.risk_level)].badge
                      }`}
                    >
                      <span
                        className={`w-1.5 h-1.5 rounded-full ${
                          riskConfig[getRisk(agent.risk_level)].dot
                        }`}
                      />
                      {agent.risk_level}
                    </span>
                  </td>

                  <td className="px-5 py-4">
                    <div className="flex items-center gap-2 flex-wrap">
                      {userRole !== "admin" && agent.risk_level !== "Risky" && (
                        <button
                          onClick={() => openFlag(agent)}
                          className="inline-flex items-center px-2.5 py-1.5 text-xs font-medium text-red-600 bg-red-50 hover:bg-red-100 border border-red-200 rounded-lg"
                        >
                          Flag Agent
                        </button>
                      )}

                      {userRole !== "admin" && agent.risk_level === "Risky" && (
                        <button
                          onClick={() => handleResetRisk(agent)}
                          className="inline-flex items-center px-2.5 py-1.5 text-xs font-medium text-green-600 bg-green-50 hover:bg-green-100 border border-green-200 rounded-lg"
                        >
                          Clear Flag
                        </button>
                      )}

                      <ActionButtons
                        onView={() => setViewTarget(agent)}
                        onEdit={() => openEdit(agent)}
                        onDelete={() => setDeleteTarget(agent)}
                        actions={userRole === "admin" ? ["view"] : ["view", "edit", "delete"]}
                      />
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
