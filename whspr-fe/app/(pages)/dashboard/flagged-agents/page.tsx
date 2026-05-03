"use client";

import { useEffect, useState } from "react";
import Modal from "@/components/ui/Modal";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface FlaggedAgent {
  id: number;
  name: string;
  email?: string;
  role?: string;
  risk_level: string;
  is_active: boolean;
  cluster_id: number;
  cluster_name: string | null;
  calls_today: number;
  total_calls: number;
  flag_comment: string | null;
  flagged_at: string | null;
}

export default function FlaggedAgentsPage() {
  const [agents, setAgents] = useState<FlaggedAgent[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");

  const [viewTarget, setViewTarget] = useState<FlaggedAgent | null>(null);
  const [clearTarget, setClearTarget] = useState<FlaggedAgent | null>(null);
  const [clearing, setClearing] = useState(false);

  const fetchFlagged = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API}/agents/flagged`);
      const data = await res.json();
      setAgents(Array.isArray(data) ? data : []);
    } catch {
      setAgents([]);
    }
    setLoading(false);
  };

  useEffect(() => {
    fetchFlagged();
  }, []);

  const handleClearFlag = async () => {
    if (!clearTarget) return;
    setClearing(true);
    try {
      await fetch(`${API}/agents/${clearTarget.id}/reset-risk`, {
        method: "PATCH",
      });
      setClearTarget(null);
      fetchFlagged();
    } catch {
      /* ignore */
    }
    setClearing(false);
  };

  const filtered = agents.filter((a) =>
    a.name.toLowerCase().includes(search.toLowerCase()),
  );

  const formatDate = (iso: string | null) => {
    if (!iso) return "—";
    return new Date(iso).toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
    });
  };

  return (
    <div>
      {viewTarget && (
        <Modal onClose={() => setViewTarget(null)} maxWidth="sm">
          <Modal.Header
            title="Flagged Agent Details"
            onClose={() => setViewTarget(null)}
          />
          <Modal.Body>
            <div className="flex items-center gap-4 p-4 bg-red-50 rounded-xl">
              <div className="w-14 h-14 rounded-full bg-red-100 text-red-600 text-xl font-bold flex items-center justify-center">
                {viewTarget.name.charAt(0)}
              </div>
              <div>
                <p className="text-base font-semibold text-gray-800">
                  {viewTarget.name}
                </p>
                <p className="text-xs text-gray-400">
                  {viewTarget.role ?? "CSR"} · {viewTarget.cluster_name ?? "—"}
                </p>
              </div>
              <span className="ml-auto inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold bg-red-100 text-red-700">
                <span className="w-1.5 h-1.5 rounded-full bg-red-500" />
                Flagged
              </span>
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

            <div className="bg-gray-50 rounded-xl p-4">
              <p className="text-xs text-gray-400 mb-1">Flagged On</p>
              <p className="text-sm font-semibold text-gray-800">
                {formatDate(viewTarget.flagged_at)}
              </p>
            </div>

            <div className="bg-red-50 border border-red-100 rounded-xl p-4">
              <p className="text-xs text-gray-400 mb-1">Flag Reason</p>
              <p className="text-sm text-gray-700">
                {viewTarget.flag_comment ?? "No comment provided."}
              </p>
            </div>
          </Modal.Body>
          <Modal.Footer>
            <button
              onClick={() => setViewTarget(null)}
              className="flex-1 py-2.5 text-sm font-medium text-gray-600 bg-gray-100 hover:bg-gray-200 rounded-xl"
            >
              Close
            </button>
            <button
              onClick={() => {
                setViewTarget(null);
                setClearTarget(viewTarget);
              }}
              className="flex-1 py-2.5 text-sm font-medium text-white bg-green-500 hover:bg-green-600 rounded-xl"
            >
              Clear Flag
            </button>
          </Modal.Footer>
        </Modal>
      )}

      {clearTarget && (
        <Modal onClose={() => setClearTarget(null)} maxWidth="sm">
          <Modal.Header
            title="Clear Flag"
            onClose={() => setClearTarget(null)}
          />
          <Modal.Body>
            <p className="text-sm text-gray-500 text-center">
              Remove the flag from{" "}
              <span className="font-medium text-gray-700">
                {clearTarget.name}
              </span>
              ? Their risk level will be reset to Safe.
            </p>
          </Modal.Body>
          <Modal.Footer>
            <button
              onClick={() => setClearTarget(null)}
              disabled={clearing}
              className="flex-1 py-2.5 text-sm font-medium text-gray-600 bg-gray-100 hover:bg-gray-200 rounded-xl disabled:opacity-50"
            >
              Cancel
            </button>
            <button
              onClick={handleClearFlag}
              disabled={clearing}
              className="flex-1 py-2.5 text-sm font-medium text-white bg-green-500 hover:bg-green-600 rounded-xl disabled:opacity-50"
            >
              {clearing ? "Clearing..." : "Confirm"}
            </button>
          </Modal.Footer>
        </Modal>
      )}

      <div className="flex flex-wrap items-center gap-3 mb-6">
        <div className="relative flex-1 min-w-48">
          <input
            type="text"
            placeholder="Search flagged agents..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full px-4 py-2.5 text-sm bg-white border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-red-400"
          />
        </div>
        <span className="text-sm text-gray-400">{filtered.length} flagged</span>
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
                "Flagged On",
                "Flag Reason",
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
                  No flagged agents
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
                        <p className="font-medium text-gray-800">{agent.name}</p>
                        <p className="text-xs text-gray-400">
                          {agent.email ?? "—"}
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

                  <td className="px-5 py-4 text-gray-500 text-xs">
                    {formatDate(agent.flagged_at)}
                  </td>

                  <td className="px-5 py-4 max-w-xs">
                    <p className="text-xs text-gray-500 truncate">
                      {agent.flag_comment ?? "—"}
                    </p>
                  </td>

                  <td className="px-5 py-4">
                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => setViewTarget(agent)}
                        className="inline-flex items-center px-2.5 py-1.5 text-xs font-medium text-indigo-600 bg-indigo-50 hover:bg-indigo-100 border border-indigo-200 rounded-lg"
                      >
                        View
                      </button>
                      <button
                        onClick={() => setClearTarget(agent)}
                        className="inline-flex items-center px-2.5 py-1.5 text-xs font-medium text-green-600 bg-green-50 hover:bg-green-100 border border-green-200 rounded-lg"
                      >
                        Clear Flag
                      </button>
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
