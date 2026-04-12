"use client";

import { useState, useEffect } from "react";
import Modal from "@/components/ui/Modal";
import ActionButtons from "@/components/ui/ActionButtons";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// ── Types ─────────────────────────────────────────────────────────────────────

type UserRole = "admin" | "supervisor" | "agent";

interface UserRow {
  id: number;
  agent_id: number;
  username: string;
  is_active: boolean;
  role: UserRole;
  agent_name: string | null;
  agent_email: string | null;
  last_login_at: string | null;
  created_at: string | null;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

const roleConfig: Record<UserRole, { badge: string; dot: string }> = {
  admin: { badge: "bg-purple-100 text-purple-700", dot: "bg-purple-500" },
  supervisor: { badge: "bg-blue-100 text-blue-700", dot: "bg-blue-500" },
  agent: { badge: "bg-gray-100 text-gray-600", dot: "bg-gray-400" },
};

function formatDate(iso: string | null): string {
  if (!iso) return "—";
  return new Date(iso).toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

// ── Page ──────────────────────────────────────────────────────────────────────

export default function UsersPage() {
  const [users, setUsers] = useState<UserRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [roleFilter, setRoleFilter] = useState("All");

  const [viewTarget, setViewTarget] = useState<UserRow | null>(null);
  const [editTarget, setEditTarget] = useState<UserRow | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<UserRow | null>(null);

  // Edit form state
  const [formPassword, setFormPassword] = useState("");
  const [formRole, setFormRole] = useState<UserRole>("agent");
  const [formError, setFormError] = useState("");
  const [showPassword, setShowPassword] = useState(false);

  // ── Fetch ──────────────────────────────────────────────────────────────────

  const fetchUsers = async () => {
    setLoading(true);
    try {
      const data: UserRow[] = await fetch(`${API}/users`).then((r) => r.json());
      setUsers(data);
    } catch {
      setUsers([]);
    }
    setLoading(false);
  };

  useEffect(() => {
    fetchUsers();
  }, []);

  // ── Handlers ───────────────────────────────────────────────────────────────

  const openEdit = (user: UserRow) => {
    setEditTarget(user);
    setFormRole(user.role);
    setFormPassword("");
    setFormError("");
    setShowPassword(false);
  };

  const saveEdit = async () => {
    if (!editTarget) return;
    setFormError("");

    const body: Record<string, unknown> = {
      role: formRole,
    };
    if (formPassword) {
      if (formPassword.length < 8) {
        setFormError("Password must be at least 8 characters");
        return;
      }
      body.password = formPassword;
    }

    try {
      const res = await fetch(`${API}/users/${editTarget.id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const e = await res.json();
        setFormError(e.detail ?? "Failed to update user");
        return;
      }
      setEditTarget(null);
      fetchUsers();
    } catch {
      setFormError("Network error");
    }
  };

  const handleDelete = async () => {
    if (!deleteTarget) return;
    await fetch(`${API}/users/${deleteTarget.id}`, { method: "DELETE" });
    setDeleteTarget(null);
    fetchUsers();
  };

  // ── Filtered list ──────────────────────────────────────────────────────────

  const filtered = users.filter((u) => {
    const matchRole =
      roleFilter === "All" || u.role === roleFilter.toLowerCase();
    const matchSearch =
      u.username.toLowerCase().includes(search.toLowerCase()) ||
      (u.agent_name ?? "").toLowerCase().includes(search.toLowerCase());
    return matchRole && matchSearch;
  });

  // ── Render ─────────────────────────────────────────────────────────────────

  return (
    <div>
      {/* ── VIEW MODAL ── */}
      {viewTarget && (
        <Modal onClose={() => setViewTarget(null)} maxWidth="sm">
          <Modal.Header
            title="User Details"
            description={`@${viewTarget.username}`}
            onClose={() => setViewTarget(null)}
          />

          <Modal.Body>
            {/* Top user info */}
            <div className="flex items-center gap-4 p-4 bg-gray-50 rounded-xl">
              <div className="w-14 h-14 rounded-full bg-red-100 text-red-600 text-lg font-bold flex items-center justify-center flex-shrink-0">
                {viewTarget.username.slice(2, 4).toUpperCase()}
              </div>

              <div className="min-w-0">
                <p className="text-base font-semibold text-gray-800">
                  {viewTarget.agent_name ?? "No agent name"}
                </p>
                <p className="text-sm text-gray-500">@{viewTarget.username}</p>
                <p className="text-sm text-gray-400 break-all">
                  {viewTarget.agent_email ?? "No email"}
                </p>
              </div>
            </div>

            {/* Info cards */}
            <div className="grid grid-cols-2 gap-3 mt-4">
              <div className="bg-gray-50 rounded-xl p-4">
                <p className="text-xs text-gray-400 mb-2">Role</p>
                <span
                  className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold capitalize ${
                    roleConfig[viewTarget.role]?.badge ??
                    "bg-gray-100 text-gray-600"
                  }`}
                >
                  <span
                    className={`w-1.5 h-1.5 rounded-full ${
                      roleConfig[viewTarget.role]?.dot ?? "bg-gray-400"
                    }`}
                  />
                  {viewTarget.role}
                </span>
              </div>

              <div className="bg-gray-50 rounded-xl p-4">
                <p className="text-xs text-gray-400 mb-2">Status</p>
                <span
                  className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold ${
                    viewTarget.is_active
                      ? "bg-green-100 text-green-700"
                      : "bg-red-100 text-red-700"
                  }`}
                >
                  <span
                    className={`w-1.5 h-1.5 rounded-full ${
                      viewTarget.is_active ? "bg-green-500" : "bg-red-500"
                    }`}
                  />
                  {viewTarget.is_active ? "Active" : "Inactive"}
                </span>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3 mt-3">
              <div className="bg-gray-50 rounded-xl p-4">
                <p className="text-xs text-gray-400 mb-1">Last Login</p>
                <p className="text-sm font-semibold text-gray-800">
                  {formatDate(viewTarget.last_login_at)}
                </p>
              </div>

              <div className="bg-gray-50 rounded-xl p-4">
                <p className="text-xs text-gray-400 mb-1">Created</p>
                <p className="text-sm font-semibold text-gray-800">
                  {formatDate(viewTarget.created_at)}
                </p>
              </div>
            </div>

            <div className="bg-gray-50 rounded-xl p-4 mt-3">
              <p className="text-xs text-gray-400 mb-1">Agent ID</p>
              <p className="text-sm font-semibold text-gray-800">
                {viewTarget.agent_id ?? "—"}
              </p>
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
            title="Edit User"
            description={`@${editTarget.username} · ${editTarget.agent_name ?? "—"}`}
            onClose={() => setEditTarget(null)}
          />
          <Modal.Body>
            {/* Role */}
            <div>
              <label className="block text-xs font-medium text-gray-600 mb-1.5">
                Role
              </label>
              <div className="flex gap-2">
                {(["agent", "supervisor", "admin"] as UserRole[]).map((r) => (
                  <button
                    key={r}
                    onClick={() => setFormRole(r)}
                    className={`flex-1 py-2 text-sm font-medium rounded-xl border capitalize transition-colors ${
                      formRole === r
                        ? roleConfig[r].badge + " border-transparent"
                        : "text-gray-500 border-gray-200 hover:bg-gray-50"
                    }`}
                  >
                    {r}
                  </button>
                ))}
              </div>
            </div>

            {/* New password (optional) */}
            <div>
              <label className="block text-xs font-medium text-gray-600 mb-1.5">
                New Password{" "}
                <span className="text-gray-400 font-normal">
                  (leave blank to keep current)
                </span>
              </label>
              <div className="relative">
                <input
                  type={showPassword ? "text" : "password"}
                  value={formPassword}
                  onChange={(e) => setFormPassword(e.target.value)}
                  placeholder="Min. 8 characters"
                  className="w-full px-3 py-2.5 pr-10 text-sm border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-red-400 focus:border-transparent"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword((p) => !p)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
                >
                  {showPassword ? (
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
                        d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 4.411m0 0L21 21"
                      />
                    </svg>
                  ) : (
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
                        d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
                      />
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"
                      />
                    </svg>
                  )}
                </button>
              </div>
            </div>

            {formError && <p className="text-xs text-red-500">{formError}</p>}
          </Modal.Body>
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

      {/* ── DELETE MODAL ── */}
      {deleteTarget && (
        <Modal onClose={() => setDeleteTarget(null)} maxWidth="sm">
          <Modal.Header
            title="Delete User"
            onClose={() => setDeleteTarget(null)}
          />
          <Modal.Body>
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
            <p className="text-sm text-gray-500 text-center">
              Are you sure you want to delete{" "}
              <span className="font-medium text-gray-700">
                @{deleteTarget.username}
              </span>
              ? This cannot be undone.
            </p>
          </Modal.Body>
          <Modal.Footer>
            <button
              onClick={() => setDeleteTarget(null)}
              className="flex-1 py-2.5 text-sm font-medium text-gray-600 bg-gray-100 hover:bg-gray-200 rounded-xl transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={handleDelete}
              className="flex-1 py-2.5 text-sm font-medium text-white bg-red-500 hover:bg-red-600 rounded-xl transition-colors"
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
            placeholder="Search by username or agent..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full pl-9 pr-4 py-2.5 text-sm bg-white border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-red-400 focus:border-transparent"
          />
        </div>
        <select
          value={roleFilter}
          onChange={(e) => setRoleFilter(e.target.value)}
          className="px-4 py-2.5 text-sm bg-white border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-red-400"
        >
          {["All", "Admin", "Supervisor", "Agent"].map((r) => (
            <option key={r}>{r}</option>
          ))}
        </select>
        <span className="text-sm text-gray-400">{filtered.length} users</span>
      </div>

      {/* ── TABLE ── */}
      <div className="bg-white rounded-2xl shadow-sm border border-gray-100 overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-100 bg-gray-50">
              {["User", "Agent", "Role", "Last Login", "Actions"].map((h) => (
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
                  colSpan={5}
                  className="text-center py-12 text-gray-400 text-sm"
                >
                  Loading...
                </td>
              </tr>
            ) : filtered.length === 0 ? (
              <tr>
                <td
                  colSpan={5}
                  className="text-center py-12 text-gray-400 text-sm"
                >
                  No users found
                </td>
              </tr>
            ) : (
              filtered.map((user) => (
                <tr
                  key={user.id}
                  className="hover:bg-gray-50 transition-colors"
                >
                  {/* User */}
                  <td className="px-5 py-4">
                    <div className="flex items-center gap-3">
                      <div className="w-8 h-8 rounded-full bg-red-100 text-red-600 text-sm font-bold flex items-center justify-center flex-shrink-0">
                        {user.username.slice(0, 2).toUpperCase()}
                      </div>
                      <p className="font-medium text-gray-800">
                        @{user.username}
                      </p>
                    </div>
                  </td>

                  {/* Agent */}
                  <td className="px-5 py-4">
                    <div>
                      <p className="text-gray-800 font-medium">
                        {user.agent_name ?? "—"}
                      </p>
                      {user.agent_email && (
                        <p className="text-xs text-gray-400">
                          {user.agent_email}
                        </p>
                      )}
                    </div>
                  </td>

                  {/* Role */}
                  <td className="px-5 py-4">
                    <span
                      className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold capitalize ${
                        roleConfig[user.role]?.badge ??
                        "bg-gray-100 text-gray-600"
                      }`}
                    >
                      <span
                        className={`w-1.5 h-1.5 rounded-full ${
                          roleConfig[user.role]?.dot ?? "bg-gray-400"
                        }`}
                      />
                      {user.role}
                    </span>
                  </td>

                  {/* Last Login */}
                  <td className="px-5 py-4 text-gray-500 text-xs">
                    {formatDate(user.last_login_at)}
                  </td>

                  {/* Actions */}
                  <td className="px-5 py-4">
                    <ActionButtons
                      onView={() => setViewTarget(user)}
                      onEdit={() => openEdit(user)}
                      onDelete={() => setDeleteTarget(user)}
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
