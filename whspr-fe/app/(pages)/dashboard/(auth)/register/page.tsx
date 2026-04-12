"use client";
import { useState, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type RiskLevel = "Risky" | "Medium" | "Safe";

interface AgentRow {
  id: number;
  name: string;
  email: string;
  role: string;
  risk_level: RiskLevel;
  cluster_name: string | null;
}
interface RoleOption {
  value: string;
  label: string;
}
// ── Animated waveform ─────────────────────────────────────────────────────────

// ── Password strength ─────────────────────────────────────────────────────────
function PasswordStrength({ password }: { password: string }) {
  if (!password) return null;
  let score = 0;
  if (password.length >= 8) score++;
  if (password.length >= 12) score++;
  if (/[A-Z]/.test(password)) score++;
  if (/[0-9]/.test(password)) score++;
  if (/[^A-Za-z0-9]/.test(password)) score++;
  const levels = [
    { label: "Weak", color: "bg-red-400", text: "text-red-500", bars: 1 },
    { label: "Weak", color: "bg-red-400", text: "text-red-500", bars: 1 },
    { label: "Fair", color: "bg-yellow-400", text: "text-yellow-600", bars: 2 },
    { label: "Good", color: "bg-blue-400", text: "text-blue-500", bars: 3 },
    { label: "Strong", color: "bg-green-500", text: "text-green-600", bars: 4 },
    { label: "Strong", color: "bg-green-500", text: "text-green-600", bars: 4 },
  ];
  const { label, color, text, bars } = levels[score];
  return (
    <div className="mt-2">
      <div className="flex gap-1 mb-1">
        {[1, 2, 3, 4].map((b) => (
          <div
            key={b}
            className={`flex-1 h-1 rounded-full transition-all duration-300 ${b <= bars ? color : "bg-gray-100"}`}
          />
        ))}
      </div>
      <p className={`text-xs font-medium ${text}`}>{label} password</p>
    </div>
  );
}

const padId = (id: number) => String(id).padStart(4, "0");

const riskConfig: Record<RiskLevel, { badge: string; dot: string }> = {
  Risky: { badge: "bg-red-100 text-red-700", dot: "bg-red-500" },
  Medium: { badge: "bg-yellow-100 text-yellow-700", dot: "bg-yellow-500" },
  Safe: { badge: "bg-green-100 text-green-700", dot: "bg-green-500" },
};

// ── Page ──────────────────────────────────────────────────────────────────────
export default function RegisterPage() {
  const router = useRouter();
  const [mounted, setMounted] = useState(false);

  const [agents, setAgents] = useState<AgentRow[]>([]);
  const [agentsLoading, setAgentsLoading] = useState(true);
  const [agentSearch, setAgentSearch] = useState("");
  const [selectedAgent, setSelectedAgent] = useState<AgentRow | null>(null);
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [showPw, setShowPw] = useState(false);
  const [showCpw, setShowCpw] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [roles, setRoles] = useState<RoleOption[]>([]);
  const [rolesLoading, setRolesLoading] = useState(true);
  const [selectedRole, setSelectedRole] = useState("");

  useEffect(() => {
    const t = setTimeout(() => setMounted(true), 50);
    return () => clearTimeout(t);
  }, []);

  useEffect(() => {
    fetch(`${API}/roles`)
      .then((r) => r.json())
      .then((data) => {
        console.log("ROLES API:", data); // 👈 DEBUG THIS

        // ✅ Handle different response formats
        if (Array.isArray(data)) {
          setRoles(data);
          if (data.length > 0) setSelectedRole(data[0].value);
        } else if (Array.isArray(data.roles)) {
          setRoles(data.roles);
          if (data.roles.length > 0) setSelectedRole(data.roles[0].value);
        } else {
          console.error("Invalid roles format:", data);
          setRoles([]);
        }

        setRolesLoading(false);
      })
      .catch((err) => {
        console.error("Roles fetch error:", err);
        setRoles([]);
        setRolesLoading(false);
      });
  }, []);

  useEffect(() => {
    fetch(`${API}/agents`)
      .then((r) => r.json())
      .then((data: AgentRow[]) => {
        setAgents(data);
        setAgentsLoading(false);
      })
      .catch(() => setAgentsLoading(false));
  }, []);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(e.target as Node)
      )
        setDropdownOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  const filteredAgents = agents.filter(
    (a) =>
      a.name.toLowerCase().includes(agentSearch.toLowerCase()) ||
      padId(a.id).includes(agentSearch),
  );

  const handleSelectAgent = (agent: AgentRow) => {
    setSelectedAgent(agent);
    setDropdownOpen(false);
    setAgentSearch("");
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");

    if (!selectedAgent) {
      setError("Please select an agent.");
      return;
    }

    if (!selectedRole) {
      setError("Please select a role.");
      return;
    }

    if (!password) {
      setError("Password is required.");
      return;
    }

    if (password.length < 8) {
      setError("Password must be at least 8 characters.");
      return;
    }

    if (password !== confirmPassword) {
      setError("Passwords do not match.");
      return;
    }

    setLoading(true);

    try {
      const res = await fetch(`${API}/auth/register`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          agent_id: selectedAgent.id,
          password,
          role: selectedRole,
        }),
      });

      const d = await res.json();

      if (!res.ok) {
        let message = "Registration failed.";

        if (typeof d?.detail === "string") {
          message = d.detail;
        } else if (Array.isArray(d?.detail)) {
          message = d.detail.map((item: any) => item.msg).join(", ");
        } else if (d?.detail && typeof d.detail === "object") {
          message = d.detail.msg ?? JSON.stringify(d.detail);
        }

        setError(message);
        setLoading(false);
        return;
      }

      router.push("/dashboard/register");
    } catch {
      setError("Network error. Please try again.");
      setLoading(false);
    }
  };
  const pwMatch = confirmPassword.length > 0 && password === confirmPassword;
  const pwMismatch = confirmPassword.length > 0 && password !== confirmPassword;

  const EyeIcon = ({ off }: { off?: boolean }) =>
    off ? (
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
          d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21"
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
    );

  return (
    <div className="w-full h-full">
      {/* ── LEFT PANEL ── */}

      {/* ── RIGHT PANEL ── */}
      <div className="w-full max-w-md mx-auto px-6 py-8">
        {" "}
        <div className="absolute top-0 right-0 w-80 h-80 bg-red-50 rounded-full blur-3xl opacity-60 -translate-y-1/2 translate-x-1/2 pointer-events-none" />
        <div
          className="w-full max-w-sm relative z-10 transition-all duration-700"
          style={{
            opacity: mounted ? 1 : 0,
            transform: mounted ? "translateY(0)" : "translateY(18px)",
          }}
        >
          {/* Mobile logo */}
          <div className="flex lg:hidden items-center gap-2 mb-10 justify-center">
            <div className="w-8 h-8 rounded-xl bg-red-500 flex items-center justify-center">
              <svg
                className="w-4 h-4 text-white"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"
                />
              </svg>
            </div>
            <span
              className="text-gray-900 text-lg font-bold"
              style={{ fontFamily: "'DM Serif Display', Georgia, serif" }}
            >
              Affecta
            </span>
          </div>

          <div className="mb-8">
            <h2
              className="text-gray-900 text-3xl font-bold mb-2"
              style={{ fontFamily: "'DM Serif Display', Georgia, serif" }}
            >
              Create account
            </h2>
            <p className="text-gray-400 text-sm">
              Select an agent and set their password
            </p>
          </div>

          <form onSubmit={handleSubmit} className="flex flex-col gap-4">
            {/* Agent selector */}
            <div>
              <label className="block text-xs font-semibold text-gray-600 mb-1.5 tracking-wide">
                Agent
              </label>
              <div className="relative" ref={dropdownRef}>
                <button
                  type="button"
                  onClick={() => setDropdownOpen((v) => !v)}
                  className="w-full px-3 py-3 text-sm border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-red-400 focus:border-transparent text-left flex items-center justify-between transition-shadow"
                >
                  {selectedAgent ? (
                    <div className="flex items-center gap-3 min-w-0 flex-1">
                      <div className="w-7 h-7 rounded-full bg-red-100 text-red-600 text-xs font-bold flex items-center justify-center flex-shrink-0">
                        {selectedAgent.name.charAt(0)}
                      </div>
                      <span className="font-medium text-gray-800 truncate">
                        {selectedAgent.name}
                      </span>
                      <span className="ml-auto font-mono text-xs text-gray-400 flex-shrink-0">
                        #{padId(selectedAgent.id)}
                      </span>
                    </div>
                  ) : (
                    <span className="text-gray-300">Select an agent...</span>
                  )}
                  <svg
                    className={`w-4 h-4 text-gray-400 flex-shrink-0 ml-2 transition-transform ${dropdownOpen ? "rotate-180" : ""}`}
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M19 9l-7 7-7-7"
                    />
                  </svg>
                </button>

                {dropdownOpen && (
                  <div className="absolute z-50 mt-1 w-full bg-white border border-gray-200 rounded-xl shadow-lg overflow-hidden">
                    <div className="p-2 border-b border-gray-100">
                      <div className="relative">
                        <svg
                          className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-gray-400"
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
                          value={agentSearch}
                          onChange={(e) => setAgentSearch(e.target.value)}
                          placeholder="Search by name or ID (e.g. 0001)..."
                          autoFocus
                          className="w-full pl-8 pr-3 py-2 text-xs border border-gray-200 rounded-lg focus:outline-none focus:ring-1 focus:ring-red-400"
                        />
                      </div>
                    </div>
                    <ul className="max-h-52 overflow-y-auto">
                      {agentsLoading ? (
                        <li className="px-4 py-3 text-xs text-gray-400 text-center">
                          Loading agents...
                        </li>
                      ) : filteredAgents.length === 0 ? (
                        <li className="px-4 py-3 text-xs text-gray-400 text-center">
                          No agents found
                        </li>
                      ) : (
                        filteredAgents.map((agent) => (
                          <li key={agent.id}>
                            <button
                              type="button"
                              onClick={() => handleSelectAgent(agent)}
                              className={`w-full px-3 py-2.5 flex items-center gap-3 hover:bg-gray-50 transition-colors text-left ${selectedAgent?.id === agent.id ? "bg-red-50" : ""}`}
                            >
                              <div className="w-7 h-7 rounded-full bg-red-100 text-red-600 text-xs font-bold flex items-center justify-center flex-shrink-0">
                                {agent.name.charAt(0)}
                              </div>
                              <div className="flex-1 min-w-0">
                                <p className="text-sm font-medium text-gray-800 truncate">
                                  {agent.name}
                                </p>
                                <p className="text-xs text-gray-400 truncate">
                                  {agent.role}
                                  {agent.cluster_name
                                    ? ` · ${agent.cluster_name}`
                                    : ""}
                                </p>
                              </div>
                              <div className="flex flex-col items-end gap-1 flex-shrink-0">
                                <span className="font-mono text-xs font-semibold text-gray-500">
                                  #{padId(agent.id)}
                                </span>
                                <span
                                  className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded-full text-[10px] font-semibold ${riskConfig[agent.risk_level].badge}`}
                                >
                                  <span
                                    className={`w-1 h-1 rounded-full ${riskConfig[agent.risk_level].dot}`}
                                  />
                                  {agent.risk_level}
                                </span>
                              </div>
                            </button>
                          </li>
                        ))
                      )}
                    </ul>
                  </div>
                )}
              </div>

              {/* Selected agent email strip */}
              {selectedAgent && (
                <div className="mt-2 px-3 py-2.5 bg-gray-50 rounded-xl flex items-center gap-3">
                  <svg
                    className="w-3.5 h-3.5 text-gray-400 flex-shrink-0"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M16 12a4 4 0 10-8 0 4 4 0 008 0zm0 0v1.5a2.5 2.5 0 005 0V12a9 9 0 10-9 9m4.5-1.206a8.959 8.959 0 01-4.5 1.207"
                    />
                  </svg>
                  <span className="text-xs text-gray-500 truncate flex-1">
                    {selectedAgent.email}
                  </span>
                  <button
                    type="button"
                    onClick={() => setSelectedAgent(null)}
                    className="text-gray-300 hover:text-gray-500 transition-colors flex-shrink-0"
                  >
                    <svg
                      className="w-3.5 h-3.5"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M6 18L18 6M6 6l12 12"
                      />
                    </svg>
                  </button>
                </div>
              )}
            </div>
            <div>
              <label className="block text-xs font-semibold text-gray-600 mb-1.5 tracking-wide">
                Role
              </label>

              <select
                value={selectedRole}
                onChange={(e) => setSelectedRole(e.target.value)}
                disabled={rolesLoading}
                className="w-full px-3 py-3 text-sm border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-red-400 focus:border-transparent text-gray-800 bg-white"
              >
                <option value="">
                  {rolesLoading ? "Loading roles..." : "Select a role..."}
                </option>

                {roles.map((role) => (
                  <option key={role.value} value={role.value}>
                    {role.label}
                  </option>
                ))}
              </select>
            </div>
            {/* Password */}
            <div>
              <label className="block text-xs font-semibold text-gray-600 mb-1.5 tracking-wide">
                Password
              </label>
              <div className="relative">
                <svg
                  className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"
                  />
                </svg>
                <input
                  type={showPw ? "text" : "password"}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="Min. 8 characters"
                  autoComplete="new-password"
                  className="w-full pl-10 pr-11 py-3 text-sm border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-red-400 focus:border-transparent placeholder-gray-300 text-gray-800 transition-shadow"
                />
                <button
                  type="button"
                  onClick={() => setShowPw((v) => !v)}
                  className="absolute right-3.5 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 transition-colors"
                >
                  <EyeIcon off={showPw} />
                </button>
              </div>
              <PasswordStrength password={password} />
            </div>

            {/* Confirm Password */}
            <div>
              <label className="block text-xs font-semibold text-gray-600 mb-1.5 tracking-wide">
                Confirm Password
              </label>
              <div className="relative">
                <svg
                  className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"
                  />
                </svg>
                <input
                  type={showCpw ? "text" : "password"}
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  placeholder="Re-enter your password"
                  autoComplete="new-password"
                  className={`w-full pl-10 pr-11 py-3 text-sm border rounded-xl focus:outline-none focus:ring-2 focus:border-transparent placeholder-gray-300 text-gray-800 transition-shadow ${pwMismatch ? "border-red-300 focus:ring-red-400" : pwMatch ? "border-green-300 focus:ring-green-400" : "border-gray-200 focus:ring-red-400"}`}
                />
                <button
                  type="button"
                  onClick={() => setShowCpw((v) => !v)}
                  className="absolute right-3.5 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 transition-colors"
                >
                  <EyeIcon off={showCpw} />
                </button>
                {pwMatch && (
                  <div className="absolute right-10 top-1/2 -translate-y-1/2">
                    <svg
                      className="w-4 h-4 text-green-500"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2.5}
                        d="M5 13l4 4L19 7"
                      />
                    </svg>
                  </div>
                )}
              </div>
              {pwMismatch && (
                <p className="text-xs text-red-500 mt-1.5">
                  Passwords do not match.
                </p>
              )}
            </div>

            {/* Error */}
            {error && (
              <div className="flex items-center gap-2 px-3.5 py-3 bg-red-50 border border-red-100 rounded-xl text-xs text-red-600 font-medium">
                <svg
                  className="w-3.5 h-3.5 flex-shrink-0"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                  />
                </svg>
                {error}
              </div>
            )}

            {/* Submit */}
            <button
              type="submit"
              disabled={loading}
              className={`w-full py-3 rounded-xl text-sm font-semibold text-white transition-all duration-200 mt-1 shadow-sm ${loading ? "bg-red-300 cursor-not-allowed" : "bg-red-500 hover:bg-red-600 hover:shadow-md hover:shadow-red-200 active:scale-[0.98]"}`}
            >
              {loading ? (
                <span className="flex items-center justify-center gap-2">
                  <svg
                    className="w-4 h-4 animate-spin"
                    fill="none"
                    viewBox="0 0 24 24"
                  >
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                    />
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8v8H4z"
                    />
                  </svg>
                  Creating account...
                </span>
              ) : (
                "Create Account"
              )}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
