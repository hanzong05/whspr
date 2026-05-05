"use client";

import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import Modal from "@/components/ui/Modal";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const pageTitles: Record<string, { title: string; description: string }> = {
  "/": { title: "Dashboard", description: "Overview of your CSR activity" },
  "/calls": {
    title: "Calls",
    description: "Manage and upload call recordings",
  },
  "/agents": { title: "Agents", description: "View and manage CSR agents" },
  "/clusters": { title: "Clusters", description: "Agent group clusters" },
  "/reports": { title: "Reports", description: "Emotional analysis reports" },
};

interface HeaderProps {
  // Optional — LandingPage passes this to get a handle on openLoginModal
  onExposeOpenLogin?: (fn: () => void) => void;
}

export default function Header({ onExposeOpenLogin }: HeaderProps) {
  const pathname = usePathname();
  const router = useRouter();

  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [user, setUser] = useState<Record<string, any> | null>(null);
  const [showLoginModal, setShowLoginModal] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPw, setShowPw] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    const token = localStorage.getItem("token");
    const stored = localStorage.getItem("user");

    if (token && stored) {
      try {
        const parsedUser = JSON.parse(stored);
        setUser(parsedUser);
        setIsLoggedIn(true);

        if (pathname === "/") {
          const role = parsedUser?.role?.toLowerCase();
          router.push(role === "agent" ? "/dashboard/record" : "/dashboard");
        }
        return;
      } catch {
        localStorage.removeItem("token");
        localStorage.removeItem("user");
      }
    }

    setIsLoggedIn(false);
    setUser(null);
  }, [pathname, router]);

  // Expose openLoginModal to parent (LandingPage) if requested
  useEffect(() => {
    if (onExposeOpenLogin) {
      onExposeOpenLogin(openLoginModal);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [onExposeOpenLogin]);

  const matchedKey = Object.keys(pageTitles)
    .filter((key) =>
      key === "/dashboard"
        ? pathname === "/dashboard"
        : pathname.startsWith(key),
    )
    .sort((a, b) => b.length - a.length)[0];

  const page = pageTitles[matchedKey] ?? {
    title: "Affecta",
    description: "CSR Call Analysis",
  };

  const openLoginModal = () => {
    const token = localStorage.getItem("token");
    const stored = localStorage.getItem("user");

    if (token && stored) {
      try {
        const parsedUser = JSON.parse(stored);
        const role = parsedUser?.role?.toLowerCase();

        setUser(parsedUser);
        setIsLoggedIn(true);

        router.push(role === "agent" ? "/dashboard/record" : "/dashboard");
        return;
      } catch {
        localStorage.removeItem("token");
        localStorage.removeItem("user");
      }
    }

    setEmail("");
    setPassword("");
    setError("");
    setShowPw(false);
    setShowLoginModal(true);
  };
  const handleLogin = async () => {
    setError("");
    if (!email || !password) {
      setError("Please fill in all fields.");
      return;
    }
    setLoading(true);
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 8000);
      const res = await fetch(`${API}/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username: email, password }),
        signal: controller.signal,
      });
      clearTimeout(timeout);
      if (!res.ok) {
        const d = await res.json().catch(() => ({ detail: "Server error" }));
        setError(d.detail ?? "Invalid email or password.");
        return;
      }
      const data = await res.json();
      localStorage.setItem("token", data.id.toString());
      localStorage.setItem("user", JSON.stringify(data));
      setUser(data);
      setIsLoggedIn(true);
      setShowLoginModal(false);
      const role = data.role?.toLowerCase();
      router.push(role === "agent" ? "/dashboard/record" : "/dashboard");
    } catch (err: any) {
      if (err.name === "AbortError") {
        setError("Request timed out. Is the backend running?");
      } else {
        setError(
          `Cannot reach server: ${API} — check your API URL or backend.`,
        );
      }
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("user");
    setIsLoggedIn(false);
    setUser(null);
    router.push("/");
  };

  const userInitial = user?.agent_name
    ? user.agent_name.charAt(0).toUpperCase()
    : "?";

  return (
    <>
      {/* LOGIN MODAL */}
      {showLoginModal && (
        <Modal onClose={() => setShowLoginModal(false)} maxWidth="sm">
          <Modal.Header
            title="Sign in to Affecta"
            description="Enter your credentials to continue"
            onClose={() => setShowLoginModal(false)}
          />
          <Modal.Body>
            <div>
              <label className="block text-xs font-semibold text-gray-600 mb-1.5 tracking-wide">
                Email or Agent ID
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
                    d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"
                  />
                </svg>
                <input
                  type="text"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="you@affecta.com or 0001"
                  autoComplete="username"
                  onKeyDown={(e) => e.key === "Enter" && handleLogin()}
                  className="w-full pl-10 pr-4 py-2.5 text-sm border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-red-400 focus:border-transparent placeholder-gray-300 text-gray-800"
                />
              </div>
            </div>

            <div>
              <div className="flex items-center justify-between mb-1.5">
                <label className="text-xs font-semibold text-gray-600 tracking-wide">
                  Password
                </label>

              </div>
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
                  placeholder="••••••••"
                  autoComplete="current-password"
                  onKeyDown={(e) => e.key === "Enter" && handleLogin()}
                  className="w-full pl-10 pr-10 py-2.5 text-sm border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-red-400 focus:border-transparent placeholder-gray-300 text-gray-800"
                />
                <button
                  type="button"
                  onClick={() => setShowPw((v) => !v)}
                  className="absolute right-3.5 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 transition-colors"
                >
                  {showPw ? (
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
                  )}
                </button>
              </div>
            </div>

            {error && (
              <div className="flex items-center gap-2 px-3.5 py-2.5 bg-red-50 border border-red-100 rounded-xl text-xs text-red-600 font-medium">
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
          </Modal.Body>

          <Modal.Footer>
            <button
              onClick={() => setShowLoginModal(false)}
              className="flex-1 py-2.5 text-sm font-medium text-gray-600 bg-gray-100 hover:bg-gray-200 rounded-xl transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={handleLogin}
              disabled={loading}
              className={`flex-1 py-2.5 text-sm font-medium text-white rounded-xl transition-colors ${loading ? "bg-red-300 cursor-not-allowed" : "bg-red-500 hover:bg-red-600"}`}
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
                  Signing in...
                </span>
              ) : (
                "Sign In"
              )}
            </button>
          </Modal.Footer>
        </Modal>
      )}

      {/* HEADER */}
      <header className="h-16 bg-white border-b border-gray-200 flex items-center justify-between px-6">
        <div>
          <h1 className="text-lg font-semibold text-gray-800">{page.title}</h1>
          <p className="text-xs text-gray-400">{page.description}</p>
        </div>

        <div className="flex items-center gap-3">
          <div className="w-px h-6 bg-gray-200" />

          <div
            className="w-8 h-8 rounded-full bg-red-500 flex items-center justify-center cursor-pointer select-none"
            title={user?.agent_name ?? ""}
          >
            {isLoggedIn ? (
              <span className="text-white text-xs font-bold">
                {userInitial}
              </span>
            ) : (
              <svg
                className="w-4 h-4 text-white"
                fill="currentColor"
                viewBox="0 0 20 20"
              >
                <path
                  fillRule="evenodd"
                  d="M10 9a3 3 0 100-6 3 3 0 000 6zm-7 9a7 7 0 1114 0H3z"
                  clipRule="evenodd"
                />
              </svg>
            )}
          </div>

          {isLoggedIn ? (
            <button
              onClick={handleLogout}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-gray-500 hover:text-red-500 hover:bg-red-50 rounded-lg border border-gray-200 hover:border-red-200 transition-colors"
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
                  d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1"
                />
              </svg>
              Logout
            </button>
          ) : (
            <button
              onClick={openLoginModal}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-white bg-red-500 hover:bg-red-600 rounded-lg transition-colors shadow-sm"
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
                  d="M11 16l-4-4m0 0l4-4m-4 4h14m-5 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h7a3 3 0 013 3v1"
                />
              </svg>
              Login
            </button>
          )}
        </div>
      </header>
    </>
  );
}
