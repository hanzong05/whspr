"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import Modal from "@/components/ui/Modal";
import ActionButtons from "@/components/ui/ActionButtons";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// ── Types ─────────────────────────────────────────────────────────────────────

type RiskLevel = "Critical" | "High" | "Medium" | "Low";
type Emotion = "angry" | "frustrated" | "sad" | "neutral" | "happy" | "satisfied";

interface AgentOption { id: number; name: string; cluster_name: string | null; }
interface ClusterOption { id: number; name: string; }

interface CallRow {
  id: number;
  filename: string;
  duration_sec: number | null;
  call_date: string | null;
  upload_status: string;
  agent: { id: number; name: string; email: string } | null;
  cluster: { id: number; name: string } | null;
  analysis: {
    predicted_emotion: Emotion;
    confidence: number;
    risk_level: RiskLevel;
    transcription_text: string | null;
  } | null;
  recommendation: {
    action: string;
    urgency: string;
    reason: string | null;
    instruction: string | null;
    recommended_tone: string | null;
  } | null;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

const riskBadge: Record<string, string> = {
  Critical: "bg-red-100 text-red-700",
  High: "bg-orange-100 text-orange-700",
  Medium: "bg-yellow-100 text-yellow-700",
  Low: "bg-green-100 text-green-700",
};

const emotionColor: Record<string, string> = {
  angry: "text-red-600",
  frustrated: "text-orange-600",
  sad: "text-blue-600",
  neutral: "text-gray-500",
  happy: "text-green-600",
  satisfied: "text-green-600",
};

function fmt(s: string) { return s.charAt(0).toUpperCase() + s.slice(1); }

function fmtDuration(sec: number | null) {
  if (!sec) return "—";
  const m = Math.floor(sec / 60), s2 = sec % 60;
  return `${m}:${String(s2).padStart(2, "0")}`;
}

function fmtDate(d: string | null) {
  if (!d) return "—";
  return new Date(d).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
}

// ── Upload modal ───────────────────────────────────────────────────────────────

type UploadStep = "select" | "analyzing" | "result";

function UploadContent({ onClose, onUploaded }: { onClose: () => void; onUploaded: () => void }) {
  const [step, setStep] = useState<UploadStep>("select");
  const [file, setFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [agentSearch, setAgentSearch] = useState("");
  const [selectedAgent, setSelectedAgent] = useState<AgentOption | null>(null);
  const [agents, setAgents] = useState<AgentOption[]>([]);
  const [result, setResult] = useState<CallRow | null>(null);
  const searchRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    fetch(`${API}/agents`).then((r) => r.json()).then((data: AgentOption[]) => setAgents(data)).catch(() => {});
    searchRef.current?.focus();
  }, []);

  const filteredAgents = agents.filter(
    (a) =>
      a.name.toLowerCase().includes(agentSearch.toLowerCase()) ||
      (a.cluster_name ?? "").toLowerCase().includes(agentSearch.toLowerCase()),
  );

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault(); e.stopPropagation();
    setIsDragging(e.type === "dragenter" || e.type === "dragover");
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault(); e.stopPropagation();
    setIsDragging(false);
    const dropped = e.dataTransfer.files[0];
    if (dropped) setFile(dropped);
  }, []);

  const handleUpload = async () => {
    if (!file || !selectedAgent) return;
    setStep("analyzing");
    const formData = new FormData();
    formData.append("file", file);
    formData.append("agent_id", String(selectedAgent.id));

    try {
      const res = await fetch(`${API}/analyze`, { method: "POST", body: formData });
      if (!res.ok) throw new Error("Failed");
      const data = await res.json();

      // Build a minimal CallRow from the analysis response for the result screen
      const callId: number | undefined = data.call_id;
      const ea = data.emotion_analysis ?? {};
      const rec = data.csr_recommendations ?? {};
      setResult({
        id: callId ?? 0,
        filename: file.name,
        duration_sec: data.transcription?.duration ?? null,
        call_date: new Date().toISOString().split("T")[0],
        upload_status: "analyzed",
        agent: { id: selectedAgent.id, name: selectedAgent.name, email: "" },
        cluster: null,
        analysis: {
          predicted_emotion: ea.predicted_emotion ?? "neutral",
          confidence: ea.confidence ?? 0,
          risk_level: ea.risk_level ?? "Low",
          transcription_text: data.transcription?.text ?? null,
        },
        recommendation: {
          action: rec.action_required?.action ?? "NONE",
          urgency: rec.action_required?.urgency ?? "LOW",
          reason: rec.action_required?.reason ?? null,
          instruction: rec.action_required?.instruction ?? null,
          recommended_tone: rec.communication_guidance?.recommended_tone ?? null,
        },
      });
      setStep("result");
    } catch {
      setStep("select");
    }
  };

  const handleDone = () => { onUploaded(); onClose(); };

  const stepTitle: Record<UploadStep, string> = { select: "Upload Call Recording", analyzing: "Upload Call Recording", result: "Analysis Complete" };
  const stepDesc:  Record<UploadStep, string> = { select: "Select a file and assign an agent", analyzing: "Analyzing your recording...", result: "Review the results below" };

  return (
    <Modal onClose={step === "result" ? handleDone : onClose} maxWidth="lg">
      <Modal.Header title={stepTitle[step]} description={stepDesc[step]} onClose={step === "result" ? handleDone : onClose} />

      {step === "select" && (
        <>
          <Modal.Body>
            <div onDragEnter={handleDrag} onDragLeave={handleDrag} onDragOver={handleDrag} onDrop={handleDrop}
              className={`border-2 border-dashed rounded-xl p-6 text-center transition-colors ${isDragging ? "border-red-400 bg-red-50" : "border-gray-200 hover:border-red-300"}`}>
              {file ? (
                <div className="flex items-center gap-3 justify-center">
                  <svg className="w-6 h-6 text-red-500" fill="currentColor" viewBox="0 0 20 20"><path d="M18 3a1 1 0 00-1.196-.98l-10 2A1 1 0 006 5v9.114A4.369 4.369 0 005 14c-1.657 0-3 .895-3 2s1.343 2 3 2 3-.895 3-2V7.82l8-1.6v5.894A4.37 4.37 0 0015 12c-1.657 0-3 .895-3 2s1.343 2 3 2 3-.895 3-2V3z" /></svg>
                  <span className="text-sm font-medium text-gray-700 truncate max-w-xs">{file.name}</span>
                  <button onClick={() => setFile(null)} className="text-gray-400 hover:text-red-500">
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg>
                  </button>
                </div>
              ) : (
                <>
                  <svg className="w-8 h-8 text-gray-300 mx-auto mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" /></svg>
                  <p className="text-sm text-gray-500">Drop audio file here or{" "}
                    <label className="text-red-500 cursor-pointer hover:text-red-600 font-medium">
                      browse<input type="file" accept="audio/*,.mp3,.wav,.m4a,.ogg,.webm" onChange={(e) => e.target.files?.[0] && setFile(e.target.files[0])} className="hidden" />
                    </label>
                  </p>
                </>
              )}
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-600 mb-2">Assign Agent</label>
              <div className="relative mb-2">
                <svg className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" /></svg>
                <input ref={searchRef} type="text" placeholder="Search agent..." value={agentSearch} onChange={(e) => setAgentSearch(e.target.value)}
                  className="w-full pl-9 pr-4 py-2.5 text-sm border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-red-400 focus:border-transparent" />
              </div>
              <div className="border border-gray-100 rounded-xl overflow-hidden max-h-40 overflow-y-auto">
                {filteredAgents.map((agent) => (
                  <button key={agent.id} onClick={() => setSelectedAgent(agent)}
                    className={`w-full flex items-center gap-3 px-3 py-2.5 text-left transition-colors ${selectedAgent?.id === agent.id ? "bg-red-50" : "hover:bg-gray-50"}`}>
                    <div className={`w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0 ${selectedAgent?.id === agent.id ? "bg-red-500 text-white" : "bg-gray-100 text-gray-600"}`}>
                      {agent.name.charAt(0)}
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className={`text-sm font-medium truncate ${selectedAgent?.id === agent.id ? "text-red-700" : "text-gray-800"}`}>{agent.name}</p>
                      <p className="text-xs text-gray-400">{agent.cluster_name ?? "—"}</p>
                    </div>
                    {selectedAgent?.id === agent.id && (
                      <svg className="w-4 h-4 text-red-500 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20"><path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" /></svg>
                    )}
                  </button>
                ))}
              </div>
            </div>
          </Modal.Body>
          <Modal.Footer>
            <button onClick={onClose} className="flex-1 py-2.5 text-sm font-medium text-gray-600 bg-gray-100 hover:bg-gray-200 rounded-xl transition-colors">Cancel</button>
            <button onClick={handleUpload} disabled={!file || !selectedAgent}
              className={`flex-1 py-2.5 text-sm font-semibold text-white rounded-xl transition-colors ${file && selectedAgent ? "bg-red-500 hover:bg-red-600" : "bg-gray-300 cursor-not-allowed"}`}>
              Upload & Analyze
            </button>
          </Modal.Footer>
        </>
      )}

      {step === "analyzing" && (
        <Modal.Body>
          <div className="flex flex-col items-center gap-4 py-8">
            <div className="w-14 h-14 rounded-full border-4 border-red-200 border-t-red-500 animate-spin" />
            <p className="text-sm font-medium text-gray-600">Analyzing recording...</p>
            <p className="text-xs text-gray-400">{file?.name}</p>
          </div>
        </Modal.Body>
      )}

      {step === "result" && result?.analysis && (
        <>
          <Modal.Body>
            <div className="flex items-center gap-3 p-3 bg-green-50 border border-green-100 rounded-xl">
              <div className="w-8 h-8 bg-green-500 rounded-full flex items-center justify-center flex-shrink-0">
                <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" /></svg>
              </div>
              <div>
                <p className="text-sm font-semibold text-green-800">Analysis Complete</p>
                <p className="text-xs text-green-600 truncate">{file?.name}</p>
              </div>
            </div>
            <div className="flex items-center gap-3 p-3 bg-gray-50 rounded-xl">
              <div className="w-8 h-8 bg-red-100 rounded-full flex items-center justify-center text-red-600 text-sm font-bold flex-shrink-0">{selectedAgent?.name.charAt(0)}</div>
              <div>
                <p className="text-sm font-medium text-gray-800">{selectedAgent?.name}</p>
                <p className="text-xs text-gray-400">{selectedAgent?.cluster_name ?? "—"}</p>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div className="bg-gray-50 rounded-xl p-4">
                <p className="text-xs text-gray-400 mb-1">Detected Emotion</p>
                <p className={`text-xl font-bold ${emotionColor[result.analysis.predicted_emotion] ?? "text-gray-700"}`}>{fmt(result.analysis.predicted_emotion)}</p>
                <p className="text-xs text-gray-400 mt-1">{(result.analysis.confidence * 100).toFixed(0)}% confidence</p>
              </div>
              <div className="bg-gray-50 rounded-xl p-4">
                <p className="text-xs text-gray-400 mb-2">Risk Level</p>
                <span className={`inline-block px-2.5 py-1 rounded-full text-sm font-semibold ${riskBadge[result.analysis.risk_level] ?? "bg-gray-100 text-gray-600"}`}>{result.analysis.risk_level}</span>
              </div>
            </div>
            {result.analysis.transcription_text && (
              <div className="bg-gray-50 rounded-xl p-4">
                <p className="text-xs font-medium text-gray-500 mb-1">Transcription</p>
                <p className="text-sm text-gray-700 line-clamp-3">{result.analysis.transcription_text}</p>
              </div>
            )}
            {result.recommendation?.recommended_tone && (
              <div className="bg-blue-50 border border-blue-100 rounded-xl p-4">
                <p className="text-xs font-medium text-blue-600 mb-1">Recommendation</p>
                <p className="text-sm text-blue-800">{result.recommendation.recommended_tone}</p>
              </div>
            )}
          </Modal.Body>
          <Modal.Footer>
            <button onClick={handleDone} className="flex-1 py-2.5 text-sm font-semibold text-white bg-red-500 hover:bg-red-600 rounded-xl transition-colors">Done</button>
          </Modal.Footer>
        </>
      )}
    </Modal>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────────

export default function CallsPage() {
  const [calls, setCalls] = useState<CallRow[]>([]);
  const [clusters, setClusters] = useState<ClusterOption[]>([]);
  const [agentOptions, setAgentOptions] = useState<AgentOption[]>([]);
  const [loading, setLoading] = useState(true);

  const [clusterFilter, setClusterFilter] = useState<string>("all");
  const [agentFilter, setAgentFilter] = useState<string>("all");
  const [search, setSearch] = useState("");

  const [showUpload, setShowUpload] = useState(false);
  const [viewTarget, setViewTarget] = useState<CallRow | null>(null);
  const [editTarget, setEditTarget] = useState<CallRow | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<CallRow | null>(null);

  const [editAgentId, setEditAgentId] = useState<string>("");

  const fetchCalls = async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams();
      if (clusterFilter !== "all") params.set("cluster_id", clusterFilter);
      if (agentFilter !== "all") params.set("agent_id", agentFilter);
      const data: CallRow[] = await fetch(`${API}/calls?${params}`).then((r) => r.json());
      setCalls(data);
    } catch { setCalls([]); }
    setLoading(false);
  };

  useEffect(() => {
    fetch(`${API}/clusters`).then((r) => r.json()).then(setClusters).catch(() => {});
    fetch(`${API}/agents`).then((r) => r.json()).then(setAgentOptions).catch(() => {});
  }, []);

  useEffect(() => { fetchCalls(); }, [clusterFilter, agentFilter]);

  const openEdit = (call: CallRow) => {
    setEditTarget(call);
    setEditAgentId(String(call.agent?.id ?? ""));
  };

  const saveEdit = async () => {
    if (!editTarget) return;
    await fetch(`${API}/calls/${editTarget.id}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ agent_id: Number(editAgentId) }),
    });
    setEditTarget(null);
    fetchCalls();
  };

  const handleDelete = async () => {
    if (!deleteTarget) return;
    await fetch(`${API}/calls/${deleteTarget.id}`, { method: "DELETE" });
    setDeleteTarget(null);
    fetchCalls();
  };

  const filtered = calls.filter((c) => {
    const matchSearch =
      c.filename.toLowerCase().includes(search.toLowerCase()) ||
      (c.agent?.name ?? "").toLowerCase().includes(search.toLowerCase());
    return matchSearch;
  });

  return (
    <div>
      {/* ── View Modal ── */}
      {viewTarget && (
        <Modal onClose={() => setViewTarget(null)} maxWidth="md">
          <Modal.Header title="Call Details" description={viewTarget.filename} onClose={() => setViewTarget(null)} />
          <Modal.Body>
            <div className="flex items-center gap-3 p-3 bg-gray-50 rounded-xl">
              <div className="w-10 h-10 rounded-full bg-red-100 text-red-600 font-bold text-sm flex items-center justify-center flex-shrink-0">
                {(viewTarget.agent?.name ?? "?").charAt(0)}
              </div>
              <div>
                <p className="text-sm font-semibold text-gray-800">{viewTarget.agent?.name ?? "—"}</p>
                <p className="text-xs text-gray-400">{viewTarget.cluster?.name ?? "—"}</p>
              </div>
              <div className="ml-auto text-right">
                <p className="text-xs text-gray-400">{fmtDate(viewTarget.call_date)}</p>
                <p className="text-xs text-gray-400">{fmtDuration(viewTarget.duration_sec)}</p>
              </div>
            </div>
            {viewTarget.analysis && (
              <>
                <div className="grid grid-cols-2 gap-3">
                  <div className="bg-gray-50 rounded-xl p-4">
                    <p className="text-xs text-gray-400 mb-1">Detected Emotion</p>
                    <p className={`text-xl font-bold ${emotionColor[viewTarget.analysis.predicted_emotion] ?? "text-gray-700"}`}>{fmt(viewTarget.analysis.predicted_emotion)}</p>
                    <p className="text-xs text-gray-400 mt-1">{(viewTarget.analysis.confidence * 100).toFixed(0)}% confidence</p>
                  </div>
                  <div className="bg-gray-50 rounded-xl p-4">
                    <p className="text-xs text-gray-400 mb-2">Risk Level</p>
                    <span className={`inline-block px-2.5 py-1 rounded-full text-sm font-semibold ${riskBadge[viewTarget.analysis.risk_level] ?? "bg-gray-100 text-gray-600"}`}>{viewTarget.analysis.risk_level}</span>
                  </div>
                </div>
                {viewTarget.analysis.transcription_text && (
                  <div className="bg-gray-50 rounded-xl p-4">
                    <p className="text-xs font-medium text-gray-500 mb-1">Transcription</p>
                    <p className="text-sm text-gray-700 leading-relaxed">{viewTarget.analysis.transcription_text}</p>
                  </div>
                )}
              </>
            )}
            {viewTarget.recommendation && (
              <div className="bg-blue-50 border border-blue-100 rounded-xl p-4">
                <p className="text-xs font-medium text-blue-500 mb-1">Recommendation</p>
                <p className="text-sm text-blue-800">{viewTarget.recommendation.recommended_tone ?? viewTarget.recommendation.reason ?? "—"}</p>
              </div>
            )}
          </Modal.Body>
          <Modal.Footer>
            <button onClick={() => setViewTarget(null)} className="flex-1 py-2.5 text-sm font-medium text-gray-600 bg-gray-100 hover:bg-gray-200 rounded-xl transition-colors">Close</button>
          </Modal.Footer>
        </Modal>
      )}

      {/* ── Edit Modal ── */}
      {editTarget && (
        <Modal onClose={() => setEditTarget(null)} maxWidth="sm">
          <Modal.Header title="Edit Call" description={editTarget.filename} onClose={() => setEditTarget(null)} />
          <Modal.Body>
            <div>
              <label className="block text-xs font-medium text-gray-600 mb-1.5">Agent</label>
              <select value={editAgentId} onChange={(e) => setEditAgentId(e.target.value)}
                className="w-full px-3 py-2.5 text-sm border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-red-400">
                {agentOptions.map((a) => <option key={a.id} value={a.id}>{a.name}</option>)}
              </select>
            </div>
          </Modal.Body>
          <Modal.Footer>
            <button onClick={() => setEditTarget(null)} className="flex-1 py-2.5 text-sm font-medium text-gray-600 bg-gray-100 hover:bg-gray-200 rounded-xl transition-colors">Cancel</button>
            <button onClick={saveEdit} className="flex-1 py-2.5 text-sm font-medium text-white bg-red-500 hover:bg-red-600 rounded-xl transition-colors">Save</button>
          </Modal.Footer>
        </Modal>
      )}

      {/* ── Delete Modal ── */}
      {deleteTarget && (
        <Modal onClose={() => setDeleteTarget(null)} maxWidth="sm">
          <Modal.Header title="Delete Call" onClose={() => setDeleteTarget(null)} />
          <Modal.Body>
            <div className="w-12 h-12 bg-red-100 rounded-full flex items-center justify-center mx-auto">
              <svg className="w-6 h-6 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" /></svg>
            </div>
            <p className="text-sm text-gray-500 text-center">
              Are you sure you want to delete <span className="font-medium text-gray-700">{deleteTarget.filename}</span>? This cannot be undone.
            </p>
          </Modal.Body>
          <Modal.Footer>
            <button onClick={() => setDeleteTarget(null)} className="flex-1 py-2.5 text-sm font-medium text-gray-600 bg-gray-100 hover:bg-gray-200 rounded-xl transition-colors">Cancel</button>
            <button onClick={handleDelete} className="flex-1 py-2.5 text-sm font-medium text-white bg-red-500 hover:bg-red-600 rounded-xl transition-colors">Delete</button>
          </Modal.Footer>
        </Modal>
      )}

      {/* ── Upload Modal ── */}
      {showUpload && (
        <UploadContent onClose={() => setShowUpload(false)} onUploaded={fetchCalls} />
      )}

      {/* Toolbar */}
      <div className="flex flex-wrap items-center gap-3 mb-6">
        <div className="relative flex-1 min-w-48">
          <svg className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" /></svg>
          <input type="text" placeholder="Search calls or agents..." value={search} onChange={(e) => setSearch(e.target.value)}
            className="w-full pl-9 pr-4 py-2.5 text-sm bg-white border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-red-400 focus:border-transparent" />
        </div>
        <select value={clusterFilter} onChange={(e) => setClusterFilter(e.target.value)} className="px-4 py-2.5 text-sm bg-white border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-red-400">
          <option value="all">All Clusters</option>
          {clusters.map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}
        </select>
        <select value={agentFilter} onChange={(e) => setAgentFilter(e.target.value)} className="px-4 py-2.5 text-sm bg-white border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-red-400">
          <option value="all">All Agents</option>
          {agentOptions.map((a) => <option key={a.id} value={a.id}>{a.name}</option>)}
        </select>
        <span className="text-sm text-gray-400">{filtered.length} calls</span>
        <button onClick={() => setShowUpload(true)} className="flex items-center gap-2 px-4 py-2.5 bg-red-500 hover:bg-red-600 text-white text-sm font-semibold rounded-xl transition-colors shadow-sm">
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" /></svg>
          Upload
        </button>
      </div>

      {/* Table */}
      <div className="bg-white rounded-2xl shadow-sm border border-gray-100 overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-100 bg-gray-50">
              {["File", "Agent", "Cluster", "Date", "Duration", "Emotion", "Risk", "Recommendation", "Actions"].map((h) => (
                <th key={h} className="text-left px-5 py-3.5 text-xs font-semibold text-gray-500 uppercase tracking-wide">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-50">
            {loading ? (
              <tr><td colSpan={9} className="text-center py-12 text-gray-400 text-sm">Loading...</td></tr>
            ) : filtered.length === 0 ? (
              <tr><td colSpan={9} className="text-center py-12 text-gray-400 text-sm">No calls found</td></tr>
            ) : (
              filtered.map((call) => (
                <tr key={call.id} className="hover:bg-gray-50 transition-colors">
                  <td className="px-5 py-4">
                    <div className="flex items-center gap-2">
                      <svg className="w-5 h-5 text-red-400 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20"><path d="M18 3a1 1 0 00-1.196-.98l-10 2A1 1 0 006 5v9.114A4.369 4.369 0 005 14c-1.657 0-3 .895-3 2s1.343 2 3 2 3-.895 3-2V7.82l8-1.6v5.894A4.37 4.37 0 0015 12c-1.657 0-3 .895-3 2s1.343 2 3 2 3-.895 3-2V3z" /></svg>
                      <span className="text-gray-700 font-medium truncate max-w-36">{call.filename}</span>
                    </div>
                  </td>
                  <td className="px-5 py-4">
                    <div className="flex items-center gap-2">
                      <div className="w-6 h-6 rounded-full bg-red-100 text-red-600 text-xs font-bold flex items-center justify-center flex-shrink-0">{(call.agent?.name ?? "?").charAt(0)}</div>
                      <span className="text-gray-700">{call.agent?.name ?? "—"}</span>
                    </div>
                  </td>
                  <td className="px-5 py-4 text-gray-500">{call.cluster?.name ?? "—"}</td>
                  <td className="px-5 py-4 text-gray-500">{fmtDate(call.call_date)}</td>
                  <td className="px-5 py-4 text-gray-500">{fmtDuration(call.duration_sec)}</td>
                  <td className="px-5 py-4">
                    {call.analysis ? (
                      <span className={`font-medium ${emotionColor[call.analysis.predicted_emotion] ?? "text-gray-500"}`}>{fmt(call.analysis.predicted_emotion)}</span>
                    ) : <span className="text-gray-300">—</span>}
                  </td>
                  <td className="px-5 py-4">
                    {call.analysis ? (
                      <span className={`px-2 py-1 rounded-full text-xs font-semibold ${riskBadge[call.analysis.risk_level] ?? "bg-gray-100 text-gray-600"}`}>{call.analysis.risk_level}</span>
                    ) : <span className="text-gray-300">—</span>}
                  </td>
                  <td className="px-5 py-4 max-w-48">
                    <span className="text-sm text-gray-600 line-clamp-1">
                      {call.recommendation?.recommended_tone ?? call.recommendation?.reason ?? "—"}
                    </span>
                  </td>
                  <td className="px-5 py-4">
                    <ActionButtons onView={() => setViewTarget(call)} onEdit={() => openEdit(call)} onDelete={() => setDeleteTarget(call)} />
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
