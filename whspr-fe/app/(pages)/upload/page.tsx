"use client";

import { useState, useCallback, useRef, useEffect } from "react";

// ── Types ─────────────────────────────────────────────────────────────────────

interface AnalysisData {
  csr_recommendations?: {
    available?: boolean;
    action_required?: {
      action: string;
      urgency: string;
      reason: string;
      instruction: string;
      color: string;
    };
    communication_guidance?: {
      recommended_tone: string;
      example_phrases: string[];
    };
    do_and_dont?: {
      do: string[];
      dont: string[];
    };
  };
  speaker_detection?: {
    mode: string;
    agent_channel?: string;
    caller_channel?: string;
    note?: string;
  };
  emotion_analysis?: {
    predicted_emotion: string;
    confidence: number;
    all_probabilities?: Record<string, number>;
    emotional_state?: {
      valence?: string;
      arousal?: string;
    };
  };
  transcription?: {
    text: string;
    duration: number;
    language: string;
  };
}

interface UploadResult {
  filename: string;
  status: string;
  progress?: number;
  error?: string;
  data?: AnalysisData;
}

const AGENTS = [
  { id: "1", name: "Maria Santos", group: "West Coast" },
  { id: "2", name: "James Reyes", group: "West Coast" },
  { id: "3", name: "Ana Cruz", group: "East Coast" },
  { id: "4", name: "Carlo Dela Cruz", group: "East Coast" },
  { id: "5", name: "Liza Ramos", group: "Midwest" },
  { id: "6", name: "Rico Fernandez", group: "Midwest" },
  { id: "7", name: "Jessa Villanueva", group: "The South" },
  { id: "8", name: "Mark Tolentino", group: "The South" },
  { id: "9", name: "Nina Bautista", group: "West Coast" },
  { id: "10", name: "Luis Garcia", group: "East Coast" },
  { id: "11", name: "Carla Mendoza", group: "Midwest" },
  { id: "12", name: "Ryan Pascual", group: "The South" },
];

function AgentSelectModal({
  onConfirm,
  onCancel,
}: {
  onConfirm: (agent: (typeof AGENTS)[0]) => void;
  onCancel: () => void;
}) {
  const [search, setSearch] = useState("");
  const [selected, setSelected] = useState<(typeof AGENTS)[0] | null>(null);
  const searchRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    searchRef.current?.focus();
  }, []);

  const filtered = AGENTS.filter(
    (a) =>
      a.name.toLowerCase().includes(search.toLowerCase()) ||
      a.group.toLowerCase().includes(search.toLowerCase()),
  );

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/40 backdrop-blur-sm"
        onClick={onCancel}
      />

      {/* Modal */}
      <div className="relative bg-white rounded-2xl shadow-2xl w-full max-w-md mx-4 overflow-hidden">
        {/* Header */}
        <div className="px-6 pt-6 pb-4 border-b border-gray-100">
          <div className="flex items-center justify-between mb-1">
            <h2 className="text-lg font-semibold text-gray-800">
              Select Agent
            </h2>
            <button
              onClick={onCancel}
              className="p-1 text-gray-400 hover:text-gray-600 rounded-lg hover:bg-gray-100 transition-colors"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
          <p className="text-sm text-gray-400">
            Assign this recording to an agent before analyzing.
          </p>

          {/* Search */}
          <div className="relative mt-4">
            <svg
              className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
            <input
              ref={searchRef}
              type="text"
              placeholder="Search by name or group..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="w-full pl-9 pr-4 py-2.5 text-sm border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-red-400 focus:border-transparent"
            />
          </div>
        </div>

        {/* Agent list */}
        <div className="overflow-y-auto max-h-64 px-3 py-3 space-y-1">
          {filtered.length === 0 ? (
            <p className="text-center text-sm text-gray-400 py-6">
              No agents found
            </p>
          ) : (
            filtered.map((agent) => (
              <button
                key={agent.id}
                onClick={() => setSelected(agent)}
                className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-left transition-colors ${
                  selected?.id === agent.id
                    ? "bg-red-50 border border-red-200"
                    : "hover:bg-gray-50 border border-transparent"
                }`}
              >
                <div
                  className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 text-sm font-semibold ${
                    selected?.id === agent.id
                      ? "bg-red-500 text-white"
                      : "bg-gray-100 text-gray-600"
                  }`}
                >
                  {agent.name.charAt(0)}
                </div>
                <div className="flex-1 min-w-0">
                  <p
                    className={`text-sm font-medium truncate ${
                      selected?.id === agent.id ? "text-red-700" : "text-gray-800"
                    }`}
                  >
                    {agent.name}
                  </p>
                  <p className="text-xs text-gray-400 truncate">{agent.group}</p>
                </div>
                {selected?.id === agent.id && (
                  <svg className="w-4 h-4 text-red-500 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                  </svg>
                )}
              </button>
            ))
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-gray-100 flex gap-3">
          <button
            onClick={onCancel}
            className="flex-1 py-2.5 px-4 text-sm font-medium text-gray-600 bg-gray-100 hover:bg-gray-200 rounded-xl transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={() => selected && onConfirm(selected)}
            disabled={!selected}
            className={`flex-1 py-2.5 px-4 text-sm font-medium text-white rounded-xl transition-colors ${
              selected
                ? "bg-red-500 hover:bg-red-600"
                : "bg-gray-300 cursor-not-allowed"
            }`}
          >
            Confirm & Analyze
          </button>
        </div>
      </div>
    </div>
  );
}

export default function RecordingUpload() {
  const [isDragging, setIsDragging] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const [uploadResults, setUploadResults] = useState<UploadResult[]>([]);
  const [showAgentModal, setShowAgentModal] = useState(false);
  const [selectedAgent, setSelectedAgent] = useState<(typeof AGENTS)[0] | null>(null);

  // API Configuration
  const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

  const handleDrag = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setIsDragging(true);
    } else if (e.type === "dragleave") {
      setIsDragging(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const files: File[] = Array.from(e.dataTransfer.files).filter(
      (file) =>
        file.type.startsWith("audio/") ||
        [".mp3", ".wav", ".m4a", ".ogg", ".webm"].some((ext) =>
          file.name.endsWith(ext),
        ),
    );

    if (files.length > 0) {
      setSelectedFiles((prev) => [...prev, ...files]);
    }
  }, []);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files ?? []) as File[];
    setSelectedFiles((prev) => [...prev, ...files]);
  };

  const removeFile = (index: number) => {
    setSelectedFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + " " + sizes[i];
  };

  const analyzeFile = async (file: File, index: number) => {
    const formData = new FormData();
    formData.append("file", file);

    try {
      // Update status to processing
      setUploadResults((prev) => {
        const newResults = [...prev];
        newResults[index] = {
          filename: file.name,
          status: "processing",
          progress: 50,
        };
        return newResults;
      });

      const response = await fetch(`${API_URL}/analyze`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Analysis failed");
      }

      const result = await response.json();

      // Update with completed results
      setUploadResults((prev) => {
        const newResults = [...prev];
        newResults[index] = {
          filename: file.name,
          status: "completed",
          progress: 100,
          data: result,
        };
        return newResults;
      });
    } catch (error) {
      setUploadResults((prev) => {
        const newResults = [...prev];
        newResults[index] = {
          filename: file.name,
          status: "failed",
          error: error instanceof Error ? error.message : String(error),
        };
        return newResults;
      });
    }
  };

  const handleUpload = async () => {
    if (selectedFiles.length === 0) return;

    setUploading(true);

    // Initialize results array
    const initialResults = selectedFiles.map((file) => ({
      filename: file.name,
      status: "queued",
      progress: 0,
    }));
    setUploadResults(initialResults);

    // Process each file
    for (let i = 0; i < selectedFiles.length; i++) {
      await analyzeFile(selectedFiles[i], i);
    }

    setUploading(false);
    setSelectedFiles([]);
  };

  const getEmotionColor = (emotion: string) => {
    const colors: Record<string, string> = {
      angry: "text-red-600",
      frustrated: "text-orange-600",
      sad: "text-blue-600",
      neutral: "text-gray-600",
      satisfied: "text-green-600",
      happy: "text-green-600",
    };
    return colors[emotion?.toLowerCase()] || "text-gray-600";
  };

  const getRiskColor = (risk: string) => {
    const colors: Record<string, string> = {
      critical: "bg-red-100 text-red-800",
      high: "bg-orange-100 text-orange-800",
      medium: "bg-yellow-100 text-yellow-800",
      low: "bg-green-100 text-green-800",
    };
    return colors[risk?.toLowerCase()] || "bg-gray-100 text-gray-800";
  };

  return (
    <div className="min-h-screen bg-gray-50 p-4">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-5xl font-bold text-gray-800 mb-4">
            CSR Call Recording Analysis
          </h1>
          <p className="text-xl text-gray-600">
            Upload call recordings for AI-powered emotional analysis and
            recommendations
          </p>
        </div>

        {/* Upload Area */}
        <div
          className={`bg-white rounded-2xl p-12 shadow-lg transition-all duration-200 mb-8 ${
            isDragging
              ? "border-4 border-red-500 bg-red-50"
              : "border-4 border-transparent"
          }`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          <div className="flex flex-col items-center gap-6">
            <label
              htmlFor="file-upload"
              className="bg-red-500 hover:bg-red-600 text-white text-xl font-semibold py-6 px-16 rounded-xl cursor-pointer transition-colors duration-200 shadow-md hover:shadow-lg"
            >
              Select Audio Recordings
            </label>
            <input
              id="file-upload"
              type="file"
              multiple
              accept="audio/*,.mp3,.wav,.m4a,.ogg,.webm"
              onChange={handleFileSelect}
              className="hidden"
            />

            <div className="flex gap-4">
              <button className="w-12 h-12 bg-red-500 hover:bg-red-600 rounded-full flex items-center justify-center text-white shadow-md transition-colors duration-200">
                <svg
                  className="w-6 h-6"
                  fill="currentColor"
                  viewBox="0 0 20 20"
                >
                  <path d="M5.5 13a3.5 3.5 0 01-.369-6.98 4 4 0 117.753-1.977A4.5 4.5 0 1113.5 13H11V9.413l1.293 1.293a1 1 0 001.414-1.414l-3-3a1 1 0 00-1.414 0l-3 3a1 1 0 001.414 1.414L9 9.414V13H5.5z" />
                </svg>
              </button>
              <button className="w-12 h-12 bg-red-500 hover:bg-red-600 rounded-full flex items-center justify-center text-white shadow-md transition-colors duration-200">
                <svg
                  className="w-6 h-6"
                  fill="currentColor"
                  viewBox="0 0 20 20"
                >
                  <path
                    fillRule="evenodd"
                    d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zM6.293 6.707a1 1 0 010-1.414l3-3a1 1 0 011.414 0l3 3a1 1 0 01-1.414 1.414L11 5.414V13a1 1 0 11-2 0V5.414L7.707 6.707a1 1 0 01-1.414 0z"
                    clipRule="evenodd"
                  />
                </svg>
              </button>
            </div>

            <p className="text-gray-500 text-lg">
              or drop audio recordings here
            </p>
          </div>
        </div>

        {/* Selected Files */}
        {selectedFiles.length > 0 && (
          <div className="bg-white rounded-2xl p-6 shadow-lg mb-8">
            <h3 className="text-xl font-semibold text-gray-800 mb-4">
              Selected Files ({selectedFiles.length})
            </h3>
            <div className="space-y-3">
              {selectedFiles.map((file, index) => (
                <div
                  key={index}
                  className="flex items-center justify-between p-4 bg-gray-50 rounded-lg"
                >
                  <div className="flex items-center gap-3 flex-1 min-w-0">
                    <svg
                      className="w-8 h-8 text-red-500 flex-shrink-0"
                      fill="currentColor"
                      viewBox="0 0 20 20"
                    >
                      <path d="M18 3a1 1 0 00-1.196-.98l-10 2A1 1 0 006 5v9.114A4.369 4.369 0 005 14c-1.657 0-3 .895-3 2s1.343 2 3 2 3-.895 3-2V7.82l8-1.6v5.894A4.37 4.37 0 0015 12c-1.657 0-3 .895-3 2s1.343 2 3 2 3-.895 3-2V3z" />
                    </svg>
                    <div className="min-w-0 flex-1">
                      <p className="text-gray-800 font-medium truncate">
                        {file.name}
                      </p>
                      <p className="text-gray-500 text-sm">
                        {formatFileSize(file.size)}
                      </p>
                    </div>
                  </div>
                  {!uploading && (
                    <button
                      onClick={() => removeFile(index)}
                      className="ml-4 text-gray-400 hover:text-red-500 transition-colors"
                    >
                      <svg
                        className="w-6 h-6"
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
                  )}
                </div>
              ))}
            </div>

            {selectedAgent && (
              <div className="mt-4 flex items-center gap-2 px-4 py-2 bg-red-50 border border-red-100 rounded-xl">
                <div className="w-7 h-7 rounded-full bg-red-500 text-white text-xs font-bold flex items-center justify-center flex-shrink-0">
                  {selectedAgent.name.charAt(0)}
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-red-700 truncate">{selectedAgent.name}</p>
                  <p className="text-xs text-red-400">{selectedAgent.group}</p>
                </div>
                <button
                  onClick={() => setSelectedAgent(null)}
                  className="text-red-300 hover:text-red-500 transition-colors"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
            )}

            <button
              onClick={() => setShowAgentModal(true)}
              disabled={uploading}
              className={`mt-4 w-full text-white font-semibold py-4 px-6 rounded-xl transition-all duration-200 ${
                uploading
                  ? "bg-gray-400 cursor-not-allowed"
                  : "bg-red-500 hover:bg-red-600"
              }`}
            >
              {uploading
                ? "Analyzing..."
                : `Analyze ${selectedFiles.length} Recording${selectedFiles.length !== 1 ? "s" : ""}`}
            </button>
          </div>
        )}

        {/* Results */}
        {uploadResults.length > 0 && (
          <div className="space-y-4">
            <h2 className="text-2xl font-bold text-gray-800 mb-4">
              Analysis Results
            </h2>

            {uploadResults.map((result, index) => (
              <div key={index} className="bg-white rounded-2xl p-6 shadow-lg">
                <div className="flex items-start justify-between mb-4">
                  <div className="flex-1">
                    <h3 className="text-lg font-semibold text-gray-800">
                      {result.filename}
                    </h3>
                  </div>
                  <span
                    className={`px-3 py-1 rounded-full text-sm font-medium ${
                      result.status === "completed"
                        ? "bg-green-100 text-green-800"
                        : result.status === "failed"
                          ? "bg-red-100 text-red-800"
                          : result.status === "processing"
                            ? "bg-blue-100 text-blue-800"
                            : "bg-gray-100 text-gray-800"
                    }`}
                  >
                    {result.status}
                  </span>
                </div>

                {/* Progress Bar */}
                {result.status === "processing" && (
                  <div className="mb-4">
                    <div className="flex justify-between text-sm text-gray-600 mb-2">
                      <span>Processing...</span>
                      <span>{result.progress}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-3">
                      <div
                        className="bg-red-500 h-3 rounded-full transition-all duration-500 animate-pulse"
                        style={{ width: `${result.progress}%` }}
                      />
                    </div>
                  </div>
                )}

                {/* Error */}
                {result.status === "failed" && (
                  <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                    <p className="text-red-800 font-medium">
                      Error: {result.error}
                    </p>
                  </div>
                )}

                {/* Completed Results */}
                {result.status === "completed" && result.data && (
                  <div className="space-y-4">
                    {/* ACTION REQUIRED - Top Priority Display */}
                    {result.data.csr_recommendations?.action_required && (
                      <div
                        className={`p-6 rounded-lg border-4 ${
                          result.data.csr_recommendations.action_required
                            .color === "red"
                            ? "bg-red-50 border-red-500"
                            : result.data.csr_recommendations.action_required
                                  .color === "orange"
                              ? "bg-orange-50 border-orange-500"
                              : result.data.csr_recommendations.action_required
                                    .color === "green"
                                ? "bg-green-50 border-green-500"
                                : "bg-yellow-50 border-yellow-500"
                        }`}
                      >
                        <div className="flex items-start gap-4">
                          <div className="flex-shrink-0 w-16 h-16 rounded-full bg-white shadow-md flex items-center justify-center">
                            <span className="text-3xl font-bold">
                              {result.data.csr_recommendations.action_required
                                .action === "ESCALATE"
                                ? "⚠️"
                                : result.data.csr_recommendations
                                      .action_required.action === "REST"
                                  ? "⏸️"
                                  : "✅"}
                            </span>
                          </div>
                          <div className="flex-1">
                            <h3
                              className={`text-2xl font-bold mb-2 ${
                                result.data.csr_recommendations.action_required
                                  .color === "red"
                                  ? "text-red-900"
                                  : result.data.csr_recommendations
                                        .action_required.color === "orange"
                                    ? "text-orange-900"
                                    : result.data.csr_recommendations
                                          .action_required.color === "green"
                                      ? "text-green-900"
                                      : "text-yellow-900"
                              }`}
                            >
                              ACTION: {
                                result.data.csr_recommendations.action_required
                                  .action
                              }
                            </h3>
                            <p
                              className={`font-medium mb-2 ${
                                result.data.csr_recommendations.action_required
                                  .color === "red"
                                  ? "text-red-800"
                                  : result.data.csr_recommendations
                                        .action_required.color === "orange"
                                    ? "text-orange-800"
                                    : result.data.csr_recommendations
                                          .action_required.color === "green"
                                      ? "text-green-800"
                                      : "text-yellow-800"
                              }`}
                            >
                              {
                                result.data.csr_recommendations.action_required
                                  .reason
                              }
                            </p>
                            <p
                              className={`text-sm ${
                                result.data.csr_recommendations.action_required
                                  .color === "red"
                                  ? "text-red-700"
                                  : result.data.csr_recommendations
                                        .action_required.color === "orange"
                                    ? "text-orange-700"
                                    : result.data.csr_recommendations
                                          .action_required.color === "green"
                                      ? "text-green-700"
                                      : "text-yellow-700"
                              }`}
                            >
                              📋{" "}
                              {
                                result.data.csr_recommendations.action_required
                                  .instruction
                              }
                            </p>
                            <div className="mt-3">
                              <span
                                className={`inline-block px-3 py-1 rounded-full text-xs font-bold ${
                                  result.data.csr_recommendations
                                    .action_required.urgency === "IMMEDIATE"
                                    ? "bg-red-900 text-white"
                                    : result.data.csr_recommendations
                                          .action_required.urgency === "HIGH"
                                      ? "bg-red-700 text-white"
                                      : result.data.csr_recommendations
                                            .action_required.urgency ===
                                          "MEDIUM"
                                        ? "bg-orange-700 text-white"
                                        : "bg-green-700 text-white"
                                }`}
                              >
                                URGENCY:{" "}
                                {
                                  result.data.csr_recommendations
                                    .action_required.urgency
                                }
                              </span>
                            </div>
                          </div>
                        </div>
                      </div>
                    )}

                    {/* Speaker Detection Info */}
                    {result.data.speaker_detection && (
                      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                        <h4 className="font-semibold text-blue-900 mb-2 flex items-center gap-2">
                          <span>🎙️</span>
                          <span>Speaker Detection</span>
                        </h4>
                        <div className="text-sm text-blue-800 space-y-1">
                          <p>
                            <strong>Mode:</strong> {result.data.speaker_detection.mode.toUpperCase()}
                          </p>
                          {result.data.speaker_detection.agent_channel && (
                            <>
                              <p>
                                <strong>Agent:</strong> {result.data.speaker_detection.agent_channel.toUpperCase()} channel
                              </p>
                              <p>
                                <strong>Caller:</strong> {result.data.speaker_detection.caller_channel?.toUpperCase()} channel
                              </p>
                            </>
                          )}
                          <p className="text-blue-700 mt-2">
                            ℹ️ {result.data.speaker_detection.note}
                          </p>
                        </div>
                      </div>
                    )}

                    {/* Emotion */}
                    {result.data.emotion_analysis && (
                      <div className="bg-gray-50 rounded-lg p-4">
                        <h4 className="font-semibold text-gray-700 mb-2">
                          Detected Emotion (Caller)
                        </h4>
                        <p
                          className={`text-2xl font-bold ${getEmotionColor(
                            result.data.emotion_analysis.predicted_emotion,
                          )}`}
                        >
                          {result.data.emotion_analysis.predicted_emotion?.toUpperCase()}
                        </p>
                        <p className="text-gray-600 text-sm mt-1">
                          Confidence:{" "}
                          {(
                            result.data.emotion_analysis.confidence * 100
                          ).toFixed(1)}
                          %
                        </p>

                        {/* All Probabilities */}
                        <div className="mt-3 space-y-1">
                          {result.data.emotion_analysis.all_probabilities &&
                            Object.entries(
                              result.data.emotion_analysis.all_probabilities,
                            ).map(([emotion, prob]: [string, number]) => (
                              <div
                                key={emotion}
                                className="flex items-center gap-2"
                              >
                                <div className="w-24 text-sm text-gray-600">
                                  {emotion}
                                </div>
                                <div className="flex-1 bg-gray-200 rounded-full h-2">
                                  <div
                                    className="bg-red-400 h-2 rounded-full"
                                    style={{ width: `${prob * 100}%` }}
                                  />
                                </div>
                                <div className="w-12 text-sm text-gray-600 text-right">
                                  {(prob * 100).toFixed(0)}%
                                </div>
                              </div>
                            ))}
                        </div>
                      </div>
                    )}

                    {/* Emotional State */}
                    {result.data.emotion_analysis?.emotional_state && (
                      <div className="bg-gray-50 rounded-lg p-4">
                        <h4 className="font-semibold text-gray-700 mb-3">
                          Emotional State Analysis
                        </h4>
                        <div className="grid grid-cols-2 gap-3">
                          <div>
                            <p className="text-xs text-gray-500">Valence</p>
                            <p className="font-medium">
                              {result.data.emotion_analysis.emotional_state
                                .valence || "N/A"}
                            </p>
                          </div>
                          <div>
                            <p className="text-xs text-gray-500">Arousal</p>
                            <p className="font-medium">
                              {result.data.emotion_analysis.emotional_state
                                .arousal || "N/A"}
                            </p>
                          </div>
                        </div>
                      </div>
                    )}

                    {/* CSR Recommendations */}
                    {result.data.csr_recommendations?.available !== false && (
                      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                        <h4 className="font-semibold text-blue-900 mb-3">
                          CSR Recommendations
                        </h4>

                        {/* Communication Guidance */}
                        {result.data.csr_recommendations
                          ?.communication_guidance && (
                          <div className="mb-3">
                            <p className="text-sm font-medium text-blue-800 mb-1">
                              Recommended Tone:
                            </p>
                            <p className="text-sm text-blue-700">
                              {
                                result.data.csr_recommendations
                                  .communication_guidance.recommended_tone
                              }
                            </p>
                          </div>
                        )}

                        {/* Example Phrases */}
                        {result.data.csr_recommendations?.communication_guidance
                          ?.example_phrases && (
                          <div className="mb-3">
                            <p className="text-sm font-medium text-blue-800 mb-1">
                              Suggested Phrases:
                            </p>
                            <ul className="space-y-1">
                              {result.data.csr_recommendations.communication_guidance.example_phrases.map(
                                (phrase: string, i: number) => (
                                  <li key={i} className="text-sm text-blue-700">
                                    • "{phrase}"
                                  </li>
                                ),
                              )}
                            </ul>
                          </div>
                        )}

                        {/* Do's */}
                        {result.data.csr_recommendations?.do_and_dont?.do && (
                          <div className="mb-3">
                            <p className="text-sm font-medium text-green-800 mb-1">
                              ✓ Do:
                            </p>
                            <ul className="space-y-1">
                              {result.data.csr_recommendations.do_and_dont.do.map(
                                (item: string, i: number) => (
                                  <li
                                    key={i}
                                    className="text-sm text-green-700"
                                  >
                                    • {item}
                                  </li>
                                ),
                              )}
                            </ul>
                          </div>
                        )}

                        {/* Don'ts */}
                        {result.data.csr_recommendations?.do_and_dont?.dont && (
                          <div>
                            <p className="text-sm font-medium text-red-800 mb-1">
                              ✗ Don't:
                            </p>
                            <ul className="space-y-1">
                              {result.data.csr_recommendations.do_and_dont.dont.map(
                                (item: string, i: number) => (
                                  <li key={i} className="text-sm text-red-700">
                                    • {item}
                                  </li>
                                ),
                              )}
                            </ul>
                          </div>
                        )}
                      </div>
                    )}

                    {/* Transcription Preview */}
                    {result.data.transcription && (
                      <div className="bg-gray-50 rounded-lg p-4">
                        <h4 className="font-semibold text-gray-700 mb-2">
                          Transcription
                        </h4>
                        <p className="text-gray-600 text-sm">
                          {result.data.transcription.text}
                        </p>
                        <div className="flex gap-4 mt-3 text-xs text-gray-500">
                          <span>
                            Duration: {result.data.transcription.duration}s
                          </span>
                          <span>
                            Language: {result.data.transcription.language}
                          </span>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
