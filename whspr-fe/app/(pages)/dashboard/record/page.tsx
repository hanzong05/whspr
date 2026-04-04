"use client";

import { useState, useEffect, useRef, useCallback } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// ── Types ─────────────────────────────────────────────────────────────────────
type AnalysisData = {
  [key: string]: unknown;
};
type RecordingState = "idle" | "recording" | "paused" | "stopped";
type SessionUser = {
  id?: number;
  agent_id?: number;
  username?: string;
  agent_name?: string;
  agent_email?: string;
  role?: string;
  cluster_id?: number;
  cluster_name?: string;
  is_active?: boolean;
  last_login_at?: string;
};

// ── Helpers ───────────────────────────────────────────────────────────────────

function formatTime(seconds: number) {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  if (h > 0)
    return `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

// ── Waveform bars (purely visual, driven by analyser) ─────────────────────────

function Waveform({
  active,
  analyser,
}: {
  active: boolean;
  analyser: AnalyserNode | null;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number>(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const BAR_COUNT = 48;
    const BAR_GAP = 3;

    const draw = () => {
      const W = canvas.width;
      const H = canvas.height;
      ctx.clearRect(0, 0, W, H);

      let amplitudes: number[];

      if (active && analyser) {
        const data = new Uint8Array(analyser.frequencyBinCount);
        analyser.getByteFrequencyData(data);
        const step = Math.floor(data.length / BAR_COUNT);
        amplitudes = Array.from(
          { length: BAR_COUNT },
          (_, i) => data[i * step] / 255,
        );
      } else {
        const t = Date.now() / 1000;
        amplitudes = Array.from({ length: BAR_COUNT }, (_, i) =>
          active ? 0 : 0.05 + 0.05 * Math.sin(t * 1.5 + i * 0.4),
        );
      }

      const barW = (W - BAR_GAP * (BAR_COUNT - 1)) / BAR_COUNT;

      amplitudes.forEach((amp, i) => {
        const barH = Math.max(4, amp * H * 0.9);
        const x = i * (barW + BAR_GAP);
        const y = (H - barH) / 2;

        const gradient = ctx.createLinearGradient(0, y, 0, y + barH);
        if (active) {
          gradient.addColorStop(0, "rgba(239,68,68,0.9)");
          gradient.addColorStop(1, "rgba(252,165,165,0.5)");
        } else {
          gradient.addColorStop(0, "rgba(209,213,219,0.8)");
          gradient.addColorStop(1, "rgba(229,231,235,0.4)");
        }

        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.roundRect(x, y, barW, barH, barW / 2);
        ctx.fill();
      });

      rafRef.current = requestAnimationFrame(draw);
    };

    rafRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(rafRef.current);
  }, [active, analyser]);

  return (
    <canvas
      ref={canvasRef}
      width={480}
      height={80}
      className="w-full max-w-lg h-20"
    />
  );
}

// ── Page ──────────────────────────────────────────────────────────────────────

export default function RecordingPage() {
  const [analysisLoading, setAnalysisLoading] = useState(false);
  const [analysisData, setAnalysisData] = useState<AnalysisData | null>(null);
  const [sessionUser, setSessionUser] = useState<SessionUser | null>(null);
  const [state, setState] = useState<RecordingState>("idle");
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [analyser, setAnalyser] = useState<AnalyserNode | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const elapsedAtPauseRef = useRef(0);

  useEffect(() => {
    try {
      const raw = localStorage.getItem("user");
      if (raw) setSessionUser(JSON.parse(raw));
    } catch (err) {
      console.error("Failed to parse user session:", err);
    }
    return () => {
      stopAll();
    };
  }, []);

  // ── Analysis ──────────────────────────────────────────────────────────────

  const analyzeRecordingInBackground = useCallback(
    async (audioBlob: Blob, duration: number) => {
      try {
        setAnalysisLoading(true);

        const userRaw = localStorage.getItem("user");
        const user: SessionUser | null = userRaw ? JSON.parse(userRaw) : null;

        const file = new File([audioBlob], `recording-${Date.now()}.webm`, {
          type: "audio/webm",
        });

        const formData = new FormData();
        formData.append("file", file);

        // Only append agent_id when it's a real value — sending an empty string
        // makes FastAPI fail to coerce it to Optional[int] and the DB save is skipped
        if (user?.agent_id) {
          formData.append("agent_id", String(user.agent_id));
        }

        formData.append("duration", String(duration));

        const response = await fetch(`${API_URL}/analyze`, {
          method: "POST",
          body: formData,
        });

        if (!response.ok) throw new Error("Analysis failed");

        const result = await response.json();
        setAnalysisData(result);
        console.log("✅ Background analysis done:", result);
      } catch (err) {
        console.error("❌ Background analysis failed:", err);
      } finally {
        setAnalysisLoading(false);
      }
    },
    [],
  );

  // ── Timer ─────────────────────────────────────────────────────────────────

  const startTimer = useCallback(() => {
    if (timerRef.current) clearInterval(timerRef.current);
    timerRef.current = setInterval(() => setElapsed((s) => s + 1), 1000);
  }, []);

  const stopTimer = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  // ── Core recording helpers ────────────────────────────────────────────────

  const stopAll = () => {
    stopTimer();
    if (
      mediaRecorderRef.current &&
      mediaRecorderRef.current.state !== "inactive"
    )
      mediaRecorderRef.current.stop();
    if (streamRef.current)
      streamRef.current.getTracks().forEach((t) => t.stop());
    if (audioCtxRef.current) {
      audioCtxRef.current.close();
      audioCtxRef.current = null;
    }
  };

  const startRecording = async () => {
    setError(null);
    setElapsed(0);
    chunksRef.current = [];

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const ctx = new AudioContext();
      audioCtxRef.current = ctx;
      const source = ctx.createMediaStreamSource(stream);
      const analyserNode = ctx.createAnalyser();
      analyserNode.fftSize = 256;
      source.connect(analyserNode);
      setAnalyser(analyserNode);

      const mr = new MediaRecorder(stream);
      mediaRecorderRef.current = mr;

      mr.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      mr.onstop = async () => {
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        const finalDuration = elapsedAtPauseRef.current || elapsed;
        try {
          await analyzeRecordingInBackground(blob, finalDuration);
        } catch (err) {
          console.error("Auto-analysis error:", err);
        }
      };

      mr.start();
      setState("recording");
      startTimer();
    } catch (err: unknown) {
      const msg =
        err instanceof Error ? err.message : "Microphone access denied";
      setError(msg);
      setState("idle");
    }
  };

  const pauseRecording = () => {
    if (mediaRecorderRef.current?.state === "recording") {
      mediaRecorderRef.current.pause();
      stopTimer();
      elapsedAtPauseRef.current = elapsed;
      setState("paused");
    }
  };

  const resumeRecording = () => {
    if (mediaRecorderRef.current?.state === "paused") {
      mediaRecorderRef.current.resume();
      startTimer();
      setState("recording");
    }
  };

  const stopRecording = () => {
    elapsedAtPauseRef.current = elapsed;
    stopAll();
    setAnalyser(null);
    setState("stopped");
  };

  const restartRecording = () => {
    stopAll();
    setAnalyser(null);
    setState("idle");
    setElapsed(0);
    elapsedAtPauseRef.current = 0;
    startRecording();
  };

  // ── Derived ───────────────────────────────────────────────────────────────

  const isRecording = state === "recording";
  const isPaused = state === "paused";
  const isStopped = state === "stopped";

  return (
    <div className="min-h-[60vh] flex flex-col items-center justify-center gap-8 py-10">
      <div className="w-full max-w-xl bg-white rounded-3xl shadow-sm border border-gray-100 overflow-hidden">
        {/* Status bar */}
        <div
          className={`h-1.5 w-full transition-all duration-500 ${
            isRecording
              ? "bg-red-500 animate-pulse"
              : isPaused
                ? "bg-yellow-400"
                : isStopped
                  ? "bg-green-400"
                  : "bg-gray-200"
          }`}
        />

        <div className="p-8 flex flex-col items-center gap-6">
          {/* Pulsing circle + mic icon */}
          <div className="relative flex items-center justify-center">
            {isRecording && (
              <>
                <span className="absolute w-28 h-28 rounded-full bg-red-100 animate-ping opacity-60" />
                <span className="absolute w-20 h-20 rounded-full bg-red-200 animate-ping opacity-40 animation-delay-150" />
              </>
            )}
            <div
              className={`relative w-20 h-20 rounded-full flex items-center justify-center shadow-lg transition-all duration-300 ${
                isRecording
                  ? "bg-red-500 shadow-red-200"
                  : isPaused
                    ? "bg-yellow-400 shadow-yellow-200"
                    : isStopped
                      ? "bg-green-500 shadow-green-200"
                      : "bg-gray-100"
              }`}
            >
              {isStopped ? (
                <svg
                  className="w-9 h-9 text-white"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M5 13l4 4L19 7"
                  />
                </svg>
              ) : (
                <svg
                  className={`w-9 h-9 ${isRecording || isPaused ? "text-white" : "text-gray-400"}`}
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
              )}
            </div>
          </div>

          {/* Status label */}
          <div className="text-center">
            <p
              className={`text-sm font-semibold uppercase tracking-widest ${
                isRecording
                  ? "text-red-500"
                  : isPaused
                    ? "text-yellow-500"
                    : isStopped
                      ? "text-green-600"
                      : "text-gray-400"
              }`}
            >
              {isRecording
                ? "● Recording"
                : isPaused
                  ? "⏸ Paused"
                  : isStopped
                    ? "✓ Saved"
                    : "Ready"}
            </p>
            {!isStopped && (
              <p className="text-4xl font-mono font-bold text-gray-800 mt-1 tabular-nums">
                {formatTime(elapsed)}
              </p>
            )}
            {isStopped && (
              <p className="text-lg text-gray-500 mt-1">
                Recording saved · {formatTime(elapsedAtPauseRef.current)}
              </p>
            )}
          </div>

          {/* Waveform */}
          <Waveform active={isRecording} analyser={analyser} />

          {/* Analysis loading indicator */}
          {analysisLoading && (
            <div className="flex items-center gap-2 text-sm text-gray-500">
              <svg
                className="w-4 h-4 animate-spin text-red-500"
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
                  d="M4 12a8 8 0 018-8v8z"
                />
              </svg>
              Analyzing recording…
            </div>
          )}

          {/* Error */}
          {error && (
            <div className="w-full flex items-center gap-2 px-4 py-3 bg-red-50 border border-red-100 rounded-xl text-sm text-red-600">
              <svg
                className="w-4 h-4 flex-shrink-0"
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

          {/* Controls */}
          <div className="flex items-center gap-3">
            {isRecording && (
              <>
                <button
                  onClick={pauseRecording}
                  className="flex items-center gap-2 px-5 py-2.5 bg-yellow-50 hover:bg-yellow-100 text-yellow-600 font-semibold text-sm rounded-xl border border-yellow-200 transition-colors"
                >
                  <svg
                    className="w-4 h-4"
                    fill="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z" />
                  </svg>
                  Pause
                </button>
                <button
                  onClick={stopRecording}
                  className="flex items-center gap-2 px-5 py-2.5 bg-gray-800 hover:bg-gray-900 text-white font-semibold text-sm rounded-xl transition-colors"
                >
                  <svg
                    className="w-4 h-4"
                    fill="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path d="M6 6h12v12H6z" />
                  </svg>
                  Stop
                </button>
              </>
            )}
            {isPaused && (
              <>
                <button
                  onClick={resumeRecording}
                  className="flex items-center gap-2 px-5 py-2.5 bg-red-500 hover:bg-red-600 text-white font-semibold text-sm rounded-xl transition-colors shadow-sm"
                >
                  <svg
                    className="w-4 h-4"
                    fill="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path d="M8 5v14l11-7z" />
                  </svg>
                  Resume
                </button>
                <button
                  onClick={stopRecording}
                  className="flex items-center gap-2 px-5 py-2.5 bg-gray-800 hover:bg-gray-900 text-white font-semibold text-sm rounded-xl transition-colors"
                >
                  <svg
                    className="w-4 h-4"
                    fill="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path d="M6 6h12v12H6z" />
                  </svg>
                  Stop
                </button>
              </>
            )}
            {(isStopped || state === "idle") && (
              <button
                onClick={restartRecording}
                className="flex items-center gap-2 px-6 py-2.5 bg-red-500 hover:bg-red-600 text-white font-semibold text-sm rounded-xl transition-colors shadow-sm"
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
                    d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"
                  />
                </svg>
                New Recording
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
