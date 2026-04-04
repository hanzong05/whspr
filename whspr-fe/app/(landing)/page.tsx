"use client";

import { useState } from "react";
import Header from "@/components/Header";
function NavBar() {
  return (
    <nav className="sticky top-0 z-50 bg-white border-b border-gray-100 px-10 h-16 flex items-center justify-between">
      <div className="flex items-center gap-2.5 font-serif text-xl tracking-tight text-gray-900">
        <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
        CallSense
      </div>
      <div className="flex items-center gap-8">
        {["Features", "How it works", "Pricing", "Docs"].map((link) => (
          <a
            key={link}
            href="#"
            className="text-sm text-gray-500 hover:text-gray-900 transition-colors"
          >
            {link}
          </a>
        ))}
        <a
          href="#"
          className="text-sm font-medium bg-gray-900 text-white px-4 py-2 rounded-lg hover:bg-red-500 transition-colors"
        >
          Get started
        </a>
      </div>
    </nav>
  );
}

function MiniDashboard() {
  const risks = [
    { name: "Maria Santos", score: 88 },
    { name: "Juan dela Cruz", score: 74 },
    { name: "Ana Reyes", score: 61 },
  ];

  return (
    <div className="bg-white border border-gray-200 rounded-2xl p-5 space-y-3 shadow-sm">
      {/* Mini header */}
      <div className="flex items-center justify-between pb-3 border-b border-gray-100">
        <div className="flex items-center gap-2 text-sm font-semibold text-gray-900">
          <span className="w-1.5 h-1.5 rounded-full bg-red-500" />
          CallSense
        </div>
        <div className="flex gap-4 text-xs text-gray-400">
          <span className="text-gray-900 font-medium border-b border-red-500 pb-0.5">
            Dashboard
          </span>
          <span>Reports</span>
          <span>Agents</span>
        </div>
      </div>

      {/* Stat cards */}
      <div className="grid grid-cols-3 gap-2">
        <div className="bg-white border border-gray-100 rounded-xl p-3 shadow-sm">
          <p className="text-[10px] text-gray-400 mb-1">Total agents</p>
          <p className="text-xl font-semibold text-gray-900">148</p>
        </div>
        <div className="bg-red-500 rounded-xl p-3">
          <p className="text-[10px] text-red-100 mb-1">High load</p>
          <p className="text-xl font-semibold text-white">23%</p>
        </div>
        <div className="bg-white border border-gray-100 rounded-xl p-3 shadow-sm">
          <p className="text-[10px] text-gray-400 mb-1">Total calls</p>
          <p className="text-xl font-semibold text-gray-900">4,821</p>
        </div>
      </div>

      {/* Sparkline */}
      <div className="bg-white border border-gray-100 rounded-xl p-3 shadow-sm">
        <p className="text-[10px] text-gray-400 mb-2">Emotional trend line</p>
        <svg viewBox="0 0 280 36" fill="none" className="w-full h-10">
          <polyline
            points="0,28 40,22 80,26 120,14 160,18 200,10 240,16 280,8"
            stroke="#ef4444"
            strokeWidth="1.5"
            fill="none"
          />
          <polyline
            points="0,30 40,28 80,24 120,26 160,20 200,22 240,18 280,20"
            stroke="#378ADD"
            strokeWidth="1.5"
            fill="none"
          />
          <polyline
            points="0,32 40,30 80,28 120,30 160,28 200,26 240,24 280,26"
            stroke="#1D9E75"
            strokeWidth="1.5"
            fill="none"
          />
        </svg>
      </div>

      {/* Risk list */}
      <div className="bg-white border border-gray-100 rounded-xl p-3 shadow-sm">
        <p className="text-[10px] text-gray-400 mb-2">Top risk agents</p>
        <div className="space-y-2">
          {risks.map((a, i) => (
            <div key={a.name} className="flex items-center gap-2">
              <span className="w-4 h-4 rounded-full bg-red-50 text-red-600 text-[9px] font-bold flex items-center justify-center flex-shrink-0">
                {i + 1}
              </span>
              <span className="text-xs text-gray-700 flex-1 truncate">
                {a.name}
              </span>
              <div className="w-14 h-1.5 bg-gray-100 rounded-full overflow-hidden">
                <div
                  className="h-full bg-red-400 rounded-full"
                  style={{ width: `${a.score}%` }}
                />
              </div>
              <span className="text-[10px] text-gray-400 w-6 text-right">
                {a.score}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

const features = [
  {
    title: "Risk scoring",
    desc: "Every agent receives a real-time risk score based on sentiment patterns, escalation history, and call behavior.",
    icon: (
      <svg
        className="w-5 h-5 text-red-500"
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
        strokeWidth={1.8}
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
        />
      </svg>
    ),
  },
  {
    title: "Emotional trend analysis",
    desc: "Track agent emotional states over time across clusters — spot burnout signals before they become a problem.",
    icon: (
      <svg
        className="w-5 h-5 text-red-500"
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
        strokeWidth={1.8}
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"
        />
      </svg>
    ),
  },
  {
    title: "Cluster management",
    desc: "Organize agents into groups and compare call volumes, risk distribution, and daily activity side by side.",
    icon: (
      <svg
        className="w-5 h-5 text-red-500"
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
        strokeWidth={1.8}
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0"
        />
      </svg>
    ),
  },
  {
    title: "Escalation alerts",
    desc: "Critical and high-risk calls surface automatically so supervisors can intervene in real time, not after the fact.",
    icon: (
      <svg
        className="w-5 h-5 text-red-500"
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
        strokeWidth={1.8}
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9"
        />
      </svg>
    ),
  },
  {
    title: "Summary reports",
    desc: "Daily and monthly breakdowns of call volumes, agent performance, and escalation trends in one clean view.",
    icon: (
      <svg
        className="w-5 h-5 text-red-500"
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
        strokeWidth={1.8}
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
        />
      </svg>
    ),
  },
  {
    title: "API-first backend",
    desc: "Connect CallSense to your existing CRM, ticketing, or telephony platform via a clean REST API.",
    icon: (
      <svg
        className="w-5 h-5 text-red-500"
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
        strokeWidth={1.8}
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4"
        />
      </svg>
    ),
  },
];

const steps = [
  {
    title: "Calls are recorded and ingested",
    desc: "CallSense connects to your telephony system and processes every call automatically, no manual uploads needed.",
  },
  {
    title: "AI analysis runs on every transcript",
    desc: "Sentiment, tone, and compliance patterns are extracted and a risk level is assigned: Low, Medium, High, or Critical.",
  },
  {
    title: "Supervisors see results in real time",
    desc: "The dashboard surfaces flagged calls, trending agents, and cluster-level summaries the moment data is ready.",
  },
  {
    title: "Teams act on data, not hunches",
    desc: "Coaches use risk scores and trend lines to prioritize 1:1s, adjust workflows, and reduce churn at the source.",
  },
];

const metrics = [
  {
    value: "40",
    suffix: "%",
    desc: "Reduction in escalation rate within 60 days",
  },
  {
    value: "2",
    suffix: "x",
    desc: "Faster supervisor response to at-risk calls",
  },
  {
    value: "98",
    suffix: "%",
    desc: "Accuracy on risk classification across clusters",
  },
  {
    value: "5K",
    suffix: "+",
    desc: "Calls analyzed daily across enterprise deployments",
  },
];

export default function LandingPage() {
  const [activeStep, setActiveStep] = useState(0);

  return (
    <div className="bg-white min-h-screen">
      <Header />
      {/* HERO */}
      <section className="max-w-6xl mx-auto px-10 py-20 grid grid-cols-2 gap-16 items-center">
        <div>
          <div className="inline-flex items-center gap-2 text-xs font-medium tracking-widest uppercase text-red-700 bg-red-50 px-3 py-1.5 rounded-full mb-6">
            <span className="w-1.5 h-1.5 rounded-full bg-red-500 animate-pulse" />
            AI-powered call monitoring
          </div>
          <h1 className="text-5xl font-serif leading-tight tracking-tight text-gray-900 mb-5">
            Hear what <em className="italic text-red-500">matters</em> in every
            call
          </h1>
          <p className="text-base text-gray-500 leading-relaxed mb-8 max-w-md">
            Real-time risk scoring, emotional trend analysis, and agent
            performance insights — so your supervisors can act before issues
            escalate.
          </p>
          <div className="flex items-center gap-4">
            <button className="bg-red-500 hover:bg-red-600 text-white text-sm font-medium px-6 py-3 rounded-lg transition-colors">
              Request a demo
            </button>
            <button className="border border-gray-200 text-gray-600 hover:border-gray-400 hover:text-gray-900 text-sm px-5 py-3 rounded-lg transition-colors">
              See the dashboard
            </button>
          </div>
        </div>
        <MiniDashboard />
      </section>

      {/* METRICS BAND */}
      <div className="max-w-6xl mx-auto px-10 mb-20">
        <div className="bg-gray-900 rounded-2xl py-10 px-12 grid grid-cols-4 gap-8">
          {metrics.map((m) => (
            <div key={m.value} className="text-center">
              <p className="text-4xl font-serif text-white mb-1">
                {m.value}
                <span className="text-red-400">{m.suffix}</span>
              </p>
              <p className="text-xs text-gray-400 leading-relaxed">{m.desc}</p>
            </div>
          ))}
        </div>
      </div>

      {/* FEATURES */}
      <section className="max-w-6xl mx-auto px-10 pb-20">
        <p className="text-xs font-medium tracking-widest uppercase text-red-700 mb-2">
          Core capabilities
        </p>
        <h2 className="text-4xl font-serif tracking-tight text-gray-900 mb-3">
          Everything your QA team needs
        </h2>
        <p className="text-base text-gray-500 leading-relaxed max-w-lg mb-10">
          Built for call centers managing high-volume, high-stakes conversations
          across distributed agent clusters.
        </p>
        <div className="grid grid-cols-3 gap-5">
          {features.map((f) => (
            <div
              key={f.title}
              className="border border-gray-100 rounded-2xl p-6 bg-white hover:border-red-200 hover:shadow-sm transition-all"
            >
              <div className="w-9 h-9 rounded-lg bg-red-50 flex items-center justify-center mb-4">
                {f.icon}
              </div>
              <h3 className="text-base font-semibold text-gray-900 mb-2">
                {f.title}
              </h3>
              <p className="text-sm text-gray-500 leading-relaxed">{f.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* HOW IT WORKS */}
      <section className="max-w-6xl mx-auto px-10 pb-20">
        <p className="text-xs font-medium tracking-widest uppercase text-red-700 mb-2">
          How it works
        </p>
        <h2 className="text-4xl font-serif tracking-tight text-gray-900 mb-10">
          From raw audio to actionable insight
        </h2>
        <div className="grid grid-cols-2 gap-16 items-center">
          <div className="space-y-3">
            {steps.map((s, i) => (
              <div
                key={s.title}
                onClick={() => setActiveStep(i)}
                className={`flex gap-4 items-start p-4 rounded-xl cursor-pointer transition-colors ${
                  activeStep === i ? "bg-red-50" : "hover:bg-gray-50"
                }`}
              >
                <span
                  className={`w-7 h-7 rounded-full flex items-center justify-center text-xs font-semibold flex-shrink-0 transition-colors ${
                    activeStep === i
                      ? "bg-red-500 text-white"
                      : "bg-gray-100 text-gray-500"
                  }`}
                >
                  {i + 1}
                </span>
                <div>
                  <p className="text-sm font-semibold text-gray-900 mb-1">
                    {s.title}
                  </p>
                  <p className="text-sm text-gray-500 leading-relaxed">
                    {s.desc}
                  </p>
                </div>
              </div>
            ))}
          </div>

          {/* Pulse visual */}
          <div className="bg-white border border-gray-100 rounded-2xl flex items-center justify-center min-h-72 shadow-sm">
            <div className="relative flex items-center justify-center">
              {[100, 150, 200].map((size, i) => (
                <span
                  key={size}
                  className="absolute rounded-full border border-red-300 animate-ping opacity-20"
                  style={{
                    width: size,
                    height: size,
                    animationDelay: `${i * 0.6}s`,
                    animationDuration: "2.5s",
                  }}
                />
              ))}
              <div className="w-16 h-16 bg-red-500 rounded-full flex items-center justify-center relative z-10">
                <svg
                  className="w-7 h-7 text-white"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={1.8}
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z"
                  />
                </svg>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="max-w-6xl mx-auto px-10 pb-20">
        <div className="bg-white border border-gray-100 rounded-2xl p-16 text-center shadow-sm relative overflow-hidden">
          <div className="absolute -top-16 -right-16 w-48 h-48 bg-red-50 rounded-full opacity-60" />
          <div className="absolute -bottom-10 -left-10 w-36 h-36 bg-gray-50 rounded-full opacity-80" />
          <div className="relative z-10">
            <h2 className="text-4xl font-serif tracking-tight text-gray-900 mb-4">
              Ready to make every call count?
            </h2>
            <p className="text-base text-gray-500 leading-relaxed max-w-md mx-auto mb-8">
              Join contact centers already using CallSense to reduce
              escalations, protect agents, and serve customers better.
            </p>
            <div className="flex items-center justify-center gap-4">
              <button className="bg-red-500 hover:bg-red-600 text-white text-sm font-medium px-6 py-3 rounded-lg transition-colors">
                Book a demo
              </button>
              <button className="border border-gray-200 text-gray-600 hover:border-gray-400 hover:text-gray-900 text-sm px-5 py-3 rounded-lg transition-colors">
                Talk to sales
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* FOOTER */}
      <footer className="border-t border-gray-100 py-6 px-10 max-w-6xl mx-auto flex items-center justify-between">
        <div className="flex items-center gap-2 text-base font-semibold text-gray-900">
          <span className="w-1.5 h-1.5 rounded-full bg-red-500" />
          CallSense
        </div>
        <p className="text-xs text-gray-400">
          © 2025 CallSense. All rights reserved.
        </p>
      </footer>
    </div>
  );
}
