"use client";

import { usePathname } from "next/navigation";

const pageTitles: Record<string, { title: string; description: string }> = {
  "/": { title: "Dashboard", description: "Overview of your CSR activity" },
  "/calls": {
    title: "Calls",
    description: "Manage and upload call recordings",
  },
  "/agents": {
    title: "Agents",
    description: "View and manage CSR agents",
  },
  "/clusters": {
    title: "Clusters",
    description: "Agent group clusters",
  },
  "/reports": {
    title: "Reports",
    description: "Emotional analysis reports",
  },
};

export default function Header() {
  const pathname = usePathname();

  const matchedKey = Object.keys(pageTitles)
    .filter((key) => (key === "/" ? pathname === "/" : pathname.startsWith(key)))
    .sort((a, b) => b.length - a.length)[0];

  const page = pageTitles[matchedKey] ?? {
    title: "whspr",
    description: "CSR Call Analysis",
  };

  return (
    <header className="h-16 bg-white border-b border-gray-200 flex items-center justify-between px-6">
      {/* Page title */}
      <div>
        <h1 className="text-lg font-semibold text-gray-800">{page.title}</h1>
        <p className="text-xs text-gray-400">{page.description}</p>
      </div>

      {/* Right actions */}
      <div className="flex items-center gap-3">
        {/* Search */}
        <button className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg transition-colors">
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
            />
          </svg>
        </button>

        {/* Notifications */}
        <button className="relative p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg transition-colors">
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9"
            />
          </svg>
          <span className="absolute top-1.5 right-1.5 w-2 h-2 bg-red-500 rounded-full" />
        </button>

        {/* Divider */}
        <div className="w-px h-6 bg-gray-200" />

        {/* Avatar */}
        <div className="w-8 h-8 rounded-full bg-red-500 flex items-center justify-center cursor-pointer">
          <svg className="w-4 h-4 text-white" fill="currentColor" viewBox="0 0 20 20">
            <path
              fillRule="evenodd"
              d="M10 9a3 3 0 100-6 3 3 0 000 6zm-7 9a7 7 0 1114 0H3z"
              clipRule="evenodd"
            />
          </svg>
        </div>
      </div>
    </header>
  );
}
