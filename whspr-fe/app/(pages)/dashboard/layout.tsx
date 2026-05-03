"use client";

import Sidebar from "@/components/Sidebar";
import Header from "@/components/Header";
import MainContainer from "@/components/MainContainer";
import { useEffect, useState } from "react";
import { usePathname, useRouter } from "next/navigation";

type AppUser = {
  id?: number;
  agent_id?: number;
  username?: string;
  agent_name?: string;
  agent_email?: string;
  role?: string;
};

const ROLE_ACCESS: Record<string, string[]> = {
  admin: [
    "/dashboard",
    "/dashboard/agents",
    "/dashboard/flagged-agents",
    "/dashboard/reports",
    "/dashboard/register",
    "/dashboard/users",
  ],
  // calls and clusters are supervisor-only

  supervisor: [
    "/dashboard",
    "/dashboard/agents",
    "/dashboard/reports",
    "/dashboard/calls",
    "/dashboard/clusters",
  ],

  agent: ["/dashboard/record"],
};

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const router = useRouter();
  const pathname = usePathname();

  const [authorized, setAuthorized] = useState(false);
  const [user, setUser] = useState<AppUser | null>(null);

  useEffect(() => {
    setAuthorized(false);

    const rawUser = localStorage.getItem("user");

    if (!rawUser) {
      router.replace("/login");
      return;
    }

    try {
      const parsedUser: AppUser = JSON.parse(rawUser);
      const role = parsedUser.role?.toLowerCase() || "";

      const allowedRoutes = ROLE_ACCESS[role];

      if (!allowedRoutes) {
        localStorage.removeItem("user");
        router.replace("/login");
        return;
      }

      const isAllowed = allowedRoutes.some((route) => {
        if (route === "/dashboard") {
          return pathname === "/dashboard";
        }

        return pathname === route || pathname.startsWith(`${route}/`);
      });

      if (!isAllowed) {
        if (role === "agent") {
          router.replace("/dashboard/record");
        } else {
          router.replace("/dashboard");
        }

        return;
      }

      setUser(parsedUser);
      setAuthorized(true);
    } catch {
      localStorage.removeItem("user");
      router.replace("/login");
    }
  }, [pathname, router]);

  if (!authorized) return null;

  const role = user?.role?.toLowerCase();
  const isAgent = role === "agent";

  return (
    <div className="flex min-h-screen">
      {!isAgent && <Sidebar userRole={role} />}

      <div className="flex min-w-0 flex-1 flex-col">
        <Header />
        <MainContainer>{children}</MainContainer>
      </div>
    </div>
  );
}
