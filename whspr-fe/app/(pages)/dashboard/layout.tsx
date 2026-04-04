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
    const rawUser = localStorage.getItem("user");

    if (!rawUser) {
      router.replace("/login");
      return;
    }

    try {
      const parsedUser: AppUser = JSON.parse(rawUser);
      setUser(parsedUser);

      const role = parsedUser.role?.toLowerCase();

      const restrictedRoles = ["agent"];

      // 🔒 restrict routes
      if (role && restrictedRoles.includes(role)) {
        if (pathname !== "/dashboard/record") {
          router.replace("/dashboard/record");
          return;
        }
      }

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
      {/* ❌ Hide Sidebar if agent */}
      {!isAgent && <Sidebar />}

      <div className="flex min-w-0 flex-1 flex-col">
        <Header />
        <MainContainer>{children}</MainContainer>
      </div>
    </div>
  );
}
