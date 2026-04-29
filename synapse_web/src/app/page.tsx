"use client";

import { useState } from "react";
import ChatPanel from "@/components/ChatPanel";
import AgentsPanel from "@/components/AgentsPanel";
import MetricsPanel from "@/components/MetricsPanel";

type Tab = "chat" | "agents" | "metrics";

const TABS: { id: Tab; label: string; icon: string }[] = [
  { id: "chat", label: "Chat", icon: "C" },
  { id: "agents", label: "Agents", icon: "A" },
  { id: "metrics", label: "Metrics", icon: "M" },
];

export default function Home() {
  const [activeTab, setActiveTab] = useState<Tab>("chat");

  return (
    <div style={{ display: "flex", height: "100vh", overflow: "hidden" }}>
      {/* Sidebar */}
      <nav style={{
        width: 200,
        background: "var(--bg-secondary)",
        borderRight: "1px solid var(--border)",
        display: "flex",
        flexDirection: "column",
        padding: "16px 12px",
        flexShrink: 0,
      }}>
        {/* Logo */}
        <div style={{
          padding: "12px 16px",
          marginBottom: 24,
          display: "flex",
          alignItems: "center",
          gap: 10,
        }}>
          <div style={{
            width: 32,
            height: 32,
            borderRadius: 8,
            background: "var(--accent)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: 16,
            fontWeight: 700,
            color: "#fff",
          }}>S</div>
          <div>
            <div style={{ fontSize: 15, fontWeight: 600 }}>Synapse</div>
            <div style={{ fontSize: 10, color: "var(--text-muted)" }}>v0.1.0</div>
          </div>
        </div>

        {/* Nav Items */}
        {TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 10,
              padding: "10px 14px",
              marginBottom: 4,
              fontSize: 13,
              background: activeTab === tab.id ? "var(--accent-soft)" : "transparent",
              border: "none",
              borderRadius: 8,
              color: activeTab === tab.id ? "var(--accent)" : "var(--text-secondary)",
              cursor: "pointer",
              width: "100%",
              textAlign: "left",
              transition: "background 0.15s",
            }}
          >
            <span style={{
              width: 24,
              height: 24,
              borderRadius: 6,
              background: activeTab === tab.id ? "var(--accent)" : "var(--bg-tertiary)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: 12,
              fontWeight: 600,
              color: activeTab === tab.id ? "#fff" : "var(--text-muted)",
            }}>{tab.icon}</span>
            {tab.label}
          </button>
        ))}

        {/* Bottom Links */}
        <div style={{ marginTop: "auto", paddingTop: 16, borderTop: "1px solid var(--border)" }}>
          <a
            href="https://github.com/LongJie686/synapse"
            target="_blank"
            rel="noopener noreferrer"
            style={{
              display: "block",
              padding: "8px 14px",
              fontSize: 12,
              color: "var(--text-muted)",
              textDecoration: "none",
            }}
          >
            GitHub Repo
          </a>
          <a
            href="http://localhost:8000/docs"
            target="_blank"
            rel="noopener noreferrer"
            style={{
              display: "block",
              padding: "8px 14px",
              fontSize: 12,
              color: "var(--text-muted)",
              textDecoration: "none",
            }}
          >
            API Docs
          </a>
        </div>
      </nav>

      {/* Main Content */}
      <main style={{ flex: 1, overflow: "hidden" }}>
        {activeTab === "chat" && <ChatPanel />}
        {activeTab === "agents" && <AgentsPanel />}
        {activeTab === "metrics" && <MetricsPanel />}
      </main>
    </div>
  );
}
