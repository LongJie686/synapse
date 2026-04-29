"use client";

import { useState, useCallback } from "react";
import ChatPanel, { Conversation } from "@/components/ChatPanel";
import AgentsPanel from "@/components/AgentsPanel";
import MetricsPanel from "@/components/MetricsPanel";
import { Lang, I18nKey, t } from "@/lib/i18n";

type Tab = "chat" | "agents" | "metrics";

const TAB_KEYS: { id: Tab; labelKey: I18nKey; icon: string }[] = [
  { id: "chat", labelKey: "chat", icon: "C" },
  { id: "agents", labelKey: "agents", icon: "A" },
  { id: "metrics", labelKey: "metrics", icon: "M" },
];

export default function Home() {
  const [activeTab, setActiveTab] = useState<Tab>("chat");
  const [lang, setLang] = useState<Lang>("zh");
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConvId, setActiveConvId] = useState<string | null>(null);

  const toggleLang = () => setLang((l) => (l === "zh" ? "en" : "zh"));

  const handleNewChat = useCallback(() => {
    const id = `conv-${Date.now()}`;
    setConversations((prev) => [{
      id,
      title: `${t(lang, "newChat")} ${prev.length + 1}`,
      agentId: "",
      agentName: "",
      messages: [],
    }, ...prev]);
    setActiveConvId(id);
  }, [lang]);

  const handleDeleteConv = useCallback((convId: string) => {
    setConversations((prev) => prev.filter((c) => c.id !== convId));
    if (activeConvId === convId) setActiveConvId(null);
  }, [activeConvId]);

  const handleUpdateConv = useCallback((convId: string, updater: (c: Conversation) => Conversation) => {
    setConversations((prev) => prev.map((c) => c.id === convId ? updater(c) : c));
  }, []);

  return (
    <div style={{ display: "flex", height: "100vh", overflow: "hidden" }}>
      {/* Sidebar */}
      <nav style={{
        width: 220,
        background: "var(--bg-secondary)",
        borderRight: "1px solid var(--border)",
        display: "flex",
        flexDirection: "column",
        flexShrink: 0,
      }}>
        {/* Logo */}
        <div style={{ padding: "16px 12px", marginBottom: 8, display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{
            width: 32, height: 32, borderRadius: 8, background: "var(--accent)",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 16, fontWeight: 700, color: "#fff",
          }}>S</div>
          <div>
            <div style={{ fontSize: 15, fontWeight: 600 }}>Synapse</div>
            <div style={{ fontSize: 10, color: "var(--text-muted)" }}>v0.1.0</div>
          </div>
        </div>

        {/* Nav tabs */}
        <div style={{ padding: "0 8px" }}>
          {TAB_KEYS.map((tab) => (
            <button key={tab.id} onClick={() => setActiveTab(tab.id)} style={{
              display: "flex", alignItems: "center", gap: 10, padding: "10px 12px",
              marginBottom: 2, fontSize: 13, width: "100%", textAlign: "left",
              background: activeTab === tab.id ? "var(--accent-soft)" : "transparent",
              border: "none", borderRadius: 8, cursor: "pointer",
              color: activeTab === tab.id ? "var(--accent)" : "var(--text-secondary)",
            }}>
              <span style={{
                width: 24, height: 24, borderRadius: 6,
                background: activeTab === tab.id ? "var(--accent)" : "var(--bg-tertiary)",
                display: "flex", alignItems: "center", justifyContent: "center",
                fontSize: 12, fontWeight: 600,
                color: activeTab === tab.id ? "#fff" : "var(--text-muted)",
              }}>{tab.icon}</span>
              {t(lang, tab.labelKey)}
            </button>
          ))}
        </div>

        {/* Conversation list under Chat tab */}
        {activeTab === "chat" && (
          <div style={{ flex: 1, overflowY: "auto", marginTop: 8, borderTop: "1px solid var(--border)", paddingTop: 8, paddingLeft: 8, paddingRight: 8 }}>
            <div style={{ padding: "4px 12px 8px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span style={{ fontSize: 11, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: 0.5 }}>{t(lang, "conversations")}</span>
              <button onClick={handleNewChat} style={{
                padding: "2px 8px", fontSize: 11, background: "transparent",
                border: "1px dashed var(--border)", borderRadius: 4, color: "var(--text-muted)",
                cursor: "pointer",
              }}>+</button>
            </div>
            {conversations.length === 0 && (
              <div style={{ padding: "8px 12px", fontSize: 12, color: "var(--text-muted)" }}>{t(lang, "noConversations")}</div>
            )}
            {conversations.map((conv) => (
              <div key={conv.id} style={{
                display: "flex", alignItems: "center", padding: "0 8px", marginBottom: 1,
                background: activeConvId === conv.id ? "var(--accent-soft)" : "transparent",
                borderRadius: 6,
              }}>
                <button onClick={() => setActiveConvId(conv.id)} style={{
                  flex: 1, padding: "8px 8px", fontSize: 12, textAlign: "left",
                  background: "transparent", border: "none", cursor: "pointer",
                  color: activeConvId === conv.id ? "var(--accent)" : "var(--text-secondary)",
                  whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis",
                }}>
                  {conv.title}
                  <span style={{ fontSize: 10, marginLeft: 4, opacity: 0.6 }}>({conv.messages.length})</span>
                </button>
                <button onClick={() => handleDeleteConv(conv.id)} style={{
                  padding: "4px 6px", fontSize: 10, background: "transparent",
                  border: "none", color: "var(--text-muted)", cursor: "pointer", opacity: 0.5,
                }}>x</button>
              </div>
            ))}
          </div>
        )}

        {/* Bottom links */}
        <div style={{ marginTop: "auto", padding: "12px 8px", borderTop: "1px solid var(--border)" }}>
          <button onClick={toggleLang} style={{
            display: "block", width: "100%", padding: "8px 12px", fontSize: 12,
            color: "var(--text-secondary)", background: "transparent", border: "none",
            cursor: "pointer", textAlign: "left", borderRadius: 6,
          }}>
            {lang === "zh" ? "English" : "中文"}
          </button>
          <a href="https://github.com/LongJie686/synapse" target="_blank" rel="noopener noreferrer"
            style={{ display: "block", padding: "8px 12px", fontSize: 12, color: "var(--text-muted)", textDecoration: "none", borderRadius: 6 }}>
            {t(lang, "githubRepo")}
          </a>
          <a href="http://localhost:8000/docs" target="_blank" rel="noopener noreferrer"
            style={{ display: "block", padding: "8px 12px", fontSize: 12, color: "var(--text-muted)", textDecoration: "none", borderRadius: 6 }}>
            {t(lang, "apiDocs")}
          </a>
        </div>
      </nav>

      <main style={{ flex: 1, overflow: "hidden" }}>
        {activeTab === "chat" && (
          <ChatPanel
            lang={lang}
            conversations={conversations}
            activeConvId={activeConvId}
            onUpdateConv={handleUpdateConv}
            onNewChat={handleNewChat}
          />
        )}
        {activeTab === "agents" && <AgentsPanel lang={lang} />}
        {activeTab === "metrics" && <MetricsPanel lang={lang} />}
      </main>
    </div>
  );
}
