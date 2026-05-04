"use client";

import { useState, useRef, useEffect } from "react";
import { useStream, StreamMessage, TitleUpdate, AttachedFile } from "@/hooks/useStream";
import { AgentInfo, fetchAgents, uploadFile } from "@/lib/api";
import { Lang, t, agentName, agentRole } from "@/lib/i18n";
import { useToast } from "@/components/Toast";

export interface Conversation {
  id: string;
  title: string;
  agentId: string;
  agentName: string;
  messages: StreamMessage[];
  loaded?: boolean;
}

interface Props {
  lang: Lang;
  conversations: Conversation[];
  activeConvId: string | null;
  onUpdateConv: (convId: string, updater: (c: Conversation) => Conversation) => void;
  onNewChat: () => Promise<string>;
  onTitleUpdate: (update: TitleUpdate) => void;
}

export default function ChatPanel({ lang, conversations, activeConvId, onUpdateConv, onNewChat, onTitleUpdate }: Props) {
  const { messages, isStreaming, sendMessage, stop, setMessages } = useStream();
  const [input, setInput] = useState("");
  const [agents, setAgents] = useState<AgentInfo[]>([]);
  const [selectedAgentId, setSelectedAgentId] = useState("");
  const [pendingAttachments, setPendingAttachments] = useState<AttachedFile[]>([]);
  const { toast } = useToast();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetchAgents()
      .then((data) => { setAgents(data); if (data.length > 0) setSelectedAgentId(data[0].id); })
      .catch(() => { /* agents unavailable */ });
  }, []);

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);

  // Load messages when switching conversation
  useEffect(() => {
    if (!activeConvId) {
      setMessages([]);
      return;
    }
    const conv = conversations.find((c) => c.id === activeConvId);
    if (!conv) return;

    if (conv.messages.length > 0) {
      setMessages(conv.messages);
      return;
    }

    if (!conv.loaded) {
      fetch(`/api/sessions/${activeConvId}`)
        .then((res) => res.json())
        .then((data) => {
          if (data.messages && data.messages.length > 0) {
            const streamMsgs: StreamMessage[] = data.messages.map((m: { role: string; content: string }) => ({
              role: m.role as "user" | "assistant",
              content: m.content,
            }));
            setMessages(streamMsgs);
            onUpdateConv(activeConvId, (c) => ({ ...c, messages: streamMsgs, loaded: true }));
          } else {
            setMessages([]);
            onUpdateConv(activeConvId, (c) => ({ ...c, messages: [], loaded: true }));
          }
        })
        .catch(() => {
          setMessages([]);
          onUpdateConv(activeConvId, (c) => ({ ...c, messages: [], loaded: true }));
        });
    } else {
      setMessages(conv.messages);
    }
  }, [activeConvId]);

  useEffect(() => {
    if (activeConvId && messages.length > 0) {
      onUpdateConv(activeConvId, (c) => ({ ...c, messages: [...messages] }));
    }
  }, [messages, activeConvId, onUpdateConv]);

  const handleSend = async () => {
    const text = input.trim();
    if ((!text && pendingAttachments.length === 0) || isStreaming) return;
    let sessionId = activeConvId;
    if (!sessionId) {
      sessionId = await onNewChat();
    }
    setInput("");
    const attachments = [...pendingAttachments];
    setPendingAttachments([]);
    sendMessage(text || "(attachment)", selectedAgentId, sessionId || undefined, onTitleUpdate, attachments.length > 0 ? attachments : undefined);
  };

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files) return;
    for (const file of Array.from(files)) {
      try {
        const result = await uploadFile(file);
        let base64Data: string | undefined;
        let mimeType: string | undefined;

        // If it's an image, read base64 for LLM vision
        if (result.is_image) {
          const arrayBuf = await file.arrayBuffer();
          const bytes = new Uint8Array(arrayBuf);
          let binary = "";
          for (let i = 0; i < bytes.length; i++) {
            binary += String.fromCharCode(bytes[i]);
          }
          base64Data = btoa(binary);
          mimeType = result.mime_type;
        }

        setPendingAttachments((prev) => [...prev, {
          filename: result.filename,
          url: (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000") + result.url,
          isImage: result.is_image,
          size: result.size,
          base64Data,
          mimeType,
        }]);
      } catch {
        toast(t(lang, "uploadFailed"), "error");
      }
    }
    // Reset input
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(); }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      {/* Messages */}
      <div style={{ flex: 1, overflowY: "auto", padding: "24px" }}>
        {messages.length === 0 && (
          <div style={{ textAlign: "center", marginTop: 120, color: "var(--text-muted)" }}>
            <div style={{ fontSize: 40, marginBottom: 16 }}>S</div>
            <p style={{ fontSize: 16 }}>{t(lang, "sendToStart")}</p>
            <p style={{ fontSize: 13, marginTop: 8 }}>{t(lang, "poweredBy")}</p>
          </div>
        )}
        {messages.map((msg, i) => (<MessageBubble key={i} message={msg} />))}
        {isStreaming && messages.length > 0 && messages[messages.length - 1]?.content === "" && (
          <div style={{ padding: "12px 16px", color: "var(--text-muted)" }}>
            <div className="typing-dots"><span /><span /><span /></div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input area */}
      <div style={{ padding: "12px 24px 16px", borderTop: "1px solid var(--border)" }}>
        {/* Pending attachments preview */}
        {pendingAttachments.length > 0 && (
          <div style={{ display: "flex", gap: 8, marginBottom: 8, flexWrap: "wrap" }}>
            {pendingAttachments.map((att, i) => (
              <div key={i} style={{
                position: "relative", padding: 4, background: "var(--bg-secondary)",
                border: "1px solid var(--border)", borderRadius: 6,
              }}>
                {att.isImage ? (
                  <img src={att.url} alt={att.filename} style={{ width: 60, height: 60, objectFit: "cover", borderRadius: 4 }} />
                ) : (
                  <div style={{ width: 60, height: 60, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 11, color: "var(--text-muted)" }}>
                    {att.filename.slice(-10)}
                  </div>
                )}
                <button onClick={() => setPendingAttachments((prev) => prev.filter((_, j) => j !== i))} style={{
                  position: "absolute", top: -4, right: -4, width: 18, height: 18, borderRadius: 9,
                  background: "var(--error)", border: "none", color: "#fff", fontSize: 10, cursor: "pointer",
                  display: "flex", alignItems: "center", justifyContent: "center",
                }}>x</button>
                <div style={{ fontSize: 9, color: "var(--text-muted)", textAlign: "center", marginTop: 2, maxWidth: 60, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {att.filename}
                </div>
              </div>
            ))}
          </div>
        )}
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
          <span style={{ fontSize: 12, color: "var(--text-muted)" }}>{t(lang, "selectAgent")}:</span>
          <select value={selectedAgentId} onChange={(e) => setSelectedAgentId(e.target.value)} style={{
            flex: 1, padding: "6px 10px", fontSize: 12, background: "var(--bg-tertiary)",
            border: "1px solid var(--border)", borderRadius: 6, color: "var(--text-primary)", outline: "none",
          }}>
            {agents.map((a) => (<option key={a.id} value={a.id}>{agentName(lang, a.id, a.name)} - {agentRole(lang, a.id, a.role)}</option>))}
            {agents.length === 0 && (<option value="">{t(lang, "defaultAgent")}</option>)}
          </select>
        </div>
        <div style={{ display: "flex", gap: 12 }}>
          <input ref={fileInputRef} type="file" onChange={handleFileSelect} accept="image/*,.pdf,.txt,.csv,.json,.md,.py,.js,.ts,.html,.css" multiple style={{ display: "none" }} />
          <button onClick={() => fileInputRef.current?.click()} disabled={isStreaming} style={{
            padding: "10px 12px", fontSize: 14, background: "var(--bg-tertiary)",
            border: "1px solid var(--border)", borderRadius: 8, color: "var(--text-secondary)", cursor: "pointer",
          }}>+</button>
          <input value={input} onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown} placeholder={t(lang, "typePlaceholder")} disabled={isStreaming}
            style={{
              flex: 1, padding: "10px 16px", fontSize: 14, background: "var(--bg-secondary)",
              border: "1px solid var(--border)", borderRadius: 8, color: "var(--text-primary)", outline: "none",
            }}
          />
          {isStreaming ? (
            <button onClick={stop} style={{
              padding: "10px 20px", fontSize: 14, background: "var(--error)",
              border: "none", borderRadius: 8, color: "#fff", cursor: "pointer",
            }}>{t(lang, "stop")}</button>
          ) : (
            <button onClick={handleSend} disabled={!input.trim()} style={{
              padding: "10px 20px", fontSize: 14,
              background: input.trim() ? "var(--accent)" : "var(--bg-tertiary)",
              border: "none", borderRadius: 8,
              color: input.trim() ? "#fff" : "var(--text-muted)",
              cursor: input.trim() ? "pointer" : "default",
            }}>{t(lang, "send")}</button>
          )}
        </div>
      </div>
    </div>
  );
}

function MessageBubble({ message }: { message: StreamMessage }) {
  const isUser = message.role === "user";
  return (
    <div style={{ display: "flex", justifyContent: isUser ? "flex-end" : "flex-start", marginBottom: 16 }}>
      <div style={{
        maxWidth: "75%", padding: "12px 16px",
        borderRadius: isUser ? "12px 12px 2px 12px" : "12px 12px 12px 2px",
        background: isUser ? "var(--accent)" : "var(--bg-secondary)",
        border: isUser ? "none" : "1px solid var(--border)",
        color: isUser ? "#fff" : "var(--text-primary)",
        fontSize: 14, lineHeight: 1.6, whiteSpace: "pre-wrap", wordBreak: "break-word",
      }}>
        {message.toolCalls && message.toolCalls.length > 0 && (
          <div style={{ marginBottom: 8 }}>
            {message.toolCalls.map((tc, i) => (
              <div key={i} style={{
                padding: "6px 10px", marginBottom: 4,
                background: "var(--bg-tertiary)", borderRadius: 6, fontSize: 12, fontFamily: "monospace",
              }}>
                <span style={{ color: "var(--accent)" }}>{tc.name}</span>
                <span style={{ color: "var(--text-muted)" }}>({tc.input})</span>
                {tc.result && <div style={{ color: "var(--success)", marginTop: 4 }}>= {tc.result}</div>}
              </div>
            ))}
          </div>
        )}
        {message.content}
        {message.attachments && message.attachments.length > 0 && (
          <div style={{ marginTop: 8, display: "flex", gap: 6, flexWrap: "wrap" }}>
            {message.attachments.map((att, i) => (
              <div key={i}>
                {att.isImage ? (
                  <img src={att.url} alt={att.filename} style={{ maxWidth: 200, maxHeight: 150, borderRadius: 6, border: "1px solid var(--border)" }} />
                ) : (
                  <span style={{ padding: "4px 8px", fontSize: 11, background: isUser ? "rgba(255,255,255,0.2)" : "var(--bg-tertiary)", borderRadius: 4 }}>
                    {att.filename}
                  </span>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
