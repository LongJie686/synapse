"use client";

import { useState, useRef, useEffect } from "react";
import { useStream, StreamMessage } from "@/hooks/useStream";

export default function ChatPanel() {
  const { messages, isStreaming, sendMessage, stop, clear } = useStream();
  const [input, setInput] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = () => {
    const text = input.trim();
    if (!text || isStreaming) return;
    setInput("");
    sendMessage(text);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      {/* Header */}
      <div style={{
        padding: "16px 24px",
        borderBottom: "1px solid var(--border)",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
      }}>
        <div>
          <h1 style={{ fontSize: 18, fontWeight: 600, margin: 0 }}>Synapse Chat</h1>
          <span style={{ fontSize: 12, color: "var(--text-muted)" }}>Multi-Agent Collaboration</span>
        </div>
        <button
          onClick={clear}
          style={{
            padding: "6px 14px",
            fontSize: 13,
            background: "var(--bg-tertiary)",
            border: "1px solid var(--border)",
            borderRadius: 6,
            color: "var(--text-secondary)",
            cursor: "pointer",
          }}
        >
          Clear
        </button>
      </div>

      {/* Messages */}
      <div style={{ flex: 1, overflowY: "auto", padding: "24px" }}>
        {messages.length === 0 && (
          <div style={{ textAlign: "center", marginTop: 120, color: "var(--text-muted)" }}>
            <div style={{ fontSize: 40, marginBottom: 16 }}>S</div>
            <p style={{ fontSize: 16 }}>Send a message to start</p>
            <p style={{ fontSize: 13, marginTop: 8 }}>Powered by Synapse Multi-Agent Framework</p>
          </div>
        )}

        {messages.map((msg, i) => (
          <MessageBubble key={i} message={msg} />
        ))}

        {isStreaming && messages[messages.length - 1]?.content === "" && (
          <div style={{ padding: "12px 16px", color: "var(--text-muted)" }}>
            <div className="typing-dots">
              <span /><span /><span />
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div style={{
        padding: "16px 24px",
        borderTop: "1px solid var(--border)",
        display: "flex",
        gap: 12,
      }}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type a message..."
          disabled={isStreaming}
          style={{
            flex: 1,
            padding: "10px 16px",
            fontSize: 14,
            background: "var(--bg-secondary)",
            border: "1px solid var(--border)",
            borderRadius: 8,
            color: "var(--text-primary)",
            outline: "none",
          }}
        />
        {isStreaming ? (
          <button
            onClick={stop}
            style={{
              padding: "10px 20px",
              fontSize: 14,
              background: "var(--error)",
              border: "none",
              borderRadius: 8,
              color: "#fff",
              cursor: "pointer",
            }}
          >
            Stop
          </button>
        ) : (
          <button
            onClick={handleSend}
            disabled={!input.trim()}
            style={{
              padding: "10px 20px",
              fontSize: 14,
              background: input.trim() ? "var(--accent)" : "var(--bg-tertiary)",
              border: "none",
              borderRadius: 8,
              color: input.trim() ? "#fff" : "var(--text-muted)",
              cursor: input.trim() ? "pointer" : "default",
            }}
          >
            Send
          </button>
        )}
      </div>
    </div>
  );
}

function MessageBubble({ message }: { message: StreamMessage }) {
  const isUser = message.role === "user";

  return (
    <div style={{
      display: "flex",
      justifyContent: isUser ? "flex-end" : "flex-start",
      marginBottom: 16,
    }}>
      <div style={{
        maxWidth: "75%",
        padding: "12px 16px",
        borderRadius: isUser ? "12px 12px 2px 12px" : "12px 12px 12px 2px",
        background: isUser ? "var(--accent)" : "var(--bg-secondary)",
        border: isUser ? "none" : "1px solid var(--border)",
        color: isUser ? "#fff" : "var(--text-primary)",
        fontSize: 14,
        lineHeight: 1.6,
        whiteSpace: "pre-wrap",
        wordBreak: "break-word",
      }}>
        {/* Tool calls */}
        {message.toolCalls && message.toolCalls.length > 0 && (
          <div style={{ marginBottom: 8 }}>
            {message.toolCalls.map((tc, i) => (
              <div key={i} style={{
                padding: "6px 10px",
                marginBottom: 4,
                background: "var(--bg-tertiary)",
                borderRadius: 6,
                fontSize: 12,
                fontFamily: "monospace",
              }}>
                <span style={{ color: "var(--accent)" }}>{tc.name}</span>
                <span style={{ color: "var(--text-muted)" }}>({tc.input})</span>
                {tc.result && (
                  <div style={{ color: "var(--success)", marginTop: 4 }}>= {tc.result}</div>
                )}
              </div>
            ))}
          </div>
        )}
        {message.content}
      </div>
    </div>
  );
}
