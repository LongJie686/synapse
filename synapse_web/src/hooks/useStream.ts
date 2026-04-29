"use client";

import { useCallback, useRef, useState } from "react";

export interface StreamMessage {
  role: "user" | "assistant";
  content: string;
  toolCalls?: { name: string; input: string; result?: string }[];
  isStreaming?: boolean;
}

export function useStream() {
  const [messages, setMessages] = useState<StreamMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const sendMessage = useCallback(async (content: string, agentId?: string, sessionId?: string) => {
    const userMsg: StreamMessage = { role: "user", content };
    const assistantMsg: StreamMessage = { role: "assistant", content: "", isStreaming: true, toolCalls: [] };

    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setIsStreaming(true);

    const currentIdx = { value: 0 };
    setMessages((prev) => {
      currentIdx.value = prev.length - 1;
      return prev;
    });

    try {
      const body: Record<string, string> = { message: content };
      if (agentId) body.agent_id = agentId;
      if (sessionId) body.session_id = sessionId;

      // SSE stream must go directly to backend to avoid Next.js proxy buffering
      const streamUrl = process.env.NEXT_PUBLIC_API_URL
        ? `${process.env.NEXT_PUBLIC_API_URL}/api/runs/stream`
        : "http://localhost:8000/api/runs/stream";

      const res = await fetch(streamUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: abortRef.current?.signal,
      });

      if (!res.ok || !res.body) throw new Error("Stream failed");

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let fullContent = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // SSE events are separated by double newlines
        const events = buffer.split("\n\n");
        buffer = events.pop() || "";

        for (const eventText of events) {
          if (!eventText.trim()) continue;

          let eventType = "";
          let eventData = "";

          for (const line of eventText.split("\n")) {
            if (line.startsWith("event:")) {
              eventType = line.slice(6).trim();
            } else if (line.startsWith("data:")) {
              eventData = line.slice(5).trim();
            }
          }

          if (!eventData) continue;

          try {
            const data = JSON.parse(eventData);

            if (eventType === "run:error") {
              fullContent = data.error || "Unknown error";
              setMessages((prev) => {
                const updated = [...prev];
                updated[currentIdx.value] = {
                  ...updated[currentIdx.value],
                  content: fullContent,
                  isStreaming: false,
                };
                return updated;
              });
              return;
            }

            if (eventType === "agent:call_tool" && data.toolName) {
              setMessages((prev) => {
                const updated = [...prev];
                const msg = { ...updated[currentIdx.value] };
                msg.toolCalls = [...(msg.toolCalls || []), { name: data.toolName, input: JSON.stringify(data.input) }];
                updated[currentIdx.value] = msg;
                return updated;
              });
            }

            if (eventType === "tool:result" && data.toolName) {
              setMessages((prev) => {
                const updated = [...prev];
                const msg = { ...updated[currentIdx.value] };
                const calls = [...(msg.toolCalls || [])];
                const lastCall = calls[calls.length - 1];
                if (lastCall) {
                  calls[calls.length - 1] = { ...lastCall, result: String(data.output) };
                }
                msg.toolCalls = calls;
                updated[currentIdx.value] = msg;
                return updated;
              });
            }

            if (eventType === "agent:message" && typeof data.content === "string") {
              fullContent = data.content;
              setMessages((prev) => {
                const updated = [...prev];
                updated[currentIdx.value] = {
                  ...updated[currentIdx.value],
                  content: fullContent,
                };
                return updated;
              });
            }
          } catch {
            // skip malformed JSON
          }
        }
      }

      setMessages((prev) => {
        const updated = [...prev];
        updated[currentIdx.value] = {
          ...updated[currentIdx.value],
          content: fullContent || "(no response)",
          isStreaming: false,
        };
        return updated;
      });
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        setMessages((prev) => {
          const updated = [...prev];
          updated[currentIdx.value] = {
            ...updated[currentIdx.value],
            content: "Error: failed to get response",
            isStreaming: false,
          };
          return updated;
        });
      }
    } finally {
      setIsStreaming(false);
    }
  }, []);

  const stop = useCallback(() => {
    abortRef.current?.abort();
    setIsStreaming(false);
  }, []);

  const clear = useCallback(() => {
    setMessages([]);
  }, []);

  return { messages, isStreaming, sendMessage, stop, clear, setMessages };
}
