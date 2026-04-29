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

  const sendMessage = useCallback(async (content: string) => {
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
      const res = await fetch("/api/runs/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: content }),
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
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (line.startsWith("data:")) {
            try {
              const data = JSON.parse(line.slice(5).trim());
              const eventType = line.match(/^event:\s*(.+)$/m)?.[1];

              if (data.agentId && data.content) {
                fullContent += data.content;
              } else if (data.agentId && data.thought) {
                // think event, skip or show
              }

              // Handle tool calls
              if (data.toolName && data.input !== undefined) {
                setMessages((prev) => {
                  const updated = [...prev];
                  const msg = { ...updated[currentIdx.value] };
                  msg.toolCalls = [...(msg.toolCalls || []), { name: data.toolName, input: JSON.stringify(data.input) }];
                  updated[currentIdx.value] = msg;
                  return updated;
                });
              }

              // Handle tool results
              if (data.toolName && data.output !== undefined) {
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

              // Handle content chunks (from SSE data with content field)
              if (typeof data.content === "string" && data.agentId) {
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
      }

      // Final update
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

  return { messages, isStreaming, sendMessage, stop, clear };
}
