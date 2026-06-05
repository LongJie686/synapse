"use client";

import { useCallback, useRef, useState } from "react";

export interface AttachedFile {
  filename: string;
  url: string;
  isImage: boolean;
  size: number;
  base64Data?: string;  // base64 encoded image data for LLM vision
  mimeType?: string;
}

export interface StreamMessage {
  role: "user" | "assistant";
  content: string;
  toolCalls?: { name: string; input: string; result?: string }[];
  isStreaming?: boolean;
  attachments?: AttachedFile[];
}

export interface TitleUpdate {
  sessionId: string;
  title: string;
}

export function useStream() {
  const [messages, setMessages] = useState<StreamMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const sendMessage = useCallback(
    async (
      content: string,
      agentId?: string,
      sessionId?: string,
      onTitleUpdate?: (update: TitleUpdate) => void,
      attachments?: AttachedFile[],
    ) => {
      // Create abort controller for this request
      const controller = new AbortController();
      abortRef.current = controller;

      const userMsg: StreamMessage = { role: "user", content, attachments };
      const assistantMsg: StreamMessage = {
        role: "assistant",
        content: "",
        isStreaming: true,
        toolCalls: [],
      };

      setMessages((prev) => [...prev, userMsg, assistantMsg]);
      setIsStreaming(true);

      // Track the assistant message index
      const currentIdx = { value: 0 };
      setMessages((prev) => {
        currentIdx.value = prev.length - 1;
        return prev;
      });

      try {
        const body: Record<string, string> = { message: content };
        if (agentId) body.agent_id = agentId;
        if (sessionId) body.session_id = sessionId;

        // Append attachment info to message for LLM context
        const images: { mime_type: string; data: string }[] = [];
        if (attachments && attachments.length > 0) {
          const attachmentDesc = attachments.map((a) => {
            if (a.isImage) return `[Image: ${a.filename}]`;
            return `[File: ${a.filename}]`;
          }).join(" ");
          body.message = `${content}\n\nAttached files:\n${attachmentDesc}`;

          // Collect base64 image data for multimodal LLM call
          for (const att of attachments) {
            if (att.isImage && att.base64Data && att.mimeType) {
              images.push({ mime_type: att.mimeType, data: att.base64Data });
            }
          }
        }

        const reqBody: Record<string, unknown> = { ...body, images: images.length > 0 ? images : undefined };

        // SSE bypasses Next.js proxy to avoid response buffering.
        // NEXT_PUBLIC_API_URL must be set in production (e.g. "http://backend:8000").
        const apiOrigin = process.env.NEXT_PUBLIC_API_URL ?? "";
        const streamUrl = `${apiOrigin}/api/runs/stream`;

        const apiKey = process.env.NEXT_PUBLIC_API_KEY;
        const res = await fetch(streamUrl, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            ...(apiKey ? { "X-API-Key": apiKey } : {}),
          },
          body: JSON.stringify(reqBody),
          signal: controller.signal,
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

          // NDJSON: each line is a JSON object
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed) continue;

            try {
              const event = JSON.parse(trimmed);
              const { type, data } = event;

              if (type === "run:error") {
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

              if (type === "agent:call_tool" && data.toolName) {
                setMessages((prev) => {
                  const updated = [...prev];
                  const msg = { ...updated[currentIdx.value] };
                  msg.toolCalls = [
                    ...(msg.toolCalls || []),
                    { name: data.toolName, input: JSON.stringify(data.input) },
                  ];
                  updated[currentIdx.value] = msg;
                  return updated;
                });
              }

              if (type === "tool:result" && data.toolName) {
                setMessages((prev) => {
                  const updated = [...prev];
                  const msg = { ...updated[currentIdx.value] };
                  const calls = [...(msg.toolCalls || [])];
                  const lastCall = calls[calls.length - 1];
                  if (lastCall) {
                    calls[calls.length - 1] = {
                      ...lastCall,
                      result: String(data.output),
                    };
                  }
                  msg.toolCalls = calls;
                  updated[currentIdx.value] = msg;
                  return updated;
                });
              }

              if (
                type === "agent:message" &&
                typeof data.content === "string"
              ) {
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

              if (type === "session:title_update" && data.session_id && data.title) {
                onTitleUpdate?.({ sessionId: data.session_id, title: data.title });
              }
            } catch {
              console.warn("SSE parse error: invalid JSON on stream line");
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
        abortRef.current = null;
      }
    },
    [],
  );

  const stop = useCallback(() => {
    abortRef.current?.abort();
    setIsStreaming(false);
  }, []);

  const clear = useCallback(() => {
    setMessages([]);
  }, []);

  return { messages, isStreaming, sendMessage, stop, clear, setMessages };
}
