"use client";

import { useState, useCallback, createContext, useContext, type ReactNode } from "react";

type ToastType = "success" | "error" | "info" | "warning";

interface Toast {
  id: string;
  message: string;
  type: ToastType;
}

interface ToastContextValue {
  toast: (message: string, type?: ToastType) => void;
  confirm: (message: string) => Promise<boolean>;
}

const ToastContext = createContext<ToastContextValue>({
  toast: () => {},
  confirm: async () => false,
});

export function useToast() {
  return useContext(ToastContext);
}

const TOAST_COLORS: Record<ToastType, { bg: string; border: string; icon: string }> = {
  success: { bg: "var(--success-soft, #f0fdf4)", border: "var(--success, #22c55e)", icon: "✓" },
  error: { bg: "var(--error-soft, #fef2f2)", border: "var(--error, #ef4444)", icon: "✗" },
  info: { bg: "var(--accent-soft, #eff6ff)", border: "var(--accent, #3b82f6)", icon: "i" },
  warning: { bg: "var(--warning-soft, #fffbeb)", border: "var(--warning, #f59e0b)", icon: "!" },
};

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const [confirmState, setConfirmState] = useState<{
    message: string;
    resolve: (value: boolean) => void;
  } | null>(null);

  const toast = useCallback((message: string, type: ToastType = "info") => {
    const id = `toast-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
    setToasts((prev) => [...prev, { id, message, type }]);
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 3500);
  }, []);

  const confirm = useCallback((message: string): Promise<boolean> => {
    return new Promise((resolve) => {
      setConfirmState({ message, resolve });
    });
  }, []);

  const handleConfirm = (value: boolean) => {
    confirmState?.resolve(value);
    setConfirmState(null);
  };

  return (
    <ToastContext.Provider value={{ toast, confirm }}>
      {children}

      {/* Toast container */}
      <div style={{ position: "fixed", top: 16, right: 16, zIndex: 9999, display: "flex", flexDirection: "column", gap: 8 }}>
        {toasts.map((t) => {
          const colors = TOAST_COLORS[t.type];
          return (
            <div key={t.id} style={{
              padding: "10px 16px",
              background: colors.bg,
              borderLeft: `3px solid ${colors.border}`,
              borderRadius: 8,
              boxShadow: "0 2px 8px rgba(0,0,0,0.12)",
              fontSize: 13,
              color: "var(--text-primary)",
              display: "flex",
              alignItems: "center",
              gap: 8,
              minWidth: 240,
              maxWidth: 400,
              animation: "slideIn 0.2s ease-out",
            }}>
              <span style={{
                width: 20, height: 20, borderRadius: 10,
                background: colors.border, color: "#fff", fontSize: 11, fontWeight: 700,
                display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0,
              }}>{colors.icon}</span>
              <span style={{ flex: 1 }}>{t.message.length > 200 ? t.message.slice(0, 200) + "..." : t.message}</span>
            </div>
          );
        })}
      </div>

      {/* Confirm dialog */}
      {confirmState && (
        <div style={{
          position: "fixed", inset: 0, zIndex: 10000,
          background: "rgba(0,0,0,0.4)", display: "flex", alignItems: "center", justifyContent: "center",
        }} onClick={() => handleConfirm(false)}>
          <div style={{
            background: "var(--bg-primary)", borderRadius: 12, padding: 24,
            boxShadow: "0 8px 32px rgba(0,0,0,0.2)", maxWidth: 400, width: "90%",
          }} onClick={(e) => e.stopPropagation()}>
            <p style={{ fontSize: 14, lineHeight: 1.6, marginBottom: 20, color: "var(--text-primary)" }}>
              {confirmState.message}
            </p>
            <div style={{ display: "flex", justifyContent: "flex-end", gap: 8 }}>
              <button onClick={() => handleConfirm(false)} style={{
                padding: "8px 16px", fontSize: 13, background: "var(--bg-tertiary)",
                border: "1px solid var(--border)", borderRadius: 6, color: "var(--text-secondary)", cursor: "pointer",
              }}>Cancel</button>
              <button onClick={() => handleConfirm(true)} style={{
                padding: "8px 16px", fontSize: 13, background: "var(--error)",
                border: "none", borderRadius: 6, color: "#fff", cursor: "pointer",
              }}>Confirm</button>
            </div>
          </div>
        </div>
      )}
    </ToastContext.Provider>
  );
}
