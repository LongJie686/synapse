"use client";

import { Component, type ReactNode } from "react";

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export default class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) return this.props.fallback;
      return (
        <div style={{
          padding: 24, textAlign: "center", color: "var(--text-secondary)",
        }}>
          <p style={{ fontSize: 14, marginBottom: 12 }}>
            Something went wrong: {this.state.error?.message || "Unknown error"}
          </p>
          <button onClick={() => this.setState({ hasError: false, error: null })} style={{
            padding: "8px 16px", fontSize: 13, background: "var(--accent)",
            border: "none", borderRadius: 6, color: "#fff", cursor: "pointer",
          }}>
            Retry
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
