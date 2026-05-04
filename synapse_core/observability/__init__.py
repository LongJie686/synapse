"""Observability: structured logging, OpenTelemetry tracing, metrics, LangSmith integration."""

from __future__ import annotations

import time
import logging
import contextlib
from typing import Any, Generator

from pydantic import BaseModel


# ── Structured Logging ──────────────────────────────────────────────────────

class StructuredFormatter(logging.Formatter):
    """JSON-friendly structured log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = self.formatException(record.exc_info)
        extra_data = getattr(record, "struct_data", {})
        log_entry.update(extra_data)
        return str(log_entry)


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure structured logging for synapse."""
    logger = logging.getLogger("synapse")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(StructuredFormatter())
        logger.addHandler(handler)
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a child logger under synapse namespace."""
    return logging.getLogger(f"synapse.{name}")


# ── Trace & Span ────────────────────────────────────────────────────────────

class Span(BaseModel):
    """A single operation within a trace."""

    span_id: str = ""
    trace_id: str = ""
    parent_span_id: str = ""
    operation: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    status: str = "ok"
    attributes: dict[str, Any] = {}
    events: list[dict[str, Any]] = []

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000 if self.end_time else 0.0

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        self.events.append({
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {},
        })


class Trace(BaseModel):
    """A complete trace containing multiple spans."""

    trace_id: str = ""
    run_id: str = ""
    session_id: str = ""
    user_id: str = ""
    spans: list[Span] = []
    start_time: float = 0.0
    end_time: float = 0.0
    metadata: dict[str, Any] = {}

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000 if self.end_time else 0.0

    def add_span(self, span: Span) -> None:
        self.spans.append(span)


class Tracer:
    """In-process tracer for synapse operations."""

    def __init__(self) -> None:
        self._current_span: Span | None = None
        self._current_trace: Trace | None = None
        self._span_counter = 0

    def start_trace(
        self,
        trace_id: str,
        run_id: str = "",
        session_id: str = "",
        user_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> Trace:
        trace = Trace(
            trace_id=trace_id,
            run_id=run_id,
            session_id=session_id,
            user_id=user_id,
            start_time=time.time(),
            metadata=metadata or {},
        )
        self._current_trace = trace
        return trace

    def end_trace(self) -> Trace | None:
        if not self._current_trace:
            return None
        self._current_trace.end_time = time.time()
        trace = self._current_trace
        self._current_trace = None
        self._current_span = None
        return trace

    def start_span(self, operation: str, attributes: dict[str, Any] | None = None) -> Span:
        self._span_counter += 1
        span = Span(
            span_id=f"span-{self._span_counter}",
            trace_id=self._current_trace.trace_id if self._current_trace else "",
            parent_span_id=self._current_span.span_id if self._current_span else "",
            operation=operation,
            start_time=time.time(),
            attributes=attributes or {},
        )
        if self._current_trace:
            self._current_trace.add_span(span)
        self._current_span = span
        return span

    def end_span(self, status: str = "ok") -> Span | None:
        if not self._current_span:
            return None
        self._current_span.end_time = time.time()
        self._current_span.status = status
        span = self._current_span
        if self._current_trace and span.parent_span_id:
            for s in reversed(self._current_trace.spans):
                if s.span_id == span.parent_span_id and s.end_time == 0.0:
                    self._current_span = s
                    return span
        self._current_span = None
        return span

    @contextlib.contextmanager
    def span(self, operation: str, **attributes: Any) -> Generator[Span, None, None]:
        """Context manager for a span."""
        s = self.start_span(operation, attributes)
        try:
            yield s
            self.end_span("ok")
        except Exception:
            self.end_span("error")
            raise


# ── Metrics ─────────────────────────────────────────────────────────────────

class MetricsCollector(BaseModel):
    """Collects runtime metrics for monitoring."""

    counters: dict[str, float] = {}
    gauges: dict[str, float] = {}
    histograms: dict[str, list[float]] = {}

    def increment(self, name: str, value: float = 1.0) -> None:
        self.counters[name] = self.counters.get(name, 0.0) + value

    def set_gauge(self, name: str, value: float) -> None:
        self.gauges[name] = value

    def observe(self, name: str, value: float) -> None:
        if name not in self.histograms:
            self.histograms[name] = []
        self.histograms[name].append(value)

    def get_summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {"counters": dict(self.counters), "gauges": dict(self.gauges)}
        for name, values in self.histograms.items():
            if values:
                summary[f"histogram_{name}"] = {
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values),
                    "p50": sorted(values)[len(values) // 2],
                }
        return summary


# ── LangSmith Integration ───────────────────────────────────────────────────

class LangSmithConfig(BaseModel):
    """LangSmith configuration."""

    api_key: str = ""
    project: str = "synapse"
    endpoint: str = "https://api.smith.langchain.com"
    enabled: bool = False
    tags: list[str] = []


class LangSmithTracker:
    """Tracks runs in LangSmith for LLM observability."""

    def __init__(self, config: LangSmithConfig | None = None) -> None:
        self.config = config or LangSmithConfig()
        self._runs: dict[str, dict[str, Any]] = {}

    def start_run(
        self,
        run_id: str,
        name: str,
        run_type: str = "chain",
        inputs: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self.config.enabled:
            return
        self._runs[run_id] = {
            "id": run_id,
            "name": name,
            "run_type": run_type,
            "inputs": inputs or {},
            "tags": list(set(self.config.tags + (tags or []))),
            "metadata": metadata or {},
            "start_time": time.time(),
        }

    def end_run(
        self,
        run_id: str,
        outputs: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        if not self.config.enabled or run_id not in self._runs:
            return
        run = self._runs[run_id]
        run["end_time"] = time.time()
        run["duration_ms"] = (run["end_time"] - run["start_time"]) * 1000
        run["outputs"] = outputs or {}
        run["error"] = error

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        return self._runs.get(run_id)

    @contextlib.contextmanager
    def track_run(
        self,
        run_id: str,
        name: str,
        run_type: str = "chain",
        **kwargs: Any,
    ) -> Generator[None, None, None]:
        """Context manager for tracking a LangSmith run."""
        self.start_run(run_id, name, run_type, **kwargs)
        try:
            yield
        except Exception as e:
            self.end_run(run_id, error=str(e))
            raise
        self.end_run(run_id)


# ── Observability Hub ───────────────────────────────────────────────────────

class ObservabilityHub:
    """Central hub for all observability concerns."""

    def __init__(
        self,
        log_level: str = "INFO",
        langsmith_config: LangSmithConfig | None = None,
    ) -> None:
        self.logger = setup_logging(log_level)
        self.tracer = Tracer()
        self.metrics = MetricsCollector()
        self.langsmith = LangSmithTracker(langsmith_config)

    def track_run_start(self, run_id: str, agent_name: str, user_input: str) -> None:
        self.metrics.increment("runs.total")
        self.metrics.increment(f"runs.by_agent.{agent_name}")
        self.tracer.start_trace(trace_id=run_id, run_id=run_id)
        self.langsmith.start_run(run_id, f"agent:{agent_name}", inputs={"user_input": user_input})
        self.logger.info("Run started", extra={"struct_data": {"run_id": run_id, "agent": agent_name}})

    def track_run_end(self, run_id: str, token_usage: dict[str, int] | None = None) -> None:
        trace = self.tracer.end_trace()
        self.langsmith.end_run(run_id, outputs={"token_usage": token_usage})
        if token_usage:
            prompt_t = token_usage.get("prompt_tokens", 0)
            completion_t = token_usage.get("completion_tokens", 0)
            total_t = token_usage.get("total_tokens", 0)
            self.metrics.increment("tokens.prompt", prompt_t)
            self.metrics.increment("tokens.completion", completion_t)
            self.metrics.increment("tokens.total", total_t)
            for key, val in token_usage.items():
                self.metrics.observe("tokens.usage", val)
        self.logger.info("Run completed", extra={
            "struct_data": {"run_id": run_id, "duration_ms": trace.duration_ms if trace else 0}
        })

    def track_tool_call(self, tool_name: str, duration_ms: float, success: bool) -> None:
        self.metrics.increment("tool_calls.total")
        self.metrics.increment(f"tool_calls.by_tool.{tool_name}")
        if not success:
            self.metrics.increment("tool_calls.errors")
        self.metrics.observe("tool_calls.duration_ms", duration_ms)

    def track_llm_call(self, provider: str, model: str, tokens_in: int, tokens_out: int, duration_ms: float) -> None:
        self.metrics.increment("llm_calls.total")
        self.metrics.increment(f"llm_calls.by_provider.{provider}")
        self.metrics.observe("llm_calls.tokens_in", tokens_in)
        self.metrics.observe("llm_calls.tokens_out", tokens_out)
        self.metrics.observe("llm_calls.duration_ms", duration_ms)

    def track_guardrail_event(self, filter_name: str, blocked: bool) -> None:
        self.metrics.increment("guardrail_events.total")
        if blocked:
            self.metrics.increment(f"guardrail_events.blocked.{filter_name}")

    def track_memory_event(self, event_type: str) -> None:
        self.metrics.increment("memory_events.total")
        self.metrics.increment(f"memory_events.{event_type}")

    def get_metrics_summary(self) -> dict[str, Any]:
        return self.metrics.get_summary()


_hub: ObservabilityHub | None = None


def get_hub() -> ObservabilityHub:
    global _hub
    if _hub is None:
        _hub = ObservabilityHub()
    return _hub


def init_hub(log_level: str = "INFO", langsmith_config: LangSmithConfig | None = None) -> ObservabilityHub:
    global _hub
    _hub = ObservabilityHub(log_level=log_level, langsmith_config=langsmith_config)
    return _hub
