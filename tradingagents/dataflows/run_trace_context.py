"""Context for run-level tracing (analyst phase, tools phase, report directory)."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from contextvars import ContextVar, Token
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

trace_report_dir: ContextVar[Path | None] = ContextVar("trace_report_dir", default=None)
trace_analyst_key: ContextVar[str | None] = ContextVar("trace_analyst_key", default=None)
trace_subphase: ContextVar[str | None] = ContextVar("trace_subphase", default=None)


def get_trace_max_chars() -> int:
    raw = os.getenv("TRADINGAGENTS_TRACE_MAX_CHARS", "80000").strip()
    try:
        return max(1000, int(raw))
    except ValueError:
        return 80000


@contextmanager
def analyst_llm_phase(analyst_key: str) -> Iterator[None]:
    t1: Token[str | None] = trace_analyst_key.set(analyst_key)
    t2: Token[str | None] = trace_subphase.set("llm")
    try:
        yield
    finally:
        trace_analyst_key.reset(t1)
        trace_subphase.reset(t2)


@contextmanager
def tools_phase(analyst_key: str) -> Iterator[None]:
    t1: Token[str | None] = trace_analyst_key.set(analyst_key)
    t2: Token[str | None] = trace_subphase.set("tools")
    try:
        yield
    finally:
        trace_analyst_key.reset(t1)
        trace_subphase.reset(t2)


def set_report_dir(path: Path | None) -> Token[Path | None]:
    return trace_report_dir.set(path)


def reset_report_dir(token: Token[Path | None]) -> None:
    trace_report_dir.reset(token)


def append_qdrant_trace(event: dict[str, Any]) -> None:
    base = trace_report_dir.get()
    if base is None:
        return
    trace_sub = base / "trace"
    trace_sub.mkdir(parents=True, exist_ok=True)
    path = trace_sub / "qdrant_retrieval.jsonl"
    row = dict(event)
    row.setdefault("ts", datetime.now(timezone.utc).isoformat())
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
