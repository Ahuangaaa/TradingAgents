"""LangChain callback handler: LLM/tool durations and I/O excerpts to JSONL."""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from tradingagents.dataflows.run_trace_context import (
    get_trace_max_chars,
    trace_analyst_key,
    trace_subphase,
)


def _truncate(text: Any, max_chars: int) -> str:
    if text is None:
        return ""
    s = text if isinstance(text, str) else repr(text)
    if len(s) <= max_chars:
        return s
    return s[: max(0, max_chars - 24)] + "\n...[truncated]..."


def _llm_text_from_result(response: LLMResult) -> str:
    try:
        gen = response.generations[0][0]
        if hasattr(gen, "text") and gen.text:
            return str(gen.text)
        if hasattr(gen, "message"):
            c = getattr(gen.message, "content", None)
            if c is not None:
                return c if isinstance(c, str) else repr(c)
    except (IndexError, TypeError):
        pass
    return ""


def _model_label(serialized: Dict[str, Any]) -> str:
    rep = serialized.get("repr") if isinstance(serialized, dict) else None
    if isinstance(rep, str) and rep:
        return rep[:200]
    kwargs = serialized.get("kwargs") if isinstance(serialized, dict) else None
    if isinstance(kwargs, dict):
        for key in ("model_name", "model", "model_id"):
            v = kwargs.get(key)
            if v:
                return str(v)
    return str(serialized.get("id", "") or "llm")


class RunTraceCallbackHandler(BaseCallbackHandler):
    """Append one JSON object per line to ``events_path`` for LLM and tool runs."""

    def __init__(self, events_path: Path, max_chars: Optional[int] = None) -> None:
        super().__init__()
        self._path = Path(events_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text("", encoding="utf-8")
        self._max = max_chars if max_chars is not None else get_trace_max_chars()
        self._lock = threading.Lock()
        self._starts: Dict[str, float] = {}
        self._tool_inputs: Dict[str, str] = {}
        self._tool_names: Dict[str, str] = {}

    def _append(self, row: Dict[str, Any]) -> None:
        row.setdefault("ts", datetime.now(timezone.utc).isoformat())
        row["analyst_key"] = trace_analyst_key.get()
        row["subphase"] = trace_subphase.get()
        line = json.dumps(row, ensure_ascii=False) + "\n"
        with self._lock:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(line)

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: Any,
        **kwargs: Any,
    ) -> None:
        rid = str(run_id)
        with self._lock:
            self._starts[rid] = time.perf_counter()

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[Any]],
        *,
        run_id: Any,
        **kwargs: Any,
    ) -> None:
        rid = str(run_id)
        with self._lock:
            self._starts[rid] = time.perf_counter()

    def on_llm_end(self, response: LLMResult, *, run_id: Any, **kwargs: Any) -> None:
        rid = str(run_id)
        with self._lock:
            start = self._starts.pop(rid, None)
        elapsed_ms = (time.perf_counter() - start) * 1000 if start is not None else -1.0
        out = _truncate(_llm_text_from_result(response), self._max)
        prompts = kwargs.get("prompts")
        inp = ""
        if isinstance(prompts, list) and prompts:
            inp = _truncate("\n---\n".join(str(p) for p in prompts[:3]), self._max)
        serialized = kwargs.get("serialized") or {}
        if not isinstance(serialized, dict):
            serialized = {}
        if not inp:
            msgs = kwargs.get("messages")
            if isinstance(msgs, list) and msgs:
                inp = _truncate(repr(msgs)[: self._max], self._max)
        self._append(
            {
                "event_type": "llm",
                "run_id": rid,
                "name": _model_label(serialized),
                "duration_ms": round(elapsed_ms, 2),
                "input_excerpt": inp,
                "output_excerpt": out,
            }
        )

    def on_llm_error(self, error: BaseException, *, run_id: Any, **kwargs: Any) -> None:
        rid = str(run_id)
        with self._lock:
            start = self._starts.pop(rid, None)
        elapsed_ms = (time.perf_counter() - start) * 1000 if start is not None else -1.0
        self._append(
            {
                "event_type": "llm_error",
                "run_id": rid,
                "name": "llm",
                "duration_ms": round(elapsed_ms, 2),
                "error": _truncate(str(error), self._max),
            }
        )

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: Any,
        **kwargs: Any,
    ) -> None:
        rid = str(run_id)
        name = ""
        if isinstance(serialized, dict):
            name = str(serialized.get("name") or serialized.get("id") or "")
        with self._lock:
            self._starts[rid] = time.perf_counter()
            self._tool_inputs[rid] = input_str
            self._tool_names[rid] = name

    def on_tool_end(self, output: Any, *, run_id: Any, **kwargs: Any) -> None:
        rid = str(run_id)
        with self._lock:
            start = self._starts.pop(rid, None)
            inp = self._tool_inputs.pop(rid, "")
            name = self._tool_names.pop(rid, "")
        elapsed_ms = (time.perf_counter() - start) * 1000 if start is not None else -1.0
        self._append(
            {
                "event_type": "tool",
                "run_id": rid,
                "name": name or "tool",
                "duration_ms": round(elapsed_ms, 2),
                "input_excerpt": _truncate(inp, self._max),
                "output_excerpt": _truncate(output, self._max),
            }
        )

    def on_tool_error(self, error: BaseException, *, run_id: Any, **kwargs: Any) -> None:
        rid = str(run_id)
        with self._lock:
            start = self._starts.pop(rid, None)
            inp = self._tool_inputs.pop(rid, "")
            name = self._tool_names.pop(rid, "")
        elapsed_ms = (time.perf_counter() - start) * 1000 if start is not None else -1.0
        self._append(
            {
                "event_type": "tool_error",
                "run_id": rid,
                "name": name or "tool",
                "duration_ms": round(elapsed_ms, 2),
                "input_excerpt": _truncate(inp, self._max),
                "error": _truncate(str(error), self._max),
            }
        )
