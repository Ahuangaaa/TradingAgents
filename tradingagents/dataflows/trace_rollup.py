"""Roll up ``trace/events.jsonl`` into per-analyst summaries and Markdown."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List


def _bucket_key(analyst_key: Any) -> str:
    if analyst_key is None or analyst_key == "":
        return "other"
    return str(analyst_key)


def rollup_events_jsonl(events_path: Path) -> Dict[str, Any]:
    """Parse events.jsonl and return a dict keyed by analyst bucket."""

    def _empty_block() -> Dict[str, Any]:
        return {
            "analyst_key": "",
            "llm_total_ms": 0.0,
            "tools_total_ms": 0.0,
            "llm_calls": 0,
            "tool_calls": 0,
            "timeline": [],
        }

    by_analyst: DefaultDict[str, Dict[str, Any]] = defaultdict(_empty_block)

    if not events_path.is_file():
        return {}

    with open(events_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            et = row.get("event_type")
            if et not in ("llm", "tool", "llm_error", "tool_error"):
                continue
            key = _bucket_key(row.get("analyst_key"))
            block = by_analyst[key]
            block["analyst_key"] = key
            dur = float(row.get("duration_ms") or 0.0)
            name = str(row.get("name") or "")
            entry: Dict[str, Any] = {
                "type": "llm" if et.startswith("llm") else "tool",
                "name": name,
                "duration_ms": dur,
                "subphase": row.get("subphase"),
            }
            if et == "llm":
                block["llm_total_ms"] += max(0.0, dur)
                block["llm_calls"] += 1
                entry["output_excerpt"] = row.get("output_excerpt", "")
                entry["input_excerpt"] = row.get("input_excerpt", "")
            elif et == "tool":
                block["tools_total_ms"] += max(0.0, dur)
                block["tool_calls"] += 1
                entry["input_excerpt"] = row.get("input_excerpt", "")
                entry["output_excerpt"] = row.get("output_excerpt", "")
            else:
                entry["error"] = row.get("error", "")
            block["timeline"].append(entry)

    out: Dict[str, Any] = {}
    for k, v in by_analyst.items():
        v["analyst_total_ms"] = round(v["llm_total_ms"] + v["tools_total_ms"], 2)
        v["llm_total_ms"] = round(v["llm_total_ms"], 2)
        v["tools_total_ms"] = round(v["tools_total_ms"], 2)
        out[k] = v
    return out


def write_analyst_summary_json(summary: Dict[str, Any], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def write_analyst_breakdown_md(summary: Dict[str, Any], dest: Path) -> None:
    lines: List[str] = ["# Run trace by analyst", ""]
    preferred = ["market", "social", "news", "fundamentals"]
    keys = [k for k in preferred if k in summary]
    keys += sorted(k for k in summary if k not in preferred)

    for key in keys:
        block = summary[key]
        lines.append(f"## {key}")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("| --- | --- |")
        lines.append(f"| analyst_total_ms | {block.get('analyst_total_ms', 0)} |")
        lines.append(f"| llm_total_ms | {block.get('llm_total_ms', 0)} |")
        lines.append(f"| tools_total_ms | {block.get('tools_total_ms', 0)} |")
        lines.append(f"| llm_calls | {block.get('llm_calls', 0)} |")
        lines.append(f"| tool_calls | {block.get('tool_calls', 0)} |")
        lines.append("")
        lines.append("### Timeline (chronological within this bucket)")
        lines.append("")
        lines.append("| # | type | name | duration_ms | subphase |")
        lines.append("| --- | --- | --- | --- | --- |")
        for i, ev in enumerate(block.get("timeline", []), 1):
            lines.append(
                f"| {i} | {ev.get('type')} | {ev.get('name', '')[:60]} | "
                f"{ev.get('duration_ms', '')} | {ev.get('subphase', '')} |"
            )
        lines.append("")
        lines.append("### LLM calls (detail)")
        lines.append("")
        llm_n = 0
        for ev in block.get("timeline", []):
            if ev.get("type") != "llm":
                continue
            llm_n += 1
            lines.append(f"#### LLM #{llm_n}: {ev.get('name', '')}")
            lines.append("")
            lines.append("**Input excerpt**")
            lines.append("")
            lines.append("```text")
            lines.append(str(ev.get("input_excerpt", ""))[:12000])
            lines.append("```")
            lines.append("")
            lines.append("**Output excerpt**")
            lines.append("")
            lines.append("```text")
            lines.append(str(ev.get("output_excerpt", ""))[:12000])
            lines.append("```")
            lines.append("")
        lines.append("### Tool calls (detail)")
        lines.append("")
        tool_n = 0
        for ev in block.get("timeline", []):
            if ev.get("type") != "tool":
                continue
            tool_n += 1
            lines.append(f"#### Tool #{tool_n}: `{ev.get('name', '')}`")
            lines.append("")
            lines.append("**Input excerpt**")
            lines.append("")
            lines.append("```text")
            lines.append(str(ev.get("input_excerpt", ""))[:12000])
            lines.append("```")
            lines.append("")
            lines.append("**Output excerpt**")
            lines.append("")
            lines.append("```text")
            lines.append(str(ev.get("output_excerpt", ""))[:12000])
            lines.append("```")
            lines.append("")

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines), encoding="utf-8")
