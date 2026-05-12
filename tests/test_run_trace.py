import json
from pathlib import Path

from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.messages import AIMessage

from cli.run_trace_handler import RunTraceCallbackHandler
from tradingagents.dataflows.run_trace_context import analyst_llm_phase, tools_phase
from tradingagents.dataflows.trace_rollup import rollup_events_jsonl


def test_run_trace_handler_tool_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "e.jsonl"
    h = RunTraceCallbackHandler(p, max_chars=500)
    with tools_phase("news"):
        h.on_tool_start({"name": "get_news"}, "ticker=FOO", run_id="r1")
        h.on_tool_end("result text", run_id="r1")
    lines = p.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["event_type"] == "tool"
    assert row["name"] == "get_news"
    assert row["analyst_key"] == "news"
    assert row["subphase"] == "tools"
    assert row["duration_ms"] >= 0
    assert "result text" in row["output_excerpt"]


def test_run_trace_handler_llm_end(tmp_path: Path) -> None:
    p = tmp_path / "e2.jsonl"
    h = RunTraceCallbackHandler(p, max_chars=200)
    with analyst_llm_phase("market"):
        h.on_chat_model_start({}, [[]], run_id="r2")
        msg = AIMessage(content="hello out")
        gen = ChatGeneration(message=msg)
        result = LLMResult(generations=[[gen]])
        h.on_llm_end(result, run_id="r2", serialized={"kwargs": {"model": "gpt-test"}})
    row = json.loads(p.read_text(encoding="utf-8").strip().splitlines()[0])
    assert row["event_type"] == "llm"
    assert row["analyst_key"] == "market"
    assert "hello out" in row["output_excerpt"]


def test_rollup_groups_by_analyst(tmp_path: Path) -> None:
    p = tmp_path / "ev.jsonl"
    p.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event_type": "tool",
                        "name": "x",
                        "duration_ms": 10,
                        "analyst_key": "news",
                        "subphase": "tools",
                    }
                ),
                json.dumps(
                    {
                        "event_type": "llm",
                        "name": "m",
                        "duration_ms": 5,
                        "analyst_key": "news",
                        "subphase": "llm",
                    }
                ),
                json.dumps(
                    {
                        "event_type": "tool",
                        "name": "y",
                        "duration_ms": 3,
                        "analyst_key": None,
                        "subphase": None,
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    s = rollup_events_jsonl(p)
    assert "news" in s and "other" in s
    assert s["news"]["tool_calls"] == 1
    assert s["news"]["llm_calls"] == 1
    assert s["other"]["tool_calls"] == 1
