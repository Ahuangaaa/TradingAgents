"""Post-analyst synthesis: one checklist-shaped report for downstream debate and risk."""

from __future__ import annotations

from tradingagents.agents.prompts.deep_fundamental_checklist import (
    CHECKLIST_OUTLINE,
    SYNTHESIS_SYSTEM_INSTRUCTIONS,
)
from tradingagents.agents.utils.agent_utils import get_language_instruction
from tradingagents.dataflows.run_trace_context import analyst_llm_phase


def create_deep_fundamental_checklist(llm):
    """Synthesize market/sentiment/news/fundamentals into the deep checklist (no tools)."""

    def deep_fundamental_checklist_node(state) -> dict:
        with analyst_llm_phase("deep_fundamental_checklist"):
            def _section(title: str, body: str) -> str:
                body = (body or "").strip()
                if not body:
                    return f"### {title}\n_(本节上游分析师输出为空。)_"
                return f"### {title}\n{body}"

            bundle = "\n\n".join(
                [
                    _section("Market / technical context", state.get("market_report", "")),
                    _section("Social sentiment", state.get("sentiment_report", "")),
                    _section("News", state.get("news_report", "")),
                    _section(
                        "Fundamentals (filings & statements)",
                        state.get("fundamentals_report", ""),
                    ),
                ]
            )

            user = f"""Use ONLY the following analyst outputs as factual sources.

---

{bundle}

---

## Checklist template to fill (reproduce ALL headings in your answer, then answer under each `####` item)

{CHECKLIST_OUTLINE}
"""

            messages = [
                (
                    "system",
                    SYNTHESIS_SYSTEM_INSTRUCTIONS + get_language_instruction(),
                ),
                ("human", user),
            ]
            response = llm.invoke(messages)
            return {"deep_fundamental_checklist_report": response.content}

    return deep_fundamental_checklist_node
