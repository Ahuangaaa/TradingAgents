"""Research Manager: turns the bull/bear debate into a structured investment plan for the trader."""

from __future__ import annotations

from tradingagents.agents.schemas import ResearchPlan, render_research_plan
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)
from tradingagents.dataflows.run_trace_context import analyst_llm_phase


def create_research_manager(llm):
    structured_llm = bind_structured(llm, ResearchPlan, "Research Manager")

    def research_manager_node(state) -> dict:
        with analyst_llm_phase("research_manager"):
            instrument_context = build_instrument_context(state["company_of_interest"])
            history = state["investment_debate_state"].get("history", "")

            investment_debate_state = state["investment_debate_state"]
            deep_checklist = state.get("deep_fundamental_checklist_report", "")

            prompt = f"""As the Research Manager and debate facilitator, your role is to critically evaluate this round of debate and deliver a clear, actionable investment plan for the trader.

{instrument_context}

---

**Rating Scale** (use exactly one):
- **Buy**: Strong conviction in the bull thesis; recommend taking or growing the position
- **Overweight**: Constructive view; recommend gradually increasing exposure
- **Hold**: Balanced view; recommend maintaining the current position
- **Underweight**: Cautious view; recommend trimming exposure
- **Sell**: Strong conviction in the bear thesis; recommend exiting or avoiding the position

Commit to a clear stance whenever the debate's strongest arguments warrant one; reserve Hold for situations where the evidence on both sides is genuinely balanced.

---

**Deep fundamental checklist (structured; align your thesis and risks with it, including explicit "information insufficient" items):**
{deep_checklist}

---

**Debate History:**
{history}{get_language_instruction()}"""
            prompt += """

Moderate news-timeliness calibration:
- Explicitly include one concise news-timeliness judgment in your plan (which news remains valid now, which is stale).
- Prefer 1-3 highest-impact recent news points (within ~30 days); if key news evidence is stale/weak, down-weight it and prioritize stronger market/fundamental signals.
- Data-first calibration:
- When using news/report evidence in your plan, prioritize quantified facts (number + date + metric definition) over narrative statements.
- If key news/report claims are not supported by measurable data, explicitly mark them as low-confidence and avoid making them thesis-critical.
"""

            investment_plan = invoke_structured_or_freetext(
                structured_llm,
                llm,
                prompt,
                render_research_plan,
                "Research Manager",
            )

            new_investment_debate_state = {
                "judge_decision": investment_plan,
                "history": investment_debate_state.get("history", ""),
                "bear_history": investment_debate_state.get("bear_history", ""),
                "bull_history": investment_debate_state.get("bull_history", ""),
                "current_response": investment_plan,
                "count": investment_debate_state["count"],
            }

            return {
                "investment_debate_state": new_investment_debate_state,
                "investment_plan": investment_plan,
            }

    return research_manager_node
