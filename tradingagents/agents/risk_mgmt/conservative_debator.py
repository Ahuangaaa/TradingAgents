from tradingagents.agents.utils.agent_utils import get_language_instruction
from tradingagents.dataflows.run_trace_context import analyst_llm_phase


def create_conservative_debator(llm):
    def conservative_node(state) -> dict:
        with analyst_llm_phase("conservative_analyst"):
            risk_debate_state = state["risk_debate_state"]
            history = risk_debate_state.get("history", "")
            conservative_history = risk_debate_state.get("conservative_history", "")

            current_aggressive_response = risk_debate_state.get("current_aggressive_response", "")
            current_neutral_response = risk_debate_state.get("current_neutral_response", "")

            market_research_report = state["market_report"]
            sentiment_report = state["sentiment_report"]
            news_report = state["news_report"]
            fundamentals_report = state["fundamentals_report"]
            deep_checklist = state.get("deep_fundamental_checklist_report", "")

            trader_decision = state["trader_investment_plan"]

            prompt = f"""As the Conservative Risk Analyst, your primary objective is to protect assets, minimize volatility, and ensure steady, reliable growth. You prioritize stability, security, and risk mitigation, carefully assessing potential losses, economic downturns, and market volatility. When evaluating the trader's decision or plan, critically examine high-risk elements, pointing out where the decision may expose the firm to undue risk and where more cautious alternatives could secure long-term gains. Here is the trader's decision:

{trader_decision}

Your task is to actively counter the arguments of the Aggressive and Neutral Analysts, highlighting where their views may overlook potential threats or fail to prioritize sustainability. Respond directly to their points, drawing from the following data sources to build a convincing case for a low-risk approach adjustment to the trader's decision:

Market Research Report: {market_research_report}
Social Media Sentiment Report: {sentiment_report}
Latest World Affairs Report: {news_report}
Company Fundamentals Report: {fundamentals_report}
Deep fundamental checklist: {deep_checklist}
Here is the current conversation history: {history} Here is the last response from the aggressive analyst: {current_aggressive_response} Here is the last response from the neutral analyst: {current_neutral_response}. If there are no responses from the other viewpoints yet, present your own argument based on the available data.

Engage by questioning their optimism and emphasizing the potential downsides they may have overlooked. Address each of their counterpoints to showcase why a conservative stance is ultimately the safest path for the firm's assets. Focus on debating and critiquing their arguments to demonstrate the strength of a low-risk strategy over their approaches. Output conversationally as if you are speaking without any special formatting.{get_language_instruction()}"""
            prompt += """

Moderate news-timeliness rule:
- In this round, explicitly evaluate recency for the main news risks and reference 3-5 highest-impact items (prefer within ~30 days).
- Avoid over-warning from stale headlines; if news is old/unclear, explicitly down-weight it and rely more on market/fundamental risk evidence.
- Data-first rule:
- For each major news/report risk you cite, include at least one concrete metric (number + date + what it measures), such as earnings miss, margin compression, capex/order slowdown, valuation drawdown, or macro indicator change.
- If a risk claim lacks measurable data support, explicitly treat it as low-confidence/low-weight.
- Balance rule (mandatory):
- After extracting news risks, explicitly anchor them in financial-fundamental risk checks (earnings quality, leverage/liquidity, valuation downside) and short-term technical risk checks (recent 5-20 trading day trend/momentum/volatility).
- If news-based risk is not confirmed by either fundamentals or short-term technicals, lower its weight and avoid over-warning.
"""

            response = llm.invoke(prompt)

            argument = f"Conservative Analyst: {response.content}"

            new_risk_debate_state = {
                "history": history + "\n" + argument,
                "aggressive_history": risk_debate_state.get("aggressive_history", ""),
                "conservative_history": conservative_history + "\n" + argument,
                "neutral_history": risk_debate_state.get("neutral_history", ""),
                "latest_speaker": "Conservative",
                "current_aggressive_response": risk_debate_state.get(
                    "current_aggressive_response", ""
                ),
                "current_conservative_response": argument,
                "current_neutral_response": risk_debate_state.get(
                    "current_neutral_response", ""
                ),
                "count": risk_debate_state["count"] + 1,
            }

            return {"risk_debate_state": new_risk_debate_state}

    return conservative_node
