from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_internal_doc_alignment_rule,
    get_language_instruction,
    get_report_branding_rules,
    get_web_fetch_tool_hint,
)
from tradingagents.agents.utils.capital_flow_tools import (
    get_moneyflow_hsgt,
    get_moneyflow_mkt_dc,
)
from tradingagents.agents.utils.report_publish import prepare_report_for_publish
from tradingagents.agents.utils.web_fetch_tool import fetch_url
from tradingagents.dataflows.run_trace_context import analyst_llm_phase


def create_broad_market_analyst(llm):

    def broad_market_analyst_node(state):
        with analyst_llm_phase("broad_market"):
            current_date = state["trade_date"]
            instrument_context = build_instrument_context(state["company_of_interest"])

            tools = [
                get_moneyflow_mkt_dc,
                get_moneyflow_hsgt,
                fetch_url,
            ]

            system_message = (
                "You are a broad-market analyst for A-shares. Write a comprehensive report on "
                "**overall market liquidity and sentiment** over the recent window (default 30 trading days "
                "ending on the analysis date). This sets context before stock-specific work.\n"
                "**Mandatory tools:** Call `get_moneyflow_mkt_dc` and `get_moneyflow_hsgt` with "
                f"`end_date` = `{current_date}` (use default lookback_days=30 unless you have a reason to change). "
                "Interpret: SH/SZ index moves vs main-force net inflow (`net_amount`, order-size splits), "
                "northbound/southbound trends, cumulative flow over the window, and whether the last ~5 sessions "
                "confirm or contradict the 30-day picture. Relate briefly to how this backdrop affects the focal "
                "instrument (beta / risk-on vs risk-off), without duplicating stock-level microstructure.\n"
                + get_internal_doc_alignment_rule()
                + get_report_branding_rules()
                + " End with a concise Markdown summary table of key takeaways.\n"
                + get_web_fetch_tool_hint()
                + get_language_instruction()
            )

            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are a helpful AI assistant, collaborating with other assistants."
                        " Use the provided tools to progress towards answering the question."
                        " If you are unable to fully answer, that's OK; another assistant with different tools"
                        " will help where you left off. Execute what you can to make progress."
                        " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                        " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                        " You have access to the following tools: {tool_names}.\n{system_message}"
                        "For your reference, the current date is {current_date}. {instrument_context}",
                    ),
                    MessagesPlaceholder(variable_name="messages"),
                ]
            )

            prompt = prompt.partial(system_message=system_message)
            prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
            prompt = prompt.partial(current_date=current_date)
            prompt = prompt.partial(instrument_context=instrument_context)

            chain = prompt | llm.bind_tools(tools)
            result = chain.invoke(state["messages"])

            report = ""
            if len(result.tool_calls) == 0:
                report = prepare_report_for_publish(result.content)

            return {
                "messages": [result],
                "broad_market_report": report,
            }

    return broad_market_analyst_node
