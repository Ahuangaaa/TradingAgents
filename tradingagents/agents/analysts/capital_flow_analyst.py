from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_internal_doc_alignment_rule,
    get_language_instruction,
    get_report_branding_rules,
    get_web_fetch_tool_hint,
)
from tradingagents.agents.utils.capital_flow_tools import (
    get_moneyflow_cnt_ths,
    get_moneyflow_hsgt,
    get_moneyflow_ind_ths,
)
from tradingagents.agents.utils.report_publish import prepare_report_for_publish
from tradingagents.agents.utils.web_fetch_tool import fetch_url
from tradingagents.dataflows.run_trace_context import analyst_llm_phase


def create_capital_flow_analyst(llm):

    def capital_flow_analyst_node(state):
        with analyst_llm_phase("capital_flow"):
            current_date = state["trade_date"]
            company = state["company_of_interest"]
            instrument_context = build_instrument_context(company)

            tools = [
                get_moneyflow_ind_ths,
                get_moneyflow_cnt_ths,
                get_moneyflow_hsgt,
                fetch_url,
            ]

            system_message = (
                "You are a capital-flow / sector-rotation analyst for A-shares. Analyze **industry and "
                "concept money flows** over the recent window (default 30 trading days ending on the analysis date).\n"
                "**Mandatory tools:** Call `get_moneyflow_ind_ths` and `get_moneyflow_cnt_ths` with "
                f"`end_date` = `{current_date}` and `ticker` = `{company}` so the rollup highlights the focal "
                "name's disclosed industry. Optionally call `get_moneyflow_hsgt` to cross-check northbound "
                "flows vs sector leadership.\n"
                "Cover: leading/lagging industries and concepts (period cumulative + recent daily shifts), "
                "rotation pace (stable vs whipsaw), whether focal industry's flow rank supports or contradicts "
                "a bullish/bearish single-name thesis, and risks if hot concepts diverge from fundamentals.\n"
                + get_internal_doc_alignment_rule()
                + get_report_branding_rules()
                + " End with a Markdown table: sector/concept | flow signal | relevance to focal name.\n"
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
                "capital_flow_report": report,
            }

    return capital_flow_analyst_node
