from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.dataflows.run_trace_context import analyst_llm_phase
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_global_news,
    get_industry_peers,
    get_language_instruction,
    get_news,
    get_web_fetch_tool_hint,
)
from tradingagents.agents.utils.web_fetch_tool import fetch_url


def create_news_analyst(llm):
    def news_analyst_node(state):
        with analyst_llm_phase("news"):
            current_date = state["trade_date"]
            company = state["company_of_interest"]
            instrument_context = build_instrument_context(company)

            tools = [
                get_industry_peers,
                get_news,
                get_global_news,
                fetch_url,
            ]

            ticker_guard = (
                f" For company-specific pulls, call `get_news` with `ticker` exactly `{company}` — "
                "not a similar code or name (e.g. 002750 vs 002751, 龙津药业 vs 龙泉股份)."
            )

            system_message = (
                "You are a news researcher tasked with analyzing recent news and trends over the past week. Please write a comprehensive report of the current state of the world that is relevant for trading and macroeconomics. Use the available tools: "
                "`get_industry_peers(ticker, curr_date, max_peers)` to list **model-selected** listed A-share competitors (DeepSeek + code validation; not a raw Tushare industry sort). Call first with the exact focal ticker and **curr_date = current analysis date**. "
                "When describing peer sourcing (Chinese or English), **must** match the tool output header: DeepSeek Chat Completions with a dedicated prompt, then `stock_basic` code check — **never** claim peers were chosen by sorting or sampling a Tushare same-industry constituent list. "
                "`get_news(ticker, start_date, end_date)` for company-specific or targeted news searches, and `get_global_news(curr_date, look_back_days, limit)` for broader macroeconomic news. Provide specific, actionable insights with supporting evidence to help traders make informed decisions."
                + " **Tool split (important):** `get_global_news` now only provides **③ npr + ⑧ macro vector topic**; `get_news` provides the remaining company-focused sections (**①②④⑤⑥⑦**, including stock/industry research reports). For ④⑤, use the merged Qdrant corpus returned by `get_news` directly (no LLM screening labels such as high/medium)."
                + " **Competitor coverage (mandatory):** Use `get_industry_peers` so peer tickers are explicit. Add a dedicated section **「主要竞争对手近期新闻与舆论要点」** tied to those peers (and the merged ④⑤ content when present); highlight contrasts vs the focal name (regulation, pricing, orders, M&A, capacity, litigation)."
                + " Also prioritize material useful for deep fundamental analysis: regulatory/product/market oversight, management or board changes, alliances/JVs/M&A, pricing power or demand shocks, labor or union developments, and tax or policy headlines affecting the company or its industry."
                + " A downstream step will consolidate findings into a structured deep fundamental checklist—surface concrete facts and dates so they can be cited there."
                + ticker_guard
                + """ Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."""
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
                report = result.content

            return {
                "messages": [result],
                "news_report": report,
            }

    return news_analyst_node
