from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.dataflows.run_trace_context import analyst_llm_phase
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_industry_peers,
    get_language_instruction,
    get_web_fetch_tool_hint,
)
from tradingagents.agents.utils.news_data_tools import (
    get_news,
    get_holder_number,
    get_stock_moneyflow,
    get_margin_detail,
)
from tradingagents.agents.utils.web_fetch_tool import fetch_url


def create_social_media_analyst(llm):
    def social_media_analyst_node(state):
        with analyst_llm_phase("social"):
            current_date = state["trade_date"]
            company = state["company_of_interest"]
            instrument_context = build_instrument_context(company)

            tools = [
                get_industry_peers,
                get_news,
                get_holder_number,
                get_stock_moneyflow,
                get_margin_detail,
                fetch_url,
            ]

            ticker_guard = (
                f" When calling `get_news`, you MUST pass `ticker` exactly as `{company}` (character-for-character). "
                "Do not substitute a nearby stock code (e.g. 002750 vs 002751) or a confused similar name "
                "(龙津药业 vs 龙泉股份); wrong `ticker` returns another company's news."
                f" Use this **exact same** `ticker` for `get_holder_number`, `get_stock_moneyflow`, and `get_margin_detail`."
            )

            system_message = (
                "You are a social media and company specific news researcher/analyst tasked with analyzing social media posts, recent company news, and public sentiment for a specific company over the past week. You will be given a company's name your objective is to write a comprehensive long report detailing your analysis, insights, and implications for traders and investors on this company's current state after looking at social media and what people are saying about that company, analyzing sentiment data of what people feel each day about the company, and looking at recent company news. "
                + " Call `get_industry_peers(ticker, curr_date)` **first** with the focal ticker and **curr_date = current analysis date**; peers come from **DeepSeek** (dedicated prompt) plus `stock_basic` code check — **not** a Tushare same-industry pool sort. Pick **at most 1–2** peer tickers for supplementary `get_news` only (互动易/短讯舆论角度；**不要**对多家竞品重复调用 `get_holder_number` / `get_stock_moneyflow` / `get_margin_detail`，以免工具爆炸). In prose, **never** claim peers were chosen by sampling Tushare industry constituents. Briefly relate peer tone vs focal when those pulls add something the News analyst may not emphasize."
                + " Use the get_news(ticker, start_date, end_date) tool to search for company-specific news and social media discussions (tool output includes **LLM-screened** long/flash news blocks ④⑤—use relevance labels and excerpts). Try to look at all sources possible from social media to sentiment to news. Provide specific, actionable insights with supporting evidence to help traders make informed decisions."
                + " **Quantitative A-share positioning (Tushare — required for this report):** (1) Call `get_holder_number` with the same `ticker` and an announcement-date window wide enough for several disclosed points (typically `start_date` ~18–24 months before the analysis date through `end_date` = analysis/trade date) and explain **股东户数** changes (concentration vs dispersion). "
                + "**Shareholder-count risk weighting (mandatory):** If holder counts **rise across multiple consecutive disclosures** or jump **materially** vs the prior report, you MUST **upgrade the prominence of this signal**—it is a **dangerous / high-priority warning**, not a minor data point. In A-share narratives, **持续攀升或大幅增加的股东户数** often implies **筹码分散**; combined with prior strength or **高位/滞涨**, analysts frequently flag it as evidence **consistent with 主力出货（distributing into retail）**—not definitive alone, but you **must not soft-pedal** it when the trend is clear or the increase is large. Cross-check `get_stock_moneyflow` and `get_margin_detail`, but **benign moneyflow or margin does not cancel** a sustained, large rise in holder count—keep the **户数风险** visible in conclusions and in your closing table (e.g. risk tier or priority row). "
                + "(2) Call `get_stock_moneyflow` for recent trading days (e.g. last ~30–60 sessions ending on the analysis date) and interpret **大单/特大单** activity and **net_mf_amount** (large-capital flow). "
                + "(3) Call `get_margin_detail` over the same recent trading window and describe **融资融券** trends: 融资余额 `rzye`, 融资买入 `rzmre`, 融资偿还 `rzche`, etc. "
                + "Include a dedicated markdown section summarizing these three pillars before your closing table."
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
                "sentiment_report": report,
            }

    return social_media_analyst_node
