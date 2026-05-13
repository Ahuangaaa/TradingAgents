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
    get_global_news,
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
                get_global_news,
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
                + " Use `get_news(ticker, start_date, end_date)` for company-focused corpus (**①②④⑤⑦**, including stock/industry research reports). You may also call `get_global_news(curr_date, look_back_days, limit)` for global macro context (**⑧ macro vector topic**) when it helps explain sentiment or narrative shifts. For ④⑤, read the merged Qdrant corpus directly (no LLM-screened high/medium labels). Try to look at all sources possible from social media to sentiment to news. Provide specific, actionable insights with supporting evidence to help traders make informed decisions."
                + " **Quantitative A-share positioning (Tushare — required for this report):** (1) Call `get_holder_number` with the same `ticker` and an announcement-date window wide enough for several disclosed points (typically `start_date` ~18–24 months before the analysis date through `end_date` = analysis/trade date), then explain **股东户数** changes (concentration vs dispersion), including latest-point recency, average period-to-period fluctuation, and net directional move. "
                + "**Shareholder-count risk weighting (mandatory):** Use a combined rule, not a single-point judgment. If the **latest holder-count change is highly recent** (e.g. latest disclosure within ~2 months of analysis date) **and** either (a) average report-to-report fluctuation is large, or (b) the series changes strongly in one direction (especially sustained upward move / sizable cumulative increase), assign **high risk weight** and make it a **high-priority warning**. In A-share narratives, this often maps to **筹码分散** and, with prior strength or **高位/滞涨**, may be **consistent with 主力出货（distributing into retail）**. Conversely, if average fluctuation is small, or disclosures are stale / far from current date, do **not** over-weight holder-count risk by itself. Cross-check `get_stock_moneyflow` and `get_margin_detail` for context, and keep your final risk tier explicit in conclusions and in the closing table. "
                + "(2) Call `get_stock_moneyflow` for recent trading days (e.g. last ~30–60 sessions ending on the analysis date) and interpret **大单/特大单** activity and **net_mf_amount** (large-capital flow). "
                + "(3) Call `get_margin_detail` over the same recent trading window and describe **融资融券** trends: 融资余额 `rzye`, 融资买入 `rzmre`, 融资偿还 `rzche`, etc. "
                + "Include a dedicated markdown section summarizing these three pillars before your closing table."
                + " **Citation block (mandatory):** Add a dedicated section named `## 引用来源` at the end. Include every material source you actually used from news/research reports/上证e互动/深交所互动易（深圳e互动） as a Markdown table with columns: `来源类别 | 日期 | 标题 | 证券/主题 | URL或渠道`. In `来源类别`, explicitly label one of: `新闻` / `研报` / `上证e互动` / `深交所互动易`. Do not fabricate links; if no URL is available, write `Tushare-<接口名>` as the channel."
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
