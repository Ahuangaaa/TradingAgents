from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.dataflows.run_trace_context import analyst_llm_phase
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_balance_sheet,
    get_cashflow,
    get_fundamentals,
    get_industry_peers,
    get_income_statement,
    get_language_instruction,
    get_web_fetch_tool_hint,
)
from tradingagents.agents.utils.web_fetch_tool import fetch_url


def create_fundamentals_analyst(llm):
    def fundamentals_analyst_node(state):
        with analyst_llm_phase("fundamentals"):
            current_date = state["trade_date"]
            instrument_context = build_instrument_context(state["company_of_interest"])

            tools = [
                get_industry_peers,
                get_fundamentals,
                get_balance_sheet,
                get_cashflow,
                get_income_statement,
                fetch_url,
            ]

            system_message = (
                "You are a researcher tasked with analyzing fundamental information over the past week about a company. Please write a comprehensive report of the company's fundamental information such as financial documents, company profile, basic company financials, and company financial history to gain a full view of the company's fundamental information to inform traders. Make sure to include as much detail as possible. Provide specific, actionable insights with supporting evidence to help traders make informed decisions."
                + " **Peer financials (mandatory):** Call `get_industry_peers(ticker, curr_date)` first with the focal ticker and **curr_date = current analysis date**. Peers are produced by **DeepSeek** (dedicated Chat Completions prompt) and validated via `stock_basic`; **do not** describe them as picked from a Tushare same-industry ranked list. Select **3–5** peers (or all if fewer) from the tool table. For the **focal** ticker run `get_fundamentals` (and statements as needed). For **each chosen peer**, call `get_fundamentals` with the same `curr_date` at minimum; add `get_income_statement` for 1–2 peers only if segment-level comparison needs more detail."
                + " **Doc alignment (mandatory):** Before interpreting fields from `get_fundamentals`, `get_balance_sheet`, `get_cashflow`, or `get_income_statement`, fetch the corresponding official Tushare docs via `fetch_url`. In the report, do not use a field unless you have aligned its meaning and unit with the doc."
                + " End the report with a markdown section **「标的 vs 竞品：关键财务对比」** containing at least one comparison table (use metrics actually returned: e.g. ROE, gross margin, net margin, revenue growth, leverage, R&D intensity—omit columns when missing and label 信息不足)."
                + " Add a short section **「字段释义与单位对齐（文档核对）」** listing key fields you used in conclusions with columns: `接口 | 字段 | 文档释义 | 单位/口径 | 本文用法`."
                + " Emphasize evidence that feeds a deep fundamental checklist: revenue or segment/product line breakdowns; cost of revenue, operating expenses, and capex (scale and cost-structure clues); R&D spending and intangible assets; working capital and supply-chain hints from payables/inventory; borrowing and tax notes; governance (board composition, related-party, share-based compensation) from filings; and patent/R&D or principal-product descriptions from annual reports. Use `fetch_url` on official IR or exchange filings when tools omit these details."
                + " A downstream step will merge this with news into a structured checklist—quote numbers and filing context where you can."
                + " Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."
                + " Use the available tools: `get_industry_peers`, `get_fundamentals`, `get_balance_sheet`, `get_cashflow`, and `get_income_statement`."
                + get_web_fetch_tool_hint()
                + get_language_instruction(),
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
                "fundamentals_report": report,
            }

    return fundamentals_analyst_node
