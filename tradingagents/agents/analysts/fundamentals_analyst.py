from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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
            + " **Peer financials (mandatory):** Call `get_industry_peers(ticker, curr_date)` first with the focal ticker and **curr_date = current analysis date**. Select **3–5** closest listed peers (or all if fewer). For the **focal** ticker run `get_fundamentals` (and statements as needed). For **each chosen peer**, call `get_fundamentals` with the same `curr_date` at minimum; add `get_income_statement` for 1–2 peers only if segment-level comparison needs more detail."
            + " End the report with a markdown section **「标的 vs 竞品：关键财务对比」** containing at least one comparison table (use metrics actually returned: e.g. ROE, gross margin, net margin, revenue growth, leverage, R&D intensity—omit columns when missing and label 信息不足)."
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
