from langchain_core.messages import HumanMessage, RemoveMessage

# Import tools from separate utility files
from tradingagents.agents.utils.core_stock_tools import (
    get_stock_data
)
from tradingagents.agents.utils.technical_indicators_tools import (
    get_indicators
)
from tradingagents.agents.utils.fundamental_data_tools import (
    get_fundamentals,
    get_industry_peers,
    get_balance_sheet,
    get_cashflow,
    get_income_statement
)
from tradingagents.agents.utils.news_data_tools import (
    get_news,
    get_insider_transactions,
    get_global_news,
    get_holder_number,
    get_stock_moneyflow,
    get_margin_detail,
)


def get_language_instruction() -> str:
    """Return a prompt instruction for the configured output language.

    Returns empty string when English (default), so no extra tokens are used.
    Applied to analysts, bull/bear researchers, research manager, trader, risk
    debators, and portfolio manager so saved markdown matches the CLI
    ``output_language`` choice. (Structured enums such
    as Buy/Hold/Sell stay in English for schema compatibility.)
    """
    from tradingagents.dataflows.config import get_config
    lang = get_config().get("output_language", "English")
    if lang.strip().lower() == "english":
        return ""
    return f" Write your entire response in {lang}."


def get_web_fetch_tool_hint() -> str:
    """Prompt snippet: when to call ``fetch_url`` for official docs (allowlisted https only)."""
    return (
        " Documentation alignment is mandatory when interpreting API fields: before using key fields from any"
        " data tool, call `fetch_url` with the official interface doc URL (full **https** on allowlisted hosts),"
        " then explicitly align each cited field with the doc definition (name, meaning, and unit)."
        " Do not guess field semantics, units, or formulas from memory."
    )


def get_internal_doc_alignment_rule() -> str:
    """Mandatory doc alignment during analysis; must not appear as a report appendix."""
    return (
        " **Field-doc alignment (mandatory, internal):** Before using API fields in conclusions,"
        " call `fetch_url` on the official interface documentation and align each field's meaning and unit."
        " Perform this verification during tool use and reasoning;"
        " **do not** add a separate 「字段释义与单位对齐」 section, table, or appendix to the deliverable report."
    )


def get_report_branding_rules() -> str:
    """Client-facing report must not name vendors or infrastructure brands."""
    return (
        " **Deliverable style:** The report is client-facing. Do not mention vendor, provider, or"
        " infrastructure brand names (data platforms, LLM products, vector databases, etc.)."
        " Describe sources generically (e.g. 行情数据、官方接口文档、新闻库、推理筛选)."
        " For citations without a URL, use `数据渠道-<简称>` instead of vendor-prefixed labels."
    )


def get_industry_peer_instruction() -> str:
    """How analysts should describe competitor sourcing in prose."""
    return (
        " Call `get_industry_peers(ticker, curr_date)` first with the focal ticker and"
        " **curr_date = current analysis date**. In the report, describe peers using the tool header:"
        " model-inferred listed competitors with exchange code validation —"
        " **not** an industry-constituent or same-industry mechanical sample."
    )


def build_instrument_context(ticker: str) -> str:
    """Describe the exact instrument so agents preserve exchange-qualified tickers."""
    return (
        f"The instrument to analyze is `{ticker}`. "
        "Use this exact ticker in every tool call, report, and recommendation, "
        "preserving any exchange suffix (e.g. `.TO`, `.L`, `.HK`, `.T`)."
    )

def create_msg_delete():
    def delete_messages(state):
        """Clear messages and add placeholder for Anthropic compatibility"""
        messages = state["messages"]

        # Remove all messages
        removal_operations = [RemoveMessage(id=m.id) for m in messages]

        # Add a minimal placeholder message
        placeholder = HumanMessage(content="Continue")

        return {"messages": removal_operations + [placeholder]}

    return delete_messages


        
