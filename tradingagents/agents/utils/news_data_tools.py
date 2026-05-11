from langchain_core.tools import tool
from typing import Annotated
from tradingagents.dataflows.interface import route_to_vendor

@tool
def get_news(
    ticker: Annotated[str, "Ticker symbol"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """
    Retrieve news and related corpus for a given ticker symbol.
    Uses the configured news_data vendor. One response may concatenate multiple Tushare APIs
    (e.g. interactive Q&A, policy, long-form news, flash news, announcements, research reports).
    When ``news_long_short_use_qdrant`` is true (or env ``NEWS_LONG_SHORT_USE_QDRANT=1``), sections ④⑤
    (``major_news`` / flash ``news``) are sourced from a **Qdrant** vector collection instead of live
    Tushare pulls; **section ⑧** adds a **separate** macro-only vector query (and optional LLM summary),
    not mixed with ④⑤. Install ``.[qdrant-news]`` and run the ingest pipeline under ``qdrant/`` first.

    Official Tushare API docs (use ``fetch_url`` on tushare.pro to align field meanings per section):
    npr — https://tushare.pro/wctapi/documents/406.md
    major_news — https://tushare.pro/wctapi/documents/195.md
    news — https://tushare.pro/wctapi/documents/143.md
    irm_qa_sh — https://tushare.pro/wctapi/documents/366.md
    irm_qa_sz —  https://tushare.pro/wctapi/documents/367.md
    anns_d —  https://tushare.pro/wctapi/documents/176.md
    research_report —  https://tushare.pro/wctapi/documents/415.md

    Args:
        ticker (str): Ticker symbol
        start_date (str): Start date in yyyy-mm-dd format
        end_date (str): End date in yyyy-mm-dd format
    Returns:
        str: A formatted string containing news data
    """
    return route_to_vendor("get_news", ticker, start_date, end_date)

@tool
def get_global_news(
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "Number of days to look back"] = 7,
    limit: Annotated[int, "Maximum number of articles to return"] = 5,
) -> str:
    """
    Retrieve global macro-oriented news corpus (no ticker-scoped interactive Q&A or ``anns_d`` in this path).
    Uses the configured news_data vendor. Output may combine ``npr``, ``major_news``, ``news``, and ``research_report``.
    When ``news_long_short_use_qdrant`` is enabled, the long-form / flash blocks are filled from **Qdrant**;
    **section ⑧** uses a **dedicated macro retrieval query** (and optional LLM), separate from ④⑤ (see ``get_news`` tool notes).

    Official Tushare API docs (use ``fetch_url`` on tushare.pro to align field meanings):
    npr — https://tushare.pro/wctapi/documents/406.md
    major_news — https://tushare.pro/wctapi/documents/195.md
    news — https://tushare.pro/wctapi/documents/143.md
    research_report — https://tushare.pro/wctapi/documents/415.md
    For per-stock announcements and full seven-section corpus, use ``get_news`` (see its docs for ``anns_d`` / ``irm_qa_*``).

    Args:
        curr_date (str): Current date in yyyy-mm-dd format
        look_back_days (int): Number of days to look back (default 7)
        limit (int): Maximum number of articles to return (default 5)
    Returns:
        str: A formatted string containing global news data
    """
    return route_to_vendor("get_global_news", curr_date, look_back_days, limit)

@tool
def get_insider_transactions(
    ticker: Annotated[str, "ticker symbol"],
) -> str:
    """
    Retrieve shareholder increase/decrease (insider-style holder trade) data for a ticker.
    Uses the configured news_data vendor (Tushare ``stk_holdertrade``).

    Official Tushare API docs (use ``fetch_url`` on tushare.pro to align column meanings):
    https://tushare.pro/wctapi/documents/175.md

    Args:
        ticker (str): Ticker symbol of the company
    Returns:
        str: A report of insider transaction data
    """
    return route_to_vendor("get_insider_transactions", ticker)


@tool
def get_holder_number(
    ticker: Annotated[str, "Ticker symbol (e.g. 600519.SH)"],
    start_date: Annotated[str, "Start of announcement-date window (yyyy-mm-dd); use ~18–24 months before analysis for trend"],
    end_date: Annotated[str, "End of announcement-date window (yyyy-mm-dd), usually the analysis / trade date"],
) -> str:
    """
    Shareholder count (户数) from Tushare ``stk_holdernumber`` (by ann_date).

    **When analyzing output:** treat **sustained or sharp increases** in holder_num as **high-weight risk**—often read as **筹码分散** and, especially after a rally or at elevated prices, as **consistent with possible 主力出货** (distribution). Do not bury this below positive flow or margin data; surface it prominently in risk sections.
    Docs: https://tushare.pro/wctapi/documents/166.md
    """
    return route_to_vendor("get_holder_number", ticker, start_date, end_date)


@tool
def get_stock_moneyflow(
    ticker: Annotated[str, "Ticker symbol"],
    start_date: Annotated[str, "First trade date (yyyy-mm-dd)"],
    end_date: Annotated[str, "Last trade date (yyyy-mm-dd)"],
) -> str:
    """
    Daily A-share money flow with large/extra-large order breakdown (Tushare ``moneyflow``).
    Use net_mf_amount, buy_lg/buy_elg vs sell for **大资金** flow interpretation.

    Requires sufficient Tushare积分 (~2000+). Docs: https://tushare.pro/wctapi/documents/170.md
    """
    return route_to_vendor("get_stock_moneyflow", ticker, start_date, end_date)


@tool
def get_margin_detail(
    ticker: Annotated[str, "Ticker symbol"],
    start_date: Annotated[str, "First trade date (yyyy-mm-dd)"],
    end_date: Annotated[str, "Last trade date (yyyy-mm-dd)"],
) -> str:
    """
    Margin trading detail per stock (Tushare ``margin_detail``): 融资余额 rzye, 融资买入 rzmre,
    融资偿还 rzche, 融券余量 rqyl, etc. Describe **融资** balance and activity trends over the window.

    Docs: https://tushare.pro/wctapi/documents/59.md
    """
    return route_to_vendor("get_margin_detail", ticker, start_date, end_date)
