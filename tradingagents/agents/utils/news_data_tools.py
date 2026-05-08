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

    Official Tushare API docs (use ``fetch_url`` on tushare.pro to align field meanings per section):
    npr — https://tushare.pro/document/2?doc_id=406 | https://tushare.pro/wctapi/documents/406.md
    major_news — https://tushare.pro/document/2?doc_id=195 | https://tushare.pro/wctapi/documents/195.md
    news — https://tushare.pro/document/2?doc_id=143 | https://tushare.pro/wctapi/documents/143.md
    irm_qa_sh — https://tushare.pro/document/2?doc_id=366 | https://tushare.pro/wctapi/documents/366.md
    irm_qa_sz — https://tushare.pro/document/2?doc_id=367 | https://tushare.pro/wctapi/documents/367.md
    anns_d — https://tushare.pro/document/2?doc_id=176 | https://tushare.pro/wctapi/documents/176.md
    research_report — https://tushare.pro/document/2?doc_id=415 | https://tushare.pro/wctapi/documents/415.md

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

    Official Tushare API docs (use ``fetch_url`` on tushare.pro to align field meanings):
    npr — https://tushare.pro/document/2?doc_id=406 | https://tushare.pro/wctapi/documents/406.md
    major_news — https://tushare.pro/document/2?doc_id=195 | https://tushare.pro/wctapi/documents/195.md
    news — https://tushare.pro/document/2?doc_id=143 | https://tushare.pro/wctapi/documents/143.md
    research_report — https://tushare.pro/document/2?doc_id=415 | https://tushare.pro/wctapi/documents/415.md
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
    https://tushare.pro/document/2?doc_id=175
    https://tushare.pro/wctapi/documents/175.md

    Args:
        ticker (str): Ticker symbol of the company
    Returns:
        str: A report of insider transaction data
    """
    return route_to_vendor("get_insider_transactions", ticker)
