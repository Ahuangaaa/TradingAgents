from langchain_core.tools import tool
from typing import Annotated
from tradingagents.dataflows.interface import route_to_vendor


@tool
def get_fundamentals(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
) -> str:
    """
    Retrieve comprehensive fundamental data for a given ticker symbol.
    Uses the configured fundamental_data vendor (Tushare ``stock_basic`` + ``daily_basic`` + ``fina_indicator``).

    Official Tushare API docs (use ``fetch_url`` on tushare.pro to align field meanings):
    stock_basic — https://tushare.pro/wctapi/documents/25.md
    daily_basic — https://tushare.pro/wctapi/documents/32.md
    fina_indicator — https://tushare.pro/wctapi/documents/79.md
    Interpretation rule: do not quote key fields (especially valuation/market-cap fields) until you
    align each cited field with the official doc definition and unit.

    Args:
        ticker (str): Ticker symbol of the company
        curr_date (str): Current date you are trading at, yyyy-mm-dd
    Returns:
        str: A formatted report containing comprehensive fundamental data
    """
    return route_to_vendor("get_fundamentals", ticker, curr_date)


@tool
def get_industry_peers(
    ticker: Annotated[str, "focal ticker symbol (same as company_of_interest)"],
    curr_date: Annotated[
        str | None,
        "YYYY-mm-dd analysis or trade date; optional cache key for the peer list.",
    ] = None,
    max_peers: Annotated[int, "max peer rows to return (default 3)"] = 3,
) -> str:
    """
    List **business competitors** as A-shares: backend calls **DeepSeek** (dedicated Chat Completions prompt),
    then Tushare ``stock_basic`` validates ``ts_code`` and fills names — **not** a raw same-industry sort from Tushare.

    Call **before** deep `get_news` or `get_fundamentals` on competitors so tickers are explicit.

    Official docs: stock_basic — https://tushare.pro/wctapi/documents/25.md
    Interpretation rule: if you cite any returned columns, align each field meaning with docs first.
    """
    return route_to_vendor("get_industry_peers", ticker, curr_date, max_peers)


@tool
def get_balance_sheet(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[str, "reporting frequency: annual/quarterly"] = "quarterly",
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"] = None,
) -> str:
    """
    Retrieve balance sheet data for a given ticker symbol.
    Uses the configured fundamental_data vendor (Tushare ``balancesheet``).

    Official Tushare API docs (use ``fetch_url`` on tushare.pro to align column meanings):
    https://tushare.pro/wctapi/documents/36.md
    Interpretation rule: align each cited column with doc definition and unit before analysis.

    Args:
        ticker (str): Ticker symbol of the company
        freq (str): Reporting frequency: annual/quarterly (default quarterly)
        curr_date (str): Current date you are trading at, yyyy-mm-dd
    Returns:
        str: A formatted report containing balance sheet data
    """
    return route_to_vendor("get_balance_sheet", ticker, freq, curr_date)


@tool
def get_cashflow(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[str, "reporting frequency: annual/quarterly"] = "quarterly",
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"] = None,
) -> str:
    """
    Retrieve cash flow statement data for a given ticker symbol.
    Uses the configured fundamental_data vendor (Tushare ``cashflow``).

    Official Tushare API docs (use ``fetch_url`` on tushare.pro to align column meanings):
    https://tushare.pro/wctapi/documents/44.md
    Interpretation rule: align each cited column with doc definition and unit before analysis.

    Args:
        ticker (str): Ticker symbol of the company
        freq (str): Reporting frequency: annual/quarterly (default quarterly)
        curr_date (str): Current date you are trading at, yyyy-mm-dd
    Returns:
        str: A formatted report containing cash flow statement data
    """
    return route_to_vendor("get_cashflow", ticker, freq, curr_date)


@tool
def get_income_statement(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[str, "reporting frequency: annual/quarterly"] = "quarterly",
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"] = None,
) -> str:
    """
    Retrieve income statement data for a given ticker symbol.
    Uses the configured fundamental_data vendor (Tushare ``income``).

    Official Tushare API docs (use ``fetch_url`` on tushare.pro to align column meanings):
    https://tushare.pro/wctapi/documents/33.md
    Interpretation rule: align each cited column with doc definition and unit before analysis.

    Args:
        ticker (str): Ticker symbol of the company
        freq (str): Reporting frequency: annual/quarterly (default quarterly)
        curr_date (str): Current date you are trading at, yyyy-mm-dd
    Returns:
        str: A formatted report containing income statement data
    """
    return route_to_vendor("get_income_statement", ticker, freq, curr_date)