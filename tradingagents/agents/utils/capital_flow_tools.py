from langchain_core.tools import tool
from typing import Annotated, Optional

from tradingagents.dataflows.interface import route_to_vendor


@tool
def get_moneyflow_mkt_dc(
    end_date: Annotated[str, "Analysis / trade date (yyyy-mm-dd); window ends on this date"],
    lookback_days: Annotated[int, "Number of SSE trading days to include"] = 30,
) -> str:
    """
    Broad-market money flow (Eastmoney ``moneyflow_mkt_dc``): SH/SZ index moves vs
    main-force net inflow and large/mid/small order splits. Default: last 30 trading days.

    Docs: https://tushare.pro/wctapi/documents/345.md
    """
    return route_to_vendor("get_moneyflow_mkt_dc", end_date, lookback_days)


@tool
def get_moneyflow_hsgt(
    end_date: Annotated[str, "Analysis / trade date (yyyy-mm-dd)"],
    lookback_days: Annotated[int, "Number of SSE trading days to include"] = 30,
) -> str:
    """
    Stock Connect northbound/southbound daily flows (``moneyflow_hsgt``). Default: 30 trading days.

    Docs: https://tushare.pro/wctapi/documents/47.md
    """
    return route_to_vendor("get_moneyflow_hsgt", end_date, lookback_days)


@tool
def get_moneyflow_ind_ths(
    end_date: Annotated[str, "Analysis / trade date (yyyy-mm-dd)"],
    lookback_days: Annotated[int, "Number of SSE trading days to include"] = 30,
    ticker: Annotated[
        Optional[str],
        "Optional focal ticker to highlight disclosed industry in rollup",
    ] = None,
) -> str:
    """
    THS industry-sector money flow (``moneyflow_ind_ths``): per-day top in/out industries
    plus period cumulative leaders. ``net_amount`` in 亿元. Default: 30 trading days.

    Docs: https://tushare.pro/wctapi/documents/343.md
    """
    return route_to_vendor("get_moneyflow_ind_ths", end_date, lookback_days, ticker)


@tool
def get_moneyflow_cnt_ths(
    end_date: Annotated[str, "Analysis / trade date (yyyy-mm-dd)"],
    lookback_days: Annotated[int, "Number of SSE trading days to include"] = 30,
    ticker: Annotated[
        Optional[str],
        "Optional focal ticker to cross-check disclosed industry vs hot concepts",
    ] = None,
) -> str:
    """
    THS concept-plate money flow (``moneyflow_cnt_ths``): per-day top in/out concepts
    plus period cumulative leaders. ``net_amount`` in 亿元. Default: 30 trading days.

    Docs: https://tushare.pro/wctapi/documents/371.md
    """
    return route_to_vendor("get_moneyflow_cnt_ths", end_date, lookback_days, ticker)
