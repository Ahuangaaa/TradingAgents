import pandas as pd
from stockstats import wrap
from typing import Annotated


def _clean_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize a stock DataFrame for stockstats: parse dates, drop invalid rows, fill price gaps."""
    # Tushare ``trade_date`` is typically YYYYMMDD. After CSV round-trips this
    # column may be inferred as int (e.g. 20260512). ``pd.to_datetime`` on ints
    # treats them as unix-ns and yields 1970 dates, so parse 8-digit values
    # explicitly as calendar dates first, then fall back to generic parsing.
    date_raw = data["Date"].astype(str).str.strip()
    ymd_mask = date_raw.str.fullmatch(r"\d{8}")
    parsed = pd.Series(pd.NaT, index=data.index, dtype="datetime64[ns]")
    if ymd_mask.any():
        parsed.loc[ymd_mask] = pd.to_datetime(
            date_raw.loc[ymd_mask], format="%Y%m%d", errors="coerce"
        )
    if (~ymd_mask).any():
        parsed.loc[~ymd_mask] = pd.to_datetime(
            date_raw.loc[~ymd_mask], errors="coerce"
        )
    data["Date"] = parsed
    data = data.dropna(subset=["Date"])

    price_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in data.columns]
    data[price_cols] = data[price_cols].apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=["Close"])
    data[price_cols] = data[price_cols].ffill().bfill()

    return data


def load_ohlcv(symbol: str, curr_date: str) -> pd.DataFrame:
    """Fetch OHLCV via Tushare with caching, filtered to prevent look-ahead bias."""
    from .tushare_data import _tushare_load_ohlcv

    return _tushare_load_ohlcv(symbol, curr_date)


def filter_financials_by_date(data: pd.DataFrame, curr_date: str) -> pd.DataFrame:
    """Drop financial statement columns (fiscal period timestamps) after curr_date.

    Some providers expose fiscal period end dates as columns; columns after
    ``curr_date`` represent future data and are removed to prevent look-ahead bias.
    """
    if not curr_date or data.empty:
        return data
    cutoff = pd.Timestamp(curr_date)
    mask = pd.to_datetime(data.columns, errors="coerce") <= cutoff
    return data.loc[:, mask]


class StockstatsUtils:
    @staticmethod
    def get_stock_stats(
        symbol: Annotated[str, "ticker symbol for the company"],
        indicator: Annotated[
            str, "quantitative indicators based off of the stock data for the company"
        ],
        curr_date: Annotated[
            str, "curr date for retrieving stock price data, YYYY-mm-dd"
        ],
    ):
        data = load_ohlcv(symbol, curr_date)
        df = wrap(data)
        df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
        curr_date_str = pd.to_datetime(curr_date).strftime("%Y-%m-%d")

        df[indicator]  # trigger stockstats to calculate the indicator
        matching_rows = df[df["Date"].str.startswith(curr_date_str)]

        if not matching_rows.empty:
            indicator_value = matching_rows[indicator].values[0]
            return indicator_value
        else:
            return "N/A: Not a trading day (weekend or holiday)"
