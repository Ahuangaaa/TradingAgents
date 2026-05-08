"""Tushare-backed implementations for ``route_to_vendor`` (A-share + Hong Kong; see project skill references)."""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Annotated

import pandas as pd
from dateutil.relativedelta import relativedelta
from stockstats import wrap

from .config import get_config
from .stockstats_utils import _clean_dataframe
from .tushare_common import (
    TushareVendorError,
    get_pro,
    resolve_tushare_equity,
    to_yyyymmdd,
)
from .utils import safe_ticker_component


def _df_to_csv_header(title: str, ts_code: str, body: str) -> str:
    header = f"# {title} for {ts_code}\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    return header + body


def _safe_pro_call(fn_name: str, **kwargs):
    pro = get_pro()
    fn = getattr(pro, fn_name, None)
    if fn is None:
        raise TushareVendorError(f"Tushare API '{fn_name}' is not available in this SDK version.")
    try:
        return fn(**kwargs)
    except Exception as exc:
        raise TushareVendorError(f"Tushare.{fn_name} failed: {exc}") from exc


def _try_pro_call(fn_name: str, **kwargs):
    """Invoke a Tushare API; return ``None`` if the call fails (permission, params, network)."""
    try:
        return _safe_pro_call(fn_name, **kwargs)
    except TushareVendorError:
        return None


def _irm_qa_lines(ts_code: str, d0: str, d1: str, api_name: str, label: str) -> list[str]:
    df = _try_pro_call(api_name, ts_code=ts_code, start_date=d0, end_date=d1)
    if df is None or df.empty:
        return []
    out: list[str] = []
    for _, r in df.head(28).iterrows():
        ts = r.get("pub_time", r.get("trade_date", ""))
        out.append(
            f"### [{label}] {ts}\n**问** {r.get('q', '')}\n**答** {r.get('a', '')}\n"
        )
    return out


def _npr_policy_lines(
    start_win: str,
    end_win: str,
    keywords: list[str],
    *,
    max_rows: int = 30,
) -> list[str]:
    """国家政策库 ``npr`` — National Policy Repository (skill: 国家政策库).

    ``start_win`` / ``end_win`` use ``YYYY-MM-DD HH:MM:SS`` as in Tushare docs.

    ``keywords`` 用于在标题中优先排序；**政策法规本身不是「个股新闻」接口**，
    若 ``keywords`` 为空则按时间返回窗口内政策条目（宏观与监管背景）。
    """
    df = _try_pro_call(
        "npr",
        start_date=start_win,
        end_date=end_win,
        fields="pubtime,title,pcode,puborg,ptype",
    )
    if df is None or df.empty:
        return []
    matched: list[str] = []
    other: list[str] = []
    for _, r in df.iterrows():
        title = str(r.get("title", ""))
        blob = f"{title}{r.get('ptype', '')}{r.get('puborg', '')}"
        pubt = r.get("pubtime", r.get("pub_time", ""))
        line = (
            f"### {pubt} | {r.get('puborg', '')}\n"
            f"{title}\n"
            f"发文字号: {r.get('pcode', '')} | 主题: {r.get('ptype', '')}\n"
        )
        if keywords and any(k and k in blob for k in keywords):
            matched.append(line)
        else:
            other.append(line)
    take_kw = min(len(matched), max_rows // 2 + 10)
    out = matched[:take_kw]
    rest_n = max(0, max_rows - len(out))
    out.extend(other[:rest_n])
    return out[:max_rows]


def _major_news_lines(
    start_win: str,
    end_win: str,
    match_fn,
    major_srcs: tuple[str, ...],
    *,
    per_src_cap: int = 10,
) -> list[str]:
    """长篇通讯 ``major_news`` (skill: 新闻通讯). ``start_win`` / ``end_win``: ``YYYY-MM-DD HH:MM:SS``."""
    lines: list[str] = []
    for src in major_srcs:
        df = _try_pro_call(
            "major_news",
            src=src,
            start_date=start_win,
            end_date=end_win,
            fields="title,content,pub_time,src",
        )
        if df is None or df.empty:
            continue
        n = 0
        for _, r in df.iterrows():
            title = str(r.get("title", ""))
            content = str(r.get("content", ""))
            if not match_fn(title, content):
                continue
            pub = r.get("pub_time", "")
            lines.append(f"### [{src}] {pub}\n{title}\n{str(content)[:2200]}\n")
            n += 1
            if n >= per_src_cap:
                break
        if len(lines) >= 45:
            break
    return lines


def _flash_news_lines(
    start_win: str,
    end_win: str,
    match_fn,
    flash_srcs: tuple[str, ...],
    *,
    per_src_cap: int = 12,
) -> list[str]:
    """短讯 ``news`` (skill: 新闻快讯). ``start_win`` / ``end_win``: ``YYYY-MM-DD HH:MM:SS``."""
    lines: list[str] = []
    for src in flash_srcs:
        df = _try_pro_call("news", src=src, start_date=start_win, end_date=end_win)
        if df is None or df.empty:
            continue
        n = 0
        for _, r in df.iterrows():
            title = str(r.get("title", ""))
            content = str(r.get("content", ""))
            if not match_fn(title, content):
                continue
            dt = r.get("datetime", "")
            lines.append(f"### [{src}] {dt}\n{title}\n{content}\n")
            n += 1
            if n >= per_src_cap:
                break
        if len(lines) >= 50:
            break
    return lines


def fetch_daily_price_frame(
    ts_code: str,
    start_date: str,
    end_date: str,
    *,
    market: str = "cn",
) -> pd.DataFrame:
    """Daily bars as a small DataFrame with ``Close`` (for return / benchmark math).

    ``start_date`` / ``end_date`` are ``YYYY-MM-DD``. Index is trade dates.
    ``market`` is ``cn`` (``daily``) or ``hk`` (``hk_daily``).
    """
    d0, d1 = to_yyyymmdd(start_date), to_yyyymmdd(end_date)
    if market == "hk":
        raw = _try_pro_call("hk_daily", ts_code=ts_code, start_date=d0, end_date=d1)
    else:
        raw = _try_pro_call("daily", ts_code=ts_code, start_date=d0, end_date=d1)
    if raw is None or raw.empty:
        return pd.DataFrame({"Close": []})
    raw = raw.sort_values("trade_date")
    out = pd.DataFrame({"Close": pd.to_numeric(raw["close"], errors="coerce").values})
    out.index = pd.to_datetime(raw["trade_date"])
    return out


def fetch_index_global_close_frame(
    index_ts_code: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """International index daily close (e.g. ``HSI``) via ``index_global`` — same shape as ``fetch_daily_price_frame``."""
    raw = _try_pro_call(
        "index_global",
        ts_code=index_ts_code,
        start_date=to_yyyymmdd(start_date),
        end_date=to_yyyymmdd(end_date),
    )
    if raw is None or raw.empty:
        return pd.DataFrame({"Close": []})
    raw = raw.sort_values("trade_date")
    out = pd.DataFrame({"Close": pd.to_numeric(raw["close"], errors="coerce").values})
    out.index = pd.to_datetime(raw["trade_date"])
    return out


def get_tushare_stock_data(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    datetime.strptime(start_date, "%Y-%m-%d")
    datetime.strptime(end_date, "%Y-%m-%d")
    resolved = resolve_tushare_equity(symbol)
    if not resolved:
        raise TushareVendorError(
            f"Tushare does not recognize '{symbol}' as A-share (6-digit) or Hong Kong (xxxxx.HK)."
        )
    ts_code, mkt = resolved
    if mkt == "hk":
        df = _safe_pro_call(
            "hk_daily",
            ts_code=ts_code,
            start_date=to_yyyymmdd(start_date),
            end_date=to_yyyymmdd(end_date),
        )
        src = "hk_daily"
    else:
        df = _safe_pro_call(
            "daily",
            ts_code=ts_code,
            start_date=to_yyyymmdd(start_date),
            end_date=to_yyyymmdd(end_date),
        )
        src = "daily"
    if df is None or df.empty:
        raise TushareVendorError(f"Tushare {src} returned no rows for {ts_code}.")

    df = df.sort_values("trade_date")
    out = pd.DataFrame(
        {
            "Date": pd.to_datetime(df["trade_date"]),
            "Open": pd.to_numeric(df["open"], errors="coerce"),
            "High": pd.to_numeric(df["high"], errors="coerce"),
            "Low": pd.to_numeric(df["low"], errors="coerce"),
            "Close": pd.to_numeric(df["close"], errors="coerce"),
            "Volume": pd.to_numeric(df["vol"], errors="coerce"),
        }
    )
    out["Adj Close"] = out["Close"]
    df = out
    df.set_index("Date", inplace=True)
    for col in ["Open", "High", "Low", "Close", "Adj Close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(2)

    csv_string = df.to_csv()
    mlabel = "Hong Kong (hk_daily)" if mkt == "hk" else "A-share (daily)"
    header = f"# Stock data (Tushare {mlabel}) for {ts_code} from {start_date} to {end_date}\n"
    header += f"# Total records: {len(df)}\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    return header + csv_string


def _tushare_load_ohlcv(symbol: str, curr_date: str) -> pd.DataFrame:
    """OHLCV DataFrame compatible with stockstats (same shape as ``load_ohlcv`` from Yahoo path)."""
    resolved = resolve_tushare_equity(symbol)
    if not resolved:
        raise TushareVendorError(
            f"Tushare does not recognize '{symbol}' as A-share or Hong Kong (xxxxx.HK)."
        )
    ts_code, mkt = resolved
    safe_symbol = safe_ticker_component(symbol)
    config = get_config()
    curr_date_dt = pd.to_datetime(curr_date)
    today_date = pd.Timestamp.today()
    start_date = today_date - pd.DateOffset(years=5)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = today_date.strftime("%Y-%m-%d")

    os.makedirs(config["data_cache_dir"], exist_ok=True)
    mtag = "hk" if mkt == "hk" else "cn"
    data_file = os.path.join(
        config["data_cache_dir"],
        f"{safe_symbol}-tushare-{mtag}-{to_yyyymmdd(start_str)}-{to_yyyymmdd(end_str)}.csv",
    )

    if os.path.exists(data_file):
        data = pd.read_csv(data_file, on_bad_lines="skip", encoding="utf-8")
    else:
        d0, d1 = to_yyyymmdd(start_str), to_yyyymmdd(end_str)
        if mkt == "hk":
            raw = _safe_pro_call("hk_daily", ts_code=ts_code, start_date=d0, end_date=d1)
        else:
            raw = _safe_pro_call("daily", ts_code=ts_code, start_date=d0, end_date=d1)
        if raw is None or raw.empty:
            raise TushareVendorError(f"Tushare {'hk_daily' if mkt == 'hk' else 'daily'} empty for {ts_code}.")
        raw = raw.sort_values("trade_date")
        data = raw.rename(
            columns={
                "trade_date": "Date",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "vol": "Volume",
            }
        )
        data.to_csv(data_file, index=False, encoding="utf-8")

    data = _clean_dataframe(data)
    data = data[data["Date"] <= curr_date_dt]
    return data


def _tushare_stock_stats_bulk(symbol: str, indicator: str, curr_date: str) -> dict:
    data = _tushare_load_ohlcv(symbol, curr_date)
    df = wrap(data)
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    df[indicator]
    out = {}
    for _, row in df.iterrows():
        v = row[indicator]
        out[row["Date"]] = "N/A" if pd.isna(v) else str(v)
    return out


def get_tushare_indicators(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    curr_date: Annotated[str, "The current trading date you are trading on, YYYY-mm-dd"],
    look_back_days: Annotated[int, "how many days to look back"],
) -> str:
    best_ind_params = {
        "close_50_sma": "50 SMA: medium-term trend.",
        "close_200_sma": "200 SMA: long-term trend benchmark.",
        "close_10_ema": "10 EMA: short-term responsive average.",
        "macd": "MACD: momentum via EMA differences.",
        "macds": "MACD Signal line.",
        "macdh": "MACD Histogram.",
        "rsi": "RSI: momentum / overbought-oversold.",
        "boll": "Bollinger middle (20 SMA).",
        "boll_ub": "Bollinger upper band.",
        "boll_lb": "Bollinger lower band.",
        "atr": "ATR: volatility measure.",
        "vwma": "VWMA: volume-weighted moving average.",
        "mfi": "MFI: money flow index.",
    }
    if indicator not in best_ind_params:
        raise ValueError(
            f"Indicator {indicator} is not supported. Please choose from: {list(best_ind_params.keys())}"
        )

    end_date = curr_date
    curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    before = curr_date_dt - relativedelta(days=look_back_days)

    try:
        indicator_data = _tushare_stock_stats_bulk(symbol, indicator, curr_date)
        ind_string = ""
        current_dt = curr_date_dt
        while current_dt >= before:
            date_str = current_dt.strftime("%Y-%m-%d")
            val = indicator_data.get(date_str, "N/A: Not a trading day (weekend or holiday)")
            ind_string += f"{date_str}: {val}\n"
            current_dt = current_dt - relativedelta(days=1)
    except Exception as exc:
        raise TushareVendorError(f"Tushare indicators failed: {exc}") from exc

    return (
        f"## {indicator} values from {before.strftime('%Y-%m-%d')} to {end_date} (Tushare OHLCV):\n\n"
        + ind_string
        + "\n\n"
        + best_ind_params[indicator]
    )


def get_tushare_fundamentals(
    ticker: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "current date (YYYY-mm-dd) for look-ahead filter"] = None,
) -> str:
    resolved = resolve_tushare_equity(ticker)
    if not resolved:
        raise TushareVendorError(
            f"Tushare does not recognize '{ticker}' as A-share (6-digit) or Hong Kong (xxxxx.HK)."
        )
    ts_code, mkt = resolved

    if mkt == "hk":
        basic = _try_pro_call("hk_basic", ts_code=ts_code, list_status="L")
        if basic is None or basic.empty:
            raise TushareVendorError("hk_basic returned no data for this HK ts_code.")
        row = basic.iloc[0]
        lines = [
            f"Name: {row.get('name', '')}",
            f"ts_code: {row.get('ts_code', '')}",
            f"Full name: {row.get('fullname', '')}",
            f"English name: {row.get('enname', '')}",
            f"Market: {row.get('market', '')}",
            f"List date: {row.get('list_date', '')}",
            f"Currency: {row.get('curr_type', '')}",
        ]
        if curr_date:
            end = to_yyyymmdd(curr_date)
            try:
                fi = _try_pro_call("hk_fina_indicator", ts_code=ts_code, end_date=end)
                if fi is not None and not fi.empty:
                    fi = fi.sort_values("end_date", ascending=False)
                    fi = fi[
                        pd.to_datetime(fi["end_date"], format="%Y%m%d", errors="coerce")
                        <= pd.Timestamp(curr_date)
                    ]
                    if not fi.empty:
                        r = fi.iloc[0]
                        for col in fi.columns:
                            val = r.get(col)
                            if pd.notna(val) and col not in ("ts_code",):
                                lines.append(f"{col}: {val}")
            except TushareVendorError:
                lines.append("(hk_fina_indicator unavailable — permission or points.)")
        header = f"# Company fundamentals (Tushare HK: hk_basic / hk_fina_indicator) for {ts_code}\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        return header + "\n".join(str(x) for x in lines)

    basic = _safe_pro_call("stock_basic", ts_code=ts_code, list_status="L")
    if basic is None or basic.empty:
        raise TushareVendorError("stock_basic returned no data.")

    row = basic.iloc[0]
    lines = [
        f"Name: {row.get('name', '')}",
        f"Symbol: {row.get('symbol', '')}",
        f"Area: {row.get('area', '')}",
        f"Industry: {row.get('industry', '')}",
        f"Market: {row.get('market', '')}",
        f"List date: {row.get('list_date', '')}",
    ]

    if curr_date:
        end = to_yyyymmdd(curr_date)
        try:
            fi = _safe_pro_call("fina_indicator", ts_code=ts_code, end_date=end)
            if fi is not None and not fi.empty:
                fi = fi.sort_values("end_date", ascending=False)
                fi = fi[pd.to_datetime(fi["end_date"], format="%Y%m%d", errors="coerce") <= pd.Timestamp(curr_date)]
                if not fi.empty:
                    r = fi.iloc[0]
                    for col in fi.columns:
                        val = r.get(col)
                        if pd.notna(val) and col not in ("ts_code",):
                            lines.append(f"{col}: {val}")
        except TushareVendorError:
            lines.append("(fina_indicator unavailable — token may lack permission or points.)")

    header = f"# Company fundamentals (Tushare) for {ts_code}\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    return header + "\n".join(str(x) for x in lines)


def _filter_stmt_by_curr_date(df: pd.DataFrame, curr_date: str | None) -> pd.DataFrame:
    if df is None or df.empty or not curr_date:
        return df
    c = pd.to_datetime(df["end_date"], format="%Y%m%d", errors="coerce")
    return df[c <= pd.Timestamp(curr_date)]


def _filter_stmt_by_freq(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if freq.lower() == "annual":
        return df[df["end_date"].astype(str).str.endswith("1231")]
    return df


def get_tushare_balance_sheet(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency of data: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None,
) -> str:
    resolved = resolve_tushare_equity(ticker)
    if not resolved:
        return f"No balance sheet: unknown symbol '{ticker}'"
    ts_code, mkt = resolved
    if mkt == "hk":
        df = _try_pro_call("hk_balancesheet", ts_code=ts_code)
        if df is None or df.empty:
            return f"No HK balance sheet data for '{ticker}' ({ts_code})."
        df = _filter_stmt_by_curr_date(df, curr_date)
        return _df_to_csv_header(f"HK balance sheet (hk_balancesheet, {freq})", ts_code, df.to_csv(index=False))

    df = _safe_pro_call("balancesheet", ts_code=ts_code)
    if df is None or df.empty:
        return f"No balance sheet data found for symbol '{ticker}'"
    df = _filter_stmt_by_curr_date(df, curr_date)
    df = _filter_stmt_by_freq(df, freq)
    return _df_to_csv_header(f"Balance sheet ({freq}, Tushare)", ts_code, df.to_csv(index=False))


def get_tushare_cashflow(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency of data: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None,
) -> str:
    resolved = resolve_tushare_equity(ticker)
    if not resolved:
        return f"No cash flow: unknown symbol '{ticker}'"
    ts_code, mkt = resolved
    if mkt == "hk":
        df = _try_pro_call("hk_cashflow", ts_code=ts_code)
        if df is None or df.empty:
            return f"No HK cash flow data for '{ticker}' ({ts_code})."
        df = _filter_stmt_by_curr_date(df, curr_date)
        return _df_to_csv_header(f"HK cash flow (hk_cashflow, {freq})", ts_code, df.to_csv(index=False))

    df = _safe_pro_call("cashflow", ts_code=ts_code)
    if df is None or df.empty:
        return f"No cash flow data found for symbol '{ticker}'"
    df = _filter_stmt_by_curr_date(df, curr_date)
    df = _filter_stmt_by_freq(df, freq)
    return _df_to_csv_header(f"Cash flow ({freq}, Tushare)", ts_code, df.to_csv(index=False))


def get_tushare_income_statement(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency of data: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None,
) -> str:
    resolved = resolve_tushare_equity(ticker)
    if not resolved:
        return f"No income statement: unknown symbol '{ticker}'"
    ts_code, mkt = resolved
    if mkt == "hk":
        df = _try_pro_call("hk_income", ts_code=ts_code)
        if df is None or df.empty:
            return f"No HK income statement for '{ticker}' ({ts_code})."
        df = _filter_stmt_by_curr_date(df, curr_date)
        return _df_to_csv_header(f"HK income statement (hk_income, {freq})", ts_code, df.to_csv(index=False))

    df = _safe_pro_call("income", ts_code=ts_code)
    if df is None or df.empty:
        return f"No income statement data found for symbol '{ticker}'"
    df = _filter_stmt_by_curr_date(df, curr_date)
    df = _filter_stmt_by_freq(df, freq)
    return _df_to_csv_header(f"Income statement ({freq}, Tushare)", ts_code, df.to_csv(index=False))


def _macro_market_keywords() -> tuple[str, ...]:
    """用于新闻「无个股命中」时的宏观/市场类补充匹配或降级填充。"""
    return (
        "A股", "沪深", "上证", "深证", "创业板", "科创", "北交所",
        "央行", "美联储", "GDP", "CPI", "PPI",
        "降息", "加息", "降准", "LPR", "MLF",
        "地缘", "原油", "黄金", "美元", "人民币", "汇率",
        "港股", "美股", "纳指", "标普",
        "国务院", "发改委", "财政部", "证监会", "金融监管", "产业政策",
        "房地产", "汽车", "新能源", "半导体", "人工智能", "消费",
    )


def _dedupe_keywords(keys: list[str], *, min_len: int = 2) -> list[str]:
    """Deduplicate case-insensitively, preserve order, drop blanks / too-short tokens."""
    seen: set[str] = set()
    out: list[str] = []
    for raw in keys:
        s = str(raw or "").strip()
        if len(s) < min_len:
            continue
        k = s.casefold()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out


def _get_tushare_hk_news(ticker: str, ts_code: str, start_date: str, end_date: str) -> str:
    """港股语料：无沪深 e互动。

    参考 Tushare 技能《港股基础信息》：``hk_basic`` 输出含 ``name``、``fullname``、``enname``、
    ``cn_spell``、``market``（市场类别）、``isin`` 等。长篇/短讯逻辑对齐 A 股：
    **代码/ts_code/多语言名称** 命中，或 **``market``** 命中（类似 A 股用 ``industry`` 扩行业稿；
    同板块其他公司亦可能出现，请以代码/简称为准）。
    """
    win_start = f"{start_date} 00:00:00"
    win_end = f"{end_date} 23:59:59"
    ph = "（本期无返回数据、或接口权限/参数不支持。）"

    hk_core = ts_code.replace(".HK", "")
    news_keys: list[str] = [ts_code]
    if hk_core.isdigit():
        news_keys.append(str(int(hk_core)))
        news_keys.append(f"{int(hk_core):05d}")

    stock_name = ""
    hk_market = ""
    basic = _try_pro_call(
        "hk_basic",
        ts_code=ts_code,
        list_status="L",
        fields="ts_code,name,fullname,enname,cn_spell,market,isin,curr_type",
    )
    if basic is None or basic.empty:
        basic = _try_pro_call("hk_basic", ts_code=ts_code, list_status="L")
    if basic is not None and not basic.empty:
        row = basic.iloc[0]
        stock_name = str(row.get("name") or "").strip()
        if stock_name:
            news_keys.append(stock_name)
        for col in ("fullname", "enname", "cn_spell"):
            v = str(row.get(col) or "").strip()
            if v:
                news_keys.append(v)
        isin = str(row.get("isin") or "").strip()
        if len(isin) >= 6:
            news_keys.append(isin)
        hk_market = str(row.get("market") or "").strip()

    news_keys = _dedupe_keywords(news_keys, min_len=2)

    def match_company_or_hk_board(title: str, content: str) -> bool:
        blob = f"{title} {content}"
        if any(k and k in blob for k in news_keys):
            return True
        if hk_market and len(hk_market) >= 2 and hk_market in blob:
            return True
        return False

    major_srcs = ("新浪财经", "财联社", "第一财经", "华尔街见闻", "中证网", "同花顺")
    flash_srcs = ("sina", "eastmoney", "10jqka", "cls", "yicai", "fenghuang", "jinrongjie", "wallstreetcn")

    irm_ph = (
        "（港股无沪深 **irm_qa_sh / irm_qa_sz**；公司互动与公告请结合港交所披露及下列中文财经语料。）"
    )

    npr_hint = (
        "> **说明**：`npr` 为**国家政策法规库**（部委公开文件），不是证券「个股新闻」检索；"
        "下列为时间窗内政策摘要，用作**宏观与监管背景**。\n\n"
    )
    npr_kw: list[str] = [hk_market] if hk_market and len(hk_market) >= 2 else []
    sec3 = _npr_policy_lines(win_start, win_end, npr_kw, max_rows=28)

    sec4 = _major_news_lines(win_start, win_end, match_company_or_hk_board, major_srcs, per_src_cap=12)
    sec5 = _flash_news_lines(win_start, win_end, match_company_or_hk_board, flash_srcs, per_src_cap=15)

    news_hdr = (
        f"窗口: {start_date} ~ {end_date} | ④⑤ 匹配（港股 ``hk_basic``）：**{', '.join(news_keys)}**"
        + (
            f" 或 **市场类别「{hk_market}」**（同市场其他标的稿件也可能命中，请以 ts_code/简称为准）"
            if hk_market
            else ""
        )
    )
    blocks = [
        f"## Tushare 大模型语料（港股 + 共用语料）— {ticker} / {ts_code}\n\n{news_hdr}",
        f"### ① 互动问答 · 上证e互动（irm_qa_sh）\n\n{irm_ph}",
        f"### ② 互动问答 · 深证互动易（irm_qa_sz）\n\n{irm_ph}",
        f"### ③ 国家政策库（npr）\n\n{npr_hint}"
        + ("\n".join(sec3) if sec3 else ph),
        f"### ④ 新闻快讯 · 长篇通讯（major_news）\n\n"
        + ("\n".join(sec4) if sec4 else ph),
        f"### ⑤ 新闻快讯 · 短讯（news）\n\n"
        + ("\n".join(sec5) if sec5 else ph),
    ]
    if not (sec3 or sec4 or sec5):
        blocks.append(
            "\n---\n**说明**：港股语料依赖 **hk_basic** 与中文财经 ``major_news``/``news`` 权限；"
            "若为空请检查 Tushare 港股与语料接口权限。"
            f"（简称：{stock_name or '未取到'}；市场类别 market：{hk_market or '未取到'}）"
        )
    return "\n\n".join(blocks).strip()


def get_tushare_news(
    ticker: str,
    start_date: str,
    end_date: str,
) -> str:
    """五项大模型语料：A 股含沪深 e互动；港股无 e互动但共用 ``npr`` + ``major_news`` + ``news``。

    - **国家政策库**为部委公开法规，**不是按代码的个股新闻**。
    - **长篇/短讯**命中规则：**证券代码、ts_code、证券简称**，以及 **`stock_basic` 行业名**（便于纳入行业动态；同行业其他公司稿件也可能命中，请以代码/简称为准）。
    - **港股**无 ``stock_basic``：用 ``hk_basic`` 的 **``name`` / ``fullname`` / ``enname`` / ``cn_spell`` / ``isin``** 与 **``market``（市场类别，技能文档字段）** 对应 A 股的「简称 + 行业扩召回」；``npr`` 排序亦可用 ``market``。
    - 无法识别为 A 股或港股 Tushare 代码时（如美股），仍返回 ``npr`` + 新闻降级结果。
    """
    win_start = f"{start_date} 00:00:00"
    win_end = f"{end_date} 23:59:59"
    d0, d1 = to_yyyymmdd(start_date), to_yyyymmdd(end_date)
    ph = "（本期无返回数据、或接口权限/参数不支持。）"
    broad_kw = _macro_market_keywords()

    resolved = resolve_tushare_equity(ticker)
    if not resolved:
        return _get_tushare_news_without_ts_code(ticker, start_date, end_date, win_start, win_end, broad_kw, ph)
    ts_code, market = resolved
    if market == "hk":
        return _get_tushare_hk_news(ticker, ts_code, start_date, end_date)

    stock_name = ""
    industry = ""
    basic = _try_pro_call("stock_basic", ts_code=ts_code, fields="ts_code,name,industry")
    if basic is not None and not basic.empty:
        row = basic.iloc[0]
        stock_name = str(row.get("name") or "").strip()
        industry = str(row.get("industry") or "").strip()

    code6 = ts_code.split(".")[0]
    news_keys: list[str] = [code6, ts_code]
    if stock_name and len(stock_name) >= 2:
        news_keys.append(stock_name)

    def match_company_or_industry(title: str, content: str) -> bool:
        blob = f"{title} {content}"
        if any(k and k in blob for k in news_keys):
            return True
        if industry and len(industry) >= 2 and industry in blob:
            return True
        return False

    major_srcs = ("新浪财经", "财联社", "第一财经", "华尔街见闻", "中证网", "同花顺")
    flash_srcs = ("sina", "eastmoney", "10jqka", "cls", "yicai", "fenghuang", "jinrongjie", "wallstreetcn")

    sec1 = _irm_qa_lines(ts_code, d0, d1, "irm_qa_sh", "上证e互动")
    sec2 = _irm_qa_lines(ts_code, d0, d1, "irm_qa_sz", "深证互动易")

    # npr：不按证券代码做「个股新闻」式过滤；可选行业词仅用于同窗口内排序靠前
    npr_hint = (
        "> **说明**：`npr` 为**国家政策法规库**（部委公开文件），不是证券资讯里的「个股新闻」。"
        "下列为时间窗口内最新政策条目，用作**宏观与监管背景**；与该股是否同名无必然关系。\n\n"
    )
    npr_kw: list[str] = [industry] if industry and len(industry) >= 2 else []
    sec3 = _npr_policy_lines(win_start, win_end, npr_kw, max_rows=28)

    # 长篇/短讯：代码/简称 **或** 行业（无宏观、无「无条件要闻」降级，减少无关头条）。
    sec4 = _major_news_lines(win_start, win_end, match_company_or_industry, major_srcs, per_src_cap=12)
    sec5 = _flash_news_lines(win_start, win_end, match_company_or_industry, flash_srcs, per_src_cap=15)

    news_hdr = (
        f"窗口: {start_date} ~ {end_date} | ④⑤ 匹配：**{', '.join(news_keys)}**"
        + (f" 或 **行业「{industry}」**（行业稿可能含同行业其他公司，请以代码/简称为准）" if industry else "")
    )
    blocks = [
        f"## Tushare 大模型语料（五项）— {ticker} / {ts_code}\n\n{news_hdr}",
        f"### ① 互动问答 · 上证e互动（irm_qa_sh）\n\n" + ("\n".join(sec1) if sec1 else ph),
        f"### ② 互动问答 · 深证互动易（irm_qa_sz）\n\n" + ("\n".join(sec2) if sec2 else ph),
        f"### ③ 国家政策库（npr）\n\n{npr_hint}"
        + ("\n".join(sec3) if sec3 else ph),
        f"### ④ 新闻快讯 · 长篇通讯（major_news）\n\n"
        + ("\n".join(sec4) if sec4 else ph),
        f"### ⑤ 新闻快讯 · 短讯（news）\n\n"
        + ("\n".join(sec5) if sec5 else ph),
    ]
    if not (sec1 or sec2 or sec3 or sec4 or sec5):
        blocks.append(
            "\n---\n**说明**：多项为空时，常见原因包括：未开通大模型语料类接口权限（见 "
            "https://tushare.pro/document/1?doc_id=290 ）；该时段数据源无返回；"
            "e互动仅覆盖上证/深证互动平台。"
            f"（公司简称：{stock_name or '未取到'}；行业：{industry or '未取到'}）"
        )
    return "\n\n".join(blocks).strip()


def _get_tushare_news_without_ts_code(
    ticker: str,
    start_date: str,
    end_date: str,
    win_start: str,
    win_end: str,
    broad_kw: tuple[str, ...],
    ph: str,
) -> str:
    """美股等：无法解析为 A 股或港股 ``xxxxx.HK`` 时，仅 ``npr`` + 财经新闻语料。"""
    raw = (ticker or "").strip()
    loose_kw = [x for x in (raw, raw.upper()) if x]

    def match_ticker_or_macro(title: str, content: str) -> bool:
        blob = f"{title} {content}"
        if any(k and k in blob for k in loose_kw):
            return True
        return any(k in blob for k in broad_kw)

    npr_hint = (
        "> **说明**：标的无法解析为 A 股或港股 Tushare 代码；**无 e互动**与 ``stock_basic``/``hk_basic``。"
        "``npr`` 为国家政策库（非个股新闻）。\n\n"
    )
    sec3 = _npr_policy_lines(win_start, win_end, [], max_rows=25)
    major_srcs = ("新浪财经", "财联社", "第一财经", "华尔街见闻", "中证网", "同花顺")
    flash_srcs = ("sina", "eastmoney", "10jqka", "cls", "yicai", "fenghuang", "jinrongjie", "wallstreetcn")
    sec4 = _major_news_lines(win_start, win_end, match_ticker_or_macro, major_srcs, per_src_cap=10)
    if not sec4:
        sec4 = _major_news_lines(
            win_start, win_end, lambda _t, _c: True, major_srcs, per_src_cap=5
        )
    sec5 = _flash_news_lines(win_start, win_end, match_ticker_or_macro, flash_srcs, per_src_cap=12)
    if not sec5:
        sec5 = _flash_news_lines(
            win_start, win_end, lambda _t, _c: True, flash_srcs, per_src_cap=6
        )

    blocks = [
        f"## Tushare 语料（未识别为 A/港）— `{ticker}`\n\n"
        f"请输入 **6 位 A 股**（如 600519 / 600519.SH）或 **港股**（如 00700.HK、HK00700）。\n\n"
        f"窗口: {start_date} ~ {end_date}",
        f"### ①② 互动问答（irm_qa_sh / irm_qa_sz）\n\n"
        "（跳过：需要 A 股 ``ts_code``。）",
        f"### ③ 国家政策库（npr）\n\n{npr_hint}"
        + ("\n".join(sec3) if sec3 else ph),
        f"### ④ 新闻快讯 · 长篇（major_news）\n\n"
        + (
            "> 优先含标的字符串或宏观类关键词；否则为时间窗要闻。\n\n"
            if sec4
            else ""
        )
        + ("\n".join(sec4) if sec4 else ph),
        f"### ⑤ 新闻快讯 · 短讯（news）\n\n" + ("\n".join(sec5) if sec5 else ph),
    ]
    return "\n\n".join(blocks).strip()


def get_tushare_global_news(
    curr_date: str,
    look_back_days: int = 7,
    limit: int = 50,
) -> str:
    """全局语料：国家政策库 ``npr`` + 长篇 ``major_news`` + 短讯 ``news``。

    互动问答接口需要 ``ts_code``，此处不调用；请对具体标的使用 ``get_tushare_news``。
    """
    curr = datetime.strptime(curr_date, "%Y-%m-%d")
    start = curr - timedelta(days=look_back_days)
    start_date = start.strftime("%Y-%m-%d")
    end_date = curr.strftime("%Y-%m-%d")
    win_start = f"{start_date} 00:00:00"
    win_end = f"{end_date} 23:59:59"

    broad_kw = _macro_market_keywords()

    def match_broad(title: str, content: str) -> bool:
        blob = f"{title} {content}"
        return any(k in blob for k in broad_kw)

    major_srcs = ("新浪财经", "财联社", "第一财经", "华尔街见闻", "中证网", "同花顺")
    flash_srcs = ("cls", "eastmoney", "wallstreetcn", "sina", "10jqka", "yicai", "fenghuang", "jinrongjie")

    per_major = max(6, min(20, limit // 4))
    per_flash = max(8, min(25, limit // 3))
    sec_npr = _npr_policy_lines(win_start, win_end, [], max_rows=max(30, min(80, limit)))
    sec_major = _major_news_lines(
        win_start, win_end, match_broad, major_srcs, per_src_cap=per_major
    )
    sec_flash = _flash_news_lines(
        win_start, win_end, match_broad, flash_srcs, per_src_cap=per_flash
    )

    ph = "（本期无返回数据或未命中宏观/市场类关键词。）"

    blocks = [
        f"## Tushare 全局语料（npr + major_news + news）\n\n{win_start} — {win_end}\n\n"
        "> **说明**：本接口为**宏观与市场要闻**；`npr` 为政策法规库，**不是按股票代码的个股新闻**。\n",
        f"### 国家政策库（npr）\n\n" + ("\n".join(sec_npr) if sec_npr else ph),
        f"### 新闻快讯 · 长篇通讯（major_news）\n\n" + ("\n".join(sec_major) if sec_major else ph),
        f"### 新闻快讯 · 短讯（news）\n\n" + ("\n".join(sec_flash) if sec_flash else ph),
        "\n*互动问答（irm_qa_sh / irm_qa_sz）需具体标的，请使用 ``get_tushare_news``。*",
    ]
    body = "\n\n".join(blocks).strip()
    if len(body) > 120_000:
        return body[:120_000] + "\n\n…（输出已截断）"
    return body


def get_tushare_insider_transactions(ticker: str) -> str:
    resolved = resolve_tushare_equity(ticker)
    if not resolved:
        return f"No shareholder trade data: unknown symbol '{ticker}'."
    ts_code, mkt = resolved
    if mkt == "hk":
        return (
            f"No HK ``stk_holdertrade`` equivalent wired for '{ticker}' ({ts_code}). "
            "Tushare ``stk_holdertrade`` is A-share oriented; use HKEX disclosures for substantial shareholders."
        )
    df = _safe_pro_call("stk_holdertrade", ts_code=ts_code)
    if df is None or df.empty:
        return f"No shareholder trade data for '{ticker}' (Tushare stk_holdertrade)."
    return _df_to_csv_header("Shareholder increase/decrease (stk_holdertrade)", ts_code, df.to_csv(index=False))
