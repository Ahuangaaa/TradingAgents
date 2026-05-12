"""Tushare-backed implementations for ``route_to_vendor`` (A-shares only; see project skill references)."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Annotated

import pandas as pd
from dateutil.relativedelta import relativedelta
from stockstats import wrap

from .config import get_config
from .macro_keywords import macro_market_keywords
from .news_long_short_llm_filter import news_llm_filter_disabled, screen_long_short_news_with_llm
from .stockstats_utils import _clean_dataframe
from .tushare_common import (
    TushareVendorError,
    get_pro,
    resolve_tushare_equity,
    to_yyyymmdd,
)
from .utils import safe_ticker_component

logger = logging.getLogger(__name__)


def _df_to_csv_header(title: str, ts_code: str, body: str) -> str:
    header = f"# {title} for {ts_code}\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    return header + body


def _df_to_markdown_table(df: pd.DataFrame) -> str:
    """Pipe table without optional ``tabulate`` dependency."""
    if df is None or df.empty:
        return "_(no rows)_"
    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows: list[str] = []
    for _, r in df.iterrows():
        cells: list[str] = []
        for c in df.columns:
            v = r.get(c)
            if v is None or pd.isna(v):
                cells.append("")
            else:
                cells.append(str(v).replace("\n", " ").strip())
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *rows])


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


def _sse_trading_days_between_inclusive(lo_yyyy_mm_dd: str, hi_yyyy_mm_dd: str) -> list[str]:
    """Return open trading dates in ``[lo, hi]`` (YYYY-MM-DD), ascending.

    Uses Tushare ``trade_cal`` for ``SSE`` (same session calendar as SZSE/BJ
    for routine A-share daily bars). See https://tushare.pro/wctapi/documents/26.md
    """
    d0, d1 = to_yyyymmdd(lo_yyyy_mm_dd), to_yyyymmdd(hi_yyyy_mm_dd)
    df = _try_pro_call(
        "trade_cal",
        exchange="SSE",
        start_date=d0,
        end_date=d1,
        fields="cal_date,is_open",
    )
    if df is None or df.empty:
        return []
    open_mask = pd.to_numeric(df["is_open"], errors="coerce").fillna(0).astype(int) == 1
    df = df.loc[open_mask]
    if df.empty:
        return []
    cal = pd.to_datetime(df["cal_date"].astype(str), format="%Y%m%d", errors="coerce")
    if cal.isna().all():
        cal = pd.to_datetime(df["cal_date"].astype(str), errors="coerce")
    dates = sorted(cal.dropna().dt.strftime("%Y-%m-%d").tolist())
    return [d for d in dates if lo_yyyy_mm_dd <= d <= hi_yyyy_mm_dd]


def _warmup_calendar_days_for_indicator(indicator: str) -> int:
    """Extra calendar days to load before ``curr_date`` so stockstats has enough bars."""
    # ~1.5–2× the dominant window, in calendar days (covers CNY gaps).
    key = (indicator or "").lower()
    if key == "close_200_sma":
        return 420
    if key == "close_50_sma":
        return 130
    if key in ("macd", "macds", "macdh"):
        return 90
    if key in ("boll", "boll_ub", "boll_lb"):
        return 70
    if key in ("rsi", "mfi", "vwma", "atr"):
        return 50
    return 90


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


def _long_short_window_strs(start_date: str, end_date: str, lookback_days: int) -> tuple[str, str]:
    """Intersect caller window with last ``lookback_days`` from ``end_date`` for ④⑤."""
    end_dt = pd.Timestamp(str(end_date)[:10])
    start_req = pd.Timestamp(str(start_date)[:10])
    start_eff = max(start_req, end_dt - pd.Timedelta(days=int(lookback_days)))
    ws = f"{start_eff.strftime('%Y-%m-%d')} 00:00:00"
    we = f"{end_dt.strftime('%Y-%m-%d')} 23:59:59"
    return ws, we


def _major_news_collect_raw(
    start_win: str,
    end_win: str,
    major_srcs: tuple[str, ...],
    per_src_cap: int,
    *,
    content_max: int = 2000,
) -> list[dict]:
    """Unfiltered ``major_news`` rows (④ raw corpus for LLM)."""
    out: list[dict] = []
    idx = 0
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
            if n >= per_src_cap:
                break
            out.append(
                {
                    "id": f"maj::{idx}",
                    "channel": "major_news",
                    "src": src,
                    "pub_time": str(r.get("pub_time", "")),
                    "title": str(r.get("title", "")),
                    "content": str(r.get("content", ""))[:content_max],
                }
            )
            idx += 1
            n += 1
        if len(out) >= 120:
            break
    return out


def _flash_news_collect_raw(
    start_win: str,
    end_win: str,
    flash_srcs: tuple[str, ...],
    per_src_cap: int,
    *,
    content_max: int = 1200,
) -> list[dict]:
    """Unfiltered ``news`` flash rows (⑤ raw corpus for LLM)."""
    out: list[dict] = []
    idx = 0
    for src in flash_srcs:
        df = _try_pro_call("news", src=src, start_date=start_win, end_date=end_win)
        if df is None or df.empty:
            continue
        n = 0
        for _, r in df.iterrows():
            if n >= per_src_cap:
                break
            out.append(
                {
                    "id": f"fl::{idx}",
                    "channel": "news",
                    "src": src,
                    "pub_time": str(r.get("datetime", "")),
                    "title": str(r.get("title", "")),
                    "content": str(r.get("content", ""))[:content_max],
                }
            )
            idx += 1
            n += 1
        if len(out) >= 150:
            break
    return out


def _anns_d_lines(ts_code: str, d0: str, d1: str, *, max_rows: int = 28) -> list[str]:
    """上市公司全量公告 ``anns_d``（日期 ``YYYYMMDD``）。"""
    df = _try_pro_call("anns_d", ts_code=ts_code, start_date=d0, end_date=d1)
    if df is None or df.empty:
        return []
    if "ann_date" in df.columns:
        df = df.sort_values("ann_date", ascending=False)
    out: list[str] = []
    for _, r in df.head(max_rows).iterrows():
        ad = r.get("ann_date", "")
        title = str(r.get("title", "")).strip()
        name = str(r.get("name", "")).strip()
        url = str(r.get("url", "")).strip()
        rec = r.get("rec_time", "")
        head = f"### {ad}"
        if rec:
            head += f" | {rec}"
        out.append(f"{head}\n**{name}** ({r.get('ts_code', '')})\n{title}\n{url}\n")
    return out


def _research_report_lines(
    ts_code: str,
    industry: str,
    d0: str,
    d1: str,
    *,
    max_stock: int = 14,
    max_industry: int = 10,
) -> list[str]:
    """券商研报 ``research_report``；个股 + 可选行业（``ind_name`` 与库内一致时才有命中）。"""
    out: list[str] = []
    seen: set[str] = set()

    df_s = _try_pro_call(
        "research_report",
        ts_code=ts_code,
        start_date=d0,
        end_date=d1,
        report_type="个股研报",
    )
    if df_s is not None and not df_s.empty:
        if "trade_date" in df_s.columns:
            df_s = df_s.sort_values("trade_date", ascending=False)
        for _, r in df_s.head(max_stock).iterrows():
            title = str(r.get("title", "") or r.get("file_name", "")).strip()
            td = r.get("trade_date", "")
            key = f"{td}|{title}"
            if key in seen:
                continue
            seen.add(key)
            abstr = str(r.get("abstr", "")).strip()
            if len(abstr) > 1200:
                abstr = abstr[:1200] + "…"
            out.append(
                f"### {td} | {r.get('inst_csname', '')} [{r.get('report_type', '')}]\n**{title}**\n"
                f"{r.get('name', '')} ({r.get('ts_code', '')}) | {r.get('author', '')}\n{abstr}\n"
                f"{str(r.get('url', '')).strip()}\n"
            )

    ind = (industry or "").strip()
    if ind and len(ind) >= 2:
        df_i = _try_pro_call(
            "research_report",
            ind_name=ind,
            start_date=d0,
            end_date=d1,
            report_type="行业研报",
        )
        if df_i is not None and not df_i.empty:
            if "trade_date" in df_i.columns:
                df_i = df_i.sort_values("trade_date", ascending=False)
            for _, r in df_i.head(max_industry).iterrows():
                title = str(r.get("title", "") or r.get("file_name", "")).strip()
                td = r.get("trade_date", "")
                key = f"{td}|{title}"
                if key in seen:
                    continue
                seen.add(key)
                abstr = str(r.get("abstr", "")).strip()
                if len(abstr) > 1200:
                    abstr = abstr[:1200] + "…"
                out.append(
                    f"### {td} | {r.get('inst_csname', '')} [{r.get('report_type', '')}]\n**{title}**\n"
                    f"{r.get('ind_name', '')} | {r.get('author', '')}\n{abstr}\n"
                    f"{str(r.get('url', '')).strip()}\n"
                )
    return out


def _research_report_global_lines(
    d0: str,
    d1: str,
    match_fn,
    *,
    max_rows: int = 36,
) -> list[str]:
    """时间窗内研报抽样，按 ``match_fn(title, abstr)`` 过滤。"""
    df = _try_pro_call("research_report", start_date=d0, end_date=d1)
    if df is None or df.empty:
        return []
    if "trade_date" in df.columns:
        df = df.sort_values("trade_date", ascending=False)
    out: list[str] = []
    for _, r in df.iterrows():
        title = str(r.get("title", "") or r.get("file_name", "")).strip()
        abstr = str(r.get("abstr", "")).strip()
        if not match_fn(title, abstr):
            continue
        if len(abstr) > 1000:
            abstr = abstr[:1000] + "…"
        out.append(
            f"### {r.get('trade_date', '')} | {r.get('inst_csname', '')} [{r.get('report_type', '')}]\n"
            f"**{title}**\n{r.get('name', '')} ({r.get('ts_code', '')}) | {r.get('author', '')}\n{abstr}\n"
            f"{str(r.get('url', '')).strip()}\n"
        )
        if len(out) >= max_rows:
            break
    return out


def fetch_daily_price_frame(
    ts_code: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Daily bars as a small DataFrame with ``Close`` (for return / benchmark math).

    ``start_date`` / ``end_date`` are ``YYYY-MM-DD``. Index is trade dates (``daily``).
    """
    d0, d1 = to_yyyymmdd(start_date), to_yyyymmdd(end_date)
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
    ts_code = resolve_tushare_equity(symbol)
    if not ts_code:
        raise TushareVendorError(
            f"Tushare does not recognize '{symbol}' as an A-share (6-digit .SH/.SZ/.BJ)."
        )
    df = _safe_pro_call(
        "daily",
        ts_code=ts_code,
        start_date=to_yyyymmdd(start_date),
        end_date=to_yyyymmdd(end_date),
    )
    if df is None or df.empty:
        raise TushareVendorError(f"Tushare daily returned no rows for {ts_code}.")

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
    header = f"# Stock data (Tushare A-share daily) for {ts_code} from {start_date} to {end_date}\n"
    header += f"# Total records: {len(df)}\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    return header + csv_string


def _tushare_load_ohlcv(
    symbol: str,
    curr_date: str,
    *,
    extend_start_calendar_days: int = 0,
) -> pd.DataFrame:
    """OHLCV DataFrame compatible with stockstats (same shape as ``load_ohlcv`` from Yahoo path).

    ``extend_start_calendar_days`` pulls the download window further into the past
    (from ``curr_date``) so moving averages / MACD have enough warm-up bars
    inside the filtered frame.
    """
    ts_code = resolve_tushare_equity(symbol)
    if not ts_code:
        raise TushareVendorError(
            f"Tushare does not recognize '{symbol}' as an A-share (6-digit .SH/.SZ/.BJ)."
        )
    safe_symbol = safe_ticker_component(symbol)
    config = get_config()
    curr_date_dt = pd.to_datetime(curr_date)
    today_date = pd.Timestamp.today().normalize()
    start_date = today_date - pd.DateOffset(years=5)
    if extend_start_calendar_days > 0:
        extended = (curr_date_dt.normalize() - pd.Timedelta(days=extend_start_calendar_days))
        start_date = min(start_date, extended)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = today_date.strftime("%Y-%m-%d")

    os.makedirs(config["data_cache_dir"], exist_ok=True)
    data_file = os.path.join(
        config["data_cache_dir"],
        f"{safe_symbol}-tushare-cn-{to_yyyymmdd(start_str)}-{to_yyyymmdd(end_str)}.csv",
    )

    if os.path.exists(data_file):
        data = pd.read_csv(data_file, on_bad_lines="skip", encoding="utf-8")
    else:
        d0, d1 = to_yyyymmdd(start_str), to_yyyymmdd(end_str)
        raw = _safe_pro_call("daily", ts_code=ts_code, start_date=d0, end_date=d1)
        if raw is None or raw.empty:
            raise TushareVendorError(f"Tushare daily empty for {ts_code}.")
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
    warmup = _warmup_calendar_days_for_indicator(indicator)
    data = _tushare_load_ohlcv(
        symbol, curr_date, extend_start_calendar_days=warmup
    )
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
        before_str = before.strftime("%Y-%m-%d")
        # ``trade_cal`` can mark a day as open before ``daily`` has that row (Tushare
        # end-of-day lag, or ``curr_date`` beyond last quote). Cap at last OHLCV date.
        if indicator_data:
            last_bar = max(indicator_data.keys())
            effective_end = min(end_date, last_bar)
        else:
            last_bar = None
            effective_end = end_date

        trading_days = _sse_trading_days_between_inclusive(before_str, effective_end)
        if not trading_days:
            trading_days = sorted(
                d for d in indicator_data if before_str <= d <= effective_end
            )
        ind_string = ""
        for date_str in reversed(trading_days):
            val = indicator_data.get(date_str)
            if val is None:
                val = "N/A: No daily bar (suspended or missing quote)"
            elif val == "N/A":
                val = (
                    "N/A: Indicator undefined (warm-up / new listing — "
                    "not enough prior trading days for this formula)"
                )
            ind_string += f"{date_str}: {val}\n"
    except Exception as exc:
        raise TushareVendorError(f"Tushare indicators failed: {exc}") from exc

    tail_note = ""
    if last_bar is not None and last_bar < end_date:
        tail_note = (
            f"\n*(OHLCV last bar {last_bar}; ``trade_cal`` may list later dates as open "
            f"before ``daily`` is published or when ``curr_date`` is after last quote.)*\n"
        )

    return (
        f"## {indicator} values — SSE trading days from {before.strftime('%Y-%m-%d')} "
        f"to {effective_end} (Tushare ``trade_cal`` + daily OHLCV):\n\n"
        + ind_string
        + tail_note
        + "\n\n"
        + best_ind_params[indicator]
    )


def get_tushare_fundamentals(
    ticker: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "current date (YYYY-mm-dd) for look-ahead filter"] = None,
) -> str:
    ts_code = resolve_tushare_equity(ticker)
    if not ts_code:
        raise TushareVendorError(
            f"Tushare does not recognize '{ticker}' as an A-share (6-digit .SH/.SZ/.BJ)."
        )

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


def get_tushare_industry_peers(
    ticker: Annotated[str, "ticker symbol of the focal A-share company"],
    curr_date: Annotated[
        str,
        "Optional YYYY-mm-dd cache key for peer list; does not change model ranking.",
    ] = None,
    max_peers: Annotated[int, "Maximum number of peer rows (excluding focal) to return"] = 8,
) -> str:
    """List **business competitors** as A-shares via **DeepSeek** + ``stock_basic`` validation.

    Not the old Tushare same-``industry`` universe; peers are model-selected then verified.
    """
    from .peers_deepseek import fetch_validated_peers

    ts_code = resolve_tushare_equity(ticker)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not ts_code:
        return (
            "# Listed competitors (DeepSeek + Tushare validation)\n\n"
            f"_Retrieved: {stamp}_\n\n"
            f"Cannot resolve `{ticker}` as an A-share ts_code (use 6-digit .SH/.SZ/.BJ).\n"
            "For non-A-share competitors, name tickers from filings or official IR.\n"
        )

    focal = _try_pro_call(
        "stock_basic",
        ts_code=ts_code,
        list_status="L",
        fields="ts_code,symbol,name,industry,market,area",
    )
    if focal is None or focal.empty:
        return (
            f"# Listed competitors for {ts_code}\n\n"
            f"_Retrieved: {stamp}_\n\n"
            "`stock_basic` returned no data for the focal symbol.\n"
        )

    row0 = focal.iloc[0]
    industry = str(row0.get("industry") or "").strip()
    focal_name = str(row0.get("name") or "").strip()

    cap = max(1, min(int(max_peers) if max_peers else 8, 12))
    peers = fetch_validated_peers(
        ts_code,
        focal_name,
        industry,
        max_peers=cap,
        curr_date=curr_date,
        use_cache=True,
    )
    note = (
        f"Focal: **{focal_name}** (`{ts_code}`) · Tushare industry field: **{industry or '（空）'}**. "
        f"Up to **{cap}** **model-selected** A-share competitors (codes verified via `stock_basic`). "
        "Use **3–5** for `get_news` / `get_fundamentals` if fewer are returned.\n\n"
    )
    if not peers:
        return (
            "# Listed competitors (DeepSeek + Tushare validation)\n\n"
            f"_Retrieved: {stamp}_\n\n"
            + note
            + "No validated peer rows (check API keys: DEEPSEEK / NEWS_TAG_LLM_* or PEER_LLM_*).\n"
        )

    lines = [
        "| ts_code | name | industry |",
        "| --- | --- | --- |",
    ]
    for p in peers:
        lines.append(f"| {p.ts_code} | {p.name} | {p.industry} |")
    return (
        "# Listed competitors (DeepSeek + Tushare validation)\n\n"
        f"_Retrieved: {stamp}_\n\n"
        + note
        + "\n"
        + "\n".join(lines)
        + "\n"
    )


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
    ts_code = resolve_tushare_equity(ticker)
    if not ts_code:
        return f"No balance sheet: unknown symbol '{ticker}'"
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
    ts_code = resolve_tushare_equity(ticker)
    if not ts_code:
        return f"No cash flow: unknown symbol '{ticker}'"
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
    ts_code = resolve_tushare_equity(ticker)
    if not ts_code:
        return f"No income statement: unknown symbol '{ticker}'"
    df = _safe_pro_call("income", ts_code=ts_code)
    if df is None or df.empty:
        return f"No income statement data found for symbol '{ticker}'"
    df = _filter_stmt_by_curr_date(df, curr_date)
    df = _filter_stmt_by_freq(df, freq)
    return _df_to_csv_header(f"Income statement ({freq}, Tushare)", ts_code, df.to_csv(index=False))


def _macro_section8_block(cfg: dict, *, win_start: str, win_end: str) -> str:
    """⑧ 宏观向量专题：专用 query 检索 + 可选 LLM；与 ④⑤ 个股/行业检索分离。"""
    from .news_qdrant_retrieval import news_long_short_use_qdrant, retrieve_macro_section_markdown

    if not cfg.get("news_macro_section8_enabled", True):
        return ""
    if not news_long_short_use_qdrant(cfg):
        return ""
    sl = int(cfg.get("news_macro_section8_search_limit", 100))
    pm = int(cfg.get("news_macro_section8_per_major", 10))
    pf = int(cfg.get("news_macro_section8_per_flash", 12))
    cm = int(cfg.get("news_macro_section8_major_content_max", 2200))
    cf = int(cfg.get("news_macro_section8_flash_content_max", 1500))
    raw_max = int(cfg.get("news_macro_section8_raw_max_chars", 12000))
    try:
        majors, flashes = retrieve_macro_section_markdown(
            win_start=win_start,
            win_end=win_end,
            search_limit=sl,
            per_major=pm,
            per_flash=pf,
            content_major_max=cm,
            content_flash_max=cf,
        )
    except Exception as exc:
        logger.warning("⑧ 宏观向量专题检索失败：%s", exc)
        return f"### ⑧ 宏观分析（向量库专题）\n\n（检索失败：{exc}）"
    raw_md = "\n".join([*majors, *flashes]).strip()
    head = (
        "### ⑧ 宏观分析（向量库专题）\n\n"
        "> **与 ④⑤ 区分**：④⑤按 **标的/行业/同业** 构造向量 query；本节使用 **宏观专用关键词** "
        "（``macro_keywords.macro_vector_search_query_text``）做**全市场**检索，概括政策、流动性、海外与大类资产环境。\n"
    )
    if not raw_md:
        return head + "（本期无命中或未入库该时间窗。）"

    refined = ""
    if cfg.get("news_macro_section8_llm_refine", True):
        try:
            from .macro_qdrant_llm_refine import (
                macro_section8_llm_refine_disabled,
                refine_macro_section8_corpus,
            )

            if not macro_section8_llm_refine_disabled():
                refined = refine_macro_section8_corpus(
                    raw_md, win_start=win_start, win_end=win_end
                ).strip()
        except Exception as exc:
            logger.warning("⑧ 宏观 LLM 精炼失败：%s", exc)
    parts = [head]
    if refined:
        parts.append("\n#### LLM 宏观摘要\n\n" + refined)
    raw_show = raw_md if len(raw_md) <= raw_max else raw_md[:raw_max] + "\n\n…（原文摘录已截断）"
    if cfg.get("news_macro_section8_include_raw_excerpt", True):
        parts.append("\n#### 语料摘录（⑧ 独立检索）\n\n" + raw_show)
    elif not refined:
        parts.append("\n#### 语料摘录（⑧ 独立检索）\n\n" + raw_show)
    return "".join(parts)


def get_tushare_news(
    ticker: str,
    start_date: str,
    end_date: str,
) -> str:
    """八项大模型语料：沪深 e互动、国家政策库、长篇/短讯新闻、公告、研报与 **⑧ 宏观向量专题**（均为 A 股 ``ts_code`` 路径下除 ⑧ 外按标的）。

    - **国家政策库**为部委公开法规，**不是按代码的个股新闻**。
    - **④ 长篇 / ⑤ 短讯**：启用 Qdrant 时，对标的与 **DeepSeek 给出的至多 5 个竞品** 各做一次短 query 向量检索后合并；
      否则从 Tushare 拉取后经 **quick LLM 分批语义筛选**（可缓存）。
      关闭方式：配置 ``news_llm_filter_long_short=False`` 或环境变量 ``TRADINGAGENTS_NEWS_LLM_FILTER=0``，
      此时退回 **仅代码/简称** 子串匹配（仍不使用行业名匹配）。
    - **⑥ 公告**：``anns_d`` 按 ``ts_code`` 与日期窗拉取；**⑦ 研报**：``research_report`` 个股研报 + 可选行业研报（``ind_name`` 与库内一致时才有行业命中）。
    - **⑧ 宏观分析**：在开启 Qdrant 时，使用 **与 ④⑤ 不同的专用检索词** 做全市场向量检索（时间窗为 **用户请求的起止日期**），可选 quick LLM 摘要；不混入 ④⑤ 语料。
    - 无法解析为 A 股代码时（如美股、港股代码），返回 ``npr`` + 新闻降级结果（无 e互动 / 公告 / 个股研报）；⑧ 仍可按同一日历窗尝试（若已开 Qdrant）。
    """
    win_start = f"{start_date} 00:00:00"
    win_end = f"{end_date} 23:59:59"
    d0, d1 = to_yyyymmdd(start_date), to_yyyymmdd(end_date)
    ph = "（本期无返回数据、或接口权限/参数不支持。）"
    broad_kw = macro_market_keywords()

    ts_code = resolve_tushare_equity(ticker)
    if not ts_code:
        return _get_tushare_news_without_ts_code(ticker, start_date, end_date, win_start, win_end, broad_kw, ph)

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

    def match_code_or_name_only(title: str, content: str) -> bool:
        """Fallback ④⑤：仅代码/简称（不含行业名子串），避免关闭 LLM 时全量灌入。"""
        blob = f"{title} {content}"
        return any(k and k in blob for k in news_keys)

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

    cfg = get_config()
    lb = int(cfg.get("news_long_short_lookback_days", 30))
    win_45_start, win_45_end = _long_short_window_strs(start_date, end_date, lb)

    from .news_qdrant_retrieval import news_long_short_use_qdrant

    use_qdrant = news_long_short_use_qdrant(cfg)
    cap_m = int(cfg.get("news_raw_major_per_src", 12))
    cap_f = int(cfg.get("news_raw_flash_per_src", 14))
    cm = int(cfg.get("news_raw_major_content_max", 2000))
    cf = int(cfg.get("news_raw_flash_content_max", 1200))
    pool_m = len(major_srcs) * cap_m
    pool_f = len(flash_srcs) * cap_f
    search_limit = int(cfg.get("news_qdrant_search_limit", 120))
    peer_n = int(cfg.get("news_llm_peer_context_max", 5))

    from .peers_deepseek import fetch_validated_peers

    if use_qdrant or not news_llm_filter_disabled():
        peer_objs = fetch_validated_peers(
            ts_code,
            stock_name,
            industry,
            max_peers=peer_n,
            curr_date=end_date,
            use_cache=True,
        )
    else:
        peer_objs = []
    peer_ts_codes = [p.ts_code for p in peer_objs]
    entities = [(ts_code, stock_name, industry)] + [(p.ts_code, p.name, p.industry) for p in peer_objs]

    used_qdrant_45 = False
    if use_qdrant:
        try:
            from .news_qdrant_retrieval import (
                retrieve_merged_equity_markdown_lines,
                retrieve_merged_equity_raw_items,
            )

            if news_llm_filter_disabled():
                sec4, sec5 = retrieve_merged_equity_markdown_lines(
                    entities=entities,
                    win_start=win_45_start,
                    win_end=win_45_end,
                    pool_major=pool_m,
                    pool_flash=pool_f,
                    max_major_lines=45,
                    max_flash_lines=50,
                    content_major_max=cm,
                    content_flash_max=cf,
                    search_limit=search_limit,
                    per_route_limit=None,
                    match_fn=match_code_or_name_only,
                )
            else:
                raw_mf = retrieve_merged_equity_raw_items(
                    entities=entities,
                    win_start=win_45_start,
                    win_end=win_45_end,
                    cap_major=pool_m,
                    cap_flash=pool_f,
                    content_major_max=cm,
                    content_flash_max=cf,
                    search_limit=search_limit,
                    per_route_limit=None,
                )
                sec4, sec5 = screen_long_short_news_with_llm(
                    raw_items=raw_mf,
                    focal_ticker=ticker,
                    ts_code=ts_code,
                    stock_name=stock_name,
                    industry=industry,
                    peer_ts_codes=peer_ts_codes,
                    win_start=win_45_start,
                    win_end=win_45_end,
                )
                sec4 = [sec4] if (sec4 or "").strip() else []
                sec5 = [sec5] if (sec5 or "").strip() else []
            used_qdrant_45 = True
        except Exception as exc:
            logger.warning("④⑤ Qdrant 检索失败，回退 Tushare：%s", exc)

    if not used_qdrant_45:
        if news_llm_filter_disabled():
            sec4 = _major_news_lines(
                win_45_start, win_45_end, match_code_or_name_only, major_srcs, per_src_cap=12
            )
            sec5 = _flash_news_lines(
                win_45_start, win_45_end, match_code_or_name_only, flash_srcs, per_src_cap=15
            )
        else:
            raw_m = _major_news_collect_raw(win_45_start, win_45_end, major_srcs, cap_m, content_max=cm)
            raw_f = _flash_news_collect_raw(win_45_start, win_45_end, flash_srcs, cap_f, content_max=cf)
            sec4, sec5 = screen_long_short_news_with_llm(
                raw_items=raw_m + raw_f,
                focal_ticker=ticker,
                ts_code=ts_code,
                stock_name=stock_name,
                industry=industry,
                peer_ts_codes=peer_ts_codes,
                win_start=win_45_start,
                win_end=win_45_end,
            )
            sec4 = [sec4] if (sec4 or "").strip() else []
            sec5 = [sec5] if (sec5 or "").strip() else []

    sec6 = _anns_d_lines(ts_code, d0, d1)
    sec7 = _research_report_lines(ts_code, industry, d0, d1)
    sec8 = _macro_section8_block(cfg, win_start=win_start, win_end=win_end)

    news_hdr = (
        f"窗口: {start_date} ~ {end_date} | ④⑤ 子窗: {win_45_start[:10]} ~ {win_45_end[:10]}（最长 {lb} 天）| "
        f"④⑤ 数据源：{'**Qdrant** 向量库' if used_qdrant_45 else 'Tushare API'} | "
        f"匹配/筛选：{'代码/简称子串（LLM 筛选已关闭）' if news_llm_filter_disabled() else 'LLM 语义筛选（已取消行业名子串匹配）'}"
        f" | ⑥⑦：**该股 ts_code** 的公告与研报（⑦ 另含与 ``stock_basic`` 行业名一致的 **行业研报** 抽样）"
        f" | ⑧：{'Qdrant 宏观专题（与④⑤ query 分离）' if sec8 else '（未生成：关闭 Qdrant 或 ``news_macro_section8_enabled`` 或未命中）'}"
    )
    blocks = [
        f"## Tushare 大模型语料（八项）— {ticker} / {ts_code}\n\n{news_hdr}",
        f"### ① 互动问答 · 上证e互动（irm_qa_sh）\n\n" + ("\n".join(sec1) if sec1 else ph),
        f"### ② 互动问答 · 深证互动易（irm_qa_sz）\n\n" + ("\n".join(sec2) if sec2 else ph),
        f"### ③ 国家政策库（npr）\n\n{npr_hint}"
        + ("\n".join(sec3) if sec3 else ph),
        f"### ④ 新闻快讯 · 长篇通讯（major_news）\n\n"
        + ("\n".join(sec4) if sec4 else ph),
        f"### ⑤ 新闻快讯 · 短讯（news）\n\n"
        + ("\n".join(sec5) if sec5 else ph),
        f"### ⑥ 上市公司公告（anns_d）\n\n" + ("\n".join(sec6) if sec6 else ph),
        f"### ⑦ 券商研究报告（research_report）\n\n"
        + (
            "> **说明**：行业研报依赖 ``ind_name`` 与研报库内名称一致；若仅有个股研报无行业条属正常。\n\n"
            if industry
            else ""
        )
        + ("\n".join(sec7) if sec7 else ph),
    ]
    if sec8:
        blocks.append(sec8)
    if not (sec1 or sec2 or sec3 or sec4 or sec5 or sec6 or sec7 or sec8):
        blocks.append(
            "\n---\n**说明**：多项为空时，常见原因包括：未开通大模型语料类接口权限（见 "
            "https://tushare.pro/wctapi/documents/290.md ）；该时段数据源无返回；"
            "e互动仅覆盖上证/深证互动平台；``anns_d`` / ``research_report`` 需单独语料权限。"
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
    """无法解析为 A 股 ``ts_code`` 时，仅 ``npr`` + 财经新闻语料；无代码则不调用公告/研报。"""
    raw = (ticker or "").strip()
    loose_kw = [x for x in (raw, raw.upper()) if x]

    def match_ticker_or_macro(title: str, content: str) -> bool:
        blob = f"{title} {content}"
        if any(k and k in blob for k in loose_kw):
            return True
        return any(k in blob for k in broad_kw)

    npr_hint = (
        "> **说明**：标的无法解析为 A 股 Tushare 代码；**无 e互动**与 ``stock_basic``。"
        "``npr`` 为国家政策库（非个股新闻）。\n\n"
    )
    sec3 = _npr_policy_lines(win_start, win_end, [], max_rows=25)
    cfg = get_config()
    from .news_qdrant_retrieval import news_long_short_use_qdrant, retrieve_markdown_loose

    used_q45 = False
    if news_long_short_use_qdrant(cfg):
        try:
            qtext = (raw or "") + " " + " ".join(str(k) for k in broad_kw[:24])
            sec4, sec5 = retrieve_markdown_loose(
                query_text=qtext,
                win_start=win_start,
                win_end=win_end,
                cap_major=10,
                cap_flash=12,
                content_major_max=2200,
                content_flash_max=1500,
                search_limit=int(cfg.get("news_qdrant_search_limit", 120)),
                match_fn=match_ticker_or_macro,
            )
            used_q45 = True
        except Exception as exc:
            logger.warning("④⑤ Qdrant（无 ts_code）失败，回退 Tushare：%s", exc)

    if not used_q45:
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
        f"## Tushare 语料（未识别为 A 股）— `{ticker}`\n\n"
        f"请输入 **6 位 A 股**代码（如 600519 / 600519.SH / 000001.SZ）。\n\n"
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
        "### ⑥ 上市公司公告（anns_d）\n\n"
        "（跳过：需要 A 股 ``ts_code``；请使用 ``get_tushare_news`` 并传入可解析代码。）",
        "### ⑦ 券商研究报告（research_report）\n\n"
        "（跳过：需要 ``ts_code``；全局宏观抽样见 ``get_tushare_global_news``。）",
    ]
    sec8 = _macro_section8_block(cfg, win_start=win_start, win_end=win_end)
    if sec8:
        blocks.append(sec8)
    return "\n\n".join(blocks).strip()


def get_tushare_global_news(
    curr_date: str,
    look_back_days: int = 7,
    limit: int = 50,
) -> str:
    """全局语料：国家政策库 ``npr`` + 长篇 ``major_news`` + 短讯 ``news`` + **⑧ 宏观向量专题** + 研报抽样（``research_report``）。

    互动问答、上市公司公告（``anns_d``）需 ``ts_code``，此处不调用；请对具体标的使用 ``get_tushare_news``。
    """
    curr = datetime.strptime(curr_date, "%Y-%m-%d")
    start = curr - timedelta(days=look_back_days)
    start_date = start.strftime("%Y-%m-%d")
    end_date = curr.strftime("%Y-%m-%d")
    win_start = f"{start_date} 00:00:00"
    win_end = f"{end_date} 23:59:59"
    d0, d1 = to_yyyymmdd(start_date), to_yyyymmdd(end_date)

    broad_kw = macro_market_keywords()

    def match_broad(title: str, content: str) -> bool:
        blob = f"{title} {content}"
        return any(k in blob for k in broad_kw)

    major_srcs = ("新浪财经", "财联社", "第一财经", "华尔街见闻", "中证网", "同花顺")
    flash_srcs = ("cls", "eastmoney", "wallstreetcn", "sina", "10jqka", "yicai", "fenghuang", "jinrongjie")

    per_major = max(6, min(20, limit // 4))
    per_flash = max(8, min(25, limit // 3))
    sec_npr = _npr_policy_lines(win_start, win_end, [], max_rows=max(30, min(80, limit)))

    cfg = get_config()
    from .news_qdrant_retrieval import news_long_short_use_qdrant, retrieve_global_markdown

    used_q45 = False
    if news_long_short_use_qdrant(cfg):
        try:
            sec_major, sec_flash = retrieve_global_markdown(
                win_start=win_start,
                win_end=win_end,
                per_major=per_major,
                per_flash=per_flash,
                limit=limit,
                match_fn=match_broad,
                broad_kw=broad_kw,
            )
            used_q45 = True
        except Exception as exc:
            logger.warning("全局 ④⑤ Qdrant 失败，回退 Tushare：%s", exc)

    if not used_q45:
        sec_major = _major_news_lines(
            win_start, win_end, match_broad, major_srcs, per_src_cap=per_major
        )
        sec_flash = _flash_news_lines(
            win_start, win_end, match_broad, flash_srcs, per_src_cap=per_flash
        )
    sec_rr = _research_report_global_lines(
        d0, d1, match_broad, max_rows=max(24, min(48, limit))
    )

    ph = "（本期无返回数据或未命中宏观/市场类关键词。）"
    ph_rr = "（本期无返回、或未命中宏观/市场类关键词、或 ``research_report`` 权限不足。）"

    sec8 = _macro_section8_block(cfg, win_start=win_start, win_end=win_end)

    blocks = [
        f"## Tushare 全局语料（npr + major_news + news + ⑧ + research_report）\n\n{win_start} — {win_end}\n\n"
        "> **说明**：本接口为**宏观与市场要闻**；`npr` 为政策法规库，**不是按股票代码的个股新闻**。"
        "④⑤ 为市场类要闻（与 ``get_tushare_news`` 中按标的检索的 ④⑤ query 不同）；**⑧** 为 **宏观专用 query** 的向量专题。"
        "``anns_d`` 全量公告需标的代码，请用 ``get_tushare_news``。\n",
        f"### 国家政策库（npr）\n\n" + ("\n".join(sec_npr) if sec_npr else ph),
        f"### 新闻快讯 · 长篇通讯（major_news）\n\n" + ("\n".join(sec_major) if sec_major else ph),
        f"### 新闻快讯 · 短讯（news）\n\n" + ("\n".join(sec_flash) if sec_flash else ph),
    ]
    if sec8:
        blocks.append(sec8)
    blocks.extend(
        [
            f"### 券商研究报告（research_report，宏观/行业关键词抽样）\n\n" + ("\n".join(sec_rr) if sec_rr else ph_rr),
            "\n*互动问答（irm_qa_sh / irm_qa_sz）与 **上市公司公告（anns_d）** 需具体标的，请使用 ``get_tushare_news``。*",
        ]
    )
    body = "\n\n".join(blocks).strip()
    if len(body) > 120_000:
        return body[:120_000] + "\n\n…（输出已截断）"
    return body


def get_tushare_insider_transactions(ticker: str) -> str:
    ts_code = resolve_tushare_equity(ticker)
    if not ts_code:
        return f"No shareholder trade data: unknown symbol '{ticker}'."
    df = _safe_pro_call("stk_holdertrade", ts_code=ts_code)
    if df is None or df.empty:
        return f"No shareholder trade data for '{ticker}' (Tushare stk_holdertrade)."
    return _df_to_csv_header("Shareholder increase/decrease (stk_holdertrade)", ts_code, df.to_csv(index=False))


def get_tushare_holder_number(ticker: str, start_date: str, end_date: str) -> str:
    """Shareholder count (户数) by announcement window — ``stk_holdernumber``."""
    ts_code = resolve_tushare_equity(ticker)
    if not ts_code:
        return f"No holder-number data: unknown symbol '{ticker}'."
    d0, d1 = to_yyyymmdd(start_date), to_yyyymmdd(end_date)
    df = _try_pro_call("stk_holdernumber", ts_code=ts_code, start_date=d0, end_date=d1)
    if df is None or df.empty:
        return (
            f"No shareholder count data for '{ticker}' (Tushare stk_holdernumber). "
            "May need higher Tushare points or a wider ann_date range; see https://tushare.pro/wctapi/documents/166.md"
        )
    if "ann_date" in df.columns:
        df = df.sort_values("ann_date", ascending=False, ignore_index=True)
    body = df.to_csv(index=False)
    if len(body) > 200_000:
        body = body[:200_000] + "\n…（输出已截断）"
    return _df_to_csv_header("Shareholder count / holder_num (stk_holdernumber)", ts_code, body)


def get_tushare_stock_moneyflow(ticker: str, start_date: str, end_date: str) -> str:
    """Per-stock money flow — large / extra-large orders via ``moneyflow``."""
    ts_code = resolve_tushare_equity(ticker)
    if not ts_code:
        return f"No moneyflow data: unknown symbol '{ticker}'."
    d0, d1 = to_yyyymmdd(start_date), to_yyyymmdd(end_date)
    df = _try_pro_call("moneyflow", ts_code=ts_code, start_date=d0, end_date=d1)
    if df is None or df.empty:
        return (
            f"No moneyflow data for '{ticker}' (Tushare moneyflow; requires ~2000+积分). "
            "See https://tushare.pro/wctapi/documents/170.md"
        )
    if "trade_date" in df.columns:
        df = df.sort_values("trade_date", ascending=False, ignore_index=True)
    body = df.to_csv(index=False)
    if len(body) > 200_000:
        body = body[:200_000] + "\n…（输出已截断）"
    return _df_to_csv_header(
        "Stock moneyflow: small/medium/large/extra-large (moneyflow)",
        ts_code,
        body,
    )


def get_tushare_margin_detail(ticker: str, start_date: str, end_date: str) -> str:
    """Margin trading detail — ``margin_detail`` (rzye 融资余额, rzmre 融资买入, etc.)."""
    ts_code = resolve_tushare_equity(ticker)
    if not ts_code:
        return f"No margin detail: unknown symbol '{ticker}'."
    d0, d1 = to_yyyymmdd(start_date), to_yyyymmdd(end_date)
    df = _try_pro_call("margin_detail", ts_code=ts_code, start_date=d0, end_date=d1)
    if df is None or df.empty:
        return (
            f"No margin detail for '{ticker}' (Tushare margin_detail). "
            "See https://tushare.pro/wctapi/documents/59.md"
        )
    if "trade_date" in df.columns:
        df = df.sort_values("trade_date", ascending=False, ignore_index=True)
    body = df.to_csv(index=False)
    if len(body) > 200_000:
        body = body[:200_000] + "\n…（输出已截断）"
    return _df_to_csv_header(
        "Margin trading detail: rzye / rzmre / rzche (margin_detail)",
        ts_code,
        body,
    )
