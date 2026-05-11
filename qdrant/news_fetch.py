"""Fetch and normalize Tushare ``news`` / ``major_news`` for ingestion."""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import tushare as ts

logger = logging.getLogger(__name__)

# Align with tradingagents/dataflows/tushare_data.py
MAJOR_SRCS = ("新浪财经", "同花顺")
FLASH_SRCS = ("sina", "eastmoney")


def _require_token() -> str:
    tok = (
        os.getenv("TUSHARE_TOKEN")
        or os.getenv("TUSHARE_API_KEY")
        or os.getenv("TUSHARE_PRO_TOKEN")
        or ""
    ).strip()
    if not tok:
        raise RuntimeError(
            "Missing Tushare token: set TUSHARE_TOKEN (or TUSHARE_API_KEY) in the environment."
        )
    return tok


def _parse_ts(val: Any) -> int | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    dt = pd.to_datetime(val, errors="coerce")
    if pd.isna(dt):
        return None
    try:
        return int(dt.timestamp())
    except Exception:  # noqa: BLE001
        return None


def _stable_point_id(title: str, pub_key: str, src: str) -> str:
    raw = f"{title}|{pub_key}|{src}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))


def fetch_tushare_news(days: int = 31) -> pd.DataFrame:
    """Pull flash ``news`` + ``major_news`` into one normalized DataFrame."""
    pro = ts.pro_api(_require_token())
    end = datetime.now()
    start = end - timedelta(days=int(days))
    start_str = start.strftime("%Y-%m-%d 00:00:00")
    end_str = end.strftime("%Y-%m-%d 23:59:59")
    logger.info(
        "Tushare: lookback=%s days, window=[%s .. %s]",
        int(days),
        start_str,
        end_str,
    )

    frames: list[pd.DataFrame] = []

    for src in FLASH_SRCS:
        try:
            df = pro.news(src=src, start_date=start_str, end_date=end_str)
            if df is not None and not df.empty:
                df = df.copy()
                df["source_type"] = "short_news"
                df["src"] = df.get("src", src)
                frames.append(df)
        except Exception as exc:  # noqa: BLE001
            logger.warning("news src=%s failed: %s", src, exc)

    for src in MAJOR_SRCS:
        try:
            df = pro.major_news(
                src=src,
                start_date=start_str,
                end_date=end_str,
                fields="title,content,pub_time,src",
            )
            if df is not None and not df.empty:
                df = df.copy()
                df["source_type"] = "major_news"
                df["src"] = df.get("src", src)
                # align time column name
                if "pub_time" in df.columns and "datetime" not in df.columns:
                    df["datetime"] = df["pub_time"]
                frames.append(df)
        except Exception as exc:  # noqa: BLE001
            logger.warning("major_news src=%s failed: %s", src, exc)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    # unified pub time string + unix
    if "datetime" in out.columns:
        out["pub_time"] = out["datetime"].astype(str)
    elif "pub_time" in out.columns:
        out["pub_time"] = out["pub_time"].astype(str)
    else:
        out["pub_time"] = ""

    out["pub_ts"] = out["pub_time"].map(_parse_ts)
    now_ts = int(datetime.now().timestamp())
    out["pub_ts"] = out["pub_ts"].fillna(now_ts).astype(int)

    out["title"] = out.get("title", "").astype(str).str.strip()
    out["content"] = out.get("content", "").astype(str)
    out["url"] = out.get("url", "").astype(str) if "url" in out.columns else ""

    out = out.drop_duplicates(subset=["title", "pub_time", "src"], keep="first")
    out["stable_id"] = [
        _stable_point_id(str(t), str(p), str(s))
        for t, p, s in zip(out["title"], out["pub_time"], out["src"])
    ]
    logger.info(
        "Tushare: concatenated + dedupe by (title, pub_time, src) -> %s rows",
        len(out),
    )
    return out.reset_index(drop=True)
