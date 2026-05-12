"""Fetch and normalize Tushare ``news`` / ``major_news`` for ingestion."""

from __future__ import annotations

import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import tushare as ts

logger = logging.getLogger(__name__)

# ``major_news`` / ``news`` 的 ``src`` 须与 Tushare 文档一致；单元素元组必须带尾逗号，否则写成 ``("x")`` 只是字符串、循环会按字符拆开。
MAJOR_SRCS = ("财联社", "第一财经")
FLASH_SRCS = ("sina", "cls")

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


def _fetch_concurrency() -> int:
    raw = (os.getenv("NEWS_FETCH_CONCURRENCY") or "4").strip()
    try:
        n = int(raw)
    except ValueError:
        n = 4
    return max(1, min(8, n))


def _fetch_one_src(kind: str, src: str, start_str: str, end_str: str) -> pd.DataFrame:
    """One Tushare source in an isolated thread (own ``pro_api``). Returns empty DF on failure."""
    pro = ts.pro_api(_require_token())
    try:
        if kind == "flash":
            df = pro.news(src=src, start_date=start_str, end_date=end_str)
            if df is None or df.empty:
                return pd.DataFrame()
            df = df.copy()
            df["source_type"] = "short_news"
            df["src"] = df.get("src", src)
            return df
        df = pro.major_news(
            src=src,
            start_date=start_str,
            end_date=end_str,
            fields="title,content,pub_time,src",
        )
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.copy()
        df["source_type"] = "major_news"
        df["src"] = df.get("src", src)
        if "pub_time" in df.columns and "datetime" not in df.columns:
            df["datetime"] = df["pub_time"]
        return df
    except Exception as exc:  # noqa: BLE001
        logger.warning("Tushare %s src=%s failed: %s", kind, src, exc)
        return pd.DataFrame()


def fetch_tushare_news(days: int = 31) -> pd.DataFrame:
    """Pull flash ``news`` + ``major_news`` into one normalized DataFrame."""
    end = datetime.now()
    start = end - timedelta(days=int(days))
    start_str = start.strftime("%Y-%m-%d 00:00:00")
    end_str = end.strftime("%Y-%m-%d 23:59:59")
    logger.info(
        "Tushare: lookback=%s days, window=[%s .. %s] fetch_concurrency=%s",
        int(days),
        start_str,
        end_str,
        _fetch_concurrency(),
    )

    tasks: list[tuple[str, str]] = [(kind, src) for src in FLASH_SRCS for kind in ("flash",)] + [
        ("major", src) for src in MAJOR_SRCS
    ]
    frames: list[pd.DataFrame] = []
    workers = _fetch_concurrency()
    if workers <= 1:
        for kind, src in tasks:
            df = _fetch_one_src(kind, src, start_str, end_str)
            if not df.empty:
                frames.append(df)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {
                pool.submit(_fetch_one_src, kind, src, start_str, end_str): (kind, src)
                for kind, src in tasks
            }
            for fut in as_completed(futs):
                kind, src = futs[fut]
                try:
                    df = fut.result()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Tushare fetch future failed kind=%s src=%s: %s", kind, src, exc)
                    continue
                if not df.empty:
                    frames.append(df)

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

    # Missing title/content from API are NaN; ``astype(str)`` alone becomes the literal ``"nan"`` string.
    if "title" in out.columns:
        out["title"] = out["title"].fillna("").astype(str).str.strip()
    else:
        out["title"] = ""
    if "content" in out.columns:
        out["content"] = out["content"].fillna("").astype(str)
    else:
        out["content"] = ""
    out["url"] = out["url"].fillna("").astype(str) if "url" in out.columns else ""

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
