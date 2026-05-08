"""Tushare Pro client and helpers (A-shares only)."""

from __future__ import annotations

import os
import re
from typing import Optional

import tushare as ts


class TushareVendorError(RuntimeError):
    """Raised when Tushare cannot fulfill the request (invalid symbol, API error, missing token)."""


_pro_api = None


def tushare_token() -> Optional[str]:
    return (
        os.getenv("TUSHARE_TOKEN")
        or os.getenv("TUSHARE_API_KEY")
        or os.getenv("TUSHARE_PRO_TOKEN")
    )


def get_pro():
    """Return a cached ``ts.pro_api()`` instance."""
    global _pro_api
    if _pro_api is not None:
        return _pro_api
    token = tushare_token()
    if not token or not str(token).strip():
        raise TushareVendorError(
            "Missing Tushare token: set TUSHARE_TOKEN or TUSHARE_API_KEY in the environment (.env)."
        )
    ts.set_token(str(token).strip())
    _pro_api = ts.pro_api()
    return _pro_api


def to_yyyymmdd(date_yyyy_mm_dd: str) -> str:
    return date_yyyy_mm_dd.replace("-", "")[:8]


def symbol_to_ts_code(symbol: str) -> Optional[str]:
    """
    Map a user ticker to Tushare **A-share** ``ts_code`` (``000001.SZ`` style).

    Returns ``None`` for symbols that are not A-share codes (e.g. ``AAPL``, ``00700.HK``).
    """
    raw = (symbol or "").strip().upper()
    if not raw:
        return None

    # Strip common US suffix
    if raw.endswith(".US"):
        raw = raw[:-3]

    m = re.match(r"^(\d{6})\.(SH|SZ|BJ)$", raw)
    if m:
        return raw

    core = re.sub(r"[^\d]", "", raw.split(".")[0])
    if len(core) != 6 or not core.isdigit():
        return None

    p2, p3 = core[:2], core[:3]
    if core.startswith("920"):
        return f"{core}.BJ"
    if p3 in ("688", "689") or p2 in ("60", "68"):
        return f"{core}.SH"
    if p2 in ("00", "30") or p3 in ("000", "001", "002", "003", "300", "301"):
        return f"{core}.SZ"
    if p2 in ("43", "83", "87", "88"):
        return f"{core}.BJ"
    if core[0] == "6":
        return f"{core}.SH"
    if core[0] in "03":
        return f"{core}.SZ"
    if core[0] in "48":
        return f"{core}.BJ"
    return None


def resolve_tushare_equity(symbol: str) -> Optional[str]:
    """Resolve user input to Tushare A-share ``ts_code`` (``daily`` / ``stock_basic`` 等）。"""
    return symbol_to_ts_code(symbol)


def require_ts_code(symbol: str) -> str:
    code = symbol_to_ts_code(symbol)
    if not code:
        raise TushareVendorError(
            f"Tushare does not recognize '{symbol}' as an A-share ts_code "
            f"(use 6-digit .SH/.SZ/.BJ, e.g. 600519 or 600519.SH)."
        )
    return code


def require_equity_ts(symbol: str) -> str:
    """Require A-share ``ts_code`` for ``daily`` and related interfaces."""
    r = resolve_tushare_equity(symbol)
    if not r:
        raise TushareVendorError(
            f"Tushare does not recognize '{symbol}' as an A-share ts_code "
            f"(use 6-digit .SH/.SZ/.BJ, e.g. 600519 or 600519.SH)."
        )
    return r
