"""Tushare Pro client and helpers (A-share and Hong Kong equities)."""

from __future__ import annotations

import os
import re
from typing import Literal, Optional

import tushare as ts

EquityMarket = Literal["cn", "hk"]


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


def symbol_to_hk_ts_code(symbol: str) -> Optional[str]:
    """
    Map a user ticker to Tushare Hong Kong ``ts_code`` (``00700.HK`` style, 5 digits + .HK).

    Accepts ``00700.HK``, ``HK00700``, ``HK9992``, ``9992``, etc. Returns ``None`` if not HK-shaped.
    """
    raw = (symbol or "").strip().upper()
    if not raw:
        return None

    m = re.match(r"^(\d{1,5})\.HK$", raw)
    if m:
        return f"{int(m.group(1)):05d}.HK"

    m = re.match(r"^HK\.?(\d{1,5})$", raw)
    if m:
        return f"{int(m.group(1)):05d}.HK"

    digits = re.sub(r"\D", "", raw.split(".")[0])
    if digits.isdigit() and 1 <= len(digits) <= 5:
        return f"{int(digits):05d}.HK"

    return None


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


def resolve_tushare_equity(symbol: str) -> Optional[tuple[str, EquityMarket]]:
    """Resolve **A-share** or **Hong Kong** listing to ``(ts_code, 'cn'|'hk')``.

    A-share is tried first (6-digit mainland codes); then HK (``xxxxx.HK`` / ``HK`` prefix / 1–5 digits).
    """
    cn = symbol_to_ts_code(symbol)
    if cn:
        return (cn, "cn")
    hk = symbol_to_hk_ts_code(symbol)
    if hk:
        return (hk, "hk")
    return None


def require_ts_code(symbol: str) -> str:
    code = symbol_to_ts_code(symbol)
    if not code:
        raise TushareVendorError(
            f"Tushare does not recognize '{symbol}' as an A-share ts_code "
            f"(use 6-digit .SH/.SZ/.BJ, or Hong Kong e.g. 00700.HK / HK00700)."
        )
    return code


def require_equity_ts(symbol: str) -> tuple[str, EquityMarket]:
    """Require A-share or Hong Kong ``ts_code`` (Tushare ``daily`` / ``hk_daily``)."""
    r = resolve_tushare_equity(symbol)
    if not r:
        raise TushareVendorError(
            f"Tushare does not recognize '{symbol}' as A-share (6-digit) or Hong Kong (xxxxx.HK)."
        )
    return r
