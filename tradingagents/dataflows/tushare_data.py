"""Tushare-backed implementations for ``route_to_vendor`` (A-shares only; see project skill references)."""

from __future__ import annotations

import logging
import os
import io
import importlib
from pathlib import Path
from datetime import datetime, timedelta
import hashlib
import json
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

logger = logging.getLogger(__name__)

_RUN_NEWS_TOOL_CACHE: dict[str, str] = {}
_RUN_RESEARCH_PDF_CACHE: dict[str, dict] = {}

# 上证 e 互动 / 深证互动易 / 研报 ``research_report``：相对 **分析结束日** 固定回溯 90 个自然日（与 ④⑤/⑧ 的窗口独立）。
NEWS_IRM_NPR_REPORT_LOOKBACK_CAL_DAYS = 90


def _irm_npr_report_window_from_end(end_date: str) -> tuple[str, str, str, str]:
    """返回时间型接口用的时间串及 Tushare ``YYYYMMDD`` 起止。

    窗口 ``[end_date - 90 自然日 00:00:00, end_date 23:59:59]``（含端点）。
    用于 ①② ``irm_qa``、⑦ ``research_report``。
    """
    end_s = str(end_date)[:10]
    end_dt = pd.Timestamp(end_s)
    start_dt = end_dt - pd.Timedelta(days=int(NEWS_IRM_NPR_REPORT_LOOKBACK_CAL_DAYS))
    ext_start = start_dt.strftime("%Y-%m-%d")
    d0_ext = to_yyyymmdd(ext_start)
    d1_ext = to_yyyymmdd(end_s)
    ext_win_start = f"{ext_start} 00:00:00"
    ext_win_end = f"{end_s} 23:59:59"
    return ext_win_start, ext_win_end, d0_ext, d1_ext


def _df_to_csv_header(title: str, ts_code: str, body: str) -> str:
    header = f"# {title} for {ts_code}\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    return header + body


def clear_run_news_tool_cache() -> None:
    """Clear per-analysis in-memory cache for get_news/get_global_news."""
    _RUN_NEWS_TOOL_CACHE.clear()


def _news_tool_cache_key(kind: str, payload: dict) -> str:
    raw = json.dumps({"kind": kind, **payload}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:48]


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
    cfg = get_config()
    use_pdf = bool(cfg.get("news_research_pdf_extract_enabled", True))
    pdf_max_bytes = int(cfg.get("news_research_pdf_max_bytes", 15_000_000))
    pdf_text_max_chars = int(cfg.get("news_research_pdf_text_max_chars", 24_000))
    pdf_page_limit = int(cfg.get("news_research_pdf_page_limit", 80))
    pdf_playwright_timeout = float(cfg.get("news_research_pdf_playwright_timeout_sec", 40))
    pdf_playwright_channel = str(cfg.get("news_research_pdf_playwright_channel", "msedge") or "msedge").strip()
    out: list[str] = []
    seen: set[str] = set()

    def _append_pdf_diag_row(row: dict) -> None:
        try:
            from .run_trace_context import trace_report_dir

            base = trace_report_dir.get()
            if base is None:
                return
            trace_sub = Path(base) / "trace"
            trace_sub.mkdir(parents=True, exist_ok=True)
            path = trace_sub / "research_pdf_extract.jsonl"
            payload = dict(row)
            payload.setdefault("ts", datetime.now().isoformat())
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            return

    def _save_pdf_artifacts(
        *,
        url: str,
        trade_date: str,
        report_type: str,
        title: str,
        pdf_bytes: bytes | None,
        text: str,
    ) -> tuple[str, str]:
        try:
            from .run_trace_context import trace_report_dir

            base = trace_report_dir.get()
            if base is None:
                return "", ""
            trace_sub = Path(base) / "trace" / "research_pdf_artifacts"
            pdf_dir = trace_sub / "pdf"
            txt_dir = trace_sub / "text"
            pdf_dir.mkdir(parents=True, exist_ok=True)
            txt_dir.mkdir(parents=True, exist_ok=True)
            key = hashlib.sha256(f"{trade_date}|{report_type}|{title}|{url}".encode("utf-8")).hexdigest()[:16]
            prefix = f"{trade_date}_{key}"
            pdf_path = pdf_dir / f"{prefix}.pdf"
            txt_path = txt_dir / f"{prefix}.txt"

            # Persist only real PDF bytes; never keep HTML/script anti-bot pages as *.pdf.
            if pdf_bytes:
                incoming_is_pdf = bool(pdf_bytes.startswith(b"%PDF-"))
                existing_is_pdf = False
                if pdf_path.exists():
                    try:
                        with open(pdf_path, "rb") as f:
                            existing_is_pdf = f.read(5) == b"%PDF-"
                    except Exception:
                        existing_is_pdf = False
                if incoming_is_pdf:
                    # Overwrite existing invalid/old file with verified PDF bytes.
                    pdf_path.write_bytes(pdf_bytes)
                elif pdf_path.exists() and not existing_is_pdf:
                    # Remove stale invalid artifact so users don't open a broken file.
                    pdf_path.unlink(missing_ok=True)

            if text and not txt_path.exists():
                txt_path.write_text(text, encoding="utf-8")
            pdf_saved = ""
            if pdf_path.exists():
                try:
                    with open(pdf_path, "rb") as f:
                        if f.read(5) == b"%PDF-":
                            pdf_saved = str(pdf_path)
                except Exception:
                    pdf_saved = ""
            txt_saved = str(txt_path) if txt_path.exists() else ""
            return pdf_saved, txt_saved
        except Exception:
            return "", ""

    def _extract_pdf_text(url: str, *, trade_date: str, report_type: str, title: str) -> str:
        def _looks_like_pdf_bytes(data: bytes) -> bool:
            return bool(data and data.startswith(b"%PDF-"))

        def _download_with_playwright(target_url: str) -> tuple[bytes, str]:
            p = importlib.import_module("playwright.sync_api")
            sync_playwright = getattr(p, "sync_playwright")
            with sync_playwright() as pw:
                browser = pw.chromium.launch(headless=True, channel=pdf_playwright_channel)
                try:
                    ctx = browser.new_context()
                    try:
                        resp = ctx.request.get(target_url, timeout=int(pdf_playwright_timeout * 1000))
                        if not resp.ok:
                            raise RuntimeError(f"playwright_request_status_{resp.status}")
                        return (resp.body() or b""), "playwright_context_request"
                    finally:
                        ctx.close()
                finally:
                    browser.close()

        u = str(url or "").strip()
        if not use_pdf or not u:
            return ""
        cached = _RUN_RESEARCH_PDF_CACHE.get(u)
        if isinstance(cached, dict):
            _append_pdf_diag_row(
                {
                    "kind": "research_pdf_extract",
                    "url": u,
                    "trade_date": trade_date,
                    "report_type": report_type,
                    "title": title,
                    "from_cache": True,
                    "download_ok": bool(cached.get("download_ok", False)),
                    "parse_ok": bool(cached.get("parse_ok", False)),
                    "download_method": str(cached.get("download_method", "") or ""),
                    "download_bytes": int(cached.get("download_bytes", 0) or 0),
                    "text_chars": len(str(cached.get("text", "") or "")),
                    "pdf_path": str(cached.get("pdf_path", "") or ""),
                    "text_path": str(cached.get("text_path", "") or ""),
                    "error": str(cached.get("error", "") or ""),
                }
            )
            return str(cached.get("text", "") or "")
        try:
            content, method = _download_with_playwright(u)
        except Exception as exc:  # noqa: BLE001
            logger.warning("⑦ 研报PDF下载失败 url=%s err=%s", u, exc)
            _RUN_RESEARCH_PDF_CACHE[u] = {
                "text": "",
                "download_ok": False,
                "parse_ok": False,
                "download_method": "",
                "download_bytes": 0,
                "pdf_path": "",
                "text_path": "",
                "error": str(exc),
            }
            _append_pdf_diag_row(
                {
                    "kind": "research_pdf_extract",
                    "url": u,
                    "trade_date": trade_date,
                    "report_type": report_type,
                    "title": title,
                    "from_cache": False,
                    "download_ok": False,
                    "parse_ok": False,
                    "download_method": "",
                    "download_bytes": 0,
                    "text_chars": 0,
                    "pdf_path": "",
                    "text_path": "",
                    "error": str(exc),
                }
            )
            return ""
        if not content:
            _RUN_RESEARCH_PDF_CACHE[u] = {
                "text": "",
                "download_ok": False,
                "parse_ok": False,
                "download_method": "",
                "download_bytes": 0,
                "pdf_path": "",
                "text_path": "",
                "error": "empty_content",
            }
            _append_pdf_diag_row(
                {
                    "kind": "research_pdf_extract",
                    "url": u,
                    "trade_date": trade_date,
                    "report_type": report_type,
                    "title": title,
                    "from_cache": False,
                    "download_ok": False,
                    "parse_ok": False,
                    "download_method": "",
                    "download_bytes": 0,
                    "text_chars": 0,
                    "pdf_path": "",
                    "text_path": "",
                    "error": "empty_content",
                }
            )
            return ""
        if not _looks_like_pdf_bytes(content):
            err = "not_pdf_content"
            pdf_path, _ = _save_pdf_artifacts(
                url=u,
                trade_date=trade_date,
                report_type=report_type,
                title=title,
                pdf_bytes=content,
                text="",
            )
            _RUN_RESEARCH_PDF_CACHE[u] = {
                "text": "",
                "download_ok": True,
                "parse_ok": False,
                "download_method": method,
                "download_bytes": len(content),
                "pdf_path": pdf_path,
                "text_path": "",
                "error": err,
            }
            _append_pdf_diag_row(
                {
                    "kind": "research_pdf_extract",
                    "url": u,
                    "trade_date": trade_date,
                    "report_type": report_type,
                    "title": title,
                    "from_cache": False,
                    "download_ok": True,
                    "parse_ok": False,
                    "download_method": method,
                    "download_bytes": len(content),
                    "text_chars": 0,
                    "pdf_path": pdf_path,
                    "text_path": "",
                    "error": err,
                }
            )
            return ""
        if len(content) > pdf_max_bytes:
            logger.warning(
                "⑦ 研报PDF过大，跳过正文解析 url=%s bytes=%s cap=%s",
                u,
                len(content),
                pdf_max_bytes,
            )
            pdf_path, _ = _save_pdf_artifacts(
                url=u,
                trade_date=trade_date,
                report_type=report_type,
                title=title,
                pdf_bytes=content,
                text="",
            )
            _RUN_RESEARCH_PDF_CACHE[u] = {
                "text": "",
                "download_ok": True,
                "parse_ok": False,
                "download_method": method,
                "download_bytes": len(content),
                "pdf_path": pdf_path,
                "text_path": "",
                "error": "pdf_too_large",
            }
            _append_pdf_diag_row(
                {
                    "kind": "research_pdf_extract",
                    "url": u,
                    "trade_date": trade_date,
                    "report_type": report_type,
                    "title": title,
                    "from_cache": False,
                    "download_ok": True,
                    "parse_ok": False,
                    "download_method": method,
                    "download_bytes": len(content),
                    "text_chars": 0,
                    "pdf_path": pdf_path,
                    "text_path": "",
                    "error": "pdf_too_large",
                }
            )
            return ""
        try:
            pypdf_mod = importlib.import_module("pypdf")
            reader = pypdf_mod.PdfReader(io.BytesIO(content))
            chunks: list[str] = []
            total = 0
            for page in reader.pages[: max(1, pdf_page_limit)]:
                text = (page.extract_text() or "").strip()
                if not text:
                    continue
                if total >= pdf_text_max_chars:
                    break
                remain = max(0, pdf_text_max_chars - total)
                if len(text) > remain:
                    text = text[:remain]
                chunks.append(text)
                total += len(text)
            txt = "\n\n".join(chunks).strip()
            pdf_path, txt_path = _save_pdf_artifacts(
                url=u,
                trade_date=trade_date,
                report_type=report_type,
                title=title,
                pdf_bytes=content,
                text=txt,
            )
            _RUN_RESEARCH_PDF_CACHE[u] = {
                "text": txt,
                "download_ok": True,
                "parse_ok": True,
                "download_method": method,
                "download_bytes": len(content),
                "pdf_path": pdf_path,
                "text_path": txt_path,
                "error": "",
            }
            _append_pdf_diag_row(
                {
                    "kind": "research_pdf_extract",
                    "url": u,
                    "trade_date": trade_date,
                    "report_type": report_type,
                    "title": title,
                    "from_cache": False,
                    "download_ok": True,
                    "parse_ok": True,
                    "download_method": method,
                    "download_bytes": len(content),
                    "text_chars": len(txt),
                    "pdf_path": pdf_path,
                    "text_path": txt_path,
                    "error": "",
                }
            )
            return txt
        except Exception as exc:  # noqa: BLE001
            logger.warning("⑦ 研报PDF解析失败 url=%s err=%s", u, exc)
            pdf_path, _ = _save_pdf_artifacts(
                url=u,
                trade_date=trade_date,
                report_type=report_type,
                title=title,
                pdf_bytes=content,
                text="",
            )
            _RUN_RESEARCH_PDF_CACHE[u] = {
                "text": "",
                "download_ok": True,
                "parse_ok": False,
                "download_method": method,
                "download_bytes": len(content),
                "pdf_path": pdf_path,
                "text_path": "",
                "error": str(exc),
            }
            _append_pdf_diag_row(
                {
                    "kind": "research_pdf_extract",
                    "url": u,
                    "trade_date": trade_date,
                    "report_type": report_type,
                    "title": title,
                    "from_cache": False,
                    "download_ok": True,
                    "parse_ok": False,
                    "download_method": method,
                    "download_bytes": len(content),
                    "text_chars": 0,
                    "pdf_path": pdf_path,
                    "text_path": "",
                    "error": str(exc),
                }
            )
            return ""

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
            url = str(r.get("url", "")).strip()
            pdf_text = _extract_pdf_text(
                url,
                trade_date=str(td),
                report_type=str(r.get("report_type", "") or "个股研报"),
                title=title,
            )
            out.append(
                f"### {td} | {r.get('inst_csname', '')} [{r.get('report_type', '')}]\n**{title}**\n"
                f"{r.get('name', '')} ({r.get('ts_code', '')}) | {r.get('author', '')}\n{abstr}\n"
                f"{url}\n"
                + (f"\n--- PDF正文提取 ---\n{pdf_text}\n" if pdf_text else "")
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
                url = str(r.get("url", "")).strip()
                pdf_text = _extract_pdf_text(
                    url,
                    trade_date=str(td),
                    report_type=str(r.get("report_type", "") or "行业研报"),
                    title=title,
                )
                out.append(
                    f"### {td} | {r.get('inst_csname', '')} [{r.get('report_type', '')}]\n**{title}**\n"
                    f"{r.get('ind_name', '')} | {r.get('author', '')}\n{abstr}\n"
                    f"{url}\n"
                    + (f"\n--- PDF正文提取 ---\n{pdf_text}\n" if pdf_text else "")
                )
    return out


def _refine_research_report_lines_with_llm(
    *,
    ticker: str,
    ts_code: str,
    stock_name: str,
    industry: str,
    win_start: str,
    win_end: str,
    raw_lines: list[str],
) -> str:
    """LLM 精简 ⑦ 研报（先逐篇，再全局压缩），总输出硬限制为配置上限。"""
    if not raw_lines:
        return ""
    cfg = get_config()
    # 语义约定：
    # - news_research_llm_input_max_chars: 每篇研报送入 LLM 的输入上限
    # - news_research_llm_output_max_chars: 所有研报汇总后的最终总输出上限
    out_cap = int(cfg.get("news_research_llm_output_max_chars", 5000))
    in_cap = int(cfg.get("news_research_llm_input_max_chars", 30000))
    trunc_suffix = "\n\n…（研报精简输出已截断）"
    raw_lines = [x.strip() for x in raw_lines if str(x or "").strip()]
    if not raw_lines:
        return ""
    raw = "\n\n".join(raw_lines)

    if not bool(cfg.get("news_research_llm_refine", True)):
        return raw[:out_cap]

    try:
        from .news_long_short_llm_filter import _get_quick_llm, _normalize_llm_content

        llm, model = _get_quick_llm()
    except Exception as exc:
        logger.warning("⑦ 研报精简：无法创建 quick LLM（%s）", exc)
        return raw[:out_cap]

    # 第1阶段：逐篇研报独立精简
    n_reports = max(1, len(raw_lines))
    per_item_cap = max(600, min(1800, out_cap // min(n_reports, 6)))
    item_system = f"""你是A股卖方研报编辑。输入是一篇研报条目，请只提取对当前标的有价值的信息。

重点（必须优先）：
1) 关键预期：机构对未来1-4个季度/1-2年的核心判断（需求、价格、份额、产品放量、订单、政策、产能、出海等）
2) 关键数据：优先提取明确数字与变化方向（同比/环比、上修/下修、区间/目标值），包括财务数据、行业数据、宏观数据
3) 大概率将发生的事件：时间点、触发条件、验证路径（如新品发布、产能投放、招投标、价格拐点、监管落地）
4) 风险点：导致预期落空的主要风险、发生条件、可能后果
5) 产品与竞争：产品特性、护城河（技术/渠道/品牌/成本/生态/资质等）、相对竞品的比较优势或劣势
6) 财务与技术面：若原文有高价值信息可保留（如盈利预测、估值口径、量价结构、关键技术位/成交特征），但只保留结论性信息

时效性规则（必须执行）：
1) 明确标注研报发布日期（trade_date/pub_time）；无日期时标注“日期缺失”并降置信度。
2) 同一主题若出现新旧观点冲突，优先采用最新日期观点，并一句话说明“旧 -> 新”的变化。
3) 近期（近30-60天）信息优先于更早信息；较旧信息仅在解释趋势延续时保留。
4) 输出结论必须带时间锚点（如“截至2026-05-12”“Q3前后”），避免无时点判断。

表达约束（去废话）：
- 不要空话、套话、重复描述，不要长段背景复述。
- 不要泛泛而谈“持续看好/长期向好”而不给数据或触发条件。
- 每条结论尽量配“日期 + 数据/事实依据”。

要求：
- 只依据输入，不编造信息。
- 尽量保留时间、机构名、关键条件与方向（上调/下调、改善/恶化）。
- 优先输出“结论+关键数据+依据”，避免流水账复述。
- 不要输出代码围栏。
- 输出 <= {per_item_cap} 字符。"""
    item_results: list[str] = []
    for idx, one in enumerate(raw_lines, start=1):
        one_in = one if len(one) <= in_cap else one[:in_cap] + "\n\n…（单篇研报输入已截断）"
        item_user = (
            f"标的: {ticker} | ts_code: {ts_code} | 名称: {stock_name or '未知'} | 行业: {industry or '未知'}\n"
            f"研报窗口: {win_start} ~ {win_end}\n"
            f"第 {idx}/{n_reports} 篇研报\n\n"
            f"--- 单篇研报原文 ---\n{one_in}"
        )
        try:
            resp = llm.invoke([("system", item_system), ("human", item_user)])
            txt = _normalize_llm_content(getattr(resp, "content", None)).strip()
        except Exception:
            txt = ""
        if not txt:
            txt = one_in[:per_item_cap]
        if len(txt) > per_item_cap:
            txt = txt[:per_item_cap]
        item_results.append(f"#### 研报{idx}\n{txt}")

    merged = "\n\n".join(item_results).strip()
    if not merged:
        merged = raw[:out_cap]

    # 第2阶段：若总长度超限，对“逐篇精简结果”继续全局压缩（最多 3 轮）
    current = merged
    global_system = """你是A股研究总编。输入是“多篇研报逐篇精简结果”的合集，请做全局去重与归纳。
必须突出：
1) 跨研报一致的核心预期（以及分歧点）
2) 未来大概率事件的时间线、触发条件、先行验证信号
3) 关键数据汇总（财务/行业/宏观）与口径差异，尽量保留数字和方向
4) 主要风险清单（触发条件+影响路径+优先级）
5) 产品特性、护城河与相对竞争优势（含可能被削弱的因素）

时效性规则（必须执行）：
1) 先做按日期排序再归纳：新到旧。
2) 同一议题出现冲突时，默认“最新研报观点优先”，并明确写出被替代的旧观点。
3) 输出需区分“近期已验证/正在验证/远期假设”三类，避免把历史结论当作当前结论。
4) 每个关键结论都要保留日期锚点（至少到天或周）。

表达约束（去废话）：
- 不要堆砌形容词或重复同义结论。
- 保留“有数据支撑的财务/技术面信息”，删掉无数字、无条件、无时点的表述。
- 冲突观点用“旧观点 -> 新观点 -> 当前判断”三段式简写。

只基于输入，不编造；不要代码围栏。"""
    for _ in range(3):
        if len(current) <= out_cap:
            break
        target = max(800, out_cap - 120)
        global_user = (
            f"标的: {ticker} | ts_code: {ts_code} | 名称: {stock_name or '未知'} | 行业: {industry or '未知'}\n"
            f"窗口: {win_start} ~ {win_end}\n"
            f"请将以下内容压缩到 <= {target} 字符。\n\n--- 逐篇精简汇总 ---\n{current}"
        )
        try:
            resp = llm.invoke([("system", global_system), ("human", global_user)])
            nxt = _normalize_llm_content(getattr(resp, "content", None)).strip()
        except Exception:
            nxt = ""
        if not nxt:
            break
        if len(nxt) >= len(current):
            # 模型未有效压缩，避免无效循环
            break
        current = nxt

    if len(current) > out_cap:
        keep = max(0, out_cap - len(trunc_suffix))
        current = current[:keep] + trunc_suffix
    logger.info(
        "⑦ 研报精简：model=%r reports=%s raw_len=%s out_len=%s",
        model,
        len(raw_lines),
        len(raw),
        len(current),
    )
    return current


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
    max_peers: Annotated[int, "Maximum number of peer rows (excluding focal) to return"] = 3,
) -> str:
    """List **business competitors** as A-shares: **DeepSeek** (dedicated prompt via Chat Completions) + ``stock_basic`` validation.

    Tushare is **not** used to rank or sample a same-industry universe; it only verifies codes and fills display fields.
    """
    from .peers_deepseek import fetch_validated_peers

    ts_code = resolve_tushare_equity(ticker)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not ts_code:
        return (
            "# Listed competitors (DeepSeek inference + Tushare code check)\n\n"
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

    cap = max(1, min(int(max_peers) if max_peers else 3, 12))
    peers = fetch_validated_peers(
        ts_code,
        focal_name,
        industry,
        max_peers=cap,
        curr_date=curr_date,
        use_cache=True,
    )
    note = (
        f"**标的**：{focal_name}（`{ts_code}`）。披露行业分类：**{industry or '（空）'}**（监管字段，**不是**竞品选股依据）。\n\n"
        "## 竞品名单如何产生（请在下游报告中如实表述）\n"
        "1. 后端使用 **DeepSeek OpenAI 兼容 Chat Completions**（专用系统/用户提示词，模型见 `PEER_LLM_MODEL` / 默认 `deepseek-v4-flash`）"
        " 由模型**独立推理**主营业务竞争关系，输出候选 `ts_code` 列表。\n"
        "2. **Tushare `stock_basic`** 仅做：代码是否存在、是否处于上市状态、补全证券简称与披露行业列；"
        "**不作为**「从 Tushare 同行业（或行业分类）列表里挑几只」的依据。\n\n"
        f"下列最多 **{cap}** 家为校码后的竞品，用于 `get_news` / `get_fundamentals` 等后续分析。\n\n"
    )
    if not peers:
        return (
            "# Listed competitors (DeepSeek inference + Tushare code check)\n\n"
            f"_Retrieved: {stamp}_\n\n"
            + note
            + "No validated peer rows (check API keys: `PEER_LLM_API_KEY` / `NEWS_TAG_LLM_API_KEY` / `DEEPSEEK_API_KEY`).\n"
        )

    lines = [
        "| ts_code | name | industry |",
        "| --- | --- | --- |",
    ]
    for p in peers:
        lines.append(f"| {p.ts_code} | {p.name} | {p.industry} |")
    return (
        "# Listed competitors (DeepSeek inference + Tushare code check)\n\n"
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
        raise TushareVendorError(f"⑧ 宏观向量专题检索失败: {exc}") from exc
    raw_md = "\n".join([*majors, *flashes]).strip()
    head = (
        "### ⑧ 宏观分析（向量库专题）\n\n"
        "> **与 ④⑤ 区分**：④⑤按 **标的/行业/同业** 构造向量 query；本节使用 **宏观专用关键词** "
        "（``macro_keywords.macro_vector_search_query_text``）做**全市场**检索。**时间窗与 ④⑤ 相同**（配置 "
        "``news_long_short_lookback_days``，与 ``start_date``/``end_date`` 求交）。\n"
    )
    if not raw_md:
        return head + "（本期无命中或未入库该时间窗。）"

    parts = [head]
    raw_show = raw_md if len(raw_md) <= raw_max else raw_md[:raw_max] + "\n\n…（原文摘录已截断）"
    parts.append("\n#### 语料摘录（⑧ 独立检索）\n\n" + raw_show)
    return "".join(parts)


def get_tushare_news(
    ticker: str,
    start_date: str,
    end_date: str,
) -> str:
    """个股新闻语料（不含全局宏观包）：①②④⑤⑦。

    - ``get_global_news`` 专门承担 **⑧ 宏观向量专题**。
    - 本函数仅返回：①② 互动问答、④⑤（Qdrant 合并召回）、⑦ 券商研报（个股+行业）。
    - ④⑤ 为 Qdrant-only：检索失败直接报错，不回退 Tushare。
    """
    cfg = get_config()
    cache_key = _news_tool_cache_key(
        "get_news",
        {
            "ticker": str(ticker).strip().upper(),
            "start_date": start_date,
            "end_date": end_date,
            "news_long_short_lookback_days": int(cfg.get("news_long_short_lookback_days", 30)),
            "news_qdrant_search_limit": int(cfg.get("news_qdrant_search_limit", 120)),
            "news_raw_major_per_src": int(cfg.get("news_raw_major_per_src", 12)),
            "news_raw_flash_per_src": int(cfg.get("news_raw_flash_per_src", 14)),
            "news_macro_section8_enabled": bool(cfg.get("news_macro_section8_enabled", True)),
            "news_macro_section8_search_limit": int(cfg.get("news_macro_section8_search_limit", 100)),
            "news_research_llm_refine": bool(cfg.get("news_research_llm_refine", True)),
            "news_research_llm_output_max_chars": int(cfg.get("news_research_llm_output_max_chars", 5000)),
            "qdrant_collection": (os.getenv("QDRANT_COLLECTION") or "financial_news"),
            "schema": "run_cache_v2",
        },
    )
    cached = _RUN_NEWS_TOOL_CACHE.get(cache_key)
    if cached:
        logger.info("get_news run-cache hit: ticker=%s %s~%s", ticker, start_date, end_date)
        return cached

    win_start = f"{start_date} 00:00:00"
    win_end = f"{end_date} 23:59:59"
    _, _, irm_rr_d0, irm_rr_d1 = _irm_npr_report_window_from_end(end_date)
    ph = "（本期无返回数据、或接口权限/参数不支持。）"

    ts_code = resolve_tushare_equity(ticker)
    if not ts_code:
        body = _get_tushare_news_without_ts_code(
            ticker, start_date, end_date, win_start, win_end, ph
        )
        _RUN_NEWS_TOOL_CACHE[cache_key] = body
        return body

    stock_name = ""
    industry = ""
    basic = _try_pro_call("stock_basic", ts_code=ts_code, fields="ts_code,name,industry")
    if basic is not None and not basic.empty:
        row = basic.iloc[0]
        stock_name = str(row.get("name") or "").strip()
        industry = str(row.get("industry") or "").strip()

    major_srcs = ("新浪财经", "财联社", "第一财经", "华尔街见闻", "中证网", "同花顺")
    flash_srcs = ("sina", "eastmoney", "10jqka", "cls", "yicai", "fenghuang", "jinrongjie", "wallstreetcn")

    sec1 = _irm_qa_lines(ts_code, irm_rr_d0, irm_rr_d1, "irm_qa_sh", "上证e互动")
    sec2 = _irm_qa_lines(ts_code, irm_rr_d0, irm_rr_d1, "irm_qa_sz", "深证互动易")

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
    peer_n = int(cfg.get("news_llm_peer_context_max", 3))

    from .peers_deepseek import fetch_validated_peers

    peer_objs = fetch_validated_peers(
        ts_code,
        stock_name,
        industry,
        max_peers=peer_n,
        curr_date=end_date,
        use_cache=True,
    )
    peer_ts_codes = [p.ts_code for p in peer_objs]
    entities = [(ts_code, stock_name, industry)] + [(p.ts_code, p.name, p.industry) for p in peer_objs]

    if not use_qdrant:
        raise TushareVendorError(
            "Qdrant-only news mode requires NEWS_LONG_SHORT_USE_QDRANT=1 "
            "(or config `news_long_short_use_qdrant=True`)."
        )
    from .news_qdrant_retrieval import retrieve_merged_equity_markdown_lines

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
        match_fn=lambda _t, _c: True,
    )

    sec7_raw = _research_report_lines(ts_code, industry, irm_rr_d0, irm_rr_d1)
    sec7 = _refine_research_report_lines_with_llm(
        ticker=ticker,
        ts_code=ts_code,
        stock_name=stock_name,
        industry=industry,
        win_start=irm_rr_d0,
        win_end=irm_rr_d1,
        raw_lines=sec7_raw,
    )
    news_hdr = (
        f"窗口: {start_date} ~ {end_date} | ④⑤ 子窗: {win_45_start[:10]} ~ {win_45_end[:10]}（最长 {lb} 天）| "
        "④⑤ 数据源：**Qdrant** 向量库 | "
        "匹配/筛选：向量召回（无代码/简称二次过滤）"
        f" | ①②⑦：相对结束日 **{NEWS_IRM_NPR_REPORT_LOOKBACK_CAL_DAYS} 自然日**"
    )
    blocks = [
        f"## Tushare 个股语料（①②④⑤⑦）— {ticker} / {ts_code}\n\n{news_hdr}",
        f"### ① 互动问答 · 上证e互动（irm_qa_sh）\n\n" + ("\n".join(sec1) if sec1 else ph),
        f"### ② 互动问答 · 深证互动易（irm_qa_sz）\n\n" + ("\n".join(sec2) if sec2 else ph),
        f"### ④ 新闻快讯 · 长篇通讯（major_news）\n\n"
        + ("\n".join(sec4) if sec4 else ph),
        f"### ⑤ 新闻快讯 · 短讯（news）\n\n"
        + ("\n".join(sec5) if sec5 else ph),
        f"### ⑦ 券商研究报告（research_report，LLM精简≤{int(cfg.get('news_research_llm_output_max_chars', 5000))}字）\n\n"
        + (
            "> **说明**：行业研报依赖 ``ind_name`` 与研报库内名称一致；若仅有个股研报无行业条属正常。\n\n"
            if industry
            else ""
        )
        + (sec7 if sec7 else ph),
    ]
    if not (sec1 or sec2 or sec4 or sec5 or sec7):
        blocks.append(
            "\n---\n**说明**：多项为空时，常见原因包括：未开通大模型语料类接口权限（见 "
            "https://tushare.pro/wctapi/documents/290.md ）；该时段数据源无返回；"
            "e互动仅覆盖上证/深证互动平台；``research_report`` 需单独语料权限。"
            f"（公司简称：{stock_name or '未取到'}；行业：{industry or '未取到'}）"
        )
    body = "\n\n".join(blocks).strip()
    _RUN_NEWS_TOOL_CACHE[cache_key] = body
    return body


def _get_tushare_news_without_ts_code(
    ticker: str,
    start_date: str,
    end_date: str,
    win_start: str,
    win_end: str,
    ph: str,
) -> str:
    """无法解析为 A 股 ``ts_code`` 时，仅返回 ④⑤ 新闻语料（不含 ⑧）。"""
    raw = (ticker or "").strip()
    loose_kw = [x for x in (raw, raw.upper()) if x]

    def match_ticker_only(title: str, content: str) -> bool:
        blob = f"{title} {content}"
        if any(k and k in blob for k in loose_kw):
            return True
        return False
    cfg = get_config()
    from .news_qdrant_retrieval import news_long_short_use_qdrant, retrieve_markdown_loose

    if not news_long_short_use_qdrant(cfg):
        raise TushareVendorError(
            "Qdrant-only news mode requires NEWS_LONG_SHORT_USE_QDRANT=1 "
            "(or config `news_long_short_use_qdrant=True`)."
        )
    qtext = (raw or "")
    sec4, sec5 = retrieve_markdown_loose(
        query_text=qtext,
        win_start=win_start,
        win_end=win_end,
        cap_major=10,
        cap_flash=12,
        content_major_max=2200,
        content_flash_max=1500,
        search_limit=int(cfg.get("news_qdrant_search_limit", 120)),
        match_fn=match_ticker_only,
    )

    blocks = [
        f"## Tushare 个股语料（未识别为 A 股）— `{ticker}`\n\n"
        f"请输入 **6 位 A 股**代码（如 600519 / 600519.SH / 000001.SZ）。\n\n"
        f"窗口: {start_date} ~ {end_date}",
        f"### ①② 互动问答（irm_qa_sh / irm_qa_sz）\n\n"
        "（跳过：需要 A 股 ``ts_code``。）",
        f"### ④ 新闻快讯 · 长篇（major_news）\n\n"
        + ("\n".join(sec4) if sec4 else ph),
        f"### ⑤ 新闻快讯 · 短讯（news）\n\n" + ("\n".join(sec5) if sec5 else ph),
        "### ⑦ 券商研究报告（research_report）\n\n"
        "（跳过：需要 ``ts_code``；全局宏观语料见 ``get_tushare_global_news``。）",
    ]
    return "\n\n".join(blocks).strip()


def get_tushare_global_news(
    curr_date: str,
    look_back_days: int = 7,
    limit: int = 50,
) -> str:
    """全局语料：仅 **⑧ 宏观向量专题**。"""
    cfg = get_config()
    cache_key = _news_tool_cache_key(
        "get_global_news",
        {
            "curr_date": curr_date,
            "look_back_days": int(look_back_days),
            "limit": int(limit),
            "news_long_short_lookback_days": int(cfg.get("news_long_short_lookback_days", 30)),
            "news_qdrant_search_limit": int(cfg.get("news_qdrant_search_limit", 120)),
            "news_macro_section8_enabled": bool(cfg.get("news_macro_section8_enabled", True)),
            "news_macro_section8_search_limit": int(cfg.get("news_macro_section8_search_limit", 100)),
            "qdrant_collection": (os.getenv("QDRANT_COLLECTION") or "financial_news"),
            "schema": "run_cache_v1",
        },
    )
    cached = _RUN_NEWS_TOOL_CACHE.get(cache_key)
    if cached:
        logger.info(
            "get_global_news run-cache hit: curr_date=%s look_back_days=%s limit=%s",
            curr_date,
            look_back_days,
            limit,
        )
        return cached

    curr = datetime.strptime(curr_date, "%Y-%m-%d")
    start = curr - timedelta(days=look_back_days)
    start_date = start.strftime("%Y-%m-%d")
    end_date = curr.strftime("%Y-%m-%d")

    lb = int(cfg.get("news_long_short_lookback_days", 30))
    win_45_start, win_45_end = _long_short_window_strs(start_date, end_date, lb)
    from .news_qdrant_retrieval import news_long_short_use_qdrant

    if not news_long_short_use_qdrant(cfg):
        raise TushareVendorError(
            "Qdrant-only news mode requires NEWS_LONG_SHORT_USE_QDRANT=1 "
            "(or config `news_long_short_use_qdrant=True`)."
        )
    ph = "（本期无返回数据。）"
    sec8 = _macro_section8_block(cfg, win_start=win_45_start, win_end=win_45_end)
    if not sec8:
        raise TushareVendorError("⑧ 宏观向量专题未生成（可能未命中语料或未开启）。")

    blocks = [
        f"## Tushare 全局语料（⑧）\n\n"
        f"⑧ 子窗：{win_45_start[:10]} ~ {win_45_end[:10]}（最长 {lb} 天）\n\n"
        "> **说明**：本接口只提供全局宏观语料（⑧）。个股/竞品新闻与研报请使用 ``get_tushare_news``。\n",
        sec8,
    ]
    body = "\n\n".join(blocks).strip()
    if len(body) > 120_000:
        body = body[:120_000] + "\n\n…（输出已截断）"
    _RUN_NEWS_TOOL_CACHE[cache_key] = body
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
    """Per-stock money flow via THS endpoint ``moneyflow_ths``."""
    ts_code = resolve_tushare_equity(ticker)
    if not ts_code:
        return f"No moneyflow data: unknown symbol '{ticker}'."
    d0, d1 = to_yyyymmdd(start_date), to_yyyymmdd(end_date)
    df = _try_pro_call("moneyflow_ths", ts_code=ts_code, start_date=d0, end_date=d1)
    if df is None or df.empty:
        return (
            f"No moneyflow data for '{ticker}' (Tushare moneyflow_ths; requires ~6000+积分). "
            "See https://tushare.pro/wctapi/documents/348.md"
        )
    if "trade_date" in df.columns:
        df = df.sort_values("trade_date", ascending=False, ignore_index=True)
    body = df.to_csv(index=False)
    if len(body) > 200_000:
        body = body[:200_000] + "\n…（输出已截断）"
    return _df_to_csv_header(
        "Stock moneyflow (THS): net_amount / net_d5_amount / buy_lg_md_sm_amount_rate (moneyflow_ths)",
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
