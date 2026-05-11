"""④⑤ / ⑧ 语料：从 Qdrant 向量库检索（替代或补充 Tushare ``major_news`` / ``news`` 拉取）。

- **④⑤**：个股/行业/宽松 query 等（见 ``retrieve_*_equity``、``retrieve_markdown_loose``、``retrieve_global_markdown``）。
- **⑧**：宏观专题专用 query（``retrieve_macro_section_markdown``），与 ④⑤ 检索词分离。

依赖：``qdrant-client``、与入库一致的嵌入（默认 ``qdrant/news_embed.embed_query_texts``）。
启用：配置 ``news_long_short_use_qdrant`` 或环境变量 ``NEWS_LONG_SHORT_USE_QDRANT=1``。
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.macro_keywords import macro_vector_search_query_text

logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_qdrant_on_path() -> None:
    qdir = _repo_root() / "qdrant"
    s = str(qdir)
    if s not in sys.path:
        sys.path.insert(0, s)


def _import_embed_query_texts():
    _ensure_qdrant_on_path()
    try:
        from news_embed import embed_query_texts
    except ImportError as exc:  # noqa: BLE001
        raise RuntimeError(
            "Qdrant news retrieval requires the repo ``qdrant/`` package and "
            "``sentence_transformers`` / optional deps — see pyproject optional ``qdrant-news``."
        ) from exc
    return embed_query_texts


def _import_qdrant_client():
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http.models import FieldCondition, Filter, MatchAny, MatchValue, Range
    except ImportError as exc:  # noqa: BLE001
        raise RuntimeError(
            "Install qdrant-client (e.g. pip install tradingagents[qdrant-news] or pip install qdrant-client)."
        ) from exc
    return QdrantClient, FieldCondition, Filter, MatchAny, MatchValue, Range


def news_long_short_use_qdrant(cfg: dict | None = None) -> bool:
    env = os.getenv("NEWS_LONG_SHORT_USE_QDRANT", "").strip().lower()
    if env in ("1", "true", "yes", "on"):
        return True
    if env in ("0", "false", "no", "off"):
        return False
    c = cfg if cfg is not None else get_config()
    return bool(c.get("news_long_short_use_qdrant", False))


def _pub_ts_range(win_start: str, win_end: str) -> tuple[int, int]:
    a = pd.to_datetime(win_start, errors="coerce")
    b = pd.to_datetime(win_end, errors="coerce")
    if pd.isna(a) or pd.isna(b):
        raise ValueError(f"Invalid time window: {win_start!r} .. {win_end!r}")
    lo = int(a.timestamp())
    hi = int(b.timestamp())
    return lo, hi


def _make_client():
    QdrantClient, *_ = _import_qdrant_client()
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    key = (os.getenv("QDRANT_API_KEY") or "").strip()
    kwargs: dict[str, Any] = {}
    if key:
        kwargs["api_key"] = key
    return QdrantClient(url, **kwargs)


def _collection_name(cfg: dict) -> str:
    return (os.getenv("QDRANT_COLLECTION") or cfg.get("news_qdrant_collection") or "financial_news").strip()


def _hit_to_channel(payload: dict[str, Any]) -> str:
    st = str(payload.get("source_type") or "").strip().lower()
    if st == "major_news":
        return "major_news"
    return "news"


def _hit_to_raw_item(
    point_id: Any,
    payload: dict[str, Any],
    *,
    content_max: int,
) -> dict[str, Any]:
    ch = _hit_to_channel(payload)
    return {
        "id": str(point_id),
        "channel": ch,
        "src": str(payload.get("source") or payload.get("src") or ""),
        "pub_time": str(payload.get("pub_time") or ""),
        "title": str(payload.get("title") or ""),
        "content": str(payload.get("content") or "")[:content_max],
    }


def _hit_to_markdown(payload: dict[str, Any], *, content_max: int) -> str:
    src = str(payload.get("source") or payload.get("src") or "")
    pub = str(payload.get("pub_time") or "")
    title = str(payload.get("title") or "")
    body = str(payload.get("content") or "")[:content_max]
    return f"### [{src}] {pub}\n{title}\n{body}\n"


def search_news_qdrant(
    *,
    query_text: str,
    win_start: str,
    win_end: str,
    limit: int,
    ts_code: str | None = None,
) -> list[Any]:
    """Return Qdrant search hits (``ScoredPoint``)."""
    cfg = get_config()
    _, FieldCondition, Filter, MatchAny, MatchValue, Range = _import_qdrant_client()
    embed_query_texts = _import_embed_query_texts()
    vec = embed_query_texts([query_text])
    if not vec or not vec[0]:
        raise RuntimeError("embed_query_texts returned empty vector")
    lo, hi = _pub_ts_range(win_start, win_end)
    must: list[Any] = [
        FieldCondition(
            key="pub_ts",
            range=Range(gte=float(lo), lte=float(hi)),
        )
    ]
    if ts_code:
        must.append(
            FieldCondition(
                key="tickers",
                match=MatchAny(any=[MatchValue(value=str(ts_code).strip())]),
            )
        )
    flt = Filter(must=must)
    client = _make_client()
    try:
        hits = client.search(
            collection_name=_collection_name(cfg),
            query_vector=vec[0],
            query_filter=flt,
            limit=int(limit),
            with_payload=True,
        )
    finally:
        client.close()
    return list(hits)


def retrieve_raw_items_equity(
    *,
    ts_code: str,
    stock_name: str,
    industry: str,
    peer_ts_codes: list[str],
    win_start: str,
    win_end: str,
    cap_major: int,
    cap_flash: int,
    content_major_max: int,
    content_flash_max: int,
    search_limit: int,
    ts_code_filter: str | None,
) -> list[dict[str, Any]]:
    """Build ``raw_items`` for ``screen_long_short_news_with_llm`` from Qdrant."""
    peer_bit = ", ".join(peer_ts_codes[:12]) if peer_ts_codes else ""
    query_text = (
        f"A股标的 {ts_code} {stock_name or ''} 行业背景:{industry or '未知'}。"
        f"相关同业代码参考:{peer_bit}。"
        "请检索与该上市公司及行业相关的财经新闻与政策舆情。"
    )
    hits = search_news_qdrant(
        query_text=query_text,
        win_start=win_start,
        win_end=win_end,
        limit=search_limit,
        ts_code=ts_code_filter,
    )

    raw_major: list[dict[str, Any]] = []
    raw_flash: list[dict[str, Any]] = []
    for h in hits:
        pl = h.payload or {}
        if not isinstance(pl, dict):
            continue
        ch = _hit_to_channel(pl)
        item = _hit_to_raw_item(
            h.id,
            pl,
            content_max=content_major_max if ch == "major_news" else content_flash_max,
        )
        if ch == "major_news":
            if len(raw_major) < cap_major:
                raw_major.append(item)
        else:
            if len(raw_flash) < cap_flash:
                raw_flash.append(item)
        if len(raw_major) >= cap_major and len(raw_flash) >= cap_flash:
            break
    out = raw_major + raw_flash
    cfg = get_config()
    logger.info(
        "Qdrant ④⑤ equity: hits=%s kept_major=%s kept_flash=%s collection=%r",
        len(hits),
        len(raw_major),
        len(raw_flash),
        _collection_name(cfg),
    )
    return out


def retrieve_markdown_lines_equity(
    *,
    ts_code: str,
    stock_name: str,
    industry: str,
    peer_ts_codes: list[str],
    win_start: str,
    win_end: str,
    pool_major: int,
    pool_flash: int,
    max_major_lines: int,
    max_flash_lines: int,
    content_major_max: int,
    content_flash_max: int,
    search_limit: int,
    ts_code_filter: str | None,
    match_fn: Callable[[str, str], bool],
) -> tuple[list[str], list[str]]:
    """Return (major_lines, flash_lines) for non-LLM ④⑤ with substring ``match_fn`` filter."""
    raw = retrieve_raw_items_equity(
        ts_code=ts_code,
        stock_name=stock_name,
        industry=industry,
        peer_ts_codes=peer_ts_codes,
        win_start=win_start,
        win_end=win_end,
        cap_major=pool_major,
        cap_flash=pool_flash,
        content_major_max=content_major_max,
        content_flash_max=content_flash_max,
        search_limit=search_limit,
        ts_code_filter=ts_code_filter,
    )
    major_lines: list[str] = []
    flash_lines: list[str] = []
    for r in raw:
        title = str(r.get("title", ""))
        content = str(r.get("content", ""))
        if not match_fn(title, content):
            continue
        line = _hit_to_markdown(
            {
                "source": r.get("src"),
                "pub_time": r.get("pub_time"),
                "title": title,
                "content": content,
            },
            content_max=content_major_max if r.get("channel") == "major_news" else content_flash_max,
        )
        if r.get("channel") == "major_news":
            if len(major_lines) < max_major_lines:
                major_lines.append(line)
        else:
            if len(flash_lines) < max_flash_lines:
                flash_lines.append(line)
    return major_lines, flash_lines


def retrieve_markdown_loose(
    *,
    query_text: str,
    win_start: str,
    win_end: str,
    cap_major: int,
    cap_flash: int,
    content_major_max: int,
    content_flash_max: int,
    search_limit: int,
    match_fn: Callable[[str, str], bool],
) -> tuple[list[str], list[str]]:
    """No ts_code: semantic query + ``match_fn`` on payload."""
    hits = search_news_qdrant(
        query_text=query_text,
        win_start=win_start,
        win_end=win_end,
        limit=search_limit,
        ts_code=None,
    )
    major_lines: list[str] = []
    flash_lines: list[str] = []
    for h in hits:
        pl = h.payload or {}
        if not isinstance(pl, dict):
            continue
        title = str(pl.get("title", ""))
        content = str(pl.get("content", ""))
        if not match_fn(title, content):
            continue
        ch = _hit_to_channel(pl)
        line = _hit_to_markdown(
            pl,
            content_max=content_major_max if ch == "major_news" else content_flash_max,
        )
        if ch == "major_news":
            if len(major_lines) < cap_major:
                major_lines.append(line)
        else:
            if len(flash_lines) < cap_flash:
                flash_lines.append(line)
        if len(major_lines) >= cap_major and len(flash_lines) >= cap_flash:
            break
    if not major_lines and not flash_lines:
        for h in hits[: max(cap_major, cap_flash)]:
            pl = h.payload or {}
            if not isinstance(pl, dict):
                continue
            ch = _hit_to_channel(pl)
            line = _hit_to_markdown(
                pl,
                content_max=content_major_max if ch == "major_news" else content_flash_max,
            )
            if ch == "major_news":
                if len(major_lines) < max(1, cap_major // 2):
                    major_lines.append(line)
            else:
                if len(flash_lines) < max(1, cap_flash // 2):
                    flash_lines.append(line)
    return major_lines, flash_lines


def retrieve_macro_section_markdown(
    *,
    win_start: str,
    win_end: str,
    search_limit: int,
    per_major: int,
    per_flash: int,
    content_major_max: int,
    content_flash_max: int,
) -> tuple[list[str], list[str]]:
    """⑧ 宏观专题：专用向量 query，不按个股过滤（与 ④⑤ ``retrieve_*_equity`` 分离）。"""
    query_text = macro_vector_search_query_text()
    lim = max(40, min(200, int(search_limit)))
    hits = search_news_qdrant(
        query_text=query_text,
        win_start=win_start,
        win_end=win_end,
        limit=lim,
        ts_code=None,
    )
    major_lines: list[str] = []
    flash_lines: list[str] = []
    for h in hits:
        pl = h.payload or {}
        if not isinstance(pl, dict):
            continue
        ch = _hit_to_channel(pl)
        line = _hit_to_markdown(
            pl,
            content_max=content_major_max if ch == "major_news" else content_flash_max,
        )
        if ch == "major_news":
            if len(major_lines) < per_major:
                major_lines.append(line)
        else:
            if len(flash_lines) < per_flash:
                flash_lines.append(line)
        if len(major_lines) >= per_major and len(flash_lines) >= per_flash:
            break
    cfg = get_config()
    logger.info(
        "Qdrant ⑧ macro section: hits=%s kept_major=%s kept_flash=%s collection=%r",
        len(hits),
        len(major_lines),
        len(flash_lines),
        _collection_name(cfg),
    )
    return major_lines, flash_lines


def retrieve_global_markdown(
    *,
    win_start: str,
    win_end: str,
    per_major: int,
    per_flash: int,
    limit: int,
    match_fn: Callable[[str, str], bool],
    broad_kw: tuple[str, ...],
) -> tuple[list[str], list[str]]:
    """Macro-style global ④⑤ from Qdrant."""
    kw = [k for k in broad_kw if k and len(str(k)) >= 2][:24]
    query_text = "宏观经济、资本市场、货币政策、行业景气与A股相关要闻：" + " ".join(str(x) for x in kw)
    search_lim = max(40, min(200, int(limit) * 3))
    hits = search_news_qdrant(
        query_text=query_text,
        win_start=win_start,
        win_end=win_end,
        limit=search_lim,
        ts_code=None,
    )
    major_lines: list[str] = []
    flash_lines: list[str] = []
    cm = 2200
    cf = 1500
    for h in hits:
        pl = h.payload or {}
        if not isinstance(pl, dict):
            continue
        title = str(pl.get("title", ""))
        content = str(pl.get("content", ""))
        if not match_fn(title, content):
            continue
        ch = _hit_to_channel(pl)
        line = _hit_to_markdown(pl, content_max=cm if ch == "major_news" else cf)
        if ch == "major_news":
            if len(major_lines) < per_major:
                major_lines.append(line)
        else:
            if len(flash_lines) < per_flash:
                flash_lines.append(line)
        if len(major_lines) >= per_major and len(flash_lines) >= per_flash:
            break
    return major_lines, flash_lines
