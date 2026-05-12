"""Qdrant 新闻向量检索：统一 ``vector_search_one`` + ``multi_search_merge``。

- **④⑤**：按「标的 + 竞争对手」每主体一条短 query，多路检索后按 point id 合并（取 max score）。
- **⑧ / 全局**：宏观词表切段多路检索，同一合并逻辑。
- 依赖 ``qdrant/news_embed.py``（DashScope ``text-embedding-v4``）。
"""

from __future__ import annotations

import importlib.util
import logging
import os
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.macro_keywords import macro_vector_search_query_texts

logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _import_embed_query_texts():
    """Load ``qdrant/news_embed.py`` by absolute path."""
    path = _repo_root() / "qdrant" / "news_embed.py"
    if not path.is_file():
        raise RuntimeError(
            "Qdrant news retrieval requires ``qdrant/news_embed.py`` next to the tradingagents package "
            f"(missing {path})."
        )
    spec = importlib.util.spec_from_file_location("tradingagents_qdrant_news_embed", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load embedding module from {path}")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except ImportError as exc:
        raise RuntimeError(
            "Qdrant news retrieval requires embedding deps (see pyproject optional ``qdrant-news`` "
            "and ``qdrant/requirements-ingest.txt``)."
        ) from exc
    fn = getattr(mod, "embed_query_texts", None)
    if not callable(fn):
        raise RuntimeError(f"{path} does not define callable embed_query_texts")
    return fn


def _import_qdrant_client():
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http.models import FieldCondition, Filter, Range
    except ImportError as exc:  # noqa: BLE001
        raise RuntimeError(
            "Install qdrant-client (e.g. pip install tradingagents[qdrant-news] or pip install qdrant-client)."
        ) from exc
    return QdrantClient, FieldCondition, Filter, Range


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


def build_entity_query(ts_code: str, stock_name: str, industry: str) -> str:
    """单主体短 query：代码 + 名称 + 行业（嵌入后与正文同空间）。"""
    ind = (industry or "").strip() or "未知"
    nm = (stock_name or "").strip()
    tc = (ts_code or "").strip()
    return (
        f"A股 {tc} {nm}，所属行业：{ind}。"
        "检索与该上市公司相关的财经新闻、公告要点、政策与舆情。"
    )


def vector_search_one(
    *,
    query_text: str,
    win_start: str,
    win_end: str,
    limit: int,
) -> list[Any]:
    """单次向量检索：仅 ``pub_ts`` 时间窗口，不做 ticker payload 过滤。"""
    cfg = get_config()
    _, FieldCondition, Filter, Range = _import_qdrant_client()
    embed_query_texts = _import_embed_query_texts()
    vec = embed_query_texts([query_text])
    if not vec or not vec[0]:
        raise RuntimeError("embed_query_texts returned empty vector")
    lo, hi = _pub_ts_range(win_start, win_end)
    flt = Filter(
        must=[
            FieldCondition(
                key="pub_ts",
                range=Range(gte=float(lo), lte=float(hi)),
            )
        ]
    )
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


def multi_search_merge(
    hit_groups: list[list[Any]],
    *,
    top_n: int | None,
) -> list[Any]:
    """多路 ``ScoredPoint`` 合并：同一 ``id`` 保留 **最高分**。"""
    best: dict[Any, tuple[float, Any]] = {}
    for group in hit_groups:
        for h in group:
            sid = getattr(h, "id", None)
            sc = float(getattr(h, "score", 0.0) or 0.0)
            prev = best.get(sid)
            if prev is None or sc > prev[0]:
                best[sid] = (sc, h)
    merged = [pair[1] for pair in sorted(best.values(), key=lambda x: -x[0])]
    if top_n is not None and top_n >= 0:
        merged = merged[: int(top_n)]
    return merged


def retrieve_merged_equity_raw_items(
    *,
    entities: list[tuple[str, str, str]],
    win_start: str,
    win_end: str,
    cap_major: int,
    cap_flash: int,
    content_major_max: int,
    content_flash_max: int,
    search_limit: int,
    per_route_limit: int | None = None,
) -> list[dict[str, Any]]:
    """④⑤：对每个主体一次 ``vector_search_one``，合并后拆成 major/flash 原始行（供 LLM 筛选）。"""
    cfg = get_config()
    pr = per_route_limit if per_route_limit is not None else int(cfg.get("news_qdrant_per_route_limit", 40))
    route_lim = max(10, min(200, int(pr)))
    if not entities:
        return []
    groups: list[list[Any]] = []
    for ts_code, name, ind in entities:
        qt = build_entity_query(ts_code, name, ind)
        hits = vector_search_one(
            query_text=qt,
            win_start=win_start,
            win_end=win_end,
            limit=route_lim,
        )
        groups.append(hits)
    merge_cap = max(int(search_limit), cap_major + cap_flash + 24)
    merged_hits = multi_search_merge(groups, top_n=merge_cap)

    raw_major: list[dict[str, Any]] = []
    raw_flash: list[dict[str, Any]] = []
    for h in merged_hits:
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
    logger.info(
        "Qdrant ④⑤ equity merged: routes=%s hits_merged=%s kept_major=%s kept_flash=%s collection=%r",
        len(entities),
        len(merged_hits),
        len(raw_major),
        len(raw_flash),
        _collection_name(cfg),
    )
    return out


def retrieve_merged_equity_markdown_lines(
    *,
    entities: list[tuple[str, str, str]],
    win_start: str,
    win_end: str,
    pool_major: int,
    pool_flash: int,
    max_major_lines: int,
    max_flash_lines: int,
    content_major_max: int,
    content_flash_max: int,
    search_limit: int,
    per_route_limit: int | None,
    match_fn: Callable[[str, str], bool],
) -> tuple[list[str], list[str]]:
    """非 LLM ④⑤：合并检索 + ``match_fn`` 子串过滤。"""
    raw = retrieve_merged_equity_raw_items(
        entities=entities,
        win_start=win_start,
        win_end=win_end,
        cap_major=pool_major,
        cap_flash=pool_flash,
        content_major_max=content_major_max,
        content_flash_max=content_flash_max,
        search_limit=search_limit,
        per_route_limit=per_route_limit,
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
    """无 A 股代码时的单 query 检索 + ``match_fn``。"""
    hits = vector_search_one(
        query_text=query_text,
        win_start=win_start,
        win_end=win_end,
        limit=search_limit,
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
    """⑧ 宏观：多路宏观词 query → merge → 长篇/快讯分行。"""
    cfg = get_config()
    chunk = int(cfg.get("news_macro_vector_terms_per_query", 12))
    queries = macro_vector_search_query_texts(terms_per_chunk=chunk)
    route_lim = max(40, min(200, int(search_limit)))
    groups = [
        vector_search_one(query_text=q, win_start=win_start, win_end=win_end, limit=route_lim)
        for q in queries
    ]
    merge_cap = max(per_major + per_flash + 20, int(search_limit))
    hits = multi_search_merge(groups, top_n=merge_cap)

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
    logger.info(
        "Qdrant ⑧ macro section: queries=%s hits_merged=%s kept_major=%s kept_flash=%s collection=%r",
        len(queries),
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
    """全局 ④⑤：多路宏观词检索 + merge + ``broad_kw`` 命中过滤。"""
    cfg = get_config()
    chunk = int(cfg.get("news_macro_vector_terms_per_query", 12))
    queries = macro_vector_search_query_texts(terms_per_chunk=chunk)
    search_lim = max(40, min(200, int(limit) * 3))
    route_lim = max(25, min(150, search_lim // max(len(queries), 1) + 20))
    groups = [
        vector_search_one(query_text=q, win_start=win_start, win_end=win_end, limit=route_lim)
        for q in queries
    ]
    hits = multi_search_merge(groups, top_n=search_lim)
    major_lines: list[str] = []
    flash_lines: list[str] = []
    cm = 2200
    cf = 1500
    kw_ok = [k for k in broad_kw if k and len(str(k)) >= 2]

    def _match(title: str, content: str) -> bool:
        blob = f"{title} {content}"
        if match_fn(title, content):
            return True
        return any(k in blob for k in kw_ok)

    for h in hits:
        pl = h.payload or {}
        if not isinstance(pl, dict):
            continue
        title = str(pl.get("title", ""))
        content = str(pl.get("content", ""))
        if not _match(title, content):
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
