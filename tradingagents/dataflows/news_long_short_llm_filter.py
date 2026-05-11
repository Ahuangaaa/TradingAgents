"""Semantic LLM screening for Tushare ④ major_news / ⑤ flash ``news`` (replaces substring match).

When ``news_llm_filter_long_short`` is True (default), ``get_tushare_news`` collects raw rows
then batches them through the configured **quick** LLM. Disabled via config or
``TRADINGAGENTS_NEWS_LLM_FILTER=0``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from tradingagents.dataflows.config import get_config


def _cache_path(cache_key: str) -> Path:
    base = Path(get_config().get("data_cache_dir", ".")) / "news_llm_long_short"
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{cache_key}.md"


def _make_cache_key(ts_code: str, win_start: str, win_end: str, model: str, raw_digest: str) -> str:
    h = hashlib.sha256()
    h.update(f"{ts_code}|{win_start}|{win_end}|{model}|{raw_digest}".encode("utf-8"))
    return h.hexdigest()[:48]


def _parse_llm_json_list(text: str) -> list[dict[str, Any]]:
    raw = (text or "").strip()
    if not raw:
        return []
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw, re.IGNORECASE)
    if m:
        raw = m.group(1).strip()
    bracket = raw.find("[")
    if bracket != -1:
        tail = raw.rfind("]")
        if tail > bracket:
            try:
                data = json.loads(raw[bracket : tail + 1])
                if isinstance(data, list):
                    return [x for x in data if isinstance(x, dict)]
            except json.JSONDecodeError:
                pass
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        if isinstance(data, dict) and isinstance(data.get("items"), list):
            return [x for x in data["items"] if isinstance(x, dict)]
    except json.JSONDecodeError:
        pass
    return []


def _normalize_llm_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(p for p in parts if p)
    return str(content or "")


def _get_quick_llm():
    from tradingagents.llm_clients import create_llm_client

    cfg = get_config()
    provider = (cfg.get("llm_provider") or "openai").lower()
    model = cfg.get("quick_think_llm") or "gpt-4o-mini"
    base_url = cfg.get("backend_url")
    kwargs: dict[str, Any] = {}
    if provider == "deepseek" and cfg.get("deepseek_quick_thinking_enabled", False):
        kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
        kwargs["reasoning_effort"] = cfg.get("deepseek_quick_reasoning_effort") or "max"
    client = create_llm_client(
        provider=provider,
        model=model,
        base_url=base_url,
        **kwargs,
    )
    return client.get_llm(), model


def _batch_items(items: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def _format_kept_block(kept_rows: list[dict[str, Any]], by_id: dict[str, dict[str, Any]]) -> str:
    if not kept_rows:
        return ""
    lines: list[str] = []
    for row in kept_rows:
        src_row = by_id[row["id"]]
        lines.append(
            f"### [{src_row.get('channel')}|{src_row.get('src')}] {src_row.get('pub_time')}\n"
            f"**相关性**: {row.get('relevance', '')}  \n"
            f"**标题**: {src_row.get('title', '')}\n\n"
            f"**摘录**: {row.get('key_excerpt', '')}\n\n"
            f"**保留理由**: {row.get('reason', '')}\n\n"
            f"**对标的的影响**: {row.get('impact_on_focal', '')}\n"
        )
    return "\n".join(lines)


def _format_raw_fallback(raw_items: list[dict[str, Any]], *, limit: int) -> str:
    lines = [f"_未筛选原始抽样（最多 {limit} 条）：_\n"]
    for r in raw_items[:limit]:
        lines.append(
            f"### [{r.get('channel')}|{r.get('src')}] {r.get('pub_time')}\n{r.get('title')}\n"
        )
    return "\n".join(lines)


def screen_long_short_news_with_llm(
    *,
    raw_items: list[dict[str, Any]],
    focal_ticker: str,
    ts_code: str,
    stock_name: str,
    industry: str,
    peer_ts_codes: list[str],
    win_start: str,
    win_end: str,
) -> tuple[str, str]:
    """Return (markdown for ④ major_news block body, markdown for ⑤ flash block body)."""
    cfg = get_config()
    header = (
        "> **④⑤ 语料说明**：已取消「标题/正文 对 **stock_basic 行业名** 的子串匹配」。"
        "默认由 **LLM 语义筛选** 保留与标的、行业、peer 相关条目；未列出者视为已丢弃。\n\n"
    )

    if not raw_items:
        empty = "_（本窗口 major_news / news 无原始返回。）_"
        return empty, empty

    digest_src = "|".join(f"{r.get('id')}:{str(r.get('title', ''))[:40]}" for r in raw_items[:220])
    try:
        llm, model = _get_quick_llm()
    except Exception as exc:
        fb = (
            f"_LLM 筛选不可用（{exc!s}）。请检查 API Key 与 llm_provider。_\n\n"
            + _format_raw_fallback(raw_items, limit=12)
        )
        return header + fb, header + fb

    cache_key = _make_cache_key(
        ts_code, win_start, win_end, model, hashlib.md5(digest_src.encode()).hexdigest()
    )
    cache_file = _cache_path(cache_key)
    if cfg.get("news_llm_filter_use_cache", True) and cache_file.is_file():
        combined = cache_file.read_text(encoding="utf-8")
        if "===SEC5===" in combined:
            a, b = combined.split("===SEC5===", 1)
            return a.strip(), b.strip()
        return combined, combined

    batch_size = int(cfg.get("news_llm_filter_batch_size", 28))
    max_kept_total = int(cfg.get("news_llm_filter_max_kept", 45))
    peer_str = ", ".join(peer_ts_codes) if peer_ts_codes else "（无自动拉取的同业代码，仅凭语义判断）"

    system = """你是金融新闻筛选助手。每条来自 Tushare 长篇通讯(major_news)或短讯(news)，已按时间窗拉取，未做关键词过滤。
判断该条对「标的」投资分析是否有用；无用则不要出现在输出中。

保留：直接涉及标的或核心子公司；行业政策/景气/技术/供需等可能影响标的（未点名也可）；peer 列表中竞争对手的重要动作；宏观、监管、产业链等可推理影响标的的新闻。
丢弃：与标的行业及 peer 无明显关系的花边、纯无关个股八卦、无增量重复。

仅输出 JSON 数组（不要用 markdown 代码围栏）。每个元素：
{"id": "<与输入一致>", "relevance": "high"|"medium", "reason": "20-80字中文", "key_excerpt": "正文摘录≤200字", "impact_on_focal": "对标的潜在影响30-120字"}
若无有用条目输出 []。"""

    focal_line = (
        f"标的入参: {focal_ticker} | ts_code: {ts_code} | 简称: {stock_name or '未知'}"
        f" | 行业(背景，勿做字面匹配): {industry or '未知'}"
    )
    peers_line = f"同行业可比 ts_code: {peer_str}"

    by_id = {str(r["id"]): r for r in raw_items}
    all_kept: list[dict[str, Any]] = []
    rank = {"high": 2, "medium": 1}

    for bi, batch in enumerate(_batch_items(raw_items, batch_size)):
        jl = [
            json.dumps(
                {
                    "id": r["id"],
                    "channel": r.get("channel"),
                    "src": r.get("src"),
                    "pub_time": r.get("pub_time"),
                    "title": (r.get("title") or "")[:500],
                    "content": (r.get("content") or "")[:1500],
                },
                ensure_ascii=False,
            )
            for r in batch
        ]
        user = (
            f"{focal_line}\n{peers_line}\n时间窗: {win_start} ~ {win_end}\n批次 {bi + 1}\n\n"
            "JSON Lines（每行一条输入）：\n" + "\n".join(jl)
        )
        try:
            resp = llm.invoke([("system", system), ("human", user)])
            content = _normalize_llm_content(getattr(resp, "content", None))
            for obj in _parse_llm_json_list(content):
                rid = str(obj.get("id", "")).strip()
                if rid not in by_id:
                    continue
                rel = str(obj.get("relevance", "")).lower().strip()
                if rel not in ("high", "medium"):
                    continue
                all_kept.append(
                    {
                        "id": rid,
                        "relevance": rel,
                        "reason": str(obj.get("reason", "")).strip(),
                        "key_excerpt": str(obj.get("key_excerpt", "")).strip(),
                        "impact_on_focal": str(obj.get("impact_on_focal", "")).strip(),
                    }
                )
        except Exception:
            continue

    merged: dict[str, dict[str, Any]] = {}
    for row in all_kept:
        rid = row["id"]
        prev = merged.get(rid)
        if not prev or rank.get(row["relevance"], 0) > rank.get(prev.get("relevance", ""), 0):
            merged[rid] = row

    kept_list = sorted(merged.values(), key=lambda x: rank.get(x.get("relevance", ""), 0), reverse=True)[
        :max_kept_total
    ]

    if not kept_list:
        fb = "_本窗口经 LLM 筛选后无 high/medium 条目。_\n\n" + _format_raw_fallback(raw_items, limit=8)
        body4 = header + fb
        body5 = header + fb
        if cfg.get("news_llm_filter_use_cache", True):
            try:
                cache_file.write_text(body4 + "\n===SEC5===\n" + body5, encoding="utf-8")
            except OSError:
                pass
        return body4, body5

    major_kept = [r for r in kept_list if by_id[r["id"]].get("channel") == "major_news"]
    flash_kept = [r for r in kept_list if by_id[r["id"]].get("channel") == "news"]

    body4 = header + (_format_kept_block(major_kept, by_id) or "_（长篇通讯：筛后无条目。）_")
    body5 = header + (_format_kept_block(flash_kept, by_id) or "_（短讯：筛后无条目。）_")

    if cfg.get("news_llm_filter_use_cache", True):
        try:
            cache_file.write_text(body4 + "\n===SEC5===\n" + body5, encoding="utf-8")
        except OSError:
            pass
    return body4, body5


def news_llm_filter_disabled() -> bool:
    env = os.getenv("TRADINGAGENTS_NEWS_LLM_FILTER", "").strip().lower()
    if env in ("0", "false", "no"):
        return True
    return not bool(get_config().get("news_llm_filter_long_short", True))
