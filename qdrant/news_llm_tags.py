"""Batch-extract tickers / industry / concept tags via OpenAI-compatible Chat API (DeepSeek)."""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

logger = logging.getLogger(__name__)


def _ensure_repo_root_on_path() -> None:
    """So ``from tradingagents...`` works when running ``python qdrant/ingest_news.py`` from repo root."""
    root = Path(__file__).resolve().parents[1]
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)


def _macro_market_keywords_tuple() -> tuple[str, ...]:
    try:
        from tradingagents.dataflows.macro_keywords import macro_market_keywords
    except ImportError:
        _ensure_repo_root_on_path()
        from tradingagents.dataflows.macro_keywords import macro_market_keywords
    return macro_market_keywords()


def _macro_keyword_fallback_row(row: dict[str, Any], *, body_chars: int) -> bool:
    """If tickers / industry / concept are all empty, fill ``concept_tags`` from title+body substring hits.

    Returns True if fallback was applied.
    """
    t = list(row.get("tickers") or [])
    i = list(row.get("industry_tags") or [])
    c = list(row.get("concept_tags") or [])
    if t or i or c:
        return False
    title = str(row.get("title", "") or "")
    body = str(row.get("content", "") or "")[: max(0, int(body_chars))]
    blob = f"{title}\n{body}"
    seen: set[str] = set()
    concepts: list[str] = []
    for kw in _macro_market_keywords_tuple():
        if not kw or kw in seen:
            continue
        if kw in blob:
            concepts.append(kw)
            seen.add(kw)
        if len(concepts) >= 12:
            break
    if concepts:
        row["concept_tags"] = concepts
        return True
    return False

SYSTEM = """你是 A 股金融新闻结构化抽取助手。
对每条新闻输出 JSON 对象，字段：
- "id": 与输入完全一致（字符串）
- "tickers": A 股证券代码列表，元素必须是 ``000001.SZ`` / ``600519.SH`` / ``920001.BJ`` 形式；无法确定则 []
- "industry_tags": 行业名或行业关键词短语（中文），最多 8 条；无则 []
- "concept_tags": 概念板块/主题关键词（中文），最多 12 条；无则 []

只输出 **一个 JSON 数组**，不要 markdown 围栏，不要解释。"""


def _chat_url() -> str:
    base = (os.getenv("NEWS_TAG_LLM_BASE_URL") or "https://api.deepseek.com/v1").rstrip("/")
    return f"{base}/chat/completions"


def _api_key() -> str:
    return (
        os.getenv("NEWS_TAG_LLM_API_KEY")
        or os.getenv("DEEPSEEK_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or ""
    ).strip()


def _model() -> str:
    return (os.getenv("NEWS_TAG_LLM_MODEL") or "deepseek-chat").strip()


def _parse_json_array(text: str) -> list[dict[str, Any]]:
    raw = (text or "").strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw, re.IGNORECASE)
    if m:
        raw = m.group(1).strip()
    i = raw.find("[")
    if i >= 0:
        j = raw.rfind("]")
        if j > i:
            try:
                data = json.loads(raw[i : j + 1])
                if isinstance(data, list):
                    return [x for x in data if isinstance(x, dict)]
            except json.JSONDecodeError:
                pass
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
    except json.JSONDecodeError:
        pass
    return []


def _normalize_tags(row: dict[str, Any]) -> dict[str, list[str]]:
    def _list_str(v: Any, cap: int) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            v = [v]
        if not isinstance(v, list):
            return []
        out = [str(x).strip() for x in v if str(x).strip()]
        return out[:cap]

    return {
        "tickers": _list_str(row.get("tickers"), 32),
        "industry_tags": _list_str(row.get("industry_tags"), 8),
        "concept_tags": _list_str(row.get("concept_tags"), 12),
    }


def tag_news_dataframe(
    df: pd.DataFrame,
    *,
    rows_per_llm_call: int = 15,
    content_chars: int = 800,
) -> pd.DataFrame:
    """Add columns tickers, industry_tags, concept_tags (list[str]) using batched LLM."""
    if df.empty:
        return df

    key = _api_key()
    if not key:
        logger.warning("No NEWS_TAG_LLM_API_KEY / DEEPSEEK_API_KEY / OPENAI_API_KEY — tags will be empty.")
        df = df.copy()
        df["tickers"] = [[] for _ in range(len(df))]
        df["industry_tags"] = [[] for _ in range(len(df))]
        df["concept_tags"] = [[] for _ in range(len(df))]
        recs = df.to_dict("records")
        blob_chars = min(8000, max(int(content_chars), 2000))
        macro_fb = 0
        for r in recs:
            r.setdefault("tickers", [])
            r.setdefault("industry_tags", [])
            r.setdefault("concept_tags", [])
            if _macro_keyword_fallback_row(r, body_chars=blob_chars):
                macro_fb += 1
        if macro_fb:
            logger.info(
                "LLM tags: macro fallback on %s rows (no LLM key; concept_tags from title/body keywords)",
                macro_fb,
            )
        return pd.DataFrame(recs)

    n = len(df)
    n_batches = (n + rows_per_llm_call - 1) // rows_per_llm_call
    logger.info(
        "LLM tags: model=%r endpoint=%s rows=%s batch_size=%s (~%s HTTP calls)",
        _model(),
        _chat_url(),
        n,
        rows_per_llm_call,
        n_batches,
    )

    out_rows: list[dict[str, Any]] = df.to_dict("records")
    for r in out_rows:
        r.setdefault("tickers", [])
        r.setdefault("industry_tags", [])
        r.setdefault("concept_tags", [])

    id_to_idx = {str(r["stable_id"]): i for i, r in enumerate(out_rows)}
    batch_idx = 0

    for start in range(0, len(out_rows), rows_per_llm_call):
        batch_idx += 1
        chunk = out_rows[start : start + rows_per_llm_call]
        end = start + len(chunk) - 1
        lines = []
        for r in chunk:
            rid = str(r["stable_id"])
            title = str(r.get("title", ""))[:400]
            body = str(r.get("content", ""))[:content_chars].replace("\n", " ")
            lines.append(json.dumps({"id": rid, "title": title, "snippet": body}, ensure_ascii=False))

        user = "输入（JSON Lines，每行一条）：\n" + "\n".join(lines)

        logger.info(
            "LLM tags: batch %s/%s rows [%s..%s] (%s items) -> POST chat/completions",
            batch_idx,
            n_batches,
            start,
            end,
            len(chunk),
        )
        try:
            with httpx.Client(timeout=120.0) as client:
                resp = client.post(
                    _chat_url(),
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json={
                        "model": _model(),
                        "messages": [
                            {"role": "system", "content": SYSTEM},
                            {"role": "user", "content": user},
                        ],
                        "temperature": 0.1,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
            content = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
            parsed = _parse_json_array(str(content))
            matched = 0
            for obj in parsed:
                rid = str(obj.get("id", "")).strip()
                if rid not in id_to_idx:
                    continue
                matched += 1
                idx = id_to_idx[rid]
                tags = _normalize_tags(obj)
                out_rows[idx]["tickers"] = tags["tickers"]
                out_rows[idx]["industry_tags"] = tags["industry_tags"]
                out_rows[idx]["concept_tags"] = tags["concept_tags"]
            logger.info(
                "LLM tags: batch %s/%s done — parsed_objects=%s matched_stable_ids=%s",
                batch_idx,
                n_batches,
                len(parsed),
                matched,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM tag batch %s-%s failed: %s", start, start + len(chunk), exc)

    # fill missing
    for r in out_rows:
        r.setdefault("tickers", [])
        r.setdefault("industry_tags", [])
        r.setdefault("concept_tags", [])

    blob_chars = min(8000, max(int(content_chars), 2000))
    macro_fb = 0
    for r in out_rows:
        if _macro_keyword_fallback_row(r, body_chars=blob_chars):
            macro_fb += 1
    if macro_fb:
        logger.info(
            "LLM tags: macro/market keyword fallback filled concept_tags on %s rows (tickers+industry+concept were empty)",
            macro_fb,
        )

    logger.info("LLM tags: finished all batches (%s rows)", len(out_rows))
    return pd.DataFrame(out_rows)
