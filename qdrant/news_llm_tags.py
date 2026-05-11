"""Batch-extract tickers / industry / concept tags via OpenAI-compatible Chat API (DeepSeek)."""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

logger = logging.getLogger(__name__)

# If ``macro_keywords.py`` is missing or unloadable, substring fallback still works (keep roughly in sync with repo file).
_MACRO_MARKET_KEYWORDS_FALLBACK: tuple[str, ...] = (
    "A股",
    "沪深",
    "上证",
    "深证",
    "创业板",
    "科创",
    "北交所",
    "央行",
    "美联储",
    "GDP",
    "CPI",
    "PPI",
    "降息",
    "加息",
    "降准",
    "LPR",
    "MLF",
    "地缘",
    "原油",
    "黄金",
    "美元",
    "人民币",
    "汇率",
    "港股",
    "美股",
    "纳指",
    "标普",
    "国务院",
    "发改委",
    "财政部",
    "证监会",
    "金融监管",
    "产业政策",
    "房地产",
    "汽车",
    "新能源",
    "半导体",
    "人工智能",
    "消费",
    "中国宏观",
    "货币政策",
    "财政政策",
    "利率",
    "流动性",
    "OMO",
    "公开市场操作",
    "欧央行",
    "日本央行",
    "美债收益率",
    "美元指数",
    "非农就业",
    "制造业PMI",
    "社融",
    "信贷",
    "房地产政策",
    "化债",
    "地方政府债务",
    "专项债",
    "资本市场改革",
    "对外开放",
    "地缘冲突",
    "能源安全",
    "大宗商品",
    "铜",
    "风险偏好",
    "全球权益",
    "外资流向",
    "北向资金",
)


def _repo_root_for_macro_keywords() -> Path:
    """Parent of ``qdrant/`` (repository root when ``news_llm_tags.py`` lives under ``qdrant/``)."""
    return Path(__file__).resolve().parent.parent


def _load_macro_market_keywords_from_file() -> tuple[str, ...] | None:
    """Load ``macro_market_keywords()`` without ``import tradingagents`` (works in Cursor worktrees / bare scripts)."""
    mp = _repo_root_for_macro_keywords() / "tradingagents" / "dataflows" / "macro_keywords.py"
    if not mp.is_file():
        return None
    try:
        spec = importlib.util.spec_from_file_location("_qdrant_ingest_macro_keywords", mp)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        fn = getattr(mod, "macro_market_keywords", None)
        if not callable(fn):
            return None
        out = fn()
        if isinstance(out, (list, tuple)):
            return tuple(str(x).strip() for x in out if str(x).strip())
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load macro_keywords from %s: %s", mp, exc)
        return None


def _ensure_repo_root_on_path() -> None:
    """Ensure repo root is on ``sys.path`` (parent of ``qdrant/``); use append so ``qdrant/`` stays first."""
    root = Path(__file__).resolve().parent.parent
    s = str(root)
    if s not in sys.path:
        sys.path.append(s)


def _macro_market_keywords_tuple() -> tuple[str, ...]:
    """Prefer loading ``macro_keywords.py`` by path (no ``tradingagents`` package on ``sys.path`` required)."""
    loaded = _load_macro_market_keywords_from_file()
    if loaded:
        return loaded
    try:
        _ensure_repo_root_on_path()
        from tradingagents.dataflows.macro_keywords import macro_market_keywords

        return tuple(macro_market_keywords())
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "macro_market_keywords import fallback after file load failed: %s — using built-in tuple",
            exc,
        )
    return _MACRO_MARKET_KEYWORDS_FALLBACK


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
  若正文明显为宏观政策、利率汇率与流动性、海外主要央行、大类资产与市场整体风险偏好评述，
  请把对应**简短标签**写入 ``concept_tags``（例如：货币政策、降准、美联储、北向资金、大宗商品），
  便于后续向量检索与过滤；纯个股经营类新闻则仍以行业/题材为主。

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


def _tag_concurrency_default() -> int:
    raw = (os.getenv("NEWS_TAG_LLM_CONCURRENCY") or "4").strip()
    try:
        n = int(raw)
    except ValueError:
        n = 4
    return max(1, min(32, n))


def _llm_tag_one_batch(
    start: int,
    chunk: list[dict[str, Any]],
    *,
    content_chars: int,
    api_key: str,
    batch_idx: int,
    n_batches: int,
) -> tuple[int, dict[str, dict[str, list[str]]], int, int]:
    """HTTP call for one batch; returns (start, stable_id -> tags, len(parsed), matched_count)."""
    lines = []
    for r in chunk:
        rid = str(r["stable_id"])
        title = str(r.get("title", ""))[:400]
        body = str(r.get("content", ""))[:content_chars].replace("\n", " ")
        lines.append(json.dumps({"id": rid, "title": title, "snippet": body}, ensure_ascii=False))
    user = "输入（JSON Lines，每行一条）：\n" + "\n".join(lines)
    end = start + len(chunk) - 1
    logger.info(
        "LLM tags: batch %s/%s rows [%s..%s] (%s items) -> POST chat/completions (parallel worker)",
        batch_idx,
        n_batches,
        start,
        end,
        len(chunk),
    )
    id_to_tags: dict[str, dict[str, list[str]]] = {}
    parsed_n = 0
    matched = 0
    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(
                _chat_url(),
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
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
        parsed_n = len(parsed)
        for obj in parsed:
            rid = str(obj.get("id", "")).strip()
            if not rid:
                continue
            tags = _normalize_tags(obj)
            id_to_tags[rid] = tags
            matched += 1
        logger.info(
            "LLM tags: batch %s/%s done — parsed_objects=%s matched_stable_ids=%s",
            batch_idx,
            n_batches,
            parsed_n,
            matched,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM tag batch [%s..%s] failed: %s", start, end, exc)
    return start, id_to_tags, parsed_n, matched


def tag_news_dataframe(
    df: pd.DataFrame,
    *,
    rows_per_llm_call: int = 15,
    content_chars: int = 800,
    tag_concurrency: int | None = None,
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
    workers = tag_concurrency if tag_concurrency is not None else _tag_concurrency_default()
    workers = max(1, min(32, int(workers)))

    batch_specs: list[tuple[int, int, list[dict[str, Any]]]] = []
    bidx = 0
    for start in range(0, len(out_rows), rows_per_llm_call):
        bidx += 1
        chunk = out_rows[start : start + rows_per_llm_call]
        batch_specs.append((bidx, start, chunk))

    if workers <= 1:
        for batch_idx, start, chunk in batch_specs:
            _, id_to_tags, _, _ = _llm_tag_one_batch(
                start,
                chunk,
                content_chars=content_chars,
                api_key=key,
                batch_idx=batch_idx,
                n_batches=n_batches,
            )
            for rid, tags in id_to_tags.items():
                if rid not in id_to_idx:
                    continue
                idx = id_to_idx[rid]
                out_rows[idx]["tickers"] = tags["tickers"]
                out_rows[idx]["industry_tags"] = tags["industry_tags"]
                out_rows[idx]["concept_tags"] = tags["concept_tags"]
    else:
        logger.info("LLM tags: parallel HTTP — workers=%s (env NEWS_TAG_LLM_CONCURRENCY or tag_concurrency)", workers)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_map = {
                pool.submit(
                    _llm_tag_one_batch,
                    start,
                    chunk,
                    content_chars=content_chars,
                    api_key=key,
                    batch_idx=batch_idx,
                    n_batches=n_batches,
                ): (batch_idx, start)
                for batch_idx, start, chunk in batch_specs
            }
            for fut in as_completed(future_map):
                try:
                    _, id_to_tags, _, _ = fut.result()
                except Exception as exc:  # noqa: BLE001
                    batch_idx, start = future_map[fut]
                    logger.warning("LLM tag batch future failed batch_idx=%s start=%s: %s", batch_idx, start, exc)
                    continue
                for rid, tags in id_to_tags.items():
                    if rid not in id_to_idx:
                        continue
                    idx = id_to_idx[rid]
                    out_rows[idx]["tickers"] = tags["tickers"]
                    out_rows[idx]["industry_tags"] = tags["industry_tags"]
                    out_rows[idx]["concept_tags"] = tags["concept_tags"]

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
