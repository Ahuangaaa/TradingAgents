"""面向业务的 A 股竞争对手列表：DeepSeek 结构化输出 + Tushare ``ts_code`` 校验。

与 ingest 标签使用相同的 OpenAI 兼容 Chat Completions 环境变量（``NEWS_TAG_LLM_*``），
亦可专用 ``PEER_LLM_*`` 覆盖。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.tushare_common import get_pro, resolve_tushare_equity

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValidatedPeer:
    ts_code: str
    name: str
    industry: str


def _chat_url() -> str:
    base = (
        os.getenv("PEER_LLM_BASE_URL") or os.getenv("NEWS_TAG_LLM_BASE_URL") or "https://api.deepseek.com/v1"
    ).rstrip("/")
    return f"{base}/chat/completions"


def _api_key() -> str:
    return (
        os.getenv("PEER_LLM_API_KEY")
        or os.getenv("NEWS_TAG_LLM_API_KEY")
        or os.getenv("DEEPSEEK_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or ""
    ).strip()


def _model() -> str:
    return (os.getenv("PEER_LLM_MODEL") or os.getenv("NEWS_TAG_LLM_MODEL") or "deepseek-v4-flash").strip()


def _parse_json_list(text: str) -> list[dict[str, Any]]:
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


def _normalize_peer_rows(raw: list[dict[str, Any]], focal_ts: str, max_n: int) -> list[ValidatedPeer]:
    """校验 ``ts_code``，并用 ``stock_basic`` 补全名称/行业。"""
    focal = str(focal_ts).strip().upper()
    out: list[ValidatedPeer] = []
    seen: set[str] = set()
    pro = get_pro()
    for row in raw[: max_n + 4]:
        if len(out) >= max_n:
            break
        t_raw = str(row.get("ts_code") or row.get("code") or "").strip().upper()
        ts = resolve_tushare_equity(t_raw) or resolve_tushare_equity(t_raw.replace(".SH", "").replace(".SZ", ""))
        if not ts or ts == focal or ts in seen:
            continue
        try:
            df = pro.stock_basic(ts_code=ts, list_status="L", fields="ts_code,name,industry")
        except Exception as exc:  # noqa: BLE001
            logger.debug("stock_basic skip %s: %s", ts, exc)
            continue
        if df is None or df.empty:
            continue
        r0 = df.iloc[0]
        name = str(r0.get("name") or row.get("name") or "").strip()
        ind = str(r0.get("industry") or row.get("industry") or "").strip()
        if not name:
            continue
        seen.add(ts)
        out.append(ValidatedPeer(ts_code=ts, name=name, industry=ind or "未知"))
    return out


def _cache_file(key: str) -> Path | None:
    cfg = get_config()
    base = Path(cfg.get("data_cache_dir") or ".") / "peers_deepseek"
    try:
        base.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    return base / f"{key}.json"


def _cache_key(focal_ts: str, curr_date: str | None) -> str:
    h = hashlib.sha256()
    h.update(f"{focal_ts}|{curr_date or ''}".encode("utf-8"))
    return h.hexdigest()[:48]


def fetch_validated_peers(
    focal_ts_code: str,
    focal_name: str,
    focal_industry: str,
    *,
    max_peers: int = 5,
    curr_date: str | None = None,
    use_cache: bool = True,
) -> list[ValidatedPeer]:
    """调用 DeepSeek 列出与标的最相关的已上市 A 股竞争对手，返回最多 ``max_peers`` 条（已校验 ``ts_code``）。"""
    focal_ts = str(focal_ts_code).strip().upper()
    cap = max(1, min(int(max_peers) if max_peers else 5, 12))

    if use_cache:
        ck = _cache_key(focal_ts, curr_date)
        cpath = _cache_file(ck)
        if cpath is not None and cpath.is_file():
            try:
                data = json.loads(cpath.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    peers = [
                        ValidatedPeer(
                            ts_code=str(x["ts_code"]),
                            name=str(x["name"]),
                            industry=str(x.get("industry") or ""),
                        )
                        for x in data
                        if isinstance(x, dict) and x.get("ts_code")
                    ]
                    if peers:
                        logger.info("Peers DeepSeek: cache hit %s (%s rows)", cpath.name, len(peers))
                        return peers[:cap]
            except (json.JSONDecodeError, OSError, KeyError) as exc:
                logger.debug("peer cache read failed: %s", exc)

    api_key = _api_key()
    if not api_key:
        logger.warning("Peers DeepSeek: no API key (PEER_LLM_API_KEY / NEWS_TAG_LLM_API_KEY / DEEPSEEK_API_KEY)")
        return []

    user = (
        f"标的：{focal_ts} {focal_name or ''}，Tushare 行业：{focal_industry or '未知'}。\n"
        f"请列出在中国大陆 A 股市场上市、与该公司主营业务**直接竞争**的最重要对手，最多 {cap} 家。\n"
        "只输出一个 JSON 数组，元素字段："
        '`"ts_code"`（形如 600519.SH）、`"name"`（中文简称）、`"industry"`（Tushare 式行业名或接近表述）。\n'
        "不要包含标的自身；不要港股美股；不确定的代码不要编造。"
    )

    payload = {
        "model": _model(),
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": "你是 A 股产业与公司研究助手，只输出 JSON 数组。"},
            {"role": "user", "content": user},
        ],
    }
    text = ""
    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(
                _chat_url(),
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
            )
            resp.raise_for_status()
            body = resp.json()
            text = (
                (body.get("choices") or [{}])[0]
                .get("message", {})
                .get("content", "")
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Peers DeepSeek: HTTP failed: %s", exc)
        return []

    rows = _parse_json_list(text)
    peers = _normalize_peer_rows(rows, focal_ts, cap)
    if use_cache and peers:
        cpath = _cache_file(_cache_key(focal_ts, curr_date))
        if cpath is not None:
            try:
                cpath.write_text(
                    json.dumps([asdict(p) for p in peers], ensure_ascii=False),
                    encoding="utf-8",
                )
            except OSError as exc:
                logger.debug("peer cache write failed: %s", exc)
    logger.info("Peers DeepSeek: validated %s peers for focal %s", len(peers), focal_ts)
    return peers
