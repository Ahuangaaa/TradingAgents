"""面向业务的 A 股竞争对手：专用提示词调用 DeepSeek Chat Completions，再用 Tushare ``stock_basic`` 校码。

端点/模型由 ``PEER_LLM_*`` 或 ``NEWS_TAG_LLM_*`` / ``DEEPSEEK_API_KEY`` 等环境变量配置。
Tushare 仅校验代码与补全简称/披露行业字段，**不**从「同行业股票池」机械选股。
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
    h.update(f"{focal_ts}|{curr_date or ''}|peer_prompt_v2".encode("utf-8"))
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
    """调用 DeepSeek（专用 system/user 提示词，Chat Completions）列出与标的最相关的已上市 A 股竞争对手，再经 ``stock_basic`` 校码，返回最多 ``max_peers`` 条。"""
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
        f"【标的】{focal_ts}  {focal_name or ''}\n"
        f"【监管披露行业分类（仅供参考，不是选股池）】{focal_industry or '未知'}\n\n"
        f"请基于**产品/客户/渠道/商业模式重叠**，列出在中国大陆 A 股上市、与上述标的**直接竞争**的最重要公司，"
        f"至多 {cap} 家。名单必须由你在本对话中推理得出，"
        "**不要**写或暗示「从 Tushare 同行业列表选取」「按行业分类筛选」等表述。\n\n"
        "只输出一个 JSON 数组。每个元素为对象，字段：\n"
        '- `"ts_code"`：形如 600519.SH / 000001.SZ / 8xxxxx.BJ 的 A 股代码；\n'
        '- `"name"`：该公司中文简称；\n'
        '- `"industry"`：你对其主营的简短概括（可与披露行业不同）。\n\n'
        "排除标的自身；不要港股/美股；无法确认的代码不要输出。"
    )

    system = (
        "你是资深 A 股行业与公司研究助手。你的唯一任务：根据用户给出的标的信息，"
        "独立推理**主营业务直接竞争**关系，列出最重要的已上市 A 股竞争对手。\n"
        "硬性规则：\n"
        "1) 禁止按「与标的相同的 Tushare 行业名」或任何「同行业股票池 / 行业成分股」做机械筛选或排序；"
        "行业分类若出现在用户消息中，仅为监管披露标签，**不得**当作选股池。\n"
        "2) 禁止编造 ts_code；不确定则不要输出该条。\n"
        "3) 只输出一个 JSON 数组，无其它说明文字、无 markdown 围栏。"
    )

    payload = {
        "model": _model(),
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": system},
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
