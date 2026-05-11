"""⑧ 宏观专题：对 ``retrieve_macro_section_markdown`` 命中的语料做 quick LLM 摘要（与 ④⑤ 无关）。"""

from __future__ import annotations

import logging
import os

from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.news_long_short_llm_filter import (
    _get_quick_llm,
    _normalize_llm_content,
)

logger = logging.getLogger(__name__)


def macro_section8_llm_refine_disabled() -> bool:
    env = os.getenv("NEWS_MACRO_SECTION8_LLM_REFINE", "").strip().lower()
    if env in ("0", "false", "no", "off"):
        return True
    return False


def refine_macro_section8_corpus(raw_markdown: str, *, win_start: str, win_end: str) -> str:
    """将 ⑧ 专用向量检索得到的 markdown 浓缩为中文宏观简报。"""
    raw = (raw_markdown or "").strip()
    if not raw:
        return ""

    cfg = get_config()
    cap = int(cfg.get("news_macro_section8_llm_max_chars", 24000))
    if len(raw) > cap:
        raw = raw[:cap] + "\n\n…（语料已截断）"

    system = """你是宏观编辑。输入为时间窗内、用「宏观专用」向量 query 从新闻库检索到的条目摘录（与个股舆情检索无关）。
请输出 **中文** 简报，面向股票研究员，建议结构：
- **国内政策与监管**
- **货币/信用/汇率**
- **海外与跨境传导**
- **大类资产与风险偏好**
- **信息缺口**（一两句）

只依据输入归纳，不编造未出现的数字或主体名；总长度约 1000 字内；不要用 markdown 代码围栏。"""

    user = f"时间窗：{win_start} — {win_end}\n\n--- ⑧ 语料 ---\n{raw}"

    try:
        llm, model = _get_quick_llm()
    except Exception as exc:
        logger.warning("⑧ 宏观精炼：无法创建 quick LLM（%s）", exc)
        return ""

    try:
        resp = llm.invoke([("system", system), ("human", user)])
        text = _normalize_llm_content(getattr(resp, "content", None)).strip()
        if not text:
            return ""
        logger.info("⑧ 宏观精炼：model=%r len=%s", model, len(text))
        return text
    except Exception as exc:
        logger.warning("⑧ 宏观精炼：invoke 失败：%s", exc)
        return ""
