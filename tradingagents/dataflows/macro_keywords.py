"""宏观 / 市场类关键词：子串匹配、入库兜底、Tushare 全局过滤与 ⑧ 向量 query **共用同一词表**。"""

from __future__ import annotations


def _macro_lexicon_merged() -> tuple[str, ...]:
    """单一权威词表：先保留原「市场/板块」短词，再并入原 ⑧ 专用检索里的补充短语（去重保序）。"""
    base: tuple[str, ...] = (
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
    )
    extra: tuple[str, ...] = (
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
    seen: set[str] = set()
    out: list[str] = []
    for group in (base, extra):
        for w in group:
            s = str(w).strip()
            if not s or s in seen:
                continue
            seen.add(s)
            out.append(s)
    return tuple(out)


def macro_market_keywords() -> tuple[str, ...]:
    """用于新闻「无个股命中」时的宏观/市场类补充匹配或降级填充（与 ``macro_vector_search_query_text`` 同源）。"""
    return _macro_lexicon_merged()


def macro_vector_search_query_text() -> str:
    """⑧ 宏观专题（单条长 query，兼容旧逻辑）：与 ``macro_market_keywords`` 同源。"""
    terms = macro_market_keywords()
    return "宏观经济与资本市场环境：" + " ".join(terms)


def macro_vector_search_query_texts(*, terms_per_chunk: int = 12) -> list[str]:
    """多路宏观向量检索：将 ``macro_market_keywords`` 切段，避免单条 query 过长、主题稀释。

    ``terms_per_chunk``：每段最多多少个关键词（实现时可由配置覆盖）。
    """
    terms = [str(t).strip() for t in macro_market_keywords() if str(t).strip()]
    if not terms:
        return [macro_vector_search_query_text()]
    n = max(4, min(40, int(terms_per_chunk)))
    out: list[str] = []
    for i in range(0, len(terms), n):
        chunk = terms[i : i + n]
        out.append("宏观经济与资本市场环境：" + " ".join(chunk))
    return out if len(out) >= 1 else [macro_vector_search_query_text()]
