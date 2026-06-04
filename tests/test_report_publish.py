"""Tests for client-facing report sanitization."""

from tradingagents.agents.utils.report_publish import (
    prepare_report_for_publish,
    redact_vendor_names,
    strip_doc_alignment_section,
)


def test_strip_doc_alignment_section_removes_appendix():
    raw = (
        "## 市场观点\n\n上涨。\n\n"
        "## 字段释义与单位对齐（文档核对）\n\n"
        "| 接口 | 字段 | 文档释义 | 单位/口径 | 本文用法 |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| daily | close | 收盘价 | 元 | 趋势 |\n\n"
        "## 引用来源\n\n| 来源 | 标题 |\n"
    )
    out = strip_doc_alignment_section(raw)
    assert "字段释义与单位对齐" not in out
    assert "## 市场观点" in out
    assert "## 引用来源" in out


def test_redact_vendor_names():
    text = "数据来自 Tushare daily；竞品由 DeepSeek 推理；渠道 Tushare-news"
    out = redact_vendor_names(text)
    assert "tushare" not in out.lower()
    assert "deepseek" not in out.lower()
    assert "数据渠道-news" in out.lower() or "数据渠道-" in out


def test_prepare_report_for_publish_composes_steps():
    raw = "## 字段释义与单位对齐\n\nx\n\n## 结论\n\n使用 Tushare 字段。"
    out = prepare_report_for_publish(raw)
    assert "字段释义与单位对齐" not in out
    assert "tushare" not in out.lower()
