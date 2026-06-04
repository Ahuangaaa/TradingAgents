"""Sanitize analyst and team reports before display or export."""

from __future__ import annotations

import re

# Section headings the model was previously asked to append; strip from deliverables.
_DOC_ALIGNMENT_HEADING = re.compile(
    r"(?im)^#{1,3}\s*\*{0,2}\s*字段释义与单位对齐[^#\n]*\*{0,2}\s*$"
    r".*?(?=^#{1,3}\s+|\Z)",
    re.MULTILINE | re.DOTALL,
)

_VENDOR_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"https?://(?:www\.)?tushare\.pro[^\s\)>]*", re.I), "官方数据文档"),
    (re.compile(r"\btushare\.pro\b", re.I), "官方数据文档"),
    (re.compile(r"Tushare-", re.I), "数据渠道-"),
    (re.compile(r"\btushare\b", re.I), "数据接口"),
    (re.compile(r"\bdeepseek\b", re.I), "推理服务"),
    (re.compile(r"\bdashscope\b", re.I), "嵌入服务"),
    (re.compile(r"\bqdrant\b", re.I), "新闻库"),
    (re.compile(r"\bopenai\b", re.I), "模型服务"),
    (re.compile(r"\banthropic\b", re.I), "模型服务"),
    (re.compile(r"\bglm\b", re.I), "模型服务"),
    (re.compile(r"\bqwen\b", re.I), "模型服务"),
)


def strip_doc_alignment_section(text: str) -> str:
    """Remove the field-definition alignment appendix if the model included it."""
    if not text or not text.strip():
        return text or ""
    cleaned = _DOC_ALIGNMENT_HEADING.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def redact_vendor_names(text: str) -> str:
    """Replace vendor / provider names with neutral labels for client-facing reports."""
    if not text or not text.strip():
        return text or ""
    out = text
    for pattern, replacement in _VENDOR_PATTERNS:
        out = pattern.sub(replacement, out)
    return out


def prepare_report_for_publish(text: str | None) -> str:
    """Apply all publish-time sanitization steps."""
    if not text:
        return ""
    return redact_vendor_names(strip_doc_alignment_section(str(text)))
