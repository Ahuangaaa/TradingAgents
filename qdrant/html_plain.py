"""本地从 HTML 得到纯文本：整段删除 ``script`` / ``style`` / ``noscript`` 及其内容，再剥掉其余标签（如 ``p``、``a``、``div``），保留标签之间的文字。"""

from __future__ import annotations

import re

# 无 BeautifulSoup 时：去掉整块 script/style/noscript，再粗暴去其余尖括号标签
_RE_SCRIPT = re.compile(r"(?is)<script[^>]*>.*?</script>")
_RE_STYLE = re.compile(r"(?is)<style[^>]*>.*?</style>")
_RE_NOSCRIPT = re.compile(r"(?is)<noscript[^>]*>.*?</noscript>")
_RE_ALL_TAGS = re.compile(r"<[^>]+>")


def extract_plain_from_html(html: str | None) -> str:
    """删除 ``style`` / ``script`` 等块及其内容，再移除所有 HTML 标签，保留可见文本。"""
    if html is None:
        return ""
    raw = str(html)
    if not raw.strip():
        return ""
    if "<" not in raw or ">" not in raw:
        return " ".join(raw.split()).strip()

    try:
        from bs4 import BeautifulSoup
    except ImportError:
        plain = _RE_SCRIPT.sub(" ", raw)
        plain = _RE_STYLE.sub(" ", plain)
        plain = _RE_NOSCRIPT.sub(" ", plain)
        plain = _RE_ALL_TAGS.sub(" ", plain)
        plain = " ".join(plain.split())
        return plain.strip()

    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    lines: list[str] = []
    for line in soup.get_text("\n", strip=False).splitlines():
        s = " ".join(line.split())
        if s:
            lines.append(s)
    return "\n".join(lines).strip()
