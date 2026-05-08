"""HTTPS document fetch for analysts (allowlisted hosts, size cap, SSRF-aware)."""

from __future__ import annotations

from typing import Annotated
from urllib.parse import urlparse

import requests
from langchain_core.tools import tool

from tradingagents.dataflows.config import get_config

_USER_AGENT = "TradingAgents/0.2.4 (analyst fetch_url; documentation retrieval)"

_TEXT_CONTENT_PREFIXES = (
    "text/",
    "application/json",
    "application/xml",
    "application/xhtml",
    "application/javascript",
    "application/ld+json",
)


def _normalize_host(hostname: str | None) -> str | None:
    if not hostname:
        return None
    host = hostname.strip().lower()
    if ":" in host and not host.startswith("["):
        host = host.split(":")[0]
    return host or None


def _allowed_hosts_from_config() -> frozenset[str]:
    cfg = get_config()
    raw = cfg.get("web_fetch_allowed_hosts")
    if not raw:
        return frozenset({"tushare.pro", "www.tushare.pro"})
    out: set[str] = set()
    for h in raw:
        n = _normalize_host(str(h))
        if n:
            out.add(n)
    return frozenset(out) if out else frozenset({"tushare.pro", "www.tushare.pro"})


def _is_textual_content_type(content_type: str) -> bool:
    ct = (content_type or "").split(";")[0].strip().lower()
    if not ct:
        return True
    if ct.startswith(_TEXT_CONTENT_PREFIXES):
        return True
    return False


def _fetch_url_impl(url: str) -> str:
    cfg = get_config()
    if not cfg.get("web_fetch_enabled", True):
        return "fetch_url is disabled (set web_fetch_enabled to True in config)."

    raw_url = (url or "").strip()
    if not raw_url:
        return "fetch_url: empty URL."

    parsed = urlparse(raw_url)
    if parsed.scheme != "https":
        return "fetch_url: only https:// URLs are allowed."

    host = _normalize_host(parsed.hostname)
    allowed = _allowed_hosts_from_config()
    if not host or host not in allowed:
        return (
            f"fetch_url: host {host!r} is not allowlisted. "
            f"Allowed hosts: {', '.join(sorted(allowed))}. "
            "Extend web_fetch_allowed_hosts in config if needed."
        )

    timeout = int(cfg.get("web_fetch_timeout_sec", 20))
    max_bytes = int(cfg.get("web_fetch_max_bytes", 524288))
    if max_bytes < 1024:
        max_bytes = 1024

    try:
        resp = requests.get(
            raw_url,
            timeout=timeout,
            stream=True,
            allow_redirects=True,
            headers={"User-Agent": _USER_AGENT, "Accept": "text/html,application/json,text/plain;q=0.9,*/*;q=0.1"},
        )
    except requests.RequestException as exc:
        return f"fetch_url: request failed: {exc}"

    final_parsed = urlparse(resp.url)
    final_host = _normalize_host(final_parsed.hostname)
    if final_host not in allowed:
        return (
            f"fetch_url: redirect landed on disallowed host {final_host!r} (final URL: {resp.url}). "
            "Open redirects to non-allowlisted hosts are blocked."
        )

    ctype = resp.headers.get("Content-Type", "")
    if not _is_textual_content_type(ctype):
        return (
            f"fetch_url: unsupported Content-Type {ctype!r} for URL {resp.url}. "
            "Only textual types (HTML, JSON, plain text, XML) are returned as body."
        )

    buf = bytearray()
    truncated = False
    try:
        for chunk in resp.iter_content(chunk_size=65536):
            if not chunk:
                continue
            if len(buf) + len(chunk) > max_bytes:
                take = max_bytes - len(buf)
                if take > 0:
                    buf.extend(chunk[:take])
                truncated = True
                break
            buf.extend(chunk)
    finally:
        resp.close()

    if not buf:
        return f"fetch_url: empty body. final_url={resp.url} status={resp.status_code}"

    enc = resp.encoding or getattr(resp, "apparent_encoding", None) or "utf-8"
    try:
        text = bytes(buf).decode(enc, errors="replace")
    except LookupError:
        text = bytes(buf).decode("utf-8", errors="replace")

    if "\x00" in text[:2000]:
        return (
            f"fetch_url: response looks binary despite Content-Type. final_url={resp.url}. "
            "Refusing to decode as text."
        )

    header = (
        f"# fetch_url\n# final_url: {resp.url}\n# status: {resp.status_code}\n"
        f"# content-type: {ctype}\n\n---\n\n"
    )
    suffix = ""
    if truncated:
        suffix = f"\n\n---\n(truncated at {max_bytes} bytes; increase web_fetch_max_bytes if needed.)"
    return header + text + suffix


@tool
def fetch_url(
    url: Annotated[str, "Full https URL on an allowlisted host (default: tushare.pro docs)."],
) -> str:
    """
    Fetch documentation or reference pages over HTTPS for field definitions and API notes.

    Use when the task requires aligning tool output with official documentation. Only
    allowlisted hosts are permitted (default: tushare.pro). Response is capped by config;
    HTML/JSON/text bodies may be truncated.
    """
    return _fetch_url_impl(url)
