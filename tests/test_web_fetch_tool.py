"""Unit tests for analyst fetch_url (mocked HTTP).

Loads ``web_fetch_tool`` via importlib so ``tradingagents.agents`` (and tushare/stockstats)
is not imported during collection.
"""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

import tradingagents.default_config as default_config
from tradingagents.dataflows.config import set_config

_REPO = Path(__file__).resolve().parents[1]
_WFT_PATH = _REPO / "tradingagents" / "agents" / "utils" / "web_fetch_tool.py"


def _load_web_fetch_module():
    spec = importlib.util.spec_from_file_location(
        "_web_fetch_tool_under_test",
        _WFT_PATH,
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def web_fetch_tool():
    return _load_web_fetch_module()


@pytest.fixture(autouse=True)
def _reset_config():
    cfg = copy.deepcopy(default_config.DEFAULT_CONFIG)
    set_config(cfg)
    yield
    set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))


class _MockResponse:
    def __init__(self, url: str, body: bytes, content_type: str = "text/html; charset=utf-8"):
        self.url = url
        self.status_code = 200
        self.headers = {"Content-Type": content_type}
        self.encoding = "utf-8"
        self.apparent_encoding = "utf-8"
        self._body = body

    def iter_content(self, chunk_size: int = 65536):
        yield self._body

    def close(self):
        pass


@pytest.mark.unit
def test_fetch_url_disabled(web_fetch_tool):
    set_config({"web_fetch_enabled": False})
    out = web_fetch_tool._fetch_url_impl("https://tushare.pro/wctapi/documents/79.md")
    assert "disabled" in out.lower()


@pytest.mark.unit
def test_fetch_url_rejects_http(web_fetch_tool):
    out = web_fetch_tool._fetch_url_impl("http://tushare.pro/foo")
    assert "only https" in out.lower()


@pytest.mark.unit
def test_fetch_url_rejects_unknown_host(web_fetch_tool):
    out = web_fetch_tool._fetch_url_impl("https://evil.example/doc")
    assert "not allowlisted" in out.lower() or "allowlist" in out.lower()


@pytest.mark.unit
def test_fetch_url_rejects_redirect_to_disallowed_host(web_fetch_tool):
    bad = _MockResponse("https://evil.example/stolen", b"<html></html>")

    def fake_get(*_a, **_k):
        return bad

    with patch.object(web_fetch_tool.requests, "get", side_effect=fake_get):
        out = web_fetch_tool._fetch_url_impl("https://tushare.pro/start")
    assert "disallowed host" in out.lower() or "redirect" in out.lower()


@pytest.mark.unit
def test_fetch_url_success_small_html(web_fetch_tool):
    good = _MockResponse("https://tushare.pro/doc", b"<html><body>ok</body></html>")

    def fake_get(*_a, **_k):
        return good

    with patch.object(web_fetch_tool.requests, "get", side_effect=fake_get):
        out = web_fetch_tool._fetch_url_impl("https://tushare.pro/wctapi/documents/79.md")
    assert "final_url:" in out
    assert "ok" in out


@pytest.mark.unit
def test_fetch_url_rejects_binary_content_type(web_fetch_tool):
    binresp = _MockResponse(
        "https://tushare.pro/x",
        b"\x00\x01",
        content_type="application/octet-stream",
    )

    def fake_get(*_a, **_k):
        return binresp

    with patch.object(web_fetch_tool.requests, "get", side_effect=fake_get):
        out = web_fetch_tool._fetch_url_impl("https://tushare.pro/x")
    assert "unsupported" in out.lower() or "content-type" in out.lower()


@pytest.mark.unit
def test_fetch_url_langchain_tool_invokes_impl(web_fetch_tool):
    good = _MockResponse("https://tushare.pro/a", b"hello")

    def fake_get(*_a, **_k):
        return good

    with patch.object(web_fetch_tool.requests, "get", side_effect=fake_get):
        out = web_fetch_tool.fetch_url.invoke({"url": "https://tushare.pro/a"})
    assert "hello" in out
