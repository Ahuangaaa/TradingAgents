import os

_TRADINGAGENTS_HOME = os.path.join(os.path.expanduser("~"), ".tradingagents")

DEFAULT_CONFIG = {
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR", os.path.join(_TRADINGAGENTS_HOME, "logs")),
    "data_cache_dir": os.getenv("TRADINGAGENTS_CACHE_DIR", os.path.join(_TRADINGAGENTS_HOME, "cache")),
    "memory_log_path": os.getenv("TRADINGAGENTS_MEMORY_LOG_PATH", os.path.join(_TRADINGAGENTS_HOME, "memory", "trading_memory.md")),
    # Optional cap on the number of resolved memory log entries. When set,
    # the oldest resolved entries are pruned once this limit is exceeded.
    # Pending entries are never pruned. None disables rotation entirely.
    "memory_log_max_entries": None,
    # LLM settings
    "llm_provider": "deepseek",
    "deep_think_llm": "deepseek-v4-pro",
    "quick_think_llm": "deepseek-v4-flash",
    # When None, each provider's client falls back to its own default endpoint
    # (api.openai.com for OpenAI, generativelanguage.googleapis.com for Gemini, ...).
    # The CLI overrides this per provider when the user picks one. Keeping a
    # provider-specific URL here would leak (e.g. OpenAI's /v1 was previously
    # being forwarded to Gemini, producing malformed request URLs).
    "backend_url": None,
    # DeepSeek V4 extended thinking: quick vs deep LLM are configured separately
    # (reasoning_effort + extra_body["thinking"]). Deep model defaults to max effort.
    "deepseek_quick_thinking_enabled": True,
    "deepseek_quick_reasoning_effort": "max",
    "deepseek_deep_thinking_enabled": True,
    "deepseek_deep_reasoning_effort": "max",
    # Checkpoint/resume: when True, LangGraph saves state after each node
    # so a crashed run can resume from the last successful step.
    "checkpoint_enabled": False,
    # Output language for saved reports (analysts, trader, risk debate, portfolio manager).
    # Structured rating enums (Buy/Hold/Sell, etc.) stay English for schema compatibility.
    "output_language": "English",
    # Debate and discussion settings
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
    # LangGraph recursion_limit (each analyst↔tools step counts). Raise when many
    # tool rounds (e.g. get_news per peer + get_industry_peers + fundamentals).
    "max_recur_limit": 400,
    # Data vendor configuration (Tushare Pro only)
    "data_vendors": {
        "core_stock_apis": "tushare",
        "technical_indicators": "tushare",
        "fundamental_data": "tushare",
        "news_data": "tushare",
    },
    # Tool-level configuration (takes precedence over category-level)
    "tool_vendors": {
        # Example: "get_stock_data": "tushare",
    },
    # Analyst tool fetch_url: HTTPS only, host allowlist, caps (SSRF mitigation).
    "web_fetch_enabled": True,
    "web_fetch_allowed_hosts": ["tushare.pro", "www.tushare.pro"],
    "web_fetch_max_bytes": 524288,
    "web_fetch_timeout_sec": 20,
    # ④⑤ major_news / news: LLM semantic filter (replaces industry substring match)
    "news_llm_filter_long_short": True,
    "news_llm_filter_use_cache": True,
    "news_llm_filter_batch_size": 28,
    "news_llm_filter_max_kept": 45,
    "news_llm_peer_context_max": 3,
    "news_long_short_lookback_days": 30,
    "news_raw_major_per_src": 12,
    "news_raw_flash_per_src": 14,
    "news_raw_major_content_max": 3000,
    "news_raw_flash_content_max": 1200,
    # ④⑤ from Qdrant (not Tushare major_news/news) when True or NEWS_LONG_SHORT_USE_QDRANT=1
    "news_long_short_use_qdrant": True,
    "news_qdrant_search_limit": 200,
    "news_qdrant_per_route_limit": 40,
    # Minimum similarity score for Qdrant merged hits (0 disables threshold).
    "news_qdrant_min_score": 0.15,
    # ⑦ 研报：LLM 精简（个股+行业）输出长度硬上限
    "news_research_llm_refine": True,
    "news_research_llm_input_max_chars": 30000,
    "news_research_llm_output_max_chars": 5000,
    # ⑦ 研报：下载 PDF 并提取正文（失败自动回退 abstr）
    "news_research_pdf_extract_enabled": True,
    "news_research_pdf_max_bytes": 15000000,
    "news_research_pdf_text_max_chars": 24000,
    "news_research_pdf_page_limit": 80,
    # ⑦ 研报 PDF 下载：仅使用 Playwright 浏览器上下文请求（Edge channel）。
    "news_research_pdf_playwright_timeout_sec": 40,
    "news_research_pdf_playwright_channel": "msedge",
    "news_macro_vector_terms_per_query": 8,
    # ⑧ 宏观向量专题（专用检索词 + 可选 LLM；与 ④⑤ query 分离）
    "news_macro_section8_enabled": True,
    "news_macro_section8_search_limit": 180,
    "news_macro_section8_per_major": 14,
    "news_macro_section8_per_flash": 18,
    "news_macro_section8_major_content_max": 2200,
    "news_macro_section8_flash_content_max": 1500,
    "news_macro_section8_llm_refine": True,
    "news_macro_section8_llm_max_chars": 24000,
    "news_macro_section8_raw_max_chars": 12000,
    "news_macro_section8_include_raw_excerpt": True,
}
