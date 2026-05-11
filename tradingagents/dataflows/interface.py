from .tushare_data import (
    get_tushare_stock_data,
    get_tushare_indicators,
    get_tushare_fundamentals,
    get_tushare_industry_peers,
    get_tushare_balance_sheet,
    get_tushare_cashflow,
    get_tushare_income_statement,
    get_tushare_news,
    get_tushare_global_news,
    get_tushare_insider_transactions,
    get_tushare_holder_number,
    get_tushare_stock_moneyflow,
    get_tushare_margin_detail,
)
from .config import get_config

# Tools organized by category
TOOLS_CATEGORIES = {
    "core_stock_apis": {
        "description": "OHLCV stock price data",
        "tools": [
            "get_stock_data"
        ]
    },
    "technical_indicators": {
        "description": "Technical analysis indicators",
        "tools": [
            "get_indicators"
        ]
    },
    "fundamental_data": {
        "description": "Company fundamentals",
        "tools": [
            "get_fundamentals",
            "get_industry_peers",
            "get_balance_sheet",
            "get_cashflow",
            "get_income_statement"
        ]
    },
    "news_data": {
        "description": "News, insider data, and sentiment-oriented market microstructure",
        "tools": [
            "get_news",
            "get_global_news",
            "get_insider_transactions",
            "get_holder_number",
            "get_stock_moneyflow",
            "get_margin_detail",
        ]
    }
}

VENDOR_LIST = [
    "tushare",
]

# All market data is served via Tushare Pro (A-share oriented).
VENDOR_METHODS = {
    "get_stock_data": {
        "tushare": get_tushare_stock_data,
    },
    "get_indicators": {
        "tushare": get_tushare_indicators,
    },
    "get_fundamentals": {
        "tushare": get_tushare_fundamentals,
    },
    "get_industry_peers": {
        "tushare": get_tushare_industry_peers,
    },
    "get_balance_sheet": {
        "tushare": get_tushare_balance_sheet,
    },
    "get_cashflow": {
        "tushare": get_tushare_cashflow,
    },
    "get_income_statement": {
        "tushare": get_tushare_income_statement,
    },
    "get_news": {
        "tushare": get_tushare_news,
    },
    "get_global_news": {
        "tushare": get_tushare_global_news,
    },
    "get_insider_transactions": {
        "tushare": get_tushare_insider_transactions,
    },
    "get_holder_number": {
        "tushare": get_tushare_holder_number,
    },
    "get_stock_moneyflow": {
        "tushare": get_tushare_stock_moneyflow,
    },
    "get_margin_detail": {
        "tushare": get_tushare_margin_detail,
    },
}


def get_category_for_method(method: str) -> str:
    """Get the category that contains the specified method."""
    for category, info in TOOLS_CATEGORIES.items():
        if method in info["tools"]:
            return category
    raise ValueError(f"Method '{method}' not found in any category")


def get_vendor(category: str, method: str = None) -> str:
    """Get the configured vendor for a data category or specific tool method.

    Only ``tushare`` is supported; other values in config are ignored.
    """
    config = get_config()

    if method:
        tool_vendors = config.get("tool_vendors", {})
        if method in tool_vendors:
            return tool_vendors[method]

    return config.get("data_vendors", {}).get(category, "tushare")


def route_to_vendor(method: str, *args, **kwargs):
    """Route method calls to Tushare implementations."""
    category = get_category_for_method(method)
    vendor_config = get_vendor(category, method)
    primary_vendors = [v.strip() for v in vendor_config.split(",") if v.strip()]

    if method not in VENDOR_METHODS:
        raise ValueError(f"Method '{method}' not supported")

    available = VENDOR_METHODS[method]
    for vendor in primary_vendors:
        if vendor in available:
            impl_func = available[vendor]
            impl_func = impl_func[0] if isinstance(impl_func, list) else impl_func
            return impl_func(*args, **kwargs)

    # Default when config lists unknown vendors (e.g. legacy "yfinance")
    impl = available["tushare"]
    impl = impl[0] if isinstance(impl, list) else impl
    return impl(*args, **kwargs)
