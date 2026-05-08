"""Unit tests for SSE trading-day helper used by Tushare indicators."""

from unittest.mock import patch

import pytest

pytest.importorskip("stockstats")
import pandas as pd


@pytest.mark.unit
def test_sse_trading_days_between_inclusive_filters_open():
    from tradingagents.dataflows import tushare_data as td

    mock_df = pd.DataFrame(
        {
            "cal_date": ["20260102", "20260103", "20260104"],
            "is_open": [1, 0, 1],
        }
    )
    with patch.object(td, "_try_pro_call", return_value=mock_df):
        out = td._sse_trading_days_between_inclusive("2026-01-02", "2026-01-05")
    assert out == ["2026-01-02", "2026-01-04"]


@pytest.mark.unit
def test_sse_trading_days_empty_returns_empty():
    from tradingagents.dataflows import tushare_data as td

    with patch.object(td, "_try_pro_call", return_value=None):
        assert td._sse_trading_days_between_inclusive("2026-01-01", "2026-01-31") == []
