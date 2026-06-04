"""Tests for market-wide capital flow data helpers."""

from unittest.mock import patch

import pandas as pd

from tradingagents.dataflows import tushare_data as td


def test_capital_flow_window_takes_last_n_trading_days():
    trading_days = [f"2024-01-{d:02d}" for d in range(1, 32)] + [
        f"2024-02-{d:02d}" for d in range(1, 10)
    ]
    with patch.object(td, "_sse_trading_days_between_inclusive") as cal:
        cal.return_value = trading_days
        d0, d1, days = td._capital_flow_window("2024-02-09", lookback_days=30)
        assert len(days) == 30
        assert days[0] == "2024-01-11"
        assert days[-1] == "2024-02-09"
        assert d0 == "20240111"
        assert d1 == "20240209"


def test_flow_rank_slices_top_in_and_out():
    df = pd.DataFrame(
        {
            "industry": ["A", "B", "C", "D"],
            "net_amount": [10.0, -5.0, 3.0, -20.0],
            "trade_date": ["2024-01-02"] * 4,
        }
    )
    top_in, top_out = td._flow_rank_slices(df, "industry", top_n=2)
    assert list(top_in["industry"]) == ["A", "C"]
    assert list(top_out["industry"]) == ["D", "B"]


def test_summarize_ths_sector_moneyflow_period_rollup():
    day1 = pd.DataFrame(
        {
            "industry": ["白酒", "银行"],
            "net_amount": [10.0, -2.0],
            "pct_change": [1.0, -0.5],
        }
    )
    day2 = pd.DataFrame(
        {
            "industry": ["白酒", "银行"],
            "net_amount": [5.0, 1.0],
            "pct_change": [0.5, 0.2],
        }
    )

    def fake_call(fn_name, **kwargs):
        d = kwargs.get("trade_date")
        if d == "20240102":
            return day1
        if d == "20240103":
            return day2
        return None

    with patch.object(td, "_try_pro_call", side_effect=fake_call):
        out = td._summarize_ths_sector_moneyflow(
            "moneyflow_ind_ths",
            "industry",
            "Test industry flow",
            ["2024-01-02", "2024-01-03"],
            focal_label="白酒",
            doc_url="https://example.com",
        )
    assert "区间累计净流入 Top" in out
    assert "白酒" in out
    assert "2024-01-02" in out
    assert "2024-01-03" in out


def test_get_tushare_moneyflow_mkt_dc_empty():
    with patch.object(td, "_capital_flow_window", return_value=("20240101", "20240131", ["2024-01-31"])):
        with patch.object(td, "_try_pro_call", return_value=None):
            out = td.get_tushare_moneyflow_mkt_dc("2024-01-31")
    assert "no data" in out.lower()
