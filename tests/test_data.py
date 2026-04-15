from __future__ import annotations

import pandas as pd
import pytest

from data import fetch_stock_data, compute_returns, compute_statistics


class _FakeYF:
    @staticmethod
    def download(*args, **kwargs):
        idx = pd.date_range("2024-01-01", periods=5, freq="D")
        cols = pd.MultiIndex.from_product([["Close"], ["AAPL", "MSFT"]])
        values = [
            [100.0, 200.0],
            [101.0, 201.0],
            [102.0, 203.0],
            [103.0, 202.0],
            [104.0, 205.0],
        ]
        return pd.DataFrame(values, index=idx, columns=cols)


def test_fetch_stock_data(monkeypatch: pytest.MonkeyPatch):
    import data as data_mod

    monkeypatch.setattr(data_mod, "yf", _FakeYF)
    prices = fetch_stock_data(["AAPL", "MSFT"], period="1y")

    assert list(prices.columns) == ["AAPL", "MSFT"]
    assert len(prices) == 5


def test_returns_and_statistics(monkeypatch: pytest.MonkeyPatch):
    import data as data_mod

    monkeypatch.setattr(data_mod, "yf", _FakeYF)
    prices = fetch_stock_data(["AAPL", "MSFT"])
    returns = compute_returns(prices)
    mu, cov = compute_statistics(returns)

    assert not returns.empty
    assert len(mu) == 2
    assert cov.shape == (2, 2)
