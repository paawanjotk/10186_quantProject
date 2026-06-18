"""Assertion checks for the trading workflow."""

from __future__ import annotations

import pandas as pd

from .data_loader import clean_data, make_synthetic
from .evaluate import buy_and_hold
from .execution import apply_spread, shifted_position, simulate_limit, simulate_market
from .indicators import add_returns
from .signals import ma_crossover


def test_clean_data_contract() -> None:
    raw = pd.DataFrame(
        {
            "Date": ["2024-01-02", "2024-01-02", "2024-01-03"],
            "Open": [100, 101, 102],
            "High": [103, 102, 103],
            "Low": [99, 100, 101],
            "Close": [101, 102, 102.5],
            "Volume": [1_000, 1_100, 1_200],
        }
    )
    cleaned = clean_data(raw)
    assert cleaned.index.name == "date"
    assert list(cleaned.columns) == ["open", "high", "low", "close", "volume"]
    assert cleaned.index.is_monotonic_increasing
    assert not cleaned.index.has_duplicates
    assert not cleaned.isna().any().any()


def test_no_lookahead_position_shift() -> None:
    df = make_synthetic(n=80, seed=7)
    df = add_returns(ma_crossover(df, short=5, long=12))
    position = shifted_position(df)
    assert position.iloc[0] == 0
    assert position.equals(df["target_position"].shift(1).fillna(0).astype(int).rename("position"))


def test_market_orders_use_spread_and_next_bar() -> None:
    dates = pd.bdate_range("2024-01-01", periods=5, name="date")
    df = pd.DataFrame(
        {
            "open": [100, 101, 102, 103, 104],
            "high": [101, 102, 103, 104, 105],
            "low": [99, 100, 101, 102, 103],
            "close": [100, 101, 102, 103, 104],
            "volume": [1_000] * 5,
            "signal": [1, 0, -1, 0, 0],
        },
        index=dates,
    )
    trades, _ = simulate_market(df, cash=10_000, spread=0.20)
    assert list(trades["side"]) == ["BUY", "SELL"]
    assert trades.index[0] == dates[1]
    assert trades.index[1] == dates[3]
    assert round(float(trades.iloc[0]["price"]), 2) == 101.10
    assert round(float(trades.iloc[1]["price"]), 2) == 102.90


def test_limit_order_fill_rules() -> None:
    df = apply_spread(make_synthetic(n=80, seed=3), spread=0.10)
    no_fill_limit = float(df["ask"].min() - 1)
    no_fill, no_fill_equity = simulate_limit(df, no_fill_limit, side="buy", cash=10_000)
    assert no_fill.empty
    assert round(float(no_fill_equity.iloc[-1]), 2) == 10_000.00

    fill_limit = float(df["ask"].iloc[0])
    filled, _ = simulate_limit(df, fill_limit, side="buy", cash=10_000)
    assert len(filled) == 1
    assert filled.iloc[0]["side"] == "BUY"
    assert float(filled.iloc[0]["price"]) <= fill_limit


def test_buy_and_hold_alignment() -> None:
    df = make_synthetic(n=100, seed=11).iloc[20:]
    bh = buy_and_hold(df, cash=10_000)
    assert bh.index.equals(df.index)
    assert round(float(bh.iloc[0]), 2) == 10_000.00


def run_all_tests() -> None:
    test_clean_data_contract()
    test_no_lookahead_position_shift()
    test_market_orders_use_spread_and_next_bar()
    test_limit_order_fill_rules()
    test_buy_and_hold_alignment()
    print("All quant trading workflow tests passed.")


if __name__ == "__main__":
    run_all_tests()

