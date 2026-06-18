"""Financial indicators used by the strategies."""

from __future__ import annotations

import pandas as pd


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple and cumulative returns."""

    out = df.copy()
    out["ret"] = out["close"].pct_change().fillna(0.0)
    out["cum_ret"] = (1 + out["ret"]).cumprod() - 1
    return out


def add_moving_averages(df: pd.DataFrame, short: int = 20, long: int = 50) -> pd.DataFrame:
    """Add short and long simple moving averages."""

    if short <= 0 or long <= 0:
        raise ValueError("moving-average windows must be positive")
    if short >= long:
        raise ValueError("short window should be smaller than long window")

    out = df.copy()
    out["ma_short"] = out["close"].rolling(short, min_periods=short).mean()
    out["ma_long"] = out["close"].rolling(long, min_periods=long).mean()
    return out


def add_volatility(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Add rolling volatility of simple returns."""

    if window <= 1:
        raise ValueError("volatility window must be greater than one")

    out = df.copy()
    if "ret" not in out:
        out = add_returns(out)
    out["vol"] = out["ret"].rolling(window, min_periods=window).std()
    return out


def add_volume_avg(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Add rolling average volume."""

    if window <= 1:
        raise ValueError("volume window must be greater than one")

    out = df.copy()
    out["volume_avg"] = out["volume"].rolling(window, min_periods=window).mean()
    return out

