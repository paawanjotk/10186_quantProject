"""Trading signal generation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import add_moving_averages


def _event_signal_from_target(target: pd.Series) -> pd.Series:
    target = target.fillna(0).astype(int).clip(lower=0, upper=1)
    previous = target.shift(1, fill_value=0)
    events = np.select(
        [(target == 1) & (previous == 0), (target == 0) & (previous == 1)],
        [1, -1],
        default=0,
    )
    return pd.Series(events, index=target.index, name="signal", dtype="int64")


def ma_crossover(df: pd.DataFrame, short: int = 20, long: int = 50) -> pd.DataFrame:
    """Moving average crossover strategy.

    BUY when the short moving average moves above the long moving average.
    SELL when the short moving average is no longer above the long average.
    """

    out = add_moving_averages(df, short=short, long=long).copy()
    valid = out["ma_short"].notna() & out["ma_long"].notna()
    target = pd.Series(0, index=out.index, dtype="int64")
    target.loc[valid & (out["ma_short"] > out["ma_long"])] = 1
    out["target_position"] = target
    out["signal"] = _event_signal_from_target(target)
    return out


def mean_reversion(df: pd.DataFrame, window: int = 20, z: float = 2.0) -> pd.DataFrame:
    """Mean reversion strategy using a rolling z-score.

    BUY when price is far below its rolling mean. SELL when price is far
    above its rolling mean. Between the two thresholds, keep the prior state.
    """

    if window <= 1:
        raise ValueError("mean-reversion window must be greater than one")
    if z <= 0:
        raise ValueError("z must be positive")

    out = df.copy()
    out["mr_mean"] = out["close"].rolling(window, min_periods=window).mean()
    out["mr_std"] = out["close"].rolling(window, min_periods=window).std()
    out["zscore"] = (out["close"] - out["mr_mean"]) / out["mr_std"]

    state = 0
    target_values: list[int] = []
    for value in out["zscore"]:
        if pd.isna(value):
            state = 0
        elif value <= -z:
            state = 1
        elif value >= z:
            state = 0
        target_values.append(state)

    target = pd.Series(target_values, index=out.index, dtype="int64")
    out["target_position"] = target
    out["signal"] = _event_signal_from_target(target)
    return out

