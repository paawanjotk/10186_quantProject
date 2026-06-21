"""Performance metrics and plotting helpers."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd


def buy_and_hold(df: pd.DataFrame, cash: float = 10_000) -> pd.Series:
    """Buy at the first close and hold on the same index as the strategy."""

    if df.empty:
        raise ValueError("df must not be empty")
    shares = float(cash) / float(df["close"].iloc[0])
    return (shares * df["close"]).rename("buy_and_hold")


def _round_trip_pnls(trades: pd.DataFrame) -> list[float]:
    if trades.empty:
        return []

    open_trade: pd.Series | None = None
    pnls: list[float] = []
    for _, trade in trades.iterrows():
        if trade["side"] == "BUY":
            open_trade = trade
        elif trade["side"] == "SELL" and open_trade is not None:
            shares = min(float(open_trade["shares"]), float(trade["shares"]))
            pnls.append((float(trade["price"]) - float(open_trade["price"])) * shares)
            open_trade = None
    return pnls


def metrics(equity: pd.Series, trades: pd.DataFrame, df: pd.DataFrame | None = None) -> dict[str, Any]:
    """Compute trade counts, total return, Sharpe ratio, and max drawdown."""

    if equity.empty:
        raise ValueError("equity must not be empty")

    pnl = _round_trip_pnls(trades)
    returns = equity.pct_change().dropna()
    volatility = returns.std(ddof=0)
    sharpe = 0.0 if volatility == 0 or math.isnan(volatility) else float((returns.mean() / volatility) * math.sqrt(252))
    drawdown = equity / equity.cummax() - 1

    return {
        "total_trades": int(len(trades)),
        "completed_round_trips": int(len(pnl)),
        "winning_trades": int(sum(value > 0 for value in pnl)),
        "losing_trades": int(sum(value < 0 for value in pnl)),
        "strategy_return": float(equity.iloc[-1] / equity.iloc[0] - 1),
        "final_equity": float(equity.iloc[-1]),
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
    }


def _matplotlib():
    import matplotlib.pyplot as plt

    return plt


def plot_price_signals(df: pd.DataFrame, ax=None):
    """Plot close price, moving averages, and BUY/SELL markers."""

    plt = _matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))

    ax.plot(df.index, df["close"], label="Close", color="#1f77b4", linewidth=1.8)
    if "ma_short" in df:
        ax.plot(df.index, df["ma_short"], label="Short MA", color="#ff7f0e", linewidth=1.2)
    if "ma_long" in df:
        ax.plot(df.index, df["ma_long"], label="Long MA", color="#2ca02c", linewidth=1.2)
    if "signal" in df:
        buys = df["signal"] == 1
        sells = df["signal"] == -1
        ax.scatter(df.index[buys], df.loc[buys, "close"], marker="^", color="#2ca02c", s=70, label="BUY")
        ax.scatter(df.index[sells], df.loc[sells, "close"], marker="v", color="#d62728", s=70, label="SELL")

    ax.set_title("Close Price with Moving Averages and Signals")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.25)
    ax.legend()
    return ax


def plot_equity_vs_bh(strategy_eq: pd.Series, bh_eq: pd.Series, ax=None):
    """Plot strategy equity against aligned buy-and-hold equity."""

    plt = _matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))

    common = strategy_eq.index.intersection(bh_eq.index)
    ax.plot(common, strategy_eq.loc[common], label="Strategy", color="#1f77b4", linewidth=1.8)
    ax.plot(common, bh_eq.loc[common], label="Buy and hold", color="#ff7f0e", linewidth=1.8)
    ax.set_title("Strategy Equity vs Buy and Hold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value")
    ax.grid(True, alpha=0.25)
    ax.legend()
    return ax


def plot_volume(df: pd.DataFrame, ax=None):
    """Plot volume and rolling average volume."""

    plt = _matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))

    ax.bar(df.index, df["volume"], color="#6baed6", alpha=0.7, label="Volume")
    if "volume_avg" in df:
        ax.plot(df.index, df["volume_avg"], color="#d62728", linewidth=1.5, label="Volume avg")
    ax.set_title("Trading Volume")
    ax.set_xlabel("Date")
    ax.set_ylabel("Shares")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    return ax


def plot_volatility(df: pd.DataFrame, ax=None):
    """Plot rolling volatility."""

    plt = _matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))

    ax.plot(df.index, df["vol"], color="#9467bd", linewidth=1.6, label="Rolling volatility")
    ax.set_title("Rolling Volatility")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily return standard deviation")
    ax.grid(True, alpha=0.25)
    ax.legend()
    return ax


def plot_cumulative_returns(strategy_eq: pd.Series, bh_eq: pd.Series, ax=None):
    """Plot cumulative returns for a strategy against buy and hold."""

    plt = _matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))

    common = strategy_eq.index.intersection(bh_eq.index)
    strategy_common = strategy_eq.loc[common]
    bh_common = bh_eq.loc[common]
    ax.plot(common, strategy_common / strategy_common.iloc[0] - 1, label="Strategy", color="#1f77b4", linewidth=1.8)
    ax.plot(common, bh_common / bh_common.iloc[0] - 1, label="Buy and hold", color="#ff7f0e", linewidth=1.8)
    ax.set_title("Cumulative Returns")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative return")
    ax.grid(True, alpha=0.25)
    ax.legend()
    return ax


def plot_strategy_comparison(series_map: dict[str, pd.Series], ax=None):
    """Plot multiple strategy equity series on the same axes."""

    plt = _matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728", "#17becf"]
    for i, (label, series) in enumerate(series_map.items()):
        ax.plot(series.index, series, label=label, color=colors[i % len(colors)], linewidth=1.8)
    ax.set_title("Strategy Comparison")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value")
    ax.grid(True, alpha=0.25)
    ax.legend()
    return ax
