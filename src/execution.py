"""Market and limit order execution simulation."""

from __future__ import annotations

import pandas as pd


def apply_spread(df: pd.DataFrame, spread: float) -> pd.DataFrame:
    """Add bid and ask columns using a fixed absolute spread."""

    if spread < 0:
        raise ValueError("spread must be non-negative")

    out = df.copy()
    out["bid"] = out["close"] - spread / 2
    out["ask"] = out["close"] + spread / 2
    return out


def _target_from_signal(df: pd.DataFrame) -> pd.Series:
    if "target_position" in df.columns:
        return df["target_position"].fillna(0).astype(int).clip(lower=0, upper=1)

    if "signal" not in df.columns:
        raise ValueError("DataFrame must include 'signal' or 'target_position'.")

    state = 0
    target_values: list[int] = []
    for signal in df["signal"].fillna(0).astype(int):
        if signal == 1:
            state = 1
        elif signal == -1:
            state = 0
        target_values.append(state)
    return pd.Series(target_values, index=df.index, dtype="int64", name="target_position")


def shifted_position(df: pd.DataFrame) -> pd.Series:
    """Return next-bar executable position, preventing lookahead bias."""

    target = _target_from_signal(df)
    return target.shift(1).fillna(0).astype(int).rename("position")


def simulate_market(
    df: pd.DataFrame,
    cash: float = 10_000,
    spread: float = 0.10,
) -> tuple[pd.DataFrame, pd.Series]:
    """Simulate long-only market orders at the correct bid/ask side."""

    if cash <= 0:
        raise ValueError("cash must be positive")

    data = df.copy()
    if "bid" not in data.columns or "ask" not in data.columns:
        data = apply_spread(data, spread)

    desired_position = shifted_position(data)
    cash_balance = float(cash)
    shares = 0.0
    trades: list[dict[str, object]] = []
    equity_values: list[float] = []

    for date, row in data.iterrows():
        desired = int(desired_position.loc[date])
        if desired == 1 and shares == 0:
            price = float(row["ask"])
            shares = cash_balance / price
            notional = shares * price
            cash_balance -= notional
            side = "BUY"
            trades.append(
                {
                    "date": date,
                    "order_type": "market",
                    "side": side,
                    "price": price,
                    "shares": shares,
                    "notional": notional,
                }
            )
        elif desired == 0 and shares > 0:
            price = float(row["bid"])
            notional = shares * price
            cash_balance += notional
            trades.append(
                {
                    "date": date,
                    "order_type": "market",
                    "side": "SELL",
                    "price": price,
                    "shares": shares,
                    "notional": notional,
                }
            )
            shares = 0.0

        equity_values.append(cash_balance + shares * float(row["close"]))

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df["date"] = pd.to_datetime(trades_df["date"])
        trades_df = trades_df.set_index("date")
    equity = pd.Series(equity_values, index=data.index, name="strategy_equity")
    return trades_df, equity


def simulate_limit(
    df: pd.DataFrame,
    limit_price: float,
    side: str,
    cash: float = 10_000,
    spread: float = 0.10,
) -> tuple[pd.DataFrame, pd.Series]:
    """Simulate a single limit order with explicit fill rules."""

    if cash <= 0:
        raise ValueError("cash must be positive")
    if limit_price <= 0:
        raise ValueError("limit_price must be positive")

    side = side.upper()
    if side not in {"BUY", "SELL"}:
        raise ValueError("side must be 'buy' or 'sell'")

    data = df.copy()
    if "bid" not in data.columns or "ask" not in data.columns:
        data = apply_spread(data, spread)

    cash_balance = float(cash) if side == "BUY" else 0.0
    shares = 0.0 if side == "BUY" else float(cash) / float(data["close"].iloc[0])
    filled = False
    trades: list[dict[str, object]] = []
    equity_values: list[float] = []

    for date, row in data.iterrows():
        if not filled and side == "BUY" and float(row["ask"]) <= limit_price:
            price = float(row["ask"])
            shares = cash_balance / price
            notional = shares * price
            cash_balance -= notional
            filled = True
            trades.append(
                {
                    "date": date,
                    "order_type": "limit",
                    "side": "BUY",
                    "limit_price": float(limit_price),
                    "price": price,
                    "shares": shares,
                    "notional": notional,
                    "filled": True,
                }
            )
        elif not filled and side == "SELL" and float(row["bid"]) >= limit_price:
            price = float(row["bid"])
            notional = shares * price
            cash_balance += notional
            filled = True
            trades.append(
                {
                    "date": date,
                    "order_type": "limit",
                    "side": "SELL",
                    "limit_price": float(limit_price),
                    "price": price,
                    "shares": shares,
                    "notional": notional,
                    "filled": True,
                }
            )
            shares = 0.0

        equity_values.append(cash_balance + shares * float(row["close"]))

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df["date"] = pd.to_datetime(trades_df["date"])
        trades_df = trades_df.set_index("date")
    equity = pd.Series(equity_values, index=data.index, name="limit_equity")
    return trades_df, equity

