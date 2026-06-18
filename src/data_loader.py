"""Market data loading, cleaning, and synthetic OHLCV generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CONTRACT_COLUMNS = ["open", "high", "low", "close", "volume"]


def _flatten_columns(columns: pd.Index) -> list[str]:
    if isinstance(columns, pd.MultiIndex):
        flattened = []
        for item in columns:
            parts = [str(part) for part in item if str(part) and str(part) != "nan"]
            flattened.append("_".join(parts))
        return flattened
    return [str(column) for column in columns]


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalised = df.copy()
    normalised.columns = [
        column.strip().lower().replace(" ", "_").replace("-", "_")
        for column in _flatten_columns(normalised.columns)
    ]
    aliases = {
        "adj_close": "close",
        "adjusted_close": "close",
        "datetime": "date",
        "timestamp": "date",
    }
    normalised = normalised.rename(columns=aliases)
    return normalised


def load_data(
    source: str = "csv",
    ticker: str = "AAPL",
    start: str | None = None,
    end: str | None = None,
    path: str | Path | None = None,
) -> pd.DataFrame:
    """Load market data and return a cleaned OHLCV DataFrame.

    Supported sources are "csv", "synthetic", and "yfinance". CSV is the
    default so the submitted project runs without network access.
    """

    source = source.lower()
    if source == "synthetic":
        return make_synthetic()

    if source == "csv":
        data_path = Path(path) if path is not None else Path("data/synthetic_ohlcv.csv")
        if not data_path.exists():
            data_path = Path(__file__).resolve().parents[1] / data_path
        if not data_path.exists():
            return make_synthetic()
        return clean_data(pd.read_csv(data_path))

    if source == "yfinance":
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ImportError("Install yfinance or use source='csv'/'synthetic'.") from exc

        raw = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            group_by="column",
        )
        if raw.empty:
            raise ValueError(f"No data returned for ticker {ticker!r}.")
        return clean_data(raw)

    raise ValueError("source must be one of: 'csv', 'synthetic', 'yfinance'")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return a sorted, duplicate-free, numeric OHLCV DataFrame."""

    cleaned = _normalise_columns(df)

    if not isinstance(cleaned.index, pd.DatetimeIndex):
        date_column = None
        for candidate in ("date", "time", "index", "unnamed:_0"):
            if candidate in cleaned.columns:
                date_column = candidate
                break
        if date_column is None:
            date_column = cleaned.columns[0]
        cleaned[date_column] = pd.to_datetime(cleaned[date_column], errors="coerce")
        cleaned = cleaned.dropna(subset=[date_column]).set_index(date_column)
    else:
        cleaned.index = pd.to_datetime(cleaned.index, errors="coerce")
        cleaned = cleaned[cleaned.index.notna()]

    missing = [column for column in CONTRACT_COLUMNS if column not in cleaned.columns]
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {missing}")

    cleaned = cleaned[CONTRACT_COLUMNS].copy()
    for column in CONTRACT_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
    cleaned = cleaned[~cleaned.index.duplicated(keep="last")].sort_index()

    for column in ("open", "high", "low", "close"):
        cleaned.loc[cleaned[column] <= 0, column] = np.nan
    cleaned.loc[cleaned["volume"] < 0, "volume"] = np.nan

    cleaned = cleaned.ffill().bfill().dropna()
    price_columns = cleaned[["open", "high", "low", "close"]]
    cleaned["high"] = price_columns.max(axis=1)
    cleaned["low"] = price_columns.min(axis=1)
    cleaned["volume"] = cleaned["volume"].round().astype("int64")
    cleaned.index.name = "date"
    return cleaned


def compute_stats(df: pd.DataFrame) -> dict[str, Any]:
    """Compute compact descriptive statistics for the cleaned dataset."""

    data = clean_data(df)
    return {
        "rows": int(len(data)),
        "columns": list(data.columns),
        "start_date": data.index.min().strftime("%Y-%m-%d"),
        "end_date": data.index.max().strftime("%Y-%m-%d"),
        "close_mean": float(data["close"].mean()),
        "close_std": float(data["close"].std(ddof=0)),
        "close_min": float(data["close"].min()),
        "close_max": float(data["close"].max()),
        "volume_mean": float(data["volume"].mean()),
        "volume_min": int(data["volume"].min()),
        "volume_max": int(data["volume"].max()),
    }


def make_synthetic(n: int = 756, seed: int = 42, regime: str = "trend") -> pd.DataFrame:
    """Create a reproducible OHLCV dataset using a simple price process."""

    if n < 60:
        raise ValueError("n must be at least 60 to support rolling indicators.")

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp("2024-12-31"), periods=n, name="date")

    if regime == "trend":
        drift = 0.00045
        volatility = 0.013
        cycle = 0.0025 * np.sin(np.linspace(0, 8 * np.pi, n))
    elif regime == "choppy":
        drift = 0.00005
        volatility = 0.019
        cycle = 0.0060 * np.sin(np.linspace(0, 18 * np.pi, n))
    else:
        drift = 0.00025
        volatility = 0.016
        cycle = 0.0035 * np.sin(np.linspace(0, 12 * np.pi, n))

    log_returns = drift + cycle + rng.normal(0, volatility, n)
    close = 100.0 * np.exp(np.cumsum(log_returns))
    previous_close = np.r_[close[0], close[:-1]]
    open_ = previous_close * (1 + rng.normal(0, 0.0035, n))
    intraday_range = np.abs(rng.normal(0.012, 0.004, n))
    high = np.maximum(open_, close) * (1 + intraday_range)
    low = np.minimum(open_, close) * (1 - intraday_range)
    volume_noise = rng.normal(0, 170_000, n)
    volume_trend = 850_000 + 250_000 * np.abs(log_returns) / max(volatility, 1e-8)
    volume = np.maximum(75_000, volume_trend + volume_noise).round().astype("int64")

    raw = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )
    return clean_data(raw)

