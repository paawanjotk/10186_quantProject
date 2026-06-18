# Simplified Quant Trading System

This project implements a reproducible quantitative trading workflow for OHLCV data: data cleaning, indicators, signal generation, bid/ask execution simulation, performance evaluation, plots, a notebook, and a PDF report.

Headline result on the included synthetic dataset: Mean Reversion finished at `$10,434.90`, Moving Average Crossover finished at `$9,738.14`, and Buy and Hold finished at `$9,432.22` from a `$10,000` start with a `$0.10` spread. The active strategies helped by reducing exposure during weaker periods, while the passive baseline stayed fully exposed to the drawdown.

## Folder Guide

| Path | Purpose |
| :-- | :-- |
| `src/` | Reusable Python modules for data, indicators, signals, execution, evaluation, and checks |
| `data/synthetic_ohlcv.csv` | Offline OHLCV dataset used by default |
| `notebook/project_notebook.ipynb` | Top-to-bottom project notebook with outputs and chart references |
| `plots/` | Saved visualizations used in the notebook and report |
| `report/final_report.pdf` | Separate final report deliverable |
| `scripts/build_artifacts.py` | Rebuilds the dataset, plots, notebook, and PDF from the source modules |

## How To Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.tests
python scripts/build_artifacts.py
```

Open `notebook/project_notebook.ipynb` and run all cells. The notebook reads the local CSV by default, so no live API key or network connection is required.

## Data Contract

Every module expects a pandas DataFrame with:

- Index: `DatetimeIndex` named `date`, sorted ascending, with no duplicates.
- Columns: `open`, `high`, `low`, `close`, `volume`.
- No missing values after `clean_data`.
- Indicators add columns such as `ret`, `cum_ret`, `ma_short`, `ma_long`, `vol`, and `volume_avg`.
- Signal functions add `signal` where `1` is BUY, `-1` is SELL, and `0` is HOLD.
- Execution uses a one-bar shift from signal to position, so a signal observed on day `t` can only trade on day `t+1`.

## Assumptions

- Starting cash: `$10,000`.
- Spread model: `bid = close - spread / 2`, `ask = close + spread / 2`.
- Market BUY fills at ask; market SELL fills at bid.
- Limit BUY fills only when `ask <= limit_price`; limit SELL fills only when `bid >= limit_price`.
- Long-only trading, fractional shares allowed, no borrowing, no taxes, and no slippage beyond spread.

## Key Files To Review

- `src/execution.py` for no-lookahead position shifting and market/limit order fill rules.
- `src/signals.py` for Moving Average Crossover and Mean Reversion logic.
- `src/tests.py` for correctness checks on data shape, no-lookahead behavior, spread fills, limit fills, and buy-and-hold alignment.
- `report/final_report.pdf` for the concise interpretation of results.
