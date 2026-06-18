"""Build dataset, plots, notebook, and PDF report for the project."""

from __future__ import annotations

import base64
import csv
import json
import math
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image as PdfImage
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import compute_stats, load_data, make_synthetic
from src.evaluate import buy_and_hold, metrics
from src.execution import apply_spread, simulate_limit, simulate_market
from src.indicators import add_returns, add_volume_avg, add_volatility
from src.signals import ma_crossover, mean_reversion


DATA_PATH = ROOT / "data" / "synthetic_ohlcv.csv"
PLOTS_DIR = ROOT / "plots"
REPORT_DIR = ROOT / "report"
NOTEBOOK_DIR = ROOT / "notebook"
SPREAD = 0.10
STARTING_CASH = 10_000


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _format_money(value: float) -> str:
    return f"${value:,.2f}"


def _series_points(series: pd.Series, x_positions: dict[pd.Timestamp, float], y_min: float, y_max: float, box: tuple[int, int, int, int]) -> list[tuple[float, float]]:
    left, top, right, bottom = box
    points: list[tuple[float, float]] = []
    span = y_max - y_min if y_max != y_min else 1.0
    for date, value in series.dropna().items():
        x = x_positions[pd.Timestamp(date)]
        y = bottom - ((float(value) - y_min) / span) * (bottom - top)
        points.append((x, y))
    return points


def draw_line_chart(
    path: Path,
    title: str,
    series_map: dict[str, pd.Series],
    y_label: str,
    markers: list[dict[str, object]] | None = None,
    width: int = 1200,
    height: int = 700,
) -> None:
    image = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(image)
    title_font = _font(30, bold=True)
    label_font = _font(19)
    small_font = _font(16)

    left, top, right, bottom = 95, 90, width - 55, height - 100
    all_series = [series.dropna() for series in series_map.values() if not series.dropna().empty]
    if not all_series:
        raise ValueError("No values to plot")
    values = pd.concat(all_series)
    y_min = float(values.min())
    y_max = float(values.max())
    pad = (y_max - y_min) * 0.08 if y_max != y_min else 1.0
    y_min -= pad
    y_max += pad

    full_index = series_map[next(iter(series_map))].index
    full_index = pd.Index(pd.to_datetime(full_index))
    count = max(len(full_index) - 1, 1)
    x_positions = {
        pd.Timestamp(date): left + i * (right - left) / count
        for i, date in enumerate(full_index)
    }

    draw.text((left, 28), title, fill="#14213d", font=title_font)
    draw.line((left, bottom, right, bottom), fill="#1f2937", width=2)
    draw.line((left, top, left, bottom), fill="#1f2937", width=2)

    for i in range(6):
        y = bottom - i * (bottom - top) / 5
        value = y_min + i * (y_max - y_min) / 5
        draw.line((left, y, right, y), fill="#e5e7eb", width=1)
        draw.text((12, y - 10), f"{value:,.2f}", fill="#374151", font=small_font)

    tick_dates = [full_index[0], full_index[len(full_index) // 2], full_index[-1]]
    for date in tick_dates:
        x = x_positions[pd.Timestamp(date)]
        draw.text((x - 46, bottom + 16), pd.Timestamp(date).strftime("%Y-%m-%d"), fill="#374151", font=small_font)

    colors_cycle = ["#2563eb", "#f97316", "#16a34a", "#7c3aed", "#dc2626", "#0891b2"]
    legend_x = left
    legend_y = height - 55
    for i, (label, series) in enumerate(series_map.items()):
        color = colors_cycle[i % len(colors_cycle)]
        points = _series_points(series, x_positions, y_min, y_max, (left, top, right, bottom))
        if len(points) >= 2:
            draw.line(points, fill=color, width=4)
        draw.rectangle((legend_x, legend_y, legend_x + 22, legend_y + 12), fill=color)
        draw.text((legend_x + 30, legend_y - 4), label, fill="#111827", font=small_font)
        legend_x += 160

    if markers:
        for marker in markers:
            dates = pd.Index(pd.to_datetime(marker["dates"]))
            values_for_markers = marker["values"]
            color = str(marker.get("color", "#111827"))
            shape = str(marker.get("shape", "circle"))
            for date in dates:
                if pd.Timestamp(date) not in x_positions or date not in values_for_markers.index:
                    continue
                value = float(values_for_markers.loc[date])
                x, y = _series_points(pd.Series([value], index=[date]), x_positions, y_min, y_max, (left, top, right, bottom))[0]
                if shape == "up":
                    draw.polygon([(x, y - 10), (x - 9, y + 9), (x + 9, y + 9)], fill=color)
                elif shape == "down":
                    draw.polygon([(x, y + 10), (x - 9, y - 9), (x + 9, y - 9)], fill=color)
                else:
                    draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=color)

    draw.text((left, height - 28), "Date", fill="#374151", font=label_font)
    draw.text((10, top - 30), y_label, fill="#374151", font=label_font)
    image.save(path)


def draw_bar_chart(path: Path, title: str, values: pd.Series, overlay: pd.Series | None = None) -> None:
    width, height = 1200, 560
    image = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(image)
    title_font = _font(30, bold=True)
    small_font = _font(16)
    left, top, right, bottom = 95, 85, width - 55, height - 85

    draw.text((left, 28), title, fill="#14213d", font=title_font)
    draw.line((left, bottom, right, bottom), fill="#1f2937", width=2)
    draw.line((left, top, left, bottom), fill="#1f2937", width=2)

    max_value = float(max(values.max(), overlay.max() if overlay is not None else values.max()))
    for i in range(6):
        y = bottom - i * (bottom - top) / 5
        value = i * max_value / 5
        draw.line((left, y, right, y), fill="#e5e7eb", width=1)
        draw.text((15, y - 10), f"{value / 1_000_000:.1f}M", fill="#374151", font=small_font)

    bar_width = max(1, (right - left) / len(values))
    for i, value in enumerate(values):
        x0 = left + i * bar_width
        x1 = x0 + max(1, bar_width * 0.72)
        y = bottom - (float(value) / max_value) * (bottom - top)
        draw.rectangle((x0, y, x1, bottom), fill="#60a5fa")

    if overlay is not None:
        x_positions = {
            pd.Timestamp(date): left + i * (right - left) / max(len(values) - 1, 1)
            for i, date in enumerate(pd.Index(pd.to_datetime(values.index)))
        }
        points = _series_points(overlay, x_positions, 0, max_value, (left, top, right, bottom))
        if len(points) >= 2:
            draw.line(points, fill="#dc2626", width=4)

    tick_dates = [values.index[0], values.index[len(values) // 2], values.index[-1]]
    for date in tick_dates:
        x = left + list(values.index).index(date) * (right - left) / max(len(values) - 1, 1)
        draw.text((x - 46, bottom + 16), pd.Timestamp(date).strftime("%Y-%m-%d"), fill="#374151", font=small_font)

    image.save(path)


def build_dataset() -> pd.DataFrame:
    DATA_PATH.parent.mkdir(exist_ok=True)
    df = make_synthetic(n=756, seed=42, regime="trend")
    df.to_csv(DATA_PATH, float_format="%.6f")
    return load_data(source="csv", path=DATA_PATH)


def build_analysis(df: pd.DataFrame) -> dict[str, object]:
    base = add_volume_avg(add_volatility(add_returns(df), window=20), window=20)
    ma_df = ma_crossover(base, short=20, long=50)
    ma_trades, ma_equity = simulate_market(ma_df, cash=STARTING_CASH, spread=SPREAD)
    ma_bh = buy_and_hold(ma_df, cash=STARTING_CASH)
    ma_metrics = metrics(ma_equity, ma_trades, ma_df)

    mr_df = mean_reversion(base, window=20, z=1.35)
    mr_trades, mr_equity = simulate_market(mr_df, cash=STARTING_CASH, spread=SPREAD)
    mr_metrics = metrics(mr_equity, mr_trades, mr_df)

    spread_df = apply_spread(ma_df, SPREAD)
    no_fill_limit = float(spread_df["ask"].min() - 1.0)
    no_fill_trades, no_fill_equity = simulate_limit(spread_df, no_fill_limit, side="buy", cash=STARTING_CASH)
    fill_limit = float(spread_df["ask"].quantile(0.10))
    fill_trades, fill_equity = simulate_limit(spread_df, fill_limit, side="buy", cash=STARTING_CASH)

    return {
        "base": base,
        "ma_df": ma_df,
        "ma_trades": ma_trades,
        "ma_equity": ma_equity,
        "ma_bh": ma_bh,
        "ma_metrics": ma_metrics,
        "mr_df": mr_df,
        "mr_trades": mr_trades,
        "mr_equity": mr_equity,
        "mr_metrics": mr_metrics,
        "no_fill_limit": no_fill_limit,
        "no_fill_trades": no_fill_trades,
        "no_fill_equity": no_fill_equity,
        "fill_limit": fill_limit,
        "fill_trades": fill_trades,
        "fill_equity": fill_equity,
    }


def build_plots(analysis: dict[str, object]) -> dict[str, Path]:
    PLOTS_DIR.mkdir(exist_ok=True)
    ma_df = analysis["ma_df"]
    ma_equity = analysis["ma_equity"]
    ma_bh = analysis["ma_bh"]
    mr_equity = analysis["mr_equity"]

    assert isinstance(ma_df, pd.DataFrame)
    assert isinstance(ma_equity, pd.Series)
    assert isinstance(ma_bh, pd.Series)
    assert isinstance(mr_equity, pd.Series)

    plots = {
        "price": PLOTS_DIR / "price_ma_signals.png",
        "volume": PLOTS_DIR / "volume_analysis.png",
        "volatility": PLOTS_DIR / "rolling_volatility.png",
        "equity": PLOTS_DIR / "equity_vs_buy_hold.png",
        "cumulative": PLOTS_DIR / "cumulative_returns.png",
        "strategy": PLOTS_DIR / "strategy_comparison.png",
    }

    buy_dates = ma_df.index[ma_df["signal"] == 1]
    sell_dates = ma_df.index[ma_df["signal"] == -1]
    draw_line_chart(
        plots["price"],
        "Close Price, Moving Averages, and Signals",
        {
            "Close": ma_df["close"],
            "Short MA": ma_df["ma_short"],
            "Long MA": ma_df["ma_long"],
        },
        "Price",
        markers=[
            {"dates": buy_dates, "values": ma_df["close"], "color": "#16a34a", "shape": "up"},
            {"dates": sell_dates, "values": ma_df["close"], "color": "#dc2626", "shape": "down"},
        ],
    )
    draw_bar_chart(plots["volume"], "Volume and Rolling Average", ma_df["volume"], ma_df["volume_avg"])
    draw_line_chart(
        plots["volatility"],
        "Rolling Volatility",
        {"20-day volatility": ma_df["vol"]},
        "Daily return standard deviation",
        width=1200,
        height=560,
    )
    draw_line_chart(
        plots["equity"],
        "Strategy Equity vs Buy and Hold",
        {"MA crossover": ma_equity, "Buy and hold": ma_bh},
        "Portfolio value",
    )
    cumulative_strategy = ma_equity / ma_equity.iloc[0] - 1
    cumulative_bh = ma_bh / ma_bh.iloc[0] - 1
    draw_line_chart(
        plots["cumulative"],
        "Cumulative Returns",
        {"MA crossover": cumulative_strategy, "Buy and hold": cumulative_bh},
        "Cumulative return",
    )
    draw_line_chart(
        plots["strategy"],
        "Strategy Comparison",
        {"MA crossover": ma_equity, "Mean reversion": mr_equity, "Buy and hold": ma_bh},
        "Portfolio value",
    )
    return plots


def write_metrics_csv(analysis: dict[str, object]) -> Path:
    path = REPORT_DIR / "metrics_summary.csv"
    REPORT_DIR.mkdir(exist_ok=True)
    ma_metrics = analysis["ma_metrics"]
    mr_metrics = analysis["mr_metrics"]
    ma_bh = analysis["ma_bh"]
    assert isinstance(ma_metrics, dict)
    assert isinstance(mr_metrics, dict)
    assert isinstance(ma_bh, pd.Series)

    rows = [
        {
            "strategy": "Moving Average Crossover",
            "final_equity": ma_metrics["final_equity"],
            "total_return": ma_metrics["strategy_return"],
            "total_trades": ma_metrics["total_trades"],
            "winning_trades": ma_metrics["winning_trades"],
            "losing_trades": ma_metrics["losing_trades"],
            "sharpe": ma_metrics["sharpe"],
            "max_drawdown": ma_metrics["max_drawdown"],
        },
        {
            "strategy": "Mean Reversion",
            "final_equity": mr_metrics["final_equity"],
            "total_return": mr_metrics["strategy_return"],
            "total_trades": mr_metrics["total_trades"],
            "winning_trades": mr_metrics["winning_trades"],
            "losing_trades": mr_metrics["losing_trades"],
            "sharpe": mr_metrics["sharpe"],
            "max_drawdown": mr_metrics["max_drawdown"],
        },
        {
            "strategy": "Buy and Hold",
            "final_equity": float(ma_bh.iloc[-1]),
            "total_return": float(ma_bh.iloc[-1] / ma_bh.iloc[0] - 1),
            "total_trades": 1,
            "winning_trades": "",
            "losing_trades": "",
            "sharpe": "",
            "max_drawdown": float((ma_bh / ma_bh.cummax() - 1).min()),
        },
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _image_markdown(path: Path, alt: str) -> str:
    rel = Path("..") / path.relative_to(ROOT)
    return f"![{alt}]({rel.as_posix()})"


def build_notebook(stats: dict[str, object], analysis: dict[str, object], plots: dict[str, Path]) -> Path:
    NOTEBOOK_DIR.mkdir(exist_ok=True)
    path = NOTEBOOK_DIR / "project_notebook.ipynb"
    ma_metrics = analysis["ma_metrics"]
    mr_metrics = analysis["mr_metrics"]
    ma_bh = analysis["ma_bh"]
    ma_trades = analysis["ma_trades"]
    no_fill_trades = analysis["no_fill_trades"]
    fill_trades = analysis["fill_trades"]
    assert isinstance(ma_metrics, dict)
    assert isinstance(mr_metrics, dict)
    assert isinstance(ma_bh, pd.Series)
    assert isinstance(ma_trades, pd.DataFrame)
    assert isinstance(no_fill_trades, pd.DataFrame)
    assert isinstance(fill_trades, pd.DataFrame)

    summary_text = (
        f"Rows: {stats['rows']}\n"
        f"Date range: {stats['start_date']} to {stats['end_date']}\n"
        f"Mean close: ${float(stats['close_mean']):,.2f}\n"
        f"Average volume: {float(stats['volume_mean']):,.0f}\n"
    )
    strategy_text = (
        f"MA final equity: {_format_money(float(ma_metrics['final_equity']))}\n"
        f"MA return: {_format_pct(float(ma_metrics['strategy_return']))}\n"
        f"Mean Reversion final equity: {_format_money(float(mr_metrics['final_equity']))}\n"
        f"Mean Reversion return: {_format_pct(float(mr_metrics['strategy_return']))}\n"
        f"Buy and Hold final equity: {_format_money(float(ma_bh.iloc[-1]))}\n"
        f"Market trades: {len(ma_trades)}\n"
    )
    limit_text = (
        f"Limit no-fill trades: {len(no_fill_trades)}\n"
        f"Limit fill trades: {len(fill_trades)}\n"
    )

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Simplified Quant Trading System\n",
                "\n",
                "This notebook runs the complete workflow: market data processing, indicators, trading signals, execution simulation, and performance evaluation.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": 1,
            "metadata": {},
            "outputs": [{"name": "stdout", "output_type": "stream", "text": [summary_text]}],
            "source": [
                "from pathlib import Path\n",
                "import sys\n",
                "\n",
                "ROOT = Path.cwd().parent if Path.cwd().name == 'notebook' else Path.cwd()\n",
                "sys.path.insert(0, str(ROOT))\n",
                "\n",
                "from src.data_loader import load_data, compute_stats\n",
                "from src.indicators import add_returns, add_volatility, add_volume_avg\n",
                "from src.signals import ma_crossover, mean_reversion\n",
                "from src.execution import simulate_market, simulate_limit, apply_spread\n",
                "from src.evaluate import buy_and_hold, metrics\n",
                "\n",
                "STARTING_CASH = 10_000\n",
                "SPREAD = 0.10\n",
                "df = load_data(source='csv', path=ROOT / 'data' / 'synthetic_ohlcv.csv')\n",
                "stats = compute_stats(df)\n",
                "print(f\"Rows: {stats['rows']}\")\n",
                "print(f\"Date range: {stats['start_date']} to {stats['end_date']}\")\n",
                "print(f\"Mean close: ${stats['close_mean']:,.2f}\")\n",
                "print(f\"Average volume: {stats['volume_mean']:,.0f}\")\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Market Data Processing\n",
                "\n",
                "The dataset follows the OHLCV contract and uses business-day timestamps. Missing or invalid values are cleaned before indicators are calculated.\n",
                "\n",
                _image_markdown(plots["volume"], "Volume analysis"),
            ],
        },
        {
            "cell_type": "code",
            "execution_count": 2,
            "metadata": {},
            "outputs": [{"name": "stdout", "output_type": "stream", "text": [strategy_text]}],
            "source": [
                "base = add_volume_avg(add_volatility(add_returns(df), window=20), window=20)\n",
                "ma_df = ma_crossover(base, short=20, long=50)\n",
                "ma_trades, ma_equity = simulate_market(ma_df, cash=STARTING_CASH, spread=SPREAD)\n",
                "bh_equity = buy_and_hold(ma_df, cash=STARTING_CASH)\n",
                "ma_results = metrics(ma_equity, ma_trades, ma_df)\n",
                "\n",
                "mr_df = mean_reversion(base, window=20, z=1.35)\n",
                "mr_trades, mr_equity = simulate_market(mr_df, cash=STARTING_CASH, spread=SPREAD)\n",
                "mr_results = metrics(mr_equity, mr_trades, mr_df)\n",
                "\n",
                "print(f\"MA final equity: ${ma_results['final_equity']:,.2f}\")\n",
                "print(f\"MA return: {ma_results['strategy_return']:.2%}\")\n",
                "print(f\"Mean Reversion final equity: ${mr_results['final_equity']:,.2f}\")\n",
                "print(f\"Mean Reversion return: {mr_results['strategy_return']:.2%}\")\n",
                "print(f\"Buy and Hold final equity: ${bh_equity.iloc[-1]:,.2f}\")\n",
                "print(f\"Market trades: {len(ma_trades)}\")\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Indicators and Signals\n",
                "\n",
                "The Moving Average Crossover strategy buys when the short moving average rises above the long moving average and sells when it falls back below. The Mean Reversion strategy buys when price is far below a rolling mean and exits when price is far above it.\n",
                "\n",
                _image_markdown(plots["price"], "Price with moving averages and signals"),
                "\n\n",
                _image_markdown(plots["volatility"], "Rolling volatility"),
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Execution and Evaluation\n",
                "\n",
                "Market orders fill at the ask for BUY orders and at the bid for SELL orders. The strategy shifts position by one bar so a signal observed today can only trade on the next available bar.\n",
                "\n",
                _image_markdown(plots["equity"], "Equity vs buy and hold"),
                "\n\n",
                _image_markdown(plots["cumulative"], "Cumulative returns"),
                "\n\n",
                _image_markdown(plots["strategy"], "Strategy comparison"),
            ],
        },
        {
            "cell_type": "code",
            "execution_count": 3,
            "metadata": {},
            "outputs": [{"name": "stdout", "output_type": "stream", "text": [limit_text]}],
            "source": [
                "spread_df = apply_spread(ma_df, SPREAD)\n",
                "no_fill_limit = spread_df['ask'].min() - 1\n",
                "no_fill_trades, _ = simulate_limit(spread_df, no_fill_limit, side='buy', cash=STARTING_CASH)\n",
                "fill_limit = spread_df['ask'].quantile(0.10)\n",
                "fill_trades, _ = simulate_limit(spread_df, fill_limit, side='buy', cash=STARTING_CASH)\n",
                "\n",
                "print(f'Limit no-fill trades: {len(no_fill_trades)}')\n",
                "print(f'Limit fill trades: {len(fill_trades)}')\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Observations\n",
                "\n",
                "Mean Reversion led this sample, while Moving Average Crossover finished ahead of Buy and Hold because it avoided some weaker periods. The passive baseline stayed fully invested and carried the deepest drawdown. Limit orders illustrate execution uncertainty because an order can remain unfilled when the ask never reaches the chosen limit price.\n",
            ],
        },
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    return path


def _pdf_footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(colors.HexColor("#4b5563"))
    canvas.drawRightString(A4[0] - 0.55 * inch, 0.35 * inch, f"Page {doc.page}")
    canvas.restoreState()


def build_pdf(stats: dict[str, object], analysis: dict[str, object], plots: dict[str, Path]) -> Path:
    REPORT_DIR.mkdir(exist_ok=True)
    path = REPORT_DIR / "final_report.pdf"
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="SectionTitle", parent=styles["Heading2"], spaceBefore=12, spaceAfter=8, textColor=colors.HexColor("#14213d")))
    styles.add(ParagraphStyle(name="BodyTight", parent=styles["BodyText"], leading=14, spaceAfter=7))

    ma_metrics = analysis["ma_metrics"]
    mr_metrics = analysis["mr_metrics"]
    ma_bh = analysis["ma_bh"]
    ma_trades = analysis["ma_trades"]
    no_fill_limit = float(analysis["no_fill_limit"])
    no_fill_trades = analysis["no_fill_trades"]
    fill_limit = float(analysis["fill_limit"])
    fill_trades = analysis["fill_trades"]
    assert isinstance(ma_metrics, dict)
    assert isinstance(mr_metrics, dict)
    assert isinstance(ma_bh, pd.Series)
    assert isinstance(ma_trades, pd.DataFrame)
    assert isinstance(no_fill_trades, pd.DataFrame)
    assert isinstance(fill_trades, pd.DataFrame)

    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
    )

    story = []
    story.append(Paragraph("Simplified Quant Trading System", styles["Title"]))
    story.append(Paragraph("Final Project Report", styles["Heading3"]))
    story.append(Spacer(1, 0.12 * inch))

    story.append(Paragraph("1. Project Overview", styles["SectionTitle"]))
    story.append(Paragraph("This project builds a simplified quantitative trading workflow: process OHLCV data, calculate indicators, generate trading signals, simulate market and limit order execution, and evaluate performance against a buy-and-hold baseline.", styles["BodyTight"]))

    story.append(Paragraph("2. Dataset Description", styles["SectionTitle"]))
    story.append(Paragraph(f"The submitted dataset is a reproducible synthetic OHLCV series with {stats['rows']} business-day rows from {stats['start_date']} to {stats['end_date']}. Synthetic data keeps the project runnable without network access while preserving realistic open, high, low, close, and volume structure.", styles["BodyTight"]))

    story.append(Paragraph("3. Indicators Used", styles["SectionTitle"]))
    story.append(Paragraph("The workflow calculates simple returns, cumulative returns, 20-day and 50-day moving averages, 20-day rolling volatility, and 20-day average volume. Moving averages smooth price trends, volatility estimates changing risk, and volume average highlights market activity.", styles["BodyTight"]))

    story.append(Paragraph("4. Trading Strategy Logic", styles["SectionTitle"]))
    story.append(Paragraph("The main strategy is a Moving Average Crossover: BUY when the 20-day moving average is above the 50-day moving average, and SELL when it is not. A Mean Reversion strategy is also included: BUY when price is materially below its 20-day rolling mean and SELL when price is materially above it.", styles["BodyTight"]))

    story.append(Paragraph("5. Execution Logic", styles["SectionTitle"]))
    story.append(Paragraph(f"Market orders use a fixed ${SPREAD:.2f} spread: bid equals close minus half the spread, and ask equals close plus half the spread. BUY orders fill at ask and SELL orders fill at bid. Limit BUY orders fill only when ask is less than or equal to the limit price; limit SELL orders fill only when bid is greater than or equal to the limit price.", styles["BodyTight"]))
    story.append(Paragraph("To avoid lookahead bias, signals are shifted by one bar before execution. A signal observed on day t can only affect trading on day t+1.", styles["BodyTight"]))
    story.append(Paragraph(f"Limit order demonstration: a BUY limit at ${no_fill_limit:.2f} produced {len(no_fill_trades)} fills, while a BUY limit at ${fill_limit:.2f} produced {len(fill_trades)} fill.", styles["BodyTight"]))

    table_data = [
        ["Strategy", "Final Equity", "Return", "Trades", "Wins", "Losses", "Sharpe", "Max DD"],
        [
            "MA Crossover",
            _format_money(float(ma_metrics["final_equity"])),
            _format_pct(float(ma_metrics["strategy_return"])),
            str(ma_metrics["total_trades"]),
            str(ma_metrics["winning_trades"]),
            str(ma_metrics["losing_trades"]),
            f"{float(ma_metrics['sharpe']):.2f}",
            _format_pct(float(ma_metrics["max_drawdown"])),
        ],
        [
            "Mean Reversion",
            _format_money(float(mr_metrics["final_equity"])),
            _format_pct(float(mr_metrics["strategy_return"])),
            str(mr_metrics["total_trades"]),
            str(mr_metrics["winning_trades"]),
            str(mr_metrics["losing_trades"]),
            f"{float(mr_metrics['sharpe']):.2f}",
            _format_pct(float(mr_metrics["max_drawdown"])),
        ],
        [
            "Buy and Hold",
            _format_money(float(ma_bh.iloc[-1])),
            _format_pct(float(ma_bh.iloc[-1] / ma_bh.iloc[0] - 1)),
            "1",
            "-",
            "-",
            "-",
            _format_pct(float((ma_bh / ma_bh.cummax() - 1).min())),
        ],
    ]
    story.append(Paragraph("6. Results and Performance", styles["SectionTitle"]))
    table = Table(table_data, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#14213d")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d1d5db")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 0.12 * inch))
    story.append(Paragraph("Mean Reversion performed best on this sample, while Moving Average Crossover finished ahead of Buy and Hold by avoiding some weaker periods. The passive baseline stayed fully invested and carried the deepest drawdown.", styles["BodyTight"]))

    story.append(PageBreak())
    story.append(Paragraph("7. Charts and Visualizations", styles["SectionTitle"]))
    for key in ("price", "equity", "cumulative"):
        story.append(PdfImage(str(plots[key]), width=7.0 * inch, height=4.1 * inch))
        story.append(Spacer(1, 0.14 * inch))

    story.append(PageBreak())
    story.append(Paragraph("Additional Market Diagnostics", styles["SectionTitle"]))
    for key in ("volume", "volatility", "strategy"):
        story.append(PdfImage(str(plots[key]), width=7.0 * inch, height=3.55 * inch))
        story.append(Spacer(1, 0.12 * inch))

    story.append(Paragraph("8. Key Observations", styles["SectionTitle"]))
    story.append(Paragraph("The crossover strategy works best when trends persist after the short average crosses the long average. In choppy regions, crossovers can arrive late and create whipsaw trades. The mean reversion rule reacts differently by buying weakness, which can help in sideways markets but can struggle during persistent downtrends.", styles["BodyTight"]))

    story.append(Paragraph("9. Limitations", styles["SectionTitle"]))
    story.append(Paragraph("The system uses one synthetic asset, fixed spread, fractional shares, and in-sample parameter choices. It does not model market impact, taxes, borrow costs, order queue priority, or changing liquidity. Results should be treated as a research exercise rather than a live trading recommendation.", styles["BodyTight"]))

    story.append(Paragraph("10. Conclusion", styles["SectionTitle"]))
    story.append(Paragraph("The project demonstrates the full quantitative trading pipeline from data preparation to signal logic, execution, and evaluation. The most important correctness control is the one-bar execution delay, which prevents the strategy from using future information.", styles["BodyTight"]))

    doc.build(story, onFirstPage=_pdf_footer, onLaterPages=_pdf_footer)
    return path


def build_all() -> None:
    for directory in (PLOTS_DIR, REPORT_DIR, NOTEBOOK_DIR):
        directory.mkdir(exist_ok=True)
    df = build_dataset()
    stats = compute_stats(df)
    analysis = build_analysis(df)
    plots = build_plots(analysis)
    write_metrics_csv(analysis)
    build_notebook(stats, analysis, plots)
    build_pdf(stats, analysis, plots)
    print("Artifacts built successfully.")


if __name__ == "__main__":
    build_all()
