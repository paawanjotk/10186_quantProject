# Quantitative Modelling Project

# Build a Simplified Quant Trading System

Quantitative Modelling Program | Final Project

---

# 1\. Project Overview

## 1.1 Background

Modern trading firms use quantitative models, market data, and automated systems to make trading decisions.

These systems process financial data and use statistical techniques to:

* generate trading signals,

* identify opportunities,

* simulate execution,

* and evaluate strategy performance.

In this project, students will build a simplified quantitative trading workflow using concepts covered during the course.

The objective is not to build a production trading system, but to understand:

* how quantitative trading workflows operate,

* how market data is processed,

* how trading decisions are modeled systematically,

* and how trading strategies are evaluated.

---

## 1.2 What Will You Build?

Students will build a simplified trading research and execution framework.

The project will include:

| Module | Description |
| :---- | :---- |
| Market Data Layer | Process OHLCV market data |
| Analytics Layer | Compute returns and indicators |
| Signal Engine | Generate trading signals |
| Execution Layer | Simulate market and limit orders |
| Evaluation Layer | Analyze strategy performance |

---

## 1.3 Recommended Technologies

Students are expected to use:

* Python 3

* Jupyter Notebook or Google Colab

Recommended Libraries:

* NumPy

* Pandas

* Matplotlib

Optional Libraries:

* Seaborn

* Plotly

* yfinance

---

# 2\. Dataset Requirements

Students may:

* use publicly available financial datasets,

* generate synthetic datasets,

* or use APIs such as Yahoo Finance.

Recommended assets:

* Stocks

* ETFs

* Indices

* Crypto assets

Minimum dataset requirements:

* Open

* High

* Low

* Close

* Volume

* Timestamp/Date

---

# 3\. Problem Statement 1

# Market Data Processing

---

## 3.1 What You Need To Do

Students must:

1. Load OHLCV market data

2. Store data using Pandas DataFrames

3. Clean missing or invalid values

4. Visualize price movement

5. Compute basic statistics

---

## 3.2 Required Tasks

| Task | Description |
| :---- | :---- |
| Load Dataset | Read CSV/API data into DataFrame |
| Data Inspection | Display rows, columns, and datatypes |
| Missing Values | Detect and handle missing values |
| Price Visualization | Plot close prices |
| Volume Analysis | Visualize trading volume |

---

## 3.3 Expected Concepts

Students should understand:

* OHLCV structure

* Financial time series

* Trading intervals

* Market activity visualization

---

# 4\. Problem Statement 2

# Financial Indicators & Trading Signals

---

## 4.1 What You Need To Do

Students must compute:

* returns,

* moving averages,

* rolling statistics,

* and trading indicators.

Students should then generate BUY/SELL signals.

---

## 4.2 Required Indicators

| Indicator | Description |
| :---- | :---- |
| Simple Returns | Percentage price movement |
| Cumulative Returns | Running total return |
| Moving Average | Trend smoothing |
| Rolling Volatility | Market fluctuation |
| Volume Average | Trading activity trend |

---

## 4.3 Signal Logic

Students must implement at least one trading strategy.

Suggested strategies:

### Strategy A — Moving Average Crossover

BUY: Short MA \> Long MA

SELL: Short MA \<= Long MA

---

### Strategy B — Mean Reversion

BUY: Price significantly below rolling mean

SELL: Price significantly above rolling mean

---

## 4.4 Expected Output

Students should:

* visualize indicators,

* plot BUY/SELL points,

* explain strategy behavior,

* and discuss market trends.

---

# 5\. Problem Statement 3

# Trade Execution Simulation

---

## 5.1 What You Need To Do

Students must simulate:

* market orders,

* limit orders,

* and execution logic.

---

## 5.2 Market Order Rules

| Order Type | Execution Price |
| :---- | :---- |
| Market BUY | Ask Price |
| Market SELL | Bid Price |

Suggested spread model:

Bid \= Close − spread/2

Ask \= Close \+ spread/2

---

## 5.3 Limit Order Rules

BUY limit order executes only if:

Ask Price \<= Limit Price

SELL limit order executes only if:

Bid Price \>= Limit Price

---

## 5.4 Expected Concepts

Students should understand:

* bid/ask spread,

* execution price,

* transaction cost,

* liquidity,

* and execution uncertainty.

---

# 6\. Problem Statement 4

# Strategy Evaluation

---

## 6.1 What You Need To Do

Students must evaluate strategy performance.

---

## 6.2 Required Metrics

| Metric | Description |
| :---- | :---- |
| Total Trades | Number of executed trades |
| Winning Trades | Profitable trades |
| Losing Trades | Loss-making trades |
| Strategy Returns | Total strategy performance |
| Buy & Hold Comparison | Compare against passive investment |

---

## 6.3 Visualization Requirements

Students are encouraged to include:

* price charts,

* moving averages,

* volatility plots,

* signal markers,

* cumulative return charts,

* and volume charts.

---

# 7\. Submission Guidelines

Students must submit the following:

## Mandatory Deliverables

1. Source Code

2. Jupyter Notebook / Google Colab Notebook

3. Final Report (PDF format preferred)

4. Charts / Visualizations used in the project

---

## 7.1 Submission Format

Students should submit a single ZIP file containing all project files.

Suggested folder structure:

text id="q1yfgf" RollNumber1\_RollNumber2\_RollNumber3\_RollNumber4\_RollNumber5\_QuantProject/ │ ├── notebook/ │   └── project\_notebook.ipynb │ ├── src/ │   └── python\_files.py │ ├── data/ │   └── dataset.csv │ ├── plots/ │   └── charts\_and\_visualizations │ ├── report/ │   └── final\_report.pdf │ └── README.md

---

## 7.2 Naming Convention

ZIP File Name:

text id="t0kk2f" RollNumber1\_RollNumber2\_RollNumber3\_RollNumber4\_RollNumber5\_QuantProject.zip

Example:

text id="5vpxdg" 220045\_220067\_220089\_220101\_220115\_QuantProject.zip

---

## 7.3 Notebook Requirements

The notebook should contain:

* Proper section headings

* Clean code structure

* Comments where necessary

* Visualizations with labels/titles

* Explanation of strategy logic

* Output cells visible before submission

Students are encouraged to keep notebooks clean and readable.

---

## 7.4 Report Requirements

The final report should briefly include:

1. Project Overview

2. Dataset Description

3. Indicators Used

4. Trading Strategy Logic

5. Execution Logic

6. Results & Performance

7. Charts & Visualizations

8. Key Observations

9. Limitations

10. Conclusion

The report does not need to be very long.

Focus on:

* clarity,

* explanation,

* and understanding of concepts.

---

## 7.5 Code Requirements

Students should ensure:

* code runs without errors,

* unnecessary files are removed,

* variables are meaningfully named,

* duplicate code is minimized,

* and outputs are reproducible.

---

## 7.6 Academic Integrity

Students may discuss ideas and concepts with peers.

However:

* submitted code must be original,

* direct copying is not allowed,

* and plagiarism may lead to disqualification.

---

## 7.7 Submission Notes

* Ensure all required files are included before submission.

* Verify notebook outputs before exporting/submitting.

* Missing files may affect evaluation.

* Students are encouraged to test notebooks on a fresh runtime before submission.

---

## 7.8 Submission Guidelines

Submission Deadline: 24 June 2026

Students must work in groups of 5 members.

Only one submission per group is required.

Naming convention should use all group roll numbers separated by underscores.

**Submission Link \- [https://forms.gle/EpN9eaPwGf2JwwrP7](https://forms.gle/EpN9eaPwGf2JwwrP7)**

# 8\. Report Requirements

The final report should include:

1. Project Overview

2. Dataset Description

3. Trading Logic

4. Indicators Used

5. Signal Explanation

6. Execution Logic

7. Results & Analysis

8. Visualizations

9. Key Observations

10. Conclusion

Students are encouraged to explain:

* why certain indicators were chosen,

* how signals behaved,

* and what observations were made from the results.

---

# 9\. Evaluation Criteria

| Component | Weightage |
| :---- | :---- |
| Correctness | 30% |
| Strategy Logic | 25% |
| Code Quality | 15% |
| Visualization & Analysis | 15% |
| Documentation | 15% |

---

# 10\. Optional Extensions

Students may additionally implement:

* multiple trading strategies,

* stop-loss logic,

* volatility filters,

* portfolio simulation,

* position sizing,

* signal comparison,

* or machine learning-based signals.

These are optional and not mandatory.

---

# 11\. Important Notes

* Focus on understanding concepts and implementation.

* Clean structure and explanation matter more than notebook length.

* Visualizations and observations are strongly encouraged.

* Students are encouraged to experiment beyond the minimum requirements.

The goal of this project is to build intuition around how quantitative trading systems work in practice.