# Quantitative Portfolio Analysis & Risk Toolkit: Comprehensive Report

## Executive Summary
This report details the development and functionality of a Python-based Quantitative Finance Toolkit designed for the analysis of the Taiwan stock market. The project aims to provide professional-grade financial analytics, transitioning from basic return calculations to advanced quantitative methodologies. Key capabilities include comprehensive risk and performance metric calculations, portfolio optimization using Modern Portfolio Theory (MPT), and predictive forecasting via Monte Carlo simulations. The system is accessible through both a Command Line Interface (CLI) and an interactive web dashboard built with Streamlit.

## 1. Introduction
In modern financial analysis, relying solely on historical returns is insufficient. Investors require sophisticated tools to understand the risk-adjusted performance of their portfolios and to project potential future outcomes. This project was developed to create a robust, end-to-end pipeline that retrieves financial data, computes standard and advanced risk metrics, evaluates company fundamentals, and provides actionable insights through portfolio optimization and simulation.

## 2. System Architecture & Technologies
The project is structured modularly to ensure maintainability and scalability.
* **Language:** Python 3
* **Data Ingestion:** `yfinance` for retrieving historical market data and financial statements.
* **Core Computation:** `pandas` and `numpy` for vectorized mathematical and statistical operations.
* **Optimization Engine:** `scipy.optimize` utilizing the Sequential Least SQuares Programming (SLSQP) algorithm.
* **Visualization:** 
  * `matplotlib` for static equity curves and distribution plots in the CLI.
  * `plotly` for highly interactive, dynamic charts in the web dashboard.
* **User Interfaces:** `streamlit` for the frontend web application.

## 3. Methodology & Calculations

### 3.1 Historical Data Retrieval
The system uses the `fetch_data.py` module to download daily adjusted closing prices for selected tickers (e.g., `2330.TW` for TSMC) and a benchmark index (`^TWII`). 

### 3.2 Return Calculations
Daily percentage returns are calculated using the adjusted close prices. These discrete daily returns form the foundation for all subsequent risk and performance metrics.

### 3.3 Risk Assessment Metrics
* **Value at Risk (VaR):** Calculates the threshold loss value that the portfolio will not exceed with a certain confidence level (default 95%). Implemented using the historical simulation method (percentiles).
* **Conditional Value at Risk (CVaR / Expected Shortfall):** Calculates the expected loss *given* that the VaR threshold has been breached. It averages the worst 5% of historical returns, providing a more conservative risk measure than VaR.
* **Maximum Drawdown (MDD):** Measures the largest single drop from peak to trough in the portfolio's value, representing the historical worst-case scenario for an investor buying at the top.
* **Annualized Volatility:** The standard deviation of daily returns scaled to an annual timeframe (multiplying by $\sqrt{252}$).
* **Beta:** Measures the portfolio's volatility and systemic risk relative to the broader market benchmark (`^TWII`). Calculated as the covariance of the portfolio and benchmark divided by the variance of the benchmark.

### 3.4 Performance Metrics
* **Sharpe Ratio:** Measures the excess return (return minus risk-free rate) per unit of total risk (volatility). A higher Sharpe Ratio indicates better risk-adjusted performance.
* **Sortino Ratio:** Similar to the Sharpe ratio, but it only penalizes downside volatility, making it a preferred metric for non-normally distributed returns.
* **Alpha:** Represents the active return on an investment compared to a market index. It is calculated using the Capital Asset Pricing Model (CAPM) framework.

### 3.5 Fundamental Analysis (Piotroski F-Score)
To blend quantitative price analysis with fundamental strength, the project calculates a discrete Piotroski F-Score based on real financial data:
1. Positive Net Income (Profitability)
2. Positive Operating Cash Flow (Liquidity)
3. Operating Cash Flow greater than Net Income (Earnings Quality)
This provides a 0-3 score reflecting the underlying health of the company.

## 4. Advanced Quantitative Features

### 4.1 Portfolio Optimization
The `optimize.py` module implements Modern Portfolio Theory. It aims to find the specific allocation of weights across the chosen assets that maximizes the portfolio's Sharpe Ratio. 
* **Objective Function:** Minimize the negative Sharpe Ratio.
* **Constraints:** All weights must sum to 1.0 (100% investment).
* **Bounds:** Individual asset weights must be between 0.0 and 1.0 (long-only portfolio, no short selling).
The SLSQP algorithm iteratively tests weight combinations until it converges on the mathematically optimal portfolio.

### 4.2 Monte Carlo Simulation
To forecast future portfolio values, the `monte_carlo.py` module runs simulations using **Geometric Brownian Motion (GBM)**.
* The model assumes returns follow a normal distribution defined by the portfolio's historical mean ($\mu$) and standard deviation ($\sigma$).
* It generates thousands of random paths (default: 1000) over a specified future horizon (default: 252 trading days).
* The output provides a probabilistic range of outcomes, highlighting the median expected value, as well as the 5th and 95th percentiles (defining a 90% confidence interval).

## 5. Application Interfaces

### 5.1 Command Line Interface (`main.py`)
The CLI provides an interactive, terminal-based experience. Upon execution, it presents a menu of available Taiwan stocks, allows for custom ticker inputs, and processes the analysis. The output is printed cleanly to the console, static graphs are displayed, and a structured CSV report is generated for record-keeping.

### 5.2 Streamlit Web Dashboard (`app.py`)
The web dashboard elevates the project to a professional product. 
* **Dynamic Inputs:** Users can easily adjust tickers, confidence levels, forecast horizons, and choose whether to auto-optimize weights via a sidebar.
* **KPI Summaries:** Critical metrics are displayed in easily digestible KPI cards.
* **Interactive Visualizations:** Plotly is used to render an interactive equity curve and a shaded confidence "cone" for the Monte Carlo simulations, allowing users to hover and inspect exact values at any point in time.

## 6. Conclusion
The Quantitative Portfolio Analysis Toolkit successfully integrates financial data retrieval, rigorous statistical risk measurement, machine-driven optimization, and stochastic forecasting into a cohesive application. By providing both a programmatic CLI and a highly interactive graphical dashboard, it serves as a powerful utility for deep financial analysis and portfolio management.
