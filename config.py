"""Configuration for the TW stock risk project."""

# Stocks to analyze (TW tickers)
STOCKS = ["2330.TW", "2317.TW", "2454.TW"]

# Visualization
VISUALIZE = True

# VaR/CVaR configuration
CONFIDENCE = 0.95  # 95% VaR/CVaR

# Horizon in years (currently for metadata/export only)
HORIZON_YEARS = 5

# CSV export
EXPORT_CSV = True
CSV_PATH = "data/exports/var_cvar_results.csv"

# Advanced Portfolio Configuration
PORTFOLIO_WEIGHTS = [0.34, 0.33, 0.33]  # Equal weighting for 3 stocks
OPTIMIZE_WEIGHTS = False  # Set to True to calculate and use optimal Sharpe Ratio weights
RISK_FREE_RATE = 0.02  # 2% annual risk-free rate for Sharpe ratio

# Benchmark Configuration
BENCHMARK_TICKER = "^TWII"  # Taiwan Capitalization Weighted Stock Index
