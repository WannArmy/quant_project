
from .src.fetch_data import get_price_data
from .src.preprocess import calculate_returns
from .src.risk import (
    calculate_var, calculate_cvar, calculate_portfolio_returns, 
    calculate_maximum_drawdown, calculate_sharpe_ratio,
    calculate_annualized_volatility, calculate_sortino_ratio,
    calculate_beta, calculate_alpha
)
from .src.f_score import calculate_f_score, get_real_financials
from .config import (
    STOCKS, VISUALIZE, CONFIDENCE, HORIZON_YEARS, EXPORT_CSV, 
    CSV_PATH, PORTFOLIO_WEIGHTS, RISK_FREE_RATE, OPTIMIZE_WEIGHTS, BENCHMARK_TICKER
)
try:
    from .src.optimize import optimize_portfolio_weights
except ImportError:
    optimize_portfolio_weights = None

import numpy as np
import pandas as pd
import os

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:  # pragma: no cover
    HAS_MPL = False


def plot_returns(returns, ticker: str) -> None:
    if not HAS_MPL:
        return
    plt.figure(figsize=(8, 4))
    plt.hist(returns, bins=50, edgecolor='k', alpha=0.7)
    plt.title(f"Return Distribution for {ticker}")
    plt.xlabel('Daily Return')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()


def plot_equity_curve(all_returns: pd.DataFrame, portfolio_returns: pd.Series) -> None:
    if not HAS_MPL:
        return
    plt.figure(figsize=(10, 5))
    for col in all_returns.columns:
        cumulative = (1 + all_returns[col]).cumprod()
        plt.plot(cumulative, label=col, alpha=0.5)
        
    port_cumulative = (1 + portfolio_returns).cumprod()
    plt.plot(port_cumulative, label='Portfolio', color='black', linewidth=2)
    
    plt.title("Cumulative Returns (Equity Curve)")
    plt.xlabel('Date')
    plt.ylabel('Growth of $1')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

AVAILABLE_STOCKS = {
    "1": ("2330.TW", "TSMC (Taiwan Semiconductor Manufacturing Company)"),
    "2": ("2317.TW", "Hon Hai Precision Industry (Foxconn)"),
    "3": ("2454.TW", "MediaTek Inc."),
    "4": ("2308.TW", "Delta Electronics"),
    "5": ("2881.TW", "Fubon Financial"),
    "6": ("2891.TW", "CTBC Financial"),
    "7": ("2002.TW", "China Steel"),
    "8": ("1301.TW", "Formosa Plastics"),
    "9": ("1303.TW", "Nan Ya Plastics"),
    "10": ("2882.TW", "Cathay Financial"),
}

def prompt_user_for_stocks():
    print("\n--- Available Taiwan Stocks ---")
    for key, (ticker, name) in AVAILABLE_STOCKS.items():
        print(f"[{key}] {ticker} - {name}")
    print("-------------------------------")
    
    print("Enter the stock numbers (e.g., '1, 2, 3') OR enter custom stock codes (e.g., '2603.TW, 2303.TW').")
    print("If you want to analyze multiple stocks, separate them with commas.")
    print("Press Enter to use the default configuration from config.py.")
    
    user_input = input("\nYour choice: ").strip()
    
    if not user_input:
        return None
        
    selections = [s.strip() for s in user_input.split(",") if s.strip()]
    
    final_stocks = []
    for s in selections:
        if s in AVAILABLE_STOCKS:
            final_stocks.append(AVAILABLE_STOCKS[s][0])
        else:
            final_stocks.append(s)
                
    # Remove duplicates but keep order
    final_stocks = list(dict.fromkeys(final_stocks))
    return final_stocks


def main(visualize: bool = VISUALIZE) -> None:
    print("TW Stock Risk Toolkit (5y Daily Returns)\n")
    
    user_stocks = prompt_user_for_stocks()
    if user_stocks:
        stocks = user_stocks
    else:
        stocks = STOCKS or ["2330.TW", "2317.TW", "2454.TW"]
        
    print(f"\nAnalyzing stocks: {', '.join(stocks)}\n")

    benchmark_returns = None
    if BENCHMARK_TICKER:
        print(f"Fetching benchmark data for {BENCHMARK_TICKER}...")
        bench_df = get_price_data(BENCHMARK_TICKER)
        if bench_df is not None and not bench_df.empty:
            benchmark_returns = calculate_returns(bench_df)
            print("Benchmark data loaded successfully.\n")

    results = []
    returns_dict = {}

    for ticker in stocks:
        df = get_price_data(ticker)
        if df is None or df.empty:
            print(f"Warning: No data for {ticker}\n")
            continue

        returns = calculate_returns(df)
        if returns is None or len(returns) == 0:
            print(f"Warning: No returns for {ticker}\n")
            continue
            
        returns_dict[ticker] = returns

        var = calculate_var(returns, confidence=CONFIDENCE)
        cvar = calculate_cvar(returns, confidence=CONFIDENCE)
        mdd = calculate_maximum_drawdown(returns)
        sharpe = calculate_sharpe_ratio(returns, risk_free_rate=RISK_FREE_RATE)
        volatility = calculate_annualized_volatility(returns)
        sortino = calculate_sortino_ratio(returns, risk_free_rate=RISK_FREE_RATE)
        
        beta = float('nan')
        alpha = float('nan')
        if benchmark_returns is not None:
            beta = calculate_beta(returns, benchmark_returns)
            alpha = calculate_alpha(returns, benchmark_returns, risk_free_rate=RISK_FREE_RATE)

        # Real fundamental data for F-Score per ticker
        fin_data = get_real_financials(ticker)
        score = calculate_f_score(fin_data)

        print(f"Stock: {ticker}")
        print(f"VaR ({int(CONFIDENCE*100)}%): {var:.4f}" if not np.isnan(var) else "VaR: NaN")
        print(f"CVaR ({int(CONFIDENCE*100)}%): {cvar:.4f}" if not np.isnan(cvar) else "CVaR: NaN")
        print(f"Sharpe Ratio: {sharpe:.4f}" if not np.isnan(sharpe) else "Sharpe Ratio: NaN")
        print(f"Sortino Ratio: {sortino:.4f}" if not np.isnan(sortino) else "Sortino Ratio: NaN")
        print(f"Annual Volatility: {volatility:.2%}" if not np.isnan(volatility) else "Annual Volatility: NaN")
        print(f"Beta: {beta:.2f}" if not np.isnan(beta) else "Beta: NaN")
        print(f"Alpha: {alpha:.4f}" if not np.isnan(alpha) else "Alpha: NaN")
        print(f"Max Drawdown: {mdd:.2%}" if not np.isnan(mdd) else "Max Drawdown: NaN")
        print(f"F-Score (Real Data) for {ticker}: {score}/3\n")

        if visualize:
            plot_returns(returns, ticker)

        results.append({
            "ticker": ticker,
            "VaR": var,
            "CVaR": cvar,
            "Sharpe": sharpe,
            "Sortino": sortino,
            "Volatility": volatility,
            "Beta": beta,
            "Alpha": alpha,
            "MDD": mdd,
            "F_Score": score,
            "Horizon_Years": HORIZON_YEARS,
            "Confidence": CONFIDENCE,
        })
        
    if returns_dict:
        # Align all returns by date
        all_returns = pd.DataFrame(returns_dict).dropna()
        
        port_weights = PORTFOLIO_WEIGHTS
        if len(port_weights) != len(all_returns.columns):
            print(f"Warning: Configured weights length ({len(port_weights)}) does not match valid stocks ({len(all_returns.columns)}).")
            print("Using equal weights instead.")
            n = len(all_returns.columns)
            port_weights = [1.0 / n] * n

        if OPTIMIZE_WEIGHTS and optimize_portfolio_weights is not None:
            print("Optimizing portfolio weights to maximize Sharpe Ratio...")
            port_weights = optimize_portfolio_weights(all_returns, risk_free_rate=RISK_FREE_RATE)
            print(f"Optimal Weights: {port_weights}\n")

        if not all_returns.empty and len(port_weights) == len(all_returns.columns):
            port_returns = calculate_portfolio_returns(all_returns, port_weights)
            port_var = calculate_var(port_returns, confidence=CONFIDENCE)
            port_cvar = calculate_cvar(port_returns, confidence=CONFIDENCE)
            port_mdd = calculate_maximum_drawdown(port_returns)
            port_sharpe = calculate_sharpe_ratio(port_returns, risk_free_rate=RISK_FREE_RATE)
            port_volatility = calculate_annualized_volatility(port_returns)
            port_sortino = calculate_sortino_ratio(port_returns, risk_free_rate=RISK_FREE_RATE)
            
            port_beta = float('nan')
            port_alpha = float('nan')
            if benchmark_returns is not None:
                port_beta = calculate_beta(port_returns, benchmark_returns)
                port_alpha = calculate_alpha(port_returns, benchmark_returns, risk_free_rate=RISK_FREE_RATE)
            
            print(f"--- Portfolio Level Analysis ---")
            print(f"Weights: {port_weights}")
            print(f"Portfolio VaR ({int(CONFIDENCE*100)}%): {port_var:.4f}")
            print(f"Portfolio CVaR ({int(CONFIDENCE*100)}%): {port_cvar:.4f}")
            print(f"Portfolio Sharpe Ratio: {port_sharpe:.4f}")
            print(f"Portfolio Sortino Ratio: {port_sortino:.4f}")
            print(f"Portfolio Volatility: {port_volatility:.2%}")
            print(f"Portfolio Beta: {port_beta:.2f}")
            print(f"Portfolio Alpha: {port_alpha:.4f}")
            print(f"Portfolio Max Drawdown: {port_mdd:.2%}\n")
            
            results.append({
                "ticker": "PORTFOLIO",
                "VaR": port_var,
                "CVaR": port_cvar,
                "Sharpe": port_sharpe,
                "Sortino": port_sortino,
                "Volatility": port_volatility,
                "Beta": port_beta,
                "Alpha": port_alpha,
                "MDD": port_mdd,
                "F_Score": None,
                "Horizon_Years": HORIZON_YEARS,
                "Confidence": CONFIDENCE,
            })
            
            if visualize:
                plot_equity_curve(all_returns, port_returns)
        else:
            print("Skipping Portfolio Analysis (weights do not match number of valid stocks or empty data).")

    # CSV export
    if EXPORT_CSV:
        export_path = CSV_PATH or "data/exports/var_cvar_results.csv"
        _ensure_dir(export_path)
        if results:
            df_out = pd.DataFrame(results)
            df_out.to_csv(export_path, index=False)
            print(f"Exported VaR/CVaR/Performance results to {export_path}\n")

    print("End of run.")


if __name__ == "__main__":
    main(visualize=VISUALIZE)
