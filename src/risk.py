"""risk.py
Risk metric calculations: VaR and CVaR.
"""
import numpy as np
import pandas as pd


def calculate_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Calculate Value-at-Risk (VaR) as the (1 - confidence) percentile of returns.

    VaR here represents the loss threshold such that we expect worse losses than VaR
    with probability (1 - confidence) over the sample.
    """
    if returns is None or len(returns) == 0:
        return float('nan')
    arr = np.asarray(returns.dropna(), dtype=float)
    if arr.size == 0:
        return float('nan')
    percentile = (1.0 - float(confidence)) * 100.0
    var = np.percentile(arr, percentile)
    return float(var)


def calculate_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """Calculate Conditional VaR (CVaR) as the mean of the worst (tail) losses beyond VaR."""
    if returns is None or len(returns) == 0:
        return float('nan')
    arr = np.asarray(returns.dropna(), dtype=float)
    if arr.size == 0:
        return float('nan')
    var = calculate_var(pd.Series(arr), confidence)
    # Tail: returns less than or equal to VaR (i.e., worse outcomes)
    tail = arr[arr <= var]
    if tail.size == 0:
        return float(var)
    return float(np.mean(tail))


def calculate_portfolio_returns(returns_df: pd.DataFrame, weights: list) -> pd.Series:
    """Calculate the returns of a portfolio given the asset returns and weights."""
    if len(weights) != len(returns_df.columns):
        raise ValueError("Number of weights must match the number of columns in returns_df")
    # Matrix multiplication: (N_days x N_assets) dot (N_assets) -> N_days
    return returns_df.dot(weights)


def calculate_maximum_drawdown(returns: pd.Series) -> float:
    """Calculate Maximum Drawdown (MDD) from a series of returns."""
    if returns is None or len(returns) == 0:
        return float('nan')
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return float(drawdown.min())


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods: int = 252) -> float:
    """Calculate the annualized Sharpe Ratio.
    
    Args:
        returns: Series of daily returns
        risk_free_rate: Annual risk-free rate
        periods: Number of trading periods in a year (default 252 for daily)
    """
    if returns is None or len(returns) == 0:
        return float('nan')
    
    mean_return = returns.mean()
    std_dev = returns.std()
    
    if std_dev == 0:
        return float('nan')
        
    daily_rf = (1 + risk_free_rate) ** (1/periods) - 1
    excess_return = mean_return - daily_rf
    
    return float(excess_return / std_dev * np.sqrt(periods))


def calculate_annualized_volatility(returns: pd.Series, periods: int = 252) -> float:
    """Calculate annualized volatility."""
    if returns is None or len(returns) == 0:
        return float('nan')
    return float(returns.std() * np.sqrt(periods))


def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods: int = 252) -> float:
    """Calculate the annualized Sortino Ratio (only penalizes downside volatility)."""
    if returns is None or len(returns) == 0:
        return float('nan')
    
    mean_return = returns.mean()
    daily_rf = (1 + risk_free_rate) ** (1/periods) - 1
    excess_return = mean_return - daily_rf
    
    # Downside deviation
    downside_returns = returns[returns < 0]
    if len(downside_returns) == 0:
        return float('nan')
    
    downside_std = downside_returns.std()
    if downside_std == 0:
        return float('nan')
        
    return float(excess_return / downside_std * np.sqrt(periods))


def calculate_beta(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Calculate Beta relative to a benchmark."""
    if returns is None or benchmark_returns is None or len(returns) == 0 or len(benchmark_returns) == 0:
        return float('nan')
    
    # Align dates
    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 2:
        return float('nan')
        
    cov_matrix = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
    if cov_matrix[1, 1] == 0:
        return float('nan')
        
    beta = cov_matrix[0, 1] / cov_matrix[1, 1]
    return float(beta)


def calculate_alpha(returns: pd.Series, benchmark_returns: pd.Series, risk_free_rate: float = 0.0, periods: int = 252) -> float:
    """Calculate annualized Alpha."""
    if returns is None or benchmark_returns is None or len(returns) == 0 or len(benchmark_returns) == 0:
        return float('nan')
        
    beta = calculate_beta(returns, benchmark_returns)
    if np.isnan(beta):
        return float('nan')
        
    mean_return = returns.mean()
    mean_bench = benchmark_returns.mean()
    
    daily_rf = (1 + risk_free_rate) ** (1/periods) - 1
    
    # CAPM Alpha (daily)
    daily_alpha = mean_return - (daily_rf + beta * (mean_bench - daily_rf))
    
    # Annualized Alpha
    return float(daily_alpha * periods)
