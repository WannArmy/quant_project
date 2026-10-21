"""optimize.py
Portfolio optimization using Modern Portfolio Theory.
"""

import numpy as np
import pandas as pd
try:
    from scipy.optimize import minimize
except ImportError:
    minimize = None

from .risk import calculate_sharpe_ratio

def optimize_portfolio_weights(returns_df: pd.DataFrame, risk_free_rate: float = 0.0, periods: int = 252) -> list:
    """
    Find the optimal portfolio weights that maximize the Sharpe ratio.
    
    Args:
        returns_df: DataFrame of daily returns for each asset.
        risk_free_rate: Annual risk-free rate.
        periods: Trading periods per year.
        
    Returns:
        List of optimal weights summing to 1.
    """
    num_assets = len(returns_df.columns)
    initial_weights = np.array([1.0 / num_assets] * num_assets)
    
    if minimize is None:
        print("scipy is not installed. Returning equal weights.")
        return initial_weights.tolist()
        
    def negative_sharpe(weights):
        port_returns = returns_df.dot(weights)
        sharpe = calculate_sharpe_ratio(port_returns, risk_free_rate, periods)
        if np.isnan(sharpe):
            return 0.0
        return -sharpe

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    
    try:
        result = minimize(
            negative_sharpe,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            return result.x.tolist()
        else:
            print(f"Optimization failed: {result.message}. Using equal weights.")
            return initial_weights.tolist()
    except Exception as e:
        print(f"Error during optimization: {e}. Using equal weights.")
        return initial_weights.tolist()
