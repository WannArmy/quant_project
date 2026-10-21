"""monte_carlo.py
Monte Carlo simulation engine for forecasting future portfolio values.
"""

import numpy as np
import pandas as pd

def run_monte_carlo_simulation(returns: pd.Series, initial_portfolio_value: float = 10000.0, days: int = 252, simulations: int = 1000) -> pd.DataFrame:
    """
    Run a Monte Carlo simulation using Geometric Brownian Motion (GBM).
    
    Args:
        returns: Historical daily returns of the portfolio or asset.
        initial_portfolio_value: Starting value of the portfolio.
        days: Number of trading days to simulate into the future.
        simulations: Number of random paths to generate.
        
    Returns:
        A DataFrame where each column is a simulated price path over time.
    """
    if returns is None or len(returns) == 0:
        return pd.DataFrame()
        
    mu = returns.mean()
    sigma = returns.std()
    
    if np.isnan(mu) or np.isnan(sigma) or sigma == 0:
        return pd.DataFrame()
        
    # Generate random shocks from a standard normal distribution
    Z = np.random.normal(0, 1, (days, simulations))
    
    # Calculate daily returns using Geometric Brownian Motion formula
    daily_returns_simulated = np.exp((mu - 0.5 * sigma**2) + sigma * Z)
    
    # Calculate cumulative product for each path
    simulated_paths = initial_portfolio_value * np.cumprod(daily_returns_simulated, axis=0)
    
    # Prepend the initial value to the paths (Day 0)
    initial_row = np.full((1, simulations), initial_portfolio_value)
    simulated_paths = np.vstack([initial_row, simulated_paths])
    
    return pd.DataFrame(simulated_paths)
