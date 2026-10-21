"""preprocess.py
Data preprocessing utilities.
"""

import pandas as pd


def calculate_returns(df: pd.DataFrame) -> pd.Series:
    """Calculate daily percentage returns from the closing price.

    Args:
        df: DataFrame containing a 'Close' column.

    Returns:
        A pandas Series of daily returns (percent changes), index aligned with df.
    """
    if df is None or 'Close' not in df.columns:
        raise ValueError("DataFrame must contain a 'Close' column to compute returns.")

    returns = df['Close'].pct_change().dropna()
    return returns
