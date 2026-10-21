"""f_score.py
Simple F-Score calculator using mock financial data.
"""
from typing import Dict


def calculate_f_score(financial_data: Dict[str, float]) -> int:
    """Compute a simple three-factor F-Score.

    Criteria (mock data):
    - net_income > 0
    - cash_flow > 0
    - debt decreasing (based on provided debt_series)

    Returns an integer score in [0, 3].
    """
    score = 0
    if financial_data.get("net_income", 0) > 0:
        score += 1
    if financial_data.get("cash_flow", 0) > 0:
        score += 1
    debt_series = financial_data.get("debt_series", [])
    if isinstance(debt_series, (list, tuple)) and len(debt_series) >= 2:
        if debt_series[-1] < debt_series[0]:  # debt decreasing over period
            score += 1
    return score


def get_real_financials(ticker: str) -> Dict[str, float]:
    """Fetch actual fundamental data using yfinance.
    
    This replaces the deterministic mock data with real reported figures.
    """
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        
        # Get financials (Income Statement)
        inc_stmt = t.financials
        net_income = 0.0
        if not inc_stmt.empty and "Net Income" in inc_stmt.index:
            net_income = inc_stmt.loc["Net Income"].dropna().iloc[0]
            
        # Get cash flow
        cf_stmt = t.cashflow
        cash_flow = 0.0
        if not cf_stmt.empty and "Operating Cash Flow" in cf_stmt.index:
            cash_flow = cf_stmt.loc["Operating Cash Flow"].dropna().iloc[0]
            
        # Get balance sheet for debt
        bs_stmt = t.balance_sheet
        debt_series = []
        if not bs_stmt.empty and "Total Debt" in bs_stmt.index:
            debt_data = bs_stmt.loc["Total Debt"].dropna()
            if len(debt_data) >= 2:
                # yfinance returns newest first usually, so we reverse it to get oldest to newest
                debt_series = [debt_data.iloc[1], debt_data.iloc[0]]
            
        return {
            "net_income": float(net_income),
            "cash_flow": float(cash_flow),
            "debt_series": debt_series
        }
    except Exception as e:
        print(f"Error fetching real financials for {ticker}: {e}")
        return {"net_income": 0.0, "cash_flow": 0.0, "debt_series": []}
