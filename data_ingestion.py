import yfinance as yf
import pandas as pd
import psycopg2

def fetch_and_save_market_data():
    """
    Automated ETL process to fetch real-world financial data from Yahoo Finance
    and store it in a local PostgreSQL database for risk analysis.
    """
    # List of major Taiwan Stock Exchange (TWSE) tickers
    tickers = [
    '2330',
    '2317',
    '2454',
    '2308',
    '2002',
    '1301',
    '1303'
    ]   
    all_data = []

    print("--- Starting Data Ingestion: Yahoo Finance API ---")

    for t in tickers:
        try:
            print(f"Extracting data for: {t}.TW...")
            stock = yf.Ticker(f"{t}.TW")
            
            # Fetch official Financial Statements
            bs = stock.balance_sheet     # Balance Sheet
            is_stmt = stock.financials    # Income Statement
            cf = stock.cashflow          # Cash Flow Statement

            # Helper function to handle dynamic key mapping and missing values
            def get_val(df, keys):
                for k in keys:
                    if k in df.index: 
                        val = df.loc[k].iloc[0]
                        # Convert NumPy types to native Python floats for SQL compatibility
                        return float(val) if pd.notnull(val) else 0.0
                return 0.0

            # Transform raw data into structured financial metrics
            data = {
                'ticker': t,

                # ===== PROFITABILITY =====
                'net_income_curr': get_val(
                    is_stmt,
                    ['Net Income', 'Net Income Common Stockholders']
                ),

                'net_income_prev': (
                    float(is_stmt.loc['Net Income'].iloc[1])
                    if 'Net Income' in is_stmt.index and len(is_stmt.columns) > 1
                    else 0.0
                ),

                'total_assets_curr': get_val(
                    bs,
                    ['Total Assets']
                ),

                'total_assets_prev': (
                    float(bs.loc['Total Assets'].iloc[1])
                    if 'Total Assets' in bs.index and len(bs.columns) > 1
                    else 0.0
                ),

                'total_assets_prev2': (
                float(bs.loc['Total Assets'].iloc[2])
                if 'Total Assets' in bs.index and len(bs.columns) > 2
                else 0.0
                ),

                'cfo_curr': get_val(
                    cf,
                    [
                        'Operating Cash Flow',
                        'Cash Flow From Operating Activities',
                        'OperatingCashFlow'
                    ]
                ),

                # ===== LEVERAGE =====
                'long_term_debt_curr': get_val(
                    bs,
                    ['Long Term Debt']
                ),

                'long_term_debt_prev': (
                    float(bs.loc['Long Term Debt'].iloc[1])
                    if 'Long Term Debt' in bs.index and len(bs.columns) > 1
                    else 0.0
                ),

                # ===== LIQUIDITY =====
                'current_assets_curr': get_val(
                    bs,
                    ['Current Assets']
                ),

                'current_assets_prev': (
                    float(bs.loc['Current Assets'].iloc[1])
                    if 'Current Assets' in bs.index and len(bs.columns) > 1
                    else 0.0
                ),

                'current_liab_curr': get_val(
                    bs,
                    ['Current Liabilities']
                ),

                'current_liab_prev': (
                    float(bs.loc['Current Liabilities'].iloc[1])
                    if 'Current Liabilities' in bs.index and len(bs.columns) > 1
                    else 0.0
                ),

                # ===== OPERATING EFFICIENCY =====
                'revenue_curr': get_val(
                    is_stmt,
                    ['Total Revenue', 'Operating Revenue']
                ),

                'revenue_prev': (
                    float(is_stmt.loc['Total Revenue'].iloc[1])
                    if 'Total Revenue' in is_stmt.index and len(is_stmt.columns) > 1
                    else 0.0
                ),

                'gross_profit_curr': get_val(
                    is_stmt,
                    ['Gross Profit']
                ),

                'gross_profit_prev': (
                    float(is_stmt.loc['Gross Profit'].iloc[1])
                    if 'Gross Profit' in is_stmt.index and len(is_stmt.columns) > 1
                    else 0.0
                ),

                # ===== SHARES =====
                'shares_curr': get_val(
                    bs,
                    ['Ordinary Shares Number']
                ),

                'shares_prev': (
                    float(bs.loc['Ordinary Shares Number'].iloc[1])
                    if 'Ordinary Shares Number' in bs.index
                    and len(bs.columns) > 1
                    else 0.0
                )
            }
            all_data.append(data)

        except Exception as e:
            print(f"[ERROR] Failed to fetch data for {t}: {e}")

    # --- Database Loading Phase: PostgreSQL ---
    try:
        # Configuration for local PostgreSQL connection
        conn = psycopg2.connect(
            host="localhost", 
            database="capstone_finance", 
            user="postgres", 
            password="12345"
        )
        cur = conn.cursor()

        print("\nLoading data into PostgreSQL...")
        
        for d in all_data:
            # Data Normalization: Ensure all NumPy values are cast to native Python floats
            clean_data = {
                'ticker': d['ticker'],

                'net_income_curr': float(d['net_income_curr']),
                'net_income_prev': float(d['net_income_prev']),

                'total_assets_curr': float(d['total_assets_curr']),
                'total_assets_prev': float(d['total_assets_prev']),
                'total_assets_prev2': float(d['total_assets_prev2']),

                'cfo_curr': float(d['cfo_curr']),

                'long_term_debt_curr': float(d['long_term_debt_curr']),
                'long_term_debt_prev': float(d['long_term_debt_prev']),

                'current_assets_curr': float(d['current_assets_curr']),
                'current_assets_prev': float(d['current_assets_prev']),

                'current_liab_curr': float(d['current_liab_curr']),
                'current_liab_prev': float(d['current_liab_prev']),

                'revenue_curr': float(d['revenue_curr']),
                'revenue_prev': float(d['revenue_prev']),

                'gross_profit_curr': float(d['gross_profit_curr']),
                'gross_profit_prev': float(d['gross_profit_prev']),

                'shares_curr': float(d['shares_curr']),
                'shares_prev': float(d['shares_prev'])

                
            }

            # Upsert logic: Insert new records or update existing ones based on ticker
            cur.execute("""
                INSERT INTO financial_data (
                    ticker,
                    net_income_curr,
                    net_income_prev,
                    total_assets_curr,
                    total_assets_prev,
                    total_assets_prev2,
                    cfo_curr,
                    long_term_debt_curr,
                    long_term_debt_prev,
                    current_assets_curr,
                    current_assets_prev,
                    current_liab_curr,
                    current_liab_prev,
                    revenue_curr,
                    revenue_prev,
                    gross_profit_curr,
                    gross_profit_prev,
                    shares_curr,
                    shares_prev
                )
                VALUES (
                    %(ticker)s,
                    %(net_income_curr)s,
                    %(net_income_prev)s,
                    %(total_assets_curr)s,
                    %(total_assets_prev)s,
                    %(total_assets_prev2)s,
                    %(cfo_curr)s,
                    %(long_term_debt_curr)s,
                    %(long_term_debt_prev)s,
                    %(current_assets_curr)s,
                    %(current_assets_prev)s,
                    %(current_liab_curr)s,
                    %(current_liab_prev)s,
                    %(revenue_curr)s,
                    %(revenue_prev)s,
                    %(gross_profit_curr)s,
                    %(gross_profit_prev)s,
                    %(shares_curr)s,
                    %(shares_prev)s
                )
                ON CONFLICT (ticker) DO UPDATE SET
                    net_income_curr = EXCLUDED.net_income_curr,
                    net_income_prev = EXCLUDED.net_income_prev,
                    total_assets_curr = EXCLUDED.total_assets_curr,
                    total_assets_prev = EXCLUDED.total_assets_prev,
                    total_assets_prev2 = EXCLUDED.total_assets_prev2,
                    cfo_curr = EXCLUDED.cfo_curr,
                    long_term_debt_curr = EXCLUDED.long_term_debt_curr,
                    long_term_debt_prev = EXCLUDED.long_term_debt_prev,
                    current_assets_curr = EXCLUDED.current_assets_curr,
                    current_assets_prev = EXCLUDED.current_assets_prev,
                    current_liab_curr = EXCLUDED.current_liab_curr,
                    current_liab_prev = EXCLUDED.current_liab_prev,
                    revenue_curr = EXCLUDED.revenue_curr,
                    revenue_prev = EXCLUDED.revenue_prev,
                    gross_profit_curr = EXCLUDED.gross_profit_curr,
                    gross_profit_prev = EXCLUDED.gross_profit_prev,
                    shares_curr = EXCLUDED.shares_curr,
                    shares_prev = EXCLUDED.shares_prev,
                    updated_at = CURRENT_TIMESTAMP;
            """, clean_data)
        
        conn.commit()
        cur.close()
        conn.close()
        print("\n[SUCCESS] Pipeline completed: Real-world Taiwan market data is now stored in SQL.")
        
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Database transaction failed: {e}")

if __name__ == "__main__":
    fetch_and_save_market_data()