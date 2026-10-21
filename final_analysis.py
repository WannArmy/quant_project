import pandas as pd
import yfinance as yf


def run_f_score_analysis():
    """
    Calculate Piotroski F-Score for eligible non-financial Taiwan stocks.
    Data is fetched directly from Yahoo Finance via yfinance
    (no database connection required).
    """

    tickers = ["2330", "2317", "2454", "2308", "2002", "1301", "1303"]

    results = []

    for t in tickers:
        try:
            stock = yf.Ticker(f"{t}.TW")

            bs = stock.balance_sheet     # Balance Sheet
            is_stmt = stock.financials   # Income Statement
            cf = stock.cashflow          # Cash Flow Statement

            # Helper: safely get a value from a DataFrame by trying multiple keys
            def get_val(df, keys, col_idx=0):
                for k in keys:
                    if k in df.index:
                        series = df.loc[k].dropna()
                        if len(series) > col_idx:
                            v = series.iloc[col_idx]
                            return float(v) if pd.notnull(v) else 0.0
                return 0.0

            # ── PROFITABILITY ──────────────────────────────────────────────
            net_income_curr = get_val(is_stmt, ["Net Income", "Net Income Common Stockholders"], 0)
            net_income_prev = get_val(is_stmt, ["Net Income", "Net Income Common Stockholders"], 1)

            total_assets_curr = get_val(bs, ["Total Assets"], 0)
            total_assets_prev = get_val(bs, ["Total Assets"], 1)
            total_assets_prev2 = get_val(bs, ["Total Assets"], 2)

            cfo_curr = get_val(cf, ["Operating Cash Flow", "Cash Flow From Operating Activities", "OperatingCashFlow"], 0)

            # ── LEVERAGE ──────────────────────────────────────────────────
            long_term_debt_curr = get_val(bs, ["Long Term Debt"], 0)
            long_term_debt_prev = get_val(bs, ["Long Term Debt"], 1)

            # ── LIQUIDITY ─────────────────────────────────────────────────
            current_assets_curr = get_val(bs, ["Current Assets"], 0)
            current_assets_prev = get_val(bs, ["Current Assets"], 1)

            current_liab_curr = get_val(bs, ["Current Liabilities"], 0)
            current_liab_prev = get_val(bs, ["Current Liabilities"], 1)

            # ── OPERATING EFFICIENCY ──────────────────────────────────────
            revenue_curr = get_val(is_stmt, ["Total Revenue", "Operating Revenue"], 0)
            revenue_prev = get_val(is_stmt, ["Total Revenue", "Operating Revenue"], 1)

            gross_profit_curr = get_val(is_stmt, ["Gross Profit"], 0)
            gross_profit_prev = get_val(is_stmt, ["Gross Profit"], 1)

            # ── SHARES ────────────────────────────────────────────────────
            shares_curr = get_val(bs, ["Ordinary Shares Number"], 0)
            shares_prev = get_val(bs, ["Ordinary Shares Number"], 1)

            # ── CALCULATE F-SCORE ─────────────────────────────────────────
            f_score = 0

            # 1. ROA > 0
            avg_assets_curr = (total_assets_curr + total_assets_prev) / 2 or 1
            avg_assets_prev = (total_assets_prev + total_assets_prev2) / 2 or 1

            roa_curr = net_income_curr / avg_assets_curr
            criterion_roa = roa_curr > 0
            if criterion_roa:
                f_score += 1

            # 2. Positive Operating Cash Flow
            criterion_cfo = cfo_curr > 0
            if criterion_cfo:
                f_score += 1

            # 3. Change in ROA > 0
            roa_prev = net_income_prev / avg_assets_prev
            criterion_delta_roa = (roa_curr - roa_prev) > 0
            if criterion_delta_roa:
                f_score += 1

            # 4. CFO > ROA (Accrual / Earnings Quality)
            cfo_to_assets = cfo_curr / avg_assets_curr
            criterion_accrual = cfo_to_assets > roa_curr
            if criterion_accrual:
                f_score += 1

            # 5. Decrease in Leverage
            leverage_curr = long_term_debt_curr / avg_assets_curr
            leverage_prev = long_term_debt_prev / avg_assets_prev
            criterion_leverage = leverage_curr < leverage_prev
            if criterion_leverage:
                f_score += 1

            # 6. Increase in Liquidity (Current Ratio)
            current_ratio_curr = current_assets_curr / current_liab_curr if current_liab_curr else 0
            current_ratio_prev = current_assets_prev / current_liab_prev if current_liab_prev else 0
            criterion_liquidity = current_ratio_curr > current_ratio_prev
            if criterion_liquidity:
                f_score += 1

            # 7. No New Shares Issued
            criterion_shares = shares_curr <= shares_prev
            if criterion_shares:
                f_score += 1

            # 8. Increase in Gross Margin
            gross_margin_curr = gross_profit_curr / revenue_curr if revenue_curr else 0
            gross_margin_prev = gross_profit_prev / revenue_prev if revenue_prev else 0
            criterion_gross_margin = gross_margin_curr > gross_margin_prev
            if criterion_gross_margin:
                f_score += 1

            # 9. Increase in Asset Turnover
            asset_turnover_curr = revenue_curr / avg_assets_curr
            asset_turnover_prev = revenue_prev / avg_assets_prev
            criterion_asset_turnover = asset_turnover_curr > asset_turnover_prev
            if criterion_asset_turnover:
                f_score += 1

            # Financial strength label
            if f_score >= 8:
                status = "Strong"
            elif f_score >= 6:
                status = "Good"
            elif f_score >= 4:
                status = "Moderate"
            else:
                status = "Weak"

            results.append({
                "Stock": t,
                "ROA": int(criterion_roa),
                "CFO": int(criterion_cfo),
                "Delta_ROA": int(criterion_delta_roa),
                "Accrual": int(criterion_accrual),
                "Leverage": int(criterion_leverage),
                "Liquidity": int(criterion_liquidity),
                "Shares": int(criterion_shares),
                "Gross_Margin": int(criterion_gross_margin),
                "Asset_Turnover": int(criterion_asset_turnover),
                "F_Score": f_score,
                "Financial_Strength": status,
            })

        except Exception as e:
            print(f"[ERROR] Failed to compute F-Score for {t}: {e}")

    if not results:
        return None

    return pd.DataFrame(results)


# ==============================================
# Test
# ==============================================
if __name__ == "__main__":

    result = run_f_score_analysis()

    if result is not None:
        print("\n========== PIOTROSKI F-SCORE ==========")
        print(result.to_string(index=False))
