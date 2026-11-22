import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Load environment variables from .env (for DATABASE_URL)
from dotenv import load_dotenv
load_dotenv()


# Custom data-processing functions in the src/ folder
from src.fetch_data import get_price_data          # fetch historical prices for one ticker
from src.preprocess import calculate_returns       # prices → daily returns
from src.risk import (
    calculate_portfolio_returns,       # weighted portfolio returns
    calculate_annualized_volatility,   # annualized volatility
    calculate_maximum_drawdown,        # maximum drawdown
)
from final_analysis import run_f_score_analysis    # returns a DataFrame of F-Scores


# =========================================================
# PAGE CONFIG
# Browser tab title, icon and wide layout.
# Must be the FIRST st.* call in the script.
# =========================================================

st.set_page_config(
    page_title="台灣量化投資分析系統",
    page_icon="📈",
    layout="wide",                      # use the full screen width
    initial_sidebar_state="collapsed",  # sidebar hidden on load
)



PRIMARY = "#2563EB"       # main blue (tabs, equity curve line…)
PRIMARY_SOFT = "#EFF6FF"  # light blue background
SUCCESS = "#16A34A"       # green: passed / profit
SUCCESS_SOFT = "#F0FDF4"
DANGER = "#DC2626"        # red: failed / loss / drawdown
DANGER_SOFT = "#FEF2F2"
WARNING = "#D97706"       # orange: warning
TEXT = "#0F172A"          # main text color (near black)
MUTED = "#334155"         # dark gray text (descriptions, secondary labels)
SUBTLE = "#475569"        # slightly lighter gray (footer, notes)
BORDER = "#E2E8F0"        # card border color

# Financial strength → (Chinese label, foreground color, background color)
# Used by the count cards, bar chart and badge
STRENGTH_STYLE = {
    "Strong": ("強", SUCCESS, SUCCESS_SOFT),
    "Good": ("良好", PRIMARY, PRIMARY_SOFT),
    "Moderate": ("中等", WARNING, "#FFFBEB"),
    "Weak": ("弱", DANGER, DANGER_SOFT),
}

st.markdown(
    f"""
""",
    unsafe_allow_html=True,
)



def html(content: str):
    """Shorthand for st.markdown(..., unsafe_allow_html=True) to render HTML."""
    st.markdown(content, unsafe_allow_html=True)


def section_header(num: str, title: str, subtitle: str = ""):
    """Section heading: number box + section title + optional description line."""
    html(
        f'<div class="section-head">'
        f'<div class="section-num">{num}</div>'
        f'<div class="section-title">{title}</div>'
        f"</div>"
        + (f'<div class="section-subtitle">{subtitle}</div>' if subtitle else "")
    )


def kpi_card(label: str, value: str, desc: str, accent: str = PRIMARY):
    """
    Metric card: name (label), large value (value), small description (desc).
    accent = color of the left vertical bar, passed to CSS via --accent.
    """
    html(
        f'<div class="kpi" style="--accent:{accent};">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-desc">{desc}</div>'
        f"</div>"
    )


def score_pill(count: int, key: str):
    """
    Card counting how many stocks fall into a strength level (Strong/Good/Moderate/Weak).
    {fg}33 = text color + alpha 0x33 (~20%) → border in the same hue but lighter.
    """
    label, fg, bg = STRENGTH_STYLE[key]
    html(
        f'<div class="score-pill" style="--fg:{fg};--bg:{bg};--border:{fg}33;">'
        f'<div class="score-pill-num">{count}</div>'
        f'<div class="score-pill-label">{label}</div>'
        f"</div>"
    )


def base_layout(fig, height=420, y_prefix=None, y_format=None):
    """
    Shared style for every Plotly chart: white background, dark text,
    light horizontal grid, hidden legend, unified hover along the x axis.
    y_prefix: prefix for y-axis ticks (e.g. "NT$ ")
    y_format: y-axis tick format (e.g. ".0%" to show percentages)
    """
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color=TEXT, size=13),
        hovermode="x unified",           # one tooltip for the whole x position
        showlegend=False,
        hoverlabel=dict(font_color=TEXT, bgcolor="#FFFFFF"),
        xaxis=dict(
            showgrid=False,
            linecolor="#CBD5E1",
            tickfont=dict(color=TEXT, size=13),
            title_font=dict(color=TEXT),
        ),
        yaxis=dict(
            tickfont=dict(color=TEXT, size=13),
            title_font=dict(color=TEXT),
            showgrid=True,
            gridcolor="#E2E8F0",
            zeroline=False,
            tickprefix=y_prefix,
            tickformat=y_format,
        ),
    )
    return fig


# =========================================================
# STOCK INFORMATION
# Stock list and Chinese company names.
# =========================================================

# Tickers with the .TW suffix (format used by the price data source, e.g. yfinance)
STOCK_NAMES = {
    "2330.TW": "台積電",
    "2317.TW": "鴻海",
    "2454.TW": "聯發科",
    "2308.TW": "台達電",
    "2881.TW": "富邦金",
    "2891.TW": "中信金",
    "2002.TW": "中鋼",
    "1301.TW": "台塑",
    "1303.TW": "南亞",
    "2882.TW": "國泰金",
}
ALL_STOCKS = list(STOCK_NAMES.keys())

# Codes WITHOUT .TW — match the "Stock" column in the F-Score results.
# Financial stocks are excluded because the F-Score does not fit
# financial-sector statements well.
F_SCORE_NAMES = {
    "2330": "台積電",
    "2317": "鴻海",
    "2454": "聯發科",
    "2308": "台達電",
    "2002": "中鋼",
    "1301": "台塑",
    "1303": "南亞",
}

# The 9 Piotroski criteria split into 3 groups.
# Each criterion: (column name in f_score_df, display name, description)
# Column value = 1 (passed) or 0 (failed).
CRITERIA_GROUPS = [
    (
        "獲利能力",  # Profitability (4 criteria)
        [
            ("ROA", "正向資產報酬率 (ROA)", "公司能透過資產產生正向獲利。"),
            ("CFO", "正向營業現金流 (CFO)", "公司營業活動產生正向現金流。"),
            ("Delta_ROA", "資產報酬率改善", "本期 ROA 高於前一期。"),
            ("Accrual", "盈餘品質", "營業現金流相對於資產的表現優於 ROA。"),
        ],
    ),
    (
        "財務結構與流動性",  # Leverage, liquidity and source of funds (3 criteria)
        [
            ("Leverage", "財務槓桿下降", "公司的長期負債相對資產比例下降。"),
            ("Liquidity", "流動性改善", "公司的流動比率較前一期提高。"),
            ("Shares", "未發行新股", "公司普通股股數沒有增加。"),
        ],
    ),
    (
        "營運效率",  # Operating efficiency (2 criteria)
        [
            ("Gross_Margin", "毛利率改善", "本期毛利率高於前一期。"),
            ("Asset_Turnover", "資產週轉率改善", "公司利用資產創造營收的效率提高。"),
        ],
    ),
]


def stock_label(code) -> str:
    """'2330' → '2330　台積電' (used in the selectbox and the bar chart's y axis)."""
    code = str(code)
    return f"{code}　{F_SCORE_NAMES.get(code, code)}"


# =========================================================
# SESSION STATE
# Flag telling whether an analysis has been run. Persists across reruns.
# =========================================================

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False


# =========================================================
# HERO — top banner + 4 guide steps
# =========================================================

html(
    """
<div class="hero">
    <div class="hero-badge">📈 TAIWAN QUANT ANALYTICS</div>
    <div class="hero-title">台灣量化投資分析系統</div>
    <div class="hero-subtitle">投資組合分析 × 風險評估 × 財務強度分析</div>
    <div class="hero-steps">
        <div class="hero-step"><b>①</b>選擇股票</div>
        <div class="hero-step"><b>②</b>設定金額與權重</div>
        <div class="hero-step"><b>③</b>點擊「開始分析」</div>
        <div class="hero-step"><b>④</b>查看績效、風險與財務強度</div>
    </div>
</div>
"""
)


# =========================================================
# SECTION 1 — PORTFOLIO CONFIGURATION
# Input area, wrapped in a bordered container.
# =========================================================

with st.container(border=True):

    section_header(
        "1",
        "投資組合設定",
        "選擇您想分析的台灣股票，並設定投資金額與配置比例。",
    )

    # Wide left column (stock picker), narrow right column (capital) — ratio 2.4 : 1
    config_col1, config_col2 = st.columns([2.4, 1], gap="large")

    with config_col1:
        # Multi-select stocks; all selected by default.
        # format_func only changes the displayed text; the value stays "2330.TW"
        selected_stocks = st.multiselect(
            "選擇股票",
            options=ALL_STOCKS,
            default=ALL_STOCKS,
            format_func=lambda x: f"{x.replace('.TW', '')}　{STOCK_NAMES[x]}",
            placeholder="請選擇股票",   # gray hint shown when no stock is selected
        )

    with config_col2:
        # Initial capital (NT$), minimum 1,000, step of 1,000
        initial_investment = st.number_input(
            "初始投資金額（NT$）",
            min_value=1000.0,
            value=10000.0,
            step=1000.0,
            format="%.0f",   # show an integer, no .00
        )

    # Defaults in case no stock is selected
    weights = {}
    normalize_weights = True

    # ----- Weight settings (only shown when at least one stock is selected) -----
    if selected_stocks:
        with st.expander("⚖️ 投資組合權重設定", expanded=False):
            st.caption("請設定每檔股票的投資比例。若不確定如何配置，可以使用等權重。")

            # Lay out the weight inputs in a 4-column grid
            weight_cols = st.columns(4)
            default_weight = 1.0 / len(selected_stocks)   # equal weight by default

            for i, ticker in enumerate(selected_stocks):
                with weight_cols[i % 4]:   # i % 4 → columns 0,1,2,3,0,1…
                    weights[ticker] = st.number_input(
                        f"{ticker.replace('.TW', '')}　{STOCK_NAMES[ticker]}",
                        min_value=0.0,
                        max_value=1.0,
                        value=default_weight,
                        step=0.01,
                        format="%.2f",
                        key=f"weight_{ticker}",   # unique key so Streamlit remembers each input
                    )

            # Show the total weight: green if = 100%, orange otherwise
            total_weight = sum(weights.values())
            w_col1, w_col2 = st.columns([1, 1])
            with w_col1:
                normalize_weights = st.checkbox("自動標準化權重", value=True)
            with w_col2:
                ok = np.isclose(total_weight, 1.0)   # safe float comparison
                color = SUCCESS if ok else WARNING
                html(
                    f'<div style="text-align:right;color:{MUTED};font-size:14px;">'
                    f'目前總權重：<b style="color:{color};">{total_weight:.1%}</b></div>'
                )

    # Run button (takes 1/4 of the width).
    # st.button returns True only once, on the rerun right after the click.
    run_col, _ = st.columns([1, 3])
    with run_col:
        run_analysis = st.button("🚀 開始分析", type="primary", use_container_width=True)


# =========================================================
# ANALYSIS PROCESS
# Runs only right after the button is clicked; results go into session_state.
# =========================================================

if run_analysis:

    total_weight = sum(weights.values())

    # ----- Input validation -----
    if not selected_stocks:
        st.error("請至少選擇一檔股票。")

    elif total_weight <= 0:
        st.error("投資權重總和必須大於 0。")

    else:
        # ----- Normalize weights so they sum to 1 -----
        # Always normalize when the total ≠ 1, even if the checkbox is unticked,
        # because weights not summing to 1 would distort the portfolio value.
        if normalize_weights or not np.isclose(total_weight, 1.0):
            weights = {t: w / total_weight for t, w in weights.items()}

        # ----- Fetch prices and compute returns per ticker, with a progress bar -----
        progress = st.progress(0, text="正在取得歷史股價資料……")
        returns_dict = {}   # {ticker: Series of daily returns}

        for i, ticker in enumerate(selected_stocks):
            price_data = get_price_data(ticker)
            if price_data is not None and not price_data.empty:
                stock_returns = calculate_returns(price_data)
                if stock_returns is not None and not stock_returns.empty:
                    returns_dict[ticker] = stock_returns
            # Update the progress bar after each ticker
            progress.progress(
                (i + 1) / len(selected_stocks),
                text=f"正在取得 {STOCK_NAMES[ticker]} 資料……",
            )

        progress.empty()   # remove the progress bar when done

        if not returns_dict:
            st.error("無法取得股票歷史資料，請稍後再試。")

        else:
            # ----- Combine all tickers' returns into one DataFrame -----
            # One column per ticker, one row per day
            returns_df = pd.concat(returns_dict, axis=1)
            # concat from a dict creates MultiIndex columns → keep the first level (ticker)
            returns_df.columns = returns_df.columns.get_level_values(0)
            # Drop days with missing data so all tickers share the same dates
            returns_df = returns_df.dropna()

            # Align weights with the column order; tickers that failed to load get weight 0
            weight_series = (
                pd.Series(weights).reindex(returns_df.columns).fillna(0)
            )

            # ----- Risk metrics -----
            portfolio_returns = calculate_portfolio_returns(returns_df, weight_series)
            annual_volatility = calculate_annualized_volatility(portfolio_returns)
            maximum_drawdown = calculate_maximum_drawdown(portfolio_returns)

            # Equity curve: portfolio value over time
            # = initial capital × cumulative product of (1 + daily return)
            equity_curve = (1 + portfolio_returns).cumprod() * initial_investment

            # F-Score is computed once on click, not re-run every time the selectbox changes
            with st.spinner("正在進行財務強度分析……"):
                f_score_df = run_f_score_analysis()

            # ----- Store all results in session_state -----
            # So the data survives reruns (switching tabs, changing the selectbox…)
            st.session_state.update(
                analysis_done=True,
                portfolio_returns=portfolio_returns,
                equity_curve=equity_curve,
                annual_volatility=annual_volatility,
                maximum_drawdown=maximum_drawdown,
                initial_investment=initial_investment,
                weights=weights,
                selected_stocks=selected_stocks,
                f_score_df=f_score_df,
            )

            st.toast("分析完成！", icon="✅")   # small notification in the corner


# =========================================================
# DISPLAY RESULTS
# Read data from session_state and render it.
# =========================================================

if not st.session_state.analysis_done:

    # No analysis yet → show an empty state with a hint
    html(
        f'<div style="text-align:center;color:{MUTED};padding:48px 0;">'
        f'<div style="font-size:40px;">📊</div>'
        f"<div>設定完成後，點擊「開始分析」即可查看結果</div></div>"
    )

else:

    ss = st.session_state   # shorthand
    equity_curve = ss.equity_curve
    initial_investment = ss.initial_investment

    # Final value and cumulative total return
    final_value = float(equity_curve.iloc[-1])
    total_return = final_value / initial_investment - 1

    # Daily drawdown = current value / running peak - 1
    # (always ≤ 0; its lowest point is the Maximum Drawdown)
    drawdown_series = equity_curve / equity_curve.cummax() - 1

    st.write("")   # spacer
    tab_risk, tab_fscore = st.tabs(["📈 投資組合與風險", "🏦 Piotroski F-Score"])

    # =====================================================
    # TAB 1 — PORTFOLIO & RISK
    # =====================================================

    with tab_risk:

        section_header(
            "2",
            "投資組合風險分析",
            "透過歷史資料評估投資組合的波動程度與潛在下跌幅度。",
        )

        # ----- 4 KPI cards -----
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            kpi_card(
                "初始投資金額",
                f"NT$ {initial_investment:,.0f}",   # :,.0f → 10,000
                "歷史績效分析的起始金額",
                MUTED,
            )
        with k2:
            # Bar color: green if profit, red if loss
            kpi_card(
                "期末價值",
                f"NT$ {final_value:,.0f}",
                f"累積報酬 {total_return:+.2%}",   # :+.2% → +12.34% (always signed)
                SUCCESS if total_return >= 0 else DANGER,
            )
        with k3:
            kpi_card(
                "年化波動率",
                f"{ss.annual_volatility:.2%}",
                "衡量投資報酬的波動程度",
                PRIMARY,
            )
        with k4:
            kpi_card(
                "最大回撤",
                f"{ss.maximum_drawdown:.2%}",
                "歷史高點至低點的最大跌幅",
                DANGER,
            )

        # Explanation of the two risk metrics
        with st.expander("📖 了解風險指標"):
            st.markdown(
                """
**年化波動率**：衡量投資組合報酬的波動程度。數值越高，代表歷史報酬的不確定性通常越高。

**最大回撤**：衡量投資組合從歷史高點下降至最低點的最大幅度，可了解過去曾經歷的最大下跌程度。
"""
            )

        # ---------- Equity curve (portfolio value over time) ----------
        html('<div class="sub-title">歷史投資組合價值</div>')

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=equity_curve.index,
                y=equity_curve.values,
                mode="lines",
                line=dict(width=2.5, color=PRIMARY),
                fill="tozeroy",                         # shade the area under the line
                fillcolor="rgba(37, 99, 235, 0.08)",
                hovertemplate="投資組合價值：NT$ %{y:,.0f}<extra></extra>",
            )
        )
        # Dotted horizontal line = initial capital, to eyeball profit/loss
        fig.add_hline(
            y=initial_investment,
            line_dash="dot",
            line_color=SUBTLE,
            annotation_text="初始金額",
            annotation_font_color=MUTED,
        )
        base_layout(fig, height=420, y_prefix="NT$ ")
        # Tighten the y range around the data (otherwise fill="tozeroy" pulls
        # the axis down to 0 and flattens the curve)
        fig.update_yaxes(range=[equity_curve.min() * 0.95, equity_curve.max() * 1.03])
        st.plotly_chart(fig, use_container_width=True)

        # ---------- Drawdown + Allocation (two charts side by side) ----------
        dd_col, alloc_col = st.columns([1.6, 1], gap="large")

        with dd_col:
            # Drawdown chart: the deeper it goes, the bigger the loss from the peak
            html('<div class="sub-title">回撤走勢</div>')
            dd_fig = go.Figure()
            dd_fig.add_trace(
                go.Scatter(
                    x=drawdown_series.index,
                    y=drawdown_series.values,
                    mode="lines",
                    line=dict(width=1.5, color=DANGER),
                    fill="tozeroy",
                    fillcolor="rgba(220, 38, 38, 0.12)",
                    hovertemplate="回撤：%{y:.2%}<extra></extra>",
                )
            )
            base_layout(dd_fig, height=300, y_format=".0%")   # y axis as percentage
            st.plotly_chart(dd_fig, use_container_width=True)

        with alloc_col:
            # Donut chart showing each ticker's weight
            html('<div class="sub-title">投資組合配置</div>')
            labels = [
                f"{t.replace('.TW', '')} {STOCK_NAMES[t]}" for t in ss.selected_stocks
            ]
            values = [ss.weights[t] for t in ss.selected_stocks]
            pie = go.Figure(
                go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.6,              # 60% hole in the middle → donut
                    textinfo="percent",
                    textfont=dict(color="#FFFFFF", size=13),
                    hovertemplate="%{label}<br>%{percent}<extra></extra>",
                )
            )
            pie.update_layout(
                height=300,
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor="#FFFFFF",
                showlegend=False,
                # Text in the center of the donut: number of tickers, e.g. "10 檔"
                annotations=[
                    dict(
                        text=f"{len(labels)} 檔",
                        x=0.5, y=0.5,
                        font=dict(size=18, color=TEXT),
                        showarrow=False,
                    )
                ],
            )
            st.plotly_chart(pie, use_container_width=True)

        # ----- Allocation detail table -----
        with st.expander("📋 配置明細表"):
            composition_df = pd.DataFrame(
                {
                    "股票代號": [t.replace(".TW", "") for t in ss.selected_stocks],
                    "公司": [STOCK_NAMES[t] for t in ss.selected_stocks],
                    # Multiply by 100 so the ProgressColumn displays e.g. 10.0%
                    "投資權重": [ss.weights[t] * 100 for t in ss.selected_stocks],
                    # Amount invested in each ticker = weight × capital
                    "投入金額 (NT$)": [
                        ss.weights[t] * initial_investment for t in ss.selected_stocks
                    ],
                }
            )
            st.dataframe(
                composition_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    # Weight column shown as a 0–100% progress bar
                    "投資權重": st.column_config.ProgressColumn(
                        "投資權重", format="%.1f%%", min_value=0, max_value=100
                    ),
                    "投入金額 (NT$)": st.column_config.NumberColumn(format="%,.0f"),
                },
            )

    # =====================================================
    # TAB 2 — F-SCORE
    # =====================================================

    with tab_fscore:

        section_header(
            "3",
            "Piotroski F-Score 財務強度分析",
            "透過 9 項財務指標評估企業的財務狀況與營運改善程度。",
        )

        # Explanation of the F-Score and its scale
        with st.expander("什麼是 Piotroski F-Score？"):
            st.markdown(
                """
Piotroski F-Score 是一套由 **0 至 9 分** 組成的財務強度評估方法。
分數越高，代表公司在獲利能力、財務狀況與營運效率方面具有較多正向訊號。

**8–9 分：強**　·　**6–7 分：良好**　·　**4–5 分：中等**　·　**0–3 分：弱**
"""
            )

        f_score_df = ss.get("f_score_df")   # .get() → None if the key doesn't exist

        if f_score_df is None or f_score_df.empty:
            st.warning("目前沒有可用的 F-Score 資料。")

        else:
            # Work on a copy so the version in session_state is not modified
            f_score_df = f_score_df.copy()
            # Cast types: code → string (to match F_SCORE_NAMES), score → int
            f_score_df["Stock"] = f_score_df["Stock"].astype(str)
            f_score_df["F_Score"] = f_score_df["F_Score"].astype(int)

            # ---------- Summary pills (count of tickers per level) ----------
            counts = f_score_df["Financial_Strength"].value_counts()
            pill_cols = st.columns(4)
            for col, key in zip(pill_cols, ["Strong", "Good", "Moderate", "Weak"]):
                with col:
                    score_pill(int(counts.get(key, 0)), key)   # 0 if level is absent

            # ---------- Ranking chart + table ----------
            # Sorted ascending because Plotly's horizontal bars are drawn bottom-up
            # → the highest score ends up on top
            ranked = f_score_df.sort_values("F_Score", ascending=True)

            chart_col, table_col = st.columns([1.3, 1], gap="large")

            with chart_col:
                # Horizontal bar chart, bar color by financial strength
                html('<div class="sub-title">財務強度排名</div>')
                bar = go.Figure(
                    go.Bar(
                        x=ranked["F_Score"],
                        y=[stock_label(s) for s in ranked["Stock"]],
                        orientation="h",
                        marker_color=[
                            STRENGTH_STYLE.get(s, ("", MUTED, ""))[1]   # [1] = foreground color
                            for s in ranked["Financial_Strength"]
                        ],
                        text=[f"{v} / 9" for v in ranked["F_Score"]],
                        textposition="outside",   # score label outside the bar end
                        textfont=dict(color=TEXT, size=13),
                        hovertemplate="%{y}<br>F-Score：%{x} / 9<extra></extra>",
                    )
                )
                # Height grows with the number of tickers (46px each), min 260px
                base_layout(bar, height=max(260, 46 * len(ranked)))
                bar.update_layout(hovermode="closest")
                # x axis 0–10 (leaves room for the "9 / 9" label on the longest bar)
                bar.update_xaxes(range=[0, 10], dtick=1, showgrid=True, gridcolor="#E2E8F0")
                bar.update_yaxes(showgrid=False)
                st.plotly_chart(bar, use_container_width=True)

            with table_col:
                # Score table, sorted descending
                html('<div class="sub-title">分數明細</div>')
                table_df = ranked.sort_values("F_Score", ascending=False)
                display_df = pd.DataFrame(
                    {
                        "股票代號": table_df["Stock"],
                        "公司": table_df["Stock"].map(
                            lambda x: F_SCORE_NAMES.get(x, x)
                        ),
                        "F-Score": table_df["F_Score"],
                        # "Strong" → "強" ([0] = Chinese label)
                        "財務強度": table_df["Financial_Strength"].map(
                            lambda s: STRENGTH_STYLE.get(s, (s,))[0]
                        ),
                    }
                )
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        # Score column shown as a 0–9 progress bar
                        "F-Score": st.column_config.ProgressColumn(
                            "F-Score", format="%d / 9", min_value=0, max_value=9
                        ),
                    },
                )

            # ---------- Detail (9 criteria for one ticker) ----------
            html('<div class="sub-title">詳細財務指標</div>')

            # Options sorted by score descending.
            # index=None → nothing selected at first, the gray placeholder is shown
            selected_code = st.selectbox(
                "選擇股票查看 9 項指標",
                options=f_score_df.sort_values("F_Score", ascending=False)["Stock"].tolist(),
                format_func=stock_label,
                index=None,
                placeholder="請選擇股票",
            )

            if selected_code is None:
                # Nothing selected yet → show a hint instead of the detail cards
                html(
                    f'<div style="text-align:center;color:{MUTED};padding:28px 0;'
                    f'border:1px dashed #CBD5E1;border-radius:12px;background:#FFFFFF;">'
                    f"請從上方選擇一檔股票，查看 9 項財務指標</div>"
                )

            else:
                # Row for the selected ticker
                row = f_score_df[f_score_df["Stock"] == selected_code].iloc[0]
                score = int(row["F_Score"])
                # Colors by strength level; unknown level falls back to gray
                label, fg, bg = STRENGTH_STYLE.get(
                    row["Financial_Strength"], (row["Financial_Strength"], MUTED, "#F1F5F9")
                )

                # Overview card: code, name, strength badge | score, progress bar
                html(
                    f'<div class="stock-card" style="--fg:{fg};--bg:{bg};">'
                    f"<div>"
                    f'<div class="stock-code">{selected_code}</div>'
                    f'<div class="stock-name">{F_SCORE_NAMES.get(selected_code, selected_code)}</div>'
                    f'<div class="badge">財務強度：{label}</div>'
                    f"</div>"
                    f'<div style="text-align:right;">'
                    f'<div class="stock-score">{score}<span> / 9</span></div>'
                    # Bar width = score / 9, e.g. 6/9 → 67%
                    f'<div class="progress"><div style="width:{score / 9:.0%};"></div></div>'
                    f"</div>"
                    f"</div>"
                )

                # 3 columns, one per criteria group
                group_cols = st.columns(3, gap="medium")

                for col, (group_name, items) in zip(group_cols, CRITERIA_GROUPS):
                    with col:
                        # Number of passed criteria in the group, e.g. "獲利能力 3 / 4"
                        passed = sum(int(row[c]) for c, _, _ in items)
                        html(f'<div class="crit-group">{group_name}　{passed} / {len(items)}</div>')

                        for column_name, title, desc in items:
                            ok = int(row[column_name]) == 1   # 1 = passed
                            # Passed → green, failed → red
                            c_fg, c_bg = (SUCCESS, SUCCESS_SOFT) if ok else (DANGER, DANGER_SOFT)
                            html(
                                f'<div class="crit" style="--fg:{c_fg};--bg:{c_bg};">'
                                f'<div class="crit-icon">{"✓" if ok else "✕"}</div>'
                                f"<div>"
                                f'<div class="crit-title">{title}'
                                f'<span class="crit-status">{"符合" if ok else "未符合"}</span></div>'
                                f'<div class="crit-desc">{desc}</div>'
                                f"</div></div>"
                            )


# =========================================================
# FOOTER — always shown at the bottom of the page
# =========================================================

html(
    '<div class="footer">'
    "台灣量化投資分析系統 · 投資組合分析與 Piotroski F-Score 財務強度評估"
    "<br>本系統僅供學術研究使用，不構成任何投資建議。"
    "</div>"
)