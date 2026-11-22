# -*- coding: utf-8 -*-
"""Streamlit app for Taiwan Quantitative Analysis.
This version has been rebuilt to be syntactically correct and runnable.
The original logic is preserved where possible; missing helper modules are
imported from the local ``src`` package.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

# Load environment variables (e.g., database URL) – optional
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Custom data‑processing functions (assumed to exist in the ``src`` package)
# ---------------------------------------------------------------------------
from src.fetch_data import get_price_data  # fetch historical prices for one ticker
from src.preprocess import calculate_returns  # prices → daily returns
from src.risk import (
    calculate_portfolio_returns,
    calculate_annualized_volatility,
    calculate_maximum_drawdown,
)
from final_analysis import run_f_score_analysis  # returns a DataFrame of F‑Scores

# ---------------------------------------------------------------------------
# Page configuration (must be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="台灣量化投資分析系統",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Theme colours (used throughout the app)
# ---------------------------------------------------------------------------
PRIMARY = "#2563EB"
PRIMARY_SOFT = "#EFF6FF"
SUCCESS = "#16A34A"
SUCCESS_SOFT = "#F0FDF4"
DANGER = "#DC2626"
DANGER_SOFT = "#FEF2F2"
WARNING = "#D97706"
TEXT = "#0F172A"
MUTED = "#334155"
SUBTLE = "#475569"
BORDER = "#E2E8F0"

# ---------------------------------------------------------------------------
# Helper rendering utilities
# ---------------------------------------------------------------------------
def html(content: str) -> None:
    """Render raw HTML via ``st.markdown`` with ``unsafe_allow_html=True``."""
    st.markdown(content, unsafe_allow_html=True)


def section_header(num: str, title: str, subtitle: str = "") -> None:
    """Render a section heading with a coloured number box."""
    html(
        f"""
        <div class='section-head'>
            <div class='section-num'>{num}</div>
            <div class='section-title'>{title}</div>
            {f'<div class="section-subtitle">{subtitle}</div>' if subtitle else ''}
        </div>
        """
    )


def kpi_card(label: str, value: str, desc: str, accent: str = PRIMARY) -> None:
    """Render a KPI card used on the results page."""
    html(
        f"""
        <div class='kpi' style='--accent:{accent};'>
            <div class='kpi-label'>{label}</div>
            <div class='kpi-value'>{value}</div>
            <div class='kpi-desc'>{desc}</div>
        </div>
        """
    )


def base_layout(fig: go.Figure, height: int = 420, y_prefix: str | None = None, y_format: str | None = None) -> go.Figure:
    """Apply a common Plotly layout (white background, dark text, grid)."""
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color=TEXT, size=13),
        hovermode="x unified",
        showlegend=False,
        hoverlabel=dict(font_color=TEXT, bgcolor="#FFFFFF"),
        xaxis=dict(showgrid=False, linecolor="#CBD5E1", tickfont=dict(color=TEXT, size=13), title_font=dict(color=TEXT)),
        yaxis=dict(
            showgrid=True,
            gridcolor="#E2E8F0",
            zeroline=False,
            tickprefix=y_prefix,
            tickformat=y_format,
            tickfont=dict(color=TEXT, size=13),
            title_font=dict(color=TEXT),
        ),
    )
    return fig

# ---------------------------------------------------------------------------
# Data dictionaries
# ---------------------------------------------------------------------------
STOCK_NAMES = {
    "2330TW": "台積電",
    "2317TW": "鴻海",
    "2454TW": "聯發科",
    "2308TW": "台達電",
    "2881TW": "富邦金",
    "2891TW": "中信金",
    "2002TW": "中鋼",
    "1301TW": "台塑",
    "1303TW": "南亞",
    "2882TW": "國泰金",
}

ALL_STOCKS = list(STOCK_NAMES.keys())

F_SCORE_NAMES = {
    "2330": "台積電",
    "2317": "鴻海",
    "2454": "聯發科",
    "2308": "台達電",
    "2002": "中鋼",
    "1301": "台塑",
    "1303": "南亞",
}

# ---------------------------------------------------------------------------
# Helper to display ticker label (e.g., "2330　台積電")
# ---------------------------------------------------------------------------
def stock_label(code: str) -> str:
    """Return ``code`` without the ``TW`` suffix plus the Chinese name.
    Example: ``"2330TW"`` → ``"2330　台積電"``.
    """
    code = str(code)
    raw = code.replace("TW", "")
    name = STOCK_NAMES.get(code, "")
    return f"{raw}　{name}"

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

# ---------------------------------------------------------------------------
# HERO banner (static HTML)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# SECTION 1 – Portfolio configuration
# ---------------------------------------------------------------------------
with st.container(border=True):
    section_header("1", "投資組合設定", "選擇您想分析的台灣股票，並設定投資金額與配置比例。")

    # Columns: left for stock selection, right for capital input
    config_col1, config_col2 = st.columns([2.4, 1], gap="large")

    with config_col1:
        selected_stocks = st.multiselect(
            "選擇股票",
            options=ALL_STOCKS,
            default=ALL_STOCKS,
            format_func=stock_label,
            placeholder="請選擇股票",
        )

    with config_col2:
        initial_investment = st.number_input(
            "初始投資金額（NT$）",
            min_value=1000.0,
            value=10000.0,
            step=1000.0,
            format="%.0f",
        )

    # Defaults in case no stock is selected
    weights: dict[str, float] = {}
    normalize_weights = True

    if selected_stocks:
        with st.expander("⚖️ 投資組合權重設定", expanded=False):
            st.caption("請設定每檔股票的投資比例。若不確定如何配置，可以使用等權重。")
            weight_cols = st.columns(4)
            default_weight = 1.0 / len(selected_stocks)
            for i, ticker in enumerate(selected_stocks):
                with weight_cols[i % 4]:
                    w = st.number_input(
                        stock_label(ticker),
                        min_value=0.0,
                        max_value=1.0,
                        value=default_weight,
                        step=0.01,
                        format="%.2f",
                        key=f"weight_{ticker}",
                    )
                    weights[ticker] = w
            total_weight = sum(weights.values())
            w_col1, w_col2 = st.columns([1, 1])
            with w_col1:
                normalize_weights = st.checkbox("自動標準化權重", value=True)
            with w_col2:
                ok = np.isclose(total_weight, 1.0)
                color = SUCCESS if ok else WARNING
                html(
                    f"""
                    <div style='text-align:right;color:{MUTED};font-size:14px;'>
                        目前總權重：<b style='color:{color};'>{total_weight:.1%}</b>
                    </div>
                    """
                )

    # Run analysis button
    run_col, _ = st.columns([1, 3])
    with run_col:
        run_analysis = st.button("🚀 開始分析", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# ANALYSIS PROCESS (executed when button pressed)
# ---------------------------------------------------------------------------
if run_analysis:
    # Validate input
    if not selected_stocks:
        st.error("請至少選擇一檔股票。")
    else:
        # Normalise weights if required
        total_weight = sum(weights.values())
        if normalize_weights or not np.isclose(total_weight, 1.0):
            if total_weight == 0:
                st.error("權重總和不能為 0。")
                st.stop()
            weights = {t: w / total_weight for t, w in weights.items()}

        # Fetch price data and compute returns
        progress = st.progress(0, text="正在取得歷史股價資料……")
        returns_dict = {}
        for i, ticker in enumerate(selected_stocks):
            price_data = get_price_data(ticker.replace("TW", ".TW"))
            if price_data is not None and not price_data.empty:
                stock_returns = calculate_returns(price_data)
                if stock_returns is not None and not stock_returns.empty:
                    returns_dict[ticker] = stock_returns
            progress.progress((i + 1) / len(selected_stocks), text=f"取得 {STOCK_NAMES[ticker]} 資料……")
        progress.empty()

        if not returns_dict:
            st.error("無法取得股票歷史資料，請稍後再試。")
            st.stop()

        # Build returns DataFrame
        returns_df = pd.concat(returns_dict, axis=1)
        returns_df = returns_df.dropna()
        weight_series = pd.Series(weights).reindex(returns_df.columns).fillna(0)

        # Risk metrics
        portfolio_returns = calculate_portfolio_returns(returns_df, weight_series)
        annual_volatility = calculate_annualized_volatility(portfolio_returns)
        maximum_drawdown = calculate_maximum_drawdown(portfolio_returns)

        # Equity curve
        equity_curve = (1 + portfolio_returns).cumprod() * initial_investment

        # F‑Score analysis (run once)
        with st.spinner("正在進行財務強度分析……"):
            f_score_df = run_f_score_analysis()

        # Store results in session state
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
        st.toast("分析完成！", icon="✅")

# ---------------------------------------------------------------------------
# DISPLAY RESULTS
# ---------------------------------------------------------------------------
if not st.session_state.get("analysis_done"):
    html(
        f"""
        <div style='text-align:center;color:{MUTED};padding:48px 0;'>
            <div style='font-size:40px;'>📊</div>
            設定完成後，點擊「開始分析」即可查看結果
        </div>
        """
    )
else:
    ss = st.session_state
    equity_curve = ss.equity_curve
    initial_investment = ss.initial_investment
    final_value = float(equity_curve.iloc[-1])
    total_return = final_value / initial_investment - 1

    # Tabs for risk and F‑Score
    tab_risk, tab_fscore = st.tabs(["📈 投資組合與風險", "🏦 Piotroski F-Score"])

    with tab_risk:
        section_header("2", "投資組合風險分析", "透過歷史資料評估投資組合的波動程度與潛在下跌幅度。")
        # KPI cards
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            kpi_card("初始投資金額", f"NT$ {initial_investment:,.0f}", "歷史績效分析的起始金額", MUTED)
        with k2:
            kpi_card(
                "期末價值",
                f"NT$ {final_value:,.0f}",
                f"累積報酬 {total_return:+.2%}",
                SUCCESS if total_return >= 0 else DANGER,
            )
        with k3:
            kpi_card("年化波動率", f"{ss.annual_volatility:.2%}", "衡量投資報酬的波動程度", PRIMARY)
        with k4:
            kpi_card("最大回撤", f"{ss.maximum_drawdown:.2%}", "歷史高點至低點的最大跌幅", DANGER)

        # Equity curve chart
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=equity_curve.index,
                y=equity_curve.values,
                mode="lines",
                line=dict(width=2, color=PRIMARY),
                fill="tozeroy",
                fillcolor="rgba(37, 99, 235, 0.08)",
                hovertemplate="投資組合價值：NT$ %{y:,.0f}<extra></extra>",
            )
        )
        fig.add_hline(y=initial_investment, line_dash="dot", line_color=SUBTLE, annotation_text="初始金額", annotation_font_color=MUTED)
        fig = base_layout(fig, height=420, y_prefix="NT$ ")
        st.plotly_chart(fig, use_container_width=True)

    with tab_fscore:
        section_header("3", "Piotroski F-Score 財務強度分析", "透過 9 項財務指標評估企業的財務狀況與營運改善程度。")
        # Explanation expander
        with st.expander("什麼是 Piotroski F-Score？"):
            st.markdown(
                """
                Piotroski F-Score 是一套由 **0 至 9 分** 組成的財務強度評估方法。
                分數越高，代表公司在獲利能力、財務狀況與營運效率方面具有較多正向訊號。

                **8–9 分：強**　·　**6–7 分：良好**　·　**4–5 分：中等**　·　**0–3 分：弱**
                """
            )
        # Show the F‑Score DataFrame (simple display)
        st.dataframe(ss.f_score_df)
