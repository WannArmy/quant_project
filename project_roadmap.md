# Quantitative Finance Toolkit: 2-Semester Roadmap

Since you will be developing this project for two more semesters, you have a fantastic opportunity to take it from a solid statistical analysis tool to an **institutional-grade algorithmic trading and predictive platform**. 

Here is a structured, semester-by-semester roadmap of major features you can add that will impress your professors and build an incredible portfolio.

---

## 🚀 Next Semester: Algorithmic Trading & Machine Learning

**Theme: From Static Analysis to Dynamic Prediction & Strategy Simulation**

For the next semester, the goal is to shift the project from simply *looking at the past* to *simulating active trading strategies* and *predicting the future*.

### 1. Build a Custom Backtesting Engine
Currently, you optimize weights based on all historical data. The next step is to simulate how an active trading strategy would have performed over time.
* **Feature:** Implement a loop that steps through history day-by-day to simulate buying and selling.
* **Metrics:** Calculate the strategy's equity curve, win rate, and drawdown, comparing it directly against a "Buy & Hold" baseline.
* **Library to Explore:** You can build a simple custom one, or integrate libraries like `Backtrader` or `vectorbt`.

### 2. Implement Active Trading Strategies
Introduce algorithms that generate "Buy" or "Sell" signals.
* **Mean Reversion / Pairs Trading:** Identify two highly correlated Taiwan stocks (e.g., TSMC and MediaTek). When their price ratio diverges significantly, short the winner and buy the loser, betting they will converge.
* **Trend Following:** Implement signals based on Exponential Moving Average (EMA) crossovers or Bollinger Bands breakouts.

### 3. Machine Learning Return Prediction
Introduce an AI/ML component to your toolkit.
* **Feature:** Use a model (like **Random Forest**, **XGBoost**, or a Neural Network like **LSTM**) to predict whether a stock will go UP or DOWN tomorrow.
* **Features (Inputs):** Feed the model your historical returns, volatility, and your fundamental F-Score as inputs.
* **Integration:** Display the ML model's daily predictions directly on your Streamlit Dashboard.

### 4. Complete the Full Piotroski F-Score
* **Feature:** Right now, you have 3 criteria for the F-Score. Expand your web scraper or API logic to fetch complete balance sheets and income statements to calculate all **9 criteria**. 

---

## 🌟 Final Semester: Derivatives, Advanced Risk & Live Markets

**Theme: Institutional Modeling & Live Market Integration**

For your final semester, you can introduce complex financial instruments, alternative data, and live tracking to make it a true capstone project.

### 1. Options Pricing & "The Greeks"
Move beyond just analyzing the underlying stocks by introducing financial derivatives.
* **Feature:** Implement the **Black-Scholes Model** to price theoretical Call and Put options for your Taiwan stocks.
* **The Greeks:** Calculate Delta, Gamma, Theta, and Vega. 
* **Dashboard:** Add an "Options Chain" tab to your Streamlit app that lets users visualize how option prices change based on time decay (Theta) or volatility (Vega).

### 2. Advanced Portfolio Optimization (Black-Litterman)
Modern Portfolio Theory (MPT - which you are using now) is great, but it assumes returns follow a normal distribution and rely heavily on historical averages.
* **Feature:** Implement the **Black-Litterman Model** or **Risk Parity**. 
* Black-Litterman allows the user (investor) to input their own "views" (e.g., "I strongly believe TSMC will grow 15% next year") and combines that with market equilibrium to generate smarter portfolio weights.

### 3. NLP Sentiment Analysis (Alternative Data)
Institutional quants use more than just price data; they use text.
* **Feature:** Integrate a news API or scrape financial headlines (e.g., Yahoo Finance News or PTT Stock board for Taiwan).
* **Processing:** Use a Natural Language Processing (NLP) library like `VADER` or `HuggingFace Transformers` to score the news as Positive, Negative, or Neutral.
* **Signal:** Use this sentiment score as a new signal in your trading strategies.

### 4. "Paper Trading" Portfolio Tracker
* **Feature:** Add a database (like SQLite) to your app. Allow users to "buy" stocks in the app with fake money at current market prices.
* **Live Updates:** When they open the Streamlit dashboard the next day, it fetches the latest prices and shows their live Profit & Loss (P&L).

---

## 🎯 Summary of Added Value
By following this roadmap, your final project will demonstrate:
1. **Software Engineering:** Full-stack development (Streamlit, Databases, APIs).
2. **Quantitative Finance:** Options pricing, Black-Litterman optimization, Backtesting.
3. **Data Science & AI:** Machine Learning classification, NLP Sentiment Analysis.
