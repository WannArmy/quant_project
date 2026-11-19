# Taiwan Quantitative Investment Analysis System

## Overview
This repository contains a **Streamlit** web application that lets users:
- Pick Taiwanese stocks and assign portfolio weights.
- Fetch historical price data via `yfinance`.
- Compute returns, volatility, drawdown, and the **Piotroski F‑Score** for each ticker.
- Visualise results with Plotly charts.

The app is split into two main parts:
- `app.py` – the Streamlit UI.
- `src/` – reusable data‑processing utilities (price fetch, risk calculations, etc.).
- `final_analysis.py` – Piotroski F‑Score implementation.
- `database_schema.md` – optional SQLite schema for persisting data.

## Prerequisites (local development)
```bash
# Clone the repo (after you push it to GitHub – see below)
# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run app.py
```

## Deploying via **Streamlit Community Cloud** (free & easiest)
1. **Push the code to a GitHub repository**
   ```bash
   cd C:/Users/Richie\ Chua/Downloads/quant_project
   # If you haven't created a repository yet, do it on GitHub first.
   # Then add the remote URL (replace <USERNAME> and <REPO>)
   git remote add origin https://github.com/<USERNAME>/quant‑project.git
   git branch -M main
   git push -u origin main
   ```
2. **Open Streamlit Cloud** – go to https://share.streamlit.io/ and sign in with your GitHub account.
3. Click **"New app"**, select the repository you just pushed, and set the entry point to `app.py`.
4. Click **Deploy**. Streamlit Cloud will:
   - Clone the repo.
   - Install the packages listed in `requirements.txt`.
   - Run `streamlit run app.py --server.port $PORT` (the `$PORT` environment variable is injected automatically).
5. Once the deployment finishes you’ll get a public URL like `https://<your‑username>.streamlit.app/` that you can share.

### Optional: Custom domain & secrets
- **Custom domain** – In Streamlit Cloud, go to *Settings → Domain* and follow the DNS instructions.
- **Secrets** – If you ever need API keys, add them under *Settings → Advanced → Secrets* (`KEY=value`). Access them in Python with `os.getenv("KEY")`.

## Deploying to **Render** (alternative) – one‑click Docker not required
1. After pushing to GitHub, create a free Render account.
2. Click **New → Web Service**, connect the GitHub repo.
3. Set:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port $PORT`
4. Render will provision a service and give you a URL like `https://quant‑project.onrender.com`.

## Keeping the SQLite DB (`quant.db`) persistent
- By default Streamlit Cloud uses an **ephemeral** filesystem – the SQLite file disappears on each redeploy.
- If you need persistent storage, consider:
  1. Switching the DB URL to a cloud‑hosted PostgreSQL instance (set `DATABASE_URL` in Streamlit Cloud *Secrets*). The same SQLAlchemy models work unchanged.
  2. Or store the DB alongside the repo and commit changes (not recommended for large data).

## Quick checklist before pushing
- Ensure `requirements.txt` lists all packages (`yfinance`, `pandas`, `numpy`, `streamlit`, `plotly`).
- Add a `.gitignore` (already present) to exclude `__pycache__`, `quant.db`, and other heavy files.
- Verify the app runs locally: `streamlit run app.py`.
- Update the README (this file) if you add new features.

---

*Happy analysing!*