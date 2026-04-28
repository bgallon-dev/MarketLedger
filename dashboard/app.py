"""MarketLedger Dashboard — home / landing page."""
import sys
from pathlib import Path

import streamlit as st

# Ensure the MarketLedger package root is on the path
_MARKET_LEDGER = Path(__file__).parent.parent
if str(_MARKET_LEDGER) not in sys.path:
    sys.path.insert(0, str(_MARKET_LEDGER))

# Ensure DB indices exist on first run (no-op if they already do)
try:
    from database.database import ensure_indices
    ensure_indices()
except Exception:
    pass

from dashboard.utils.data_loaders import (
    load_decision_log,
    load_pipeline_results,
    load_positions,
    load_portfolio_meta,
    load_research_csvs,
)
from dashboard.utils.formatting import SIGNAL_COLORS
from dashboard.utils.refresh import render_refresh_controls

st.set_page_config(
    page_title="MarketLedger",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 MarketLedger")

    # Current regime banner
    research = load_research_csvs()
    regime_label = "Unknown"
    regime_color = "#6c757d"
    if "regime_labels" in research:
        rl = research["regime_labels"]
        if not rl.empty:
            last = rl.iloc[-1]
            regime_label = str(last.get("Market_Regime", last.iloc[0] if len(last) else "Unknown"))
            if "BULL" in regime_label.upper():
                regime_color = "#198754"
            elif "BEAR" in regime_label.upper():
                regime_color = "#dc3545"
            else:
                regime_color = "#fd7e14"
    st.markdown(
        f'<div style="background:{regime_color};color:#fff;padding:6px 12px;'
        f'border-radius:6px;text-align:center;font-weight:600;margin-bottom:8px">'
        f'Regime: {regime_label}</div>',
        unsafe_allow_html=True,
    )


render_refresh_controls()

# ── Home summary ──────────────────────────────────────────────────────────────
st.title("MarketLedger")
st.caption("Local-first quantitative equity research pipeline")

# Pipeline run timestamp
from pathlib import Path as _Path
import os

_YAHOO_PROJECT = _MARKET_LEDGER.parent
_csv = _YAHOO_PROJECT / "buy_list_with_projections.csv"
last_run = "No pipeline output found"
if _csv.exists():
    ts = os.path.getmtime(_csv)
    from datetime import datetime
    last_run = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")

# Metrics
decision_log = load_decision_log()
positions = load_positions()
meta = load_portfolio_meta()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Last Pipeline Run", last_run)

with col2:
    selected = 0
    strong_buy = 0
    if not decision_log.empty:
        if "Decision" in decision_log.columns:
            selected = int((decision_log["Decision"] == "selected").sum())
        if "Investment Signal" in decision_log.columns:
            strong_buy = int((decision_log["Investment Signal"] == "Strong Buy").sum())
    st.metric("Selected Stocks", selected)

with col3:
    st.metric("Strong Buys", strong_buy)

with col4:
    open_pos = 0
    if not positions.empty and "status" in positions.columns:
        open_pos = int((positions["status"] == "open").sum())
    portfolio_val = meta.get("cash", "—")
    try:
        portfolio_val = f"${float(portfolio_val):,.0f} cash"
    except (TypeError, ValueError):
        portfolio_val = "—"
    st.metric("Open Positions", open_pos, delta=portfolio_val)

st.markdown("---")

# Signal distribution mini-chart
if not decision_log.empty and "Investment Signal" in decision_log.columns:
    import plotly.express as px
    sig_counts = decision_log["Investment Signal"].value_counts().reset_index()
    sig_counts.columns = ["Signal", "Count"]
    fig = px.bar(
        sig_counts,
        x="Signal",
        y="Count",
        color="Signal",
        color_discrete_map=SIGNAL_COLORS,
        title="Signal Distribution (latest run)",
        height=300,
    )
    fig.update_layout(showlegend=False, margin=dict(t=40, b=20))
    st.plotly_chart(fig, width='stretch')
else:
    st.info("Run the pipeline (`py -m main`) to generate output, then refresh.")

st.markdown("---")
st.caption("Navigate using the sidebar pages. Use **Refresh all data** to reload after a new pipeline run.")
