"""Page 1 — Pipeline Results: browse the latest pipeline run."""
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

_MARKET_LEDGER = Path(__file__).parent.parent.parent
if str(_MARKET_LEDGER) not in sys.path:
    sys.path.insert(0, str(_MARKET_LEDGER))

from dashboard.utils.data_loaders import load_decision_log, load_latest_closes
from dashboard.utils.formatting import SIGNAL_COLORS, SIGNAL_ORDER
from dashboard.utils.refresh import render_refresh_controls

st.set_page_config(page_title="Pipeline Results — MarketLedger", layout="wide")
st.title("Pipeline Results")
render_refresh_controls()

df = load_decision_log()

if df.empty:
    st.warning("No pipeline output found. Run `py -m main` first.")
    st.stop()

# Fill missing Current Price: coalesce with the earlier strategy-stage Price column,
# then fall back to the DB for any still-missing tickers.
if "Current Price" in df.columns:
    if "Price" in df.columns:
        df["Current Price"] = df["Current Price"].fillna(df["Price"])
    missing_px = df.loc[df["Current Price"].isna(), "Ticker"].dropna().unique().tolist()
    if missing_px:
        closes = load_latest_closes(tuple(missing_px))
        if not closes.empty:
            df.loc[df["Current Price"].isna(), "Current Price"] = (
                df.loc[df["Current Price"].isna(), "Ticker"].map(closes)
            )

# ── Top metrics ────────────────────────────────────────────────────────────────
total = len(df)
selected = int((df.get("Decision", pd.Series()) == "selected").sum()) if "Decision" in df.columns else 0
rejected = total - selected
strong_buy = int((df.get("Investment Signal", pd.Series()) == "Strong Buy").sum()) if "Investment Signal" in df.columns else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Screened", total)
c2.metric("Selected", selected)
c3.metric("Rejected", rejected)
c4.metric("Strong Buys", strong_buy)

st.markdown("---")

# ── Sidebar filters ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")

    signals = sorted(df["Investment Signal"].dropna().unique().tolist()) if "Investment Signal" in df.columns else []
    sel_signals = st.multiselect("Investment Signal", options=signals, default=signals)

    decisions = sorted(df["Decision"].dropna().unique().tolist()) if "Decision" in df.columns else []
    sel_decisions = st.multiselect("Decision", options=decisions, default=decisions)

    strategies = sorted(df["Strategy"].dropna().unique().tolist()) if "Strategy" in df.columns else []
    sel_strategies = st.multiselect("Strategy", options=strategies, default=strategies)

    distress_opts = sorted(df["Distress Risk"].dropna().unique().tolist()) if "Distress Risk" in df.columns else []
    sel_distress = st.multiselect("Distress Risk", options=distress_opts, default=distress_opts)

    tail_opts = sorted(df["Tail_Risk"].dropna().unique().tolist()) if "Tail_Risk" in df.columns else []
    sel_tail = st.multiselect("Tail Risk", options=tail_opts, default=tail_opts)

    gate_opts = sorted(df["RV_Gate_Count"].dropna().unique().tolist()) if "RV_Gate_Count" in df.columns else []
    sel_gates = st.multiselect("Gate Count (failed)", options=gate_opts, default=gate_opts)

    min_mos = st.slider(
        "Min Undervalued %",
        min_value=-200,
        max_value=200,
        value=-200,
        step=5,
    )

# Apply filters
filtered = df.copy()
if sel_signals and "Investment Signal" in filtered.columns:
    filtered = filtered[filtered["Investment Signal"].isin(sel_signals)]
if sel_decisions and "Decision" in filtered.columns:
    filtered = filtered[filtered["Decision"].isin(sel_decisions)]
if sel_strategies and "Strategy" in filtered.columns:
    filtered = filtered[filtered["Strategy"].isin(sel_strategies)]
if sel_distress and "Distress Risk" in filtered.columns:
    filtered = filtered[filtered["Distress Risk"].isin(sel_distress)]
if sel_tail and "Tail_Risk" in filtered.columns:
    filtered = filtered[filtered["Tail_Risk"].isin(sel_tail)]
if sel_gates and "RV_Gate_Count" in filtered.columns:
    filtered = filtered[filtered["RV_Gate_Count"].isin(sel_gates)]
if "Undervalued %" in filtered.columns:
    # Column is stored as strings like "85.2%" — strip suffix before parsing.
    # NaN rows (rejected tickers with no valuation) always pass through.
    mos = pd.to_numeric(
        filtered["Undervalued %"].astype(str).str.rstrip("%"),
        errors="coerce",
    )
    filtered = filtered[mos.isna() | (mos >= min_mos)]

# ── Charts row ─────────────────────────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Signal Distribution")
    if "Investment Signal" in df.columns:
        counts = df["Investment Signal"].value_counts().reindex(SIGNAL_ORDER).dropna()
        fig = px.bar(
            x=counts.index,
            y=counts.values,
            color=counts.index,
            color_discrete_map=SIGNAL_COLORS,
            labels={"x": "Signal", "y": "Count"},
            height=280,
        )
        fig.update_layout(showlegend=False, margin=dict(t=10, b=20))
        st.plotly_chart(fig, width='stretch')

with col_right:
    st.subheader("Rejection Funnel")
    if "DecisionStage" in df.columns:
        stage_counts = df["DecisionStage"].value_counts()
        # Build funnel order: start with total, then rejection stages, end with selected
        funnel_stages = []
        funnel_values = []
        funnel_stages.append("Total Screened")
        funnel_values.append(total)
        for stage in ["momentum", "distress", "tail_risk", "valuation_sanity"]:
            n = int((df["DecisionStage"] == stage).sum())
            if n:
                funnel_stages.append(f"Rejected: {stage}")
                funnel_values.append(n)
        funnel_stages.append("Selected")
        funnel_values.append(selected)
        fig_f = go.Figure(go.Funnel(
            y=funnel_stages,
            x=funnel_values,
            textinfo="value+percent initial",
        ))
        fig_f.update_layout(height=280, margin=dict(t=10, b=20))
        st.plotly_chart(fig_f, width='stretch')

# ── Gate pass rates ────────────────────────────────────────────────────────────
gate_cols = ["RV_Gate_Distress", "RV_Gate_Tail", "RV_Gate_Momentum", "RV_Gate_Valuation"]
present_gates = [g for g in gate_cols if g in df.columns]
if present_gates:
    st.subheader("Gate Trip Rates")
    gate_data = []
    for g in present_gates:
        tripped = df[g].sum() if df[g].dtype == bool else (df[g] == True).sum()  # noqa: E712
        gate_data.append({
            "Gate": g.replace("RV_Gate_", ""),
            "Tripped": int(tripped),
            "Rate %": round(100 * tripped / max(total, 1), 1),
        })
    st.dataframe(
        pd.DataFrame(gate_data),
        column_config={
            "Rate %": st.column_config.ProgressColumn("Trip Rate %", min_value=0, max_value=100),
        },
        hide_index=True,
        width='content',
    )

st.markdown(f"**{len(filtered)} rows** after filters")

# ── Main table ─────────────────────────────────────────────────────────────────
DEFAULT_COLS = [
    "Ticker", "Strategy", "Investment Signal", "Current Price",
    "Fair Value (Base)", "Undervalued %", "Altman Z-Score",
    "Distress Risk", "Tail_Risk", "RV_Gate_Count",
    "DecisionStage", "RejectedReason",
]
available_cols = [c for c in DEFAULT_COLS if c in filtered.columns]
all_cols = list(filtered.columns)

with st.expander("Column selector", expanded=False):
    shown_cols = st.multiselect(
        "Columns to show",
        options=all_cols,
        default=available_cols,
    )

display_cols = shown_cols if shown_cols else available_cols
display_df = filtered[display_cols].reset_index(drop=True)

st.dataframe(display_df, width='stretch', height=500)

# ── Drill-in ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Drill into a stock")
ticker_options = sorted(filtered["Ticker"].dropna().unique().tolist()) if "Ticker" in filtered.columns else []
if ticker_options:
    drill_ticker = st.selectbox("Select ticker", options=ticker_options)
    if st.button("Go to Stock Detail →"):
        st.session_state["selected_ticker"] = drill_ticker
        st.switch_page("pages/2_Stock_Detail.py")
