"""Page 3 — Paper Portfolio: positions, P&L, and rebalance history."""
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

_MARKET_LEDGER = Path(__file__).parent.parent.parent
if str(_MARKET_LEDGER) not in sys.path:
    sys.path.insert(0, str(_MARKET_LEDGER))

from dashboard.utils.data_loaders import (
    load_latest_closes,
    load_portfolio_meta,
    load_positions,
    load_rebalance_log,
    load_transactions,
)
from dashboard.utils.formatting import SIGNAL_COLORS, fmt_currency, fmt_pct
from dashboard.utils.refresh import render_refresh_controls

st.set_page_config(page_title="Paper Portfolio — MarketLedger", layout="wide")
st.title("Paper Portfolio")
render_refresh_controls()

positions = load_positions()
transactions = load_transactions()
rebalance_log = load_rebalance_log()
meta = load_portfolio_meta()

if positions.empty and not meta:
    st.warning("No paper trading data found. Run `py -m paper_trading_engine.py --run-now` to start.")
    st.stop()

# ── Portfolio summary ──────────────────────────────────────────────────────────
starting_cash = float(meta.get("starting_cash", 0) or 0)
current_cash = float(meta.get("cash", 0) or 0)

open_pos = pd.DataFrame()
closed_pos = pd.DataFrame()
if not positions.empty and "status" in positions.columns:
    open_pos = positions[positions["status"] == "open"].copy()
    closed_pos = positions[positions["status"] == "closed"].copy()

# MTM pricing for open positions
mtm_closes = pd.Series(dtype=float)
if not open_pos.empty and "ticker" in open_pos.columns:
    symbols = tuple(open_pos["ticker"].dropna().unique().tolist())
    mtm_closes = load_latest_closes(symbols)

open_pos_value = 0.0
if not open_pos.empty:
    open_pos = open_pos.copy()
    open_pos["current_price"] = open_pos["ticker"].map(mtm_closes)
    # Fall back to fill_price when live price is unavailable; flag stale rows
    open_pos["price_stale"] = open_pos["current_price"].isna()
    open_pos["current_price"] = open_pos["current_price"].fillna(open_pos["fill_price"])
    open_pos["position_value"] = open_pos["shares"] * open_pos["current_price"]
    open_pos["unrealized_pnl_pct"] = (
        (open_pos["current_price"] / open_pos["fill_price"]) - 1
    ) * 100
    open_pos_value = open_pos["position_value"].sum()

total_portfolio = current_cash + open_pos_value
total_pnl_pct = (total_portfolio / starting_cash - 1) * 100 if starting_cash else 0

closed_hit_rate = 0.0
if not closed_pos.empty and "pnl_pct" in closed_pos.columns:
    closed_pnl = pd.to_numeric(closed_pos["pnl_pct"], errors="coerce")
    closed_hit_rate = (closed_pnl > 0).mean() * 100

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Starting Cash", fmt_currency(starting_cash))
m2.metric("Current Cash", fmt_currency(current_cash))
m3.metric("Portfolio Value", fmt_currency(total_portfolio))
m4.metric("Total P&L", fmt_pct(total_pnl_pct), delta=fmt_currency(total_portfolio - starting_cash))
m5.metric("Open Positions", len(open_pos), delta=f"Closed hit: {fmt_pct(closed_hit_rate)}")

st.markdown("---")

# ── Open positions table ───────────────────────────────────────────────────────
st.subheader("Open Positions")
if open_pos.empty:
    st.info("No open positions.")
else:
    stale_tickers = open_pos.loc[open_pos.get("price_stale", False), "ticker"].tolist()
    if stale_tickers:
        st.warning(
            f"No recent price data for: **{', '.join(stale_tickers)}** — "
            "showing entry fill price. Run `py -m data.data` to refresh."
        )

    display_open_cols = [c for c in [
        "ticker", "strategy", "signal", "entry_date",
        "fill_price", "current_price", "unrealized_pnl_pct",
        "shares", "position_value",
        "rv_gate_count", "rv_distress", "rv_tail", "rv_momentum", "regime",
    ] if c in open_pos.columns]
    st.dataframe(
        open_pos[display_open_cols].reset_index(drop=True),
        column_config={
            "fill_price":        st.column_config.NumberColumn("Entry Fill", format="$%.2f"),
            "current_price":     st.column_config.NumberColumn("Current Price", format="$%.2f"),
            "unrealized_pnl_pct": st.column_config.NumberColumn("Unrealized P&L %", format="%.1f%%"),
            "position_value":    st.column_config.NumberColumn("Position Value", format="$%.0f"),
            "shares":            st.column_config.NumberColumn("Shares", format="%.2f"),
        },
        width='stretch',
    )

    # Composition donut
    if "position_value" in open_pos.columns and "strategy" in open_pos.columns:
        st.subheader("Portfolio Composition by Strategy")
        comp_df = open_pos.groupby("strategy")["position_value"].sum().reset_index()
        fig_pie = px.pie(comp_df, names="strategy", values="position_value",
                         title="Open position value by strategy", height=300)
        fig_pie.update_layout(margin=dict(t=40, b=10))
        st.plotly_chart(fig_pie, width='stretch')

st.markdown("---")

# ── P&L analysis ──────────────────────────────────────────────────────────────
st.subheader("P&L Analysis")
tab_closed, tab_by_signal, tab_by_strategy = st.tabs(
    ["Closed Positions", "P&L by Signal", "P&L by Strategy"]
)

with tab_closed:
    if closed_pos.empty:
        st.info("No closed positions yet.")
    else:
        display_closed = [c for c in [
            "ticker", "strategy", "signal", "entry_date", "exit_date",
            "hold_days", "fill_price", "exit_price", "pnl_pct",
            "mos_at_entry", "rv_gate_count",
        ] if c in closed_pos.columns]
        st.dataframe(
            closed_pos[display_closed].reset_index(drop=True),
            column_config={
                "pnl_pct":     st.column_config.NumberColumn("P&L %", format="%.1f%%"),
                "fill_price":  st.column_config.NumberColumn("Entry", format="$%.2f"),
                "exit_price":  st.column_config.NumberColumn("Exit", format="$%.2f"),
                "mos_at_entry": st.column_config.NumberColumn("MOS at Entry", format="%.1f%%"),
            },
            width='stretch',
        )

with tab_by_signal:
    if closed_pos.empty or "signal" not in closed_pos.columns or "pnl_pct" not in closed_pos.columns:
        st.info("Insufficient data.")
    else:
        grp = closed_pos.groupby("signal")["pnl_pct"].agg(["mean", "count"]).reset_index()
        grp.columns = ["Signal", "Mean P&L %", "Count"]
        fig_sig = px.bar(grp, x="Signal", y="Mean P&L %", color="Signal",
                         color_discrete_map=SIGNAL_COLORS,
                         text="Count", title="Mean P&L % by Investment Signal at Entry",
                         height=300)
        fig_sig.update_layout(showlegend=False, margin=dict(t=40, b=20))
        st.plotly_chart(fig_sig, width='stretch')

with tab_by_strategy:
    if closed_pos.empty or "strategy" not in closed_pos.columns or "pnl_pct" not in closed_pos.columns:
        st.info("Insufficient data.")
    else:
        grp_s = closed_pos.groupby("strategy")["pnl_pct"].agg(["mean", "count"]).reset_index()
        grp_s.columns = ["Strategy", "Mean P&L %", "Count"]
        fig_strat = px.bar(grp_s, x="Strategy", y="Mean P&L %", text="Count",
                           title="Mean P&L % by Strategy", height=300)
        fig_strat.update_layout(margin=dict(t=40, b=20))
        st.plotly_chart(fig_strat, width='stretch')

st.markdown("---")

# ── Transaction log ────────────────────────────────────────────────────────────
st.subheader("Transaction Log")
if transactions.empty:
    st.info("No transactions recorded.")
else:
    tx_display = transactions.sort_values("id", ascending=False) if "id" in transactions.columns else transactions
    st.dataframe(
        tx_display.reset_index(drop=True),
        column_config={
            "fill_price":   st.column_config.NumberColumn("Fill Price", format="$%.2f"),
            "market_price": st.column_config.NumberColumn("Market Price", format="$%.2f"),
            "value":        st.column_config.NumberColumn("Value", format="$%.0f"),
            "cash_after":   st.column_config.NumberColumn("Cash After", format="$%.0f"),
        },
        width='stretch',
        height=300,
    )

    # Slippage summary
    with st.expander("Slippage analysis"):
        if "slippage_bps" in transactions.columns:
            slip = pd.to_numeric(transactions["slippage_bps"], errors="coerce")
            spread = pd.to_numeric(transactions.get("spread_bps", pd.Series(dtype=float)), errors="coerce")
            cs1, cs2, cs3 = st.columns(3)
            cs1.metric("Mean Slippage (bps)", f"{slip.mean():.1f}")
            cs2.metric("Mean Spread (bps)", f"{spread.mean():.1f}" if not spread.isna().all() else "—")
            total_tx_cost = (slip + spread).sum() if not spread.isna().all() else slip.sum()
            cs3.metric("Total Cost (bps·sum)", f"{total_tx_cost:.0f}")

st.markdown("---")

# ── Rebalance history ──────────────────────────────────────────────────────────
st.subheader("Rebalance History")
if rebalance_log.empty:
    st.info("No rebalance records yet.")
else:
    st.dataframe(rebalance_log.reset_index(drop=True), width='stretch')

    if "portfolio_value" in rebalance_log.columns:
        rb = rebalance_log.copy()
        # Use wall-clock timestamp (ts) so multiple rebalances on the same trade_date
        # are distinct x-axis points; fall back to trade_date if ts is absent.
        x_col = "ts" if "ts" in rb.columns else "trade_date"
        rb[x_col] = pd.to_datetime(rb[x_col], errors="coerce")
        rb = rb.dropna(subset=[x_col, "portfolio_value"])
        if not rb.empty:
            hover = {"trade_date": True} if x_col == "ts" and "trade_date" in rb.columns else {}
            fig_eq = px.line(rb, x=x_col, y="portfolio_value",
                             markers=True, title="Portfolio Value Over Rebalances",
                             labels={x_col: "Rebalance Time", "portfolio_value": "Value ($)"},
                             hover_data=hover,
                             height=280)
            fig_eq.update_layout(margin=dict(t=40, b=20))
            st.plotly_chart(fig_eq, width='stretch')
