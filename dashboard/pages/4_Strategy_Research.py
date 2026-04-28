"""Page 4 — Strategy Research: backtester outputs and regime analysis."""
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

_MARKET_LEDGER = Path(__file__).parent.parent.parent
if str(_MARKET_LEDGER) not in sys.path:
    sys.path.insert(0, str(_MARKET_LEDGER))

from dashboard.utils.data_loaders import load_research_csvs
from dashboard.utils.refresh import render_refresh_controls

st.set_page_config(page_title="Strategy Research — MarketLedger", layout="wide")
st.title("Strategy Research")
render_refresh_controls()

research = load_research_csvs()

if not research:
    st.warning("No research output CSVs found in `research_outputs/`. Run the backtester first.")
    st.stop()

# ── Strategy Leaderboard ──────────────────────────────────────────────────────
st.subheader("Strategy Leaderboard")
metrics_df = research.get("strategy_risk_metrics")
if metrics_df is not None and not metrics_df.empty:
    # Identify the strategy name column
    strat_col = next((c for c in ["Strategy", "strategy", "Name", "name"] if c in metrics_df.columns), None)
    if strat_col:
        metrics_df = metrics_df.rename(columns={strat_col: "Strategy"})

    col_l, col_r = st.columns([2, 3])
    with col_l:
        sharpe_col = next((c for c in ["Sharpe", "sharpe", "Sharpe_Ratio"] if c in metrics_df.columns), None)
        pbar_config = {}
        if sharpe_col:
            pbar_config[sharpe_col] = st.column_config.ProgressColumn(
                sharpe_col, min_value=0, max_value=2.0, format="%.2f"
            )
        st.dataframe(
            metrics_df.sort_values(sharpe_col, ascending=False).reset_index(drop=True) if sharpe_col else metrics_df,
            column_config=pbar_config,
            width='stretch',
            height=400,
        )

    with col_r:
        if sharpe_col and "Strategy" in metrics_df.columns:
            sorted_df = metrics_df.sort_values(sharpe_col, ascending=True)
            fig = px.bar(
                sorted_df, x=sharpe_col, y="Strategy",
                orientation="h",
                title=f"{sharpe_col} by Strategy",
                color=sharpe_col,
                color_continuous_scale="RdYlGn",
                height=400,
            )
            fig.update_layout(margin=dict(t=40, b=20), coloraxis_showscale=False)
            st.plotly_chart(fig, width='stretch')
else:
    st.info("strategy_risk_metrics.csv not found or empty.")

# ── Risk-Return scatter ────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Risk-Return Scatter")
if metrics_df is not None and not metrics_df.empty:
    vol_col   = next((c for c in ["Volatility", "volatility", "Annual_Vol"] if c in metrics_df.columns), None)
    cagr_col  = next((c for c in ["CAGR", "cagr", "Annual_Return"] if c in metrics_df.columns), None)
    dd_col    = next((c for c in ["Max_Drawdown", "max_drawdown", "MaxDrawdown"] if c in metrics_df.columns), None)

    if vol_col and cagr_col and "Strategy" in metrics_df.columns:
        scatter_kwargs = dict(
            x=vol_col, y=cagr_col,
            text="Strategy",
            title="Risk-Return: Volatility vs CAGR",
            labels={vol_col: "Volatility", cagr_col: "CAGR"},
            height=400,
        )
        if sharpe_col:
            scatter_kwargs["color"] = sharpe_col
            scatter_kwargs["color_continuous_scale"] = "RdYlGn"
        if dd_col:
            scatter_kwargs["size"] = metrics_df[dd_col].abs()
        fig_sc = px.scatter(metrics_df, **scatter_kwargs)
        fig_sc.update_traces(textposition="top center")
        fig_sc.update_layout(margin=dict(t=40, b=20))
        st.plotly_chart(fig_sc, width='stretch')
    else:
        st.info("Need Volatility and CAGR columns for scatter.")

# ── Regime Analysis ────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Regime Analysis")
tab_winners, tab_heatmap, tab_returns = st.tabs(
    ["Regime Winners", "Strategy × Regime Heatmap", "Monthly Returns"]
)

with tab_winners:
    winners_df = research.get("regime_winners")
    if winners_df is None or winners_df.empty:
        st.info("regime_winners.csv not found.")
    else:
        # Detect active regime from regime_labels
        active_regime = None
        rl = research.get("regime_labels")
        if rl is not None and not rl.empty:
            last = rl.iloc[-1]
            active_regime = str(last.get("Market_Regime", ""))
            st.caption(f"Current detected regime: **{active_regime}**")

        def _highlight_active(row):
            regime_val = str(row.get("Regime_Bucket", row.get("regime_bucket", "")))
            if active_regime and active_regime in regime_val:
                return ["background-color: #d1e7dd"] * len(row)
            return [""] * len(row)

        st.dataframe(
            winners_df.style.apply(_highlight_active, axis=1),
            width='stretch',
        )

with tab_heatmap:
    regime_metrics = research.get("regime_strategy_metrics")
    if regime_metrics is None or regime_metrics.empty:
        st.info("regime_strategy_metrics.csv not found.")
    else:
        strat_c = next((c for c in ["Strategy", "strategy"] if c in regime_metrics.columns), None)
        regime_c = next((c for c in ["Regime_Bucket", "regime_bucket", "Regime"] if c in regime_metrics.columns), None)
        sharpe_c = next((c for c in ["Sharpe", "sharpe", "Sharpe_Ratio"] if c in regime_metrics.columns), None)

        if strat_c and regime_c and sharpe_c:
            pivot = regime_metrics.pivot_table(
                index=strat_c, columns=regime_c, values=sharpe_c, aggfunc="mean"
            )
            fig_hm = px.imshow(
                pivot,
                text_auto=".2f",
                color_continuous_scale="RdYlGn",
                title="Sharpe Ratio: Strategy × Regime",
                height=450,
            )
            fig_hm.update_layout(margin=dict(t=40, b=20))
            st.plotly_chart(fig_hm, width='stretch')
        else:
            st.info("Need Strategy, Regime_Bucket, and Sharpe columns for heatmap.")

with tab_returns:
    returns_df = research.get("tournament_monthly_returns")
    if returns_df is None or returns_df.empty:
        st.info("tournament_monthly_returns.csv not found.")
    else:
        strat_c = next((c for c in ["Strategy", "strategy"] if c in returns_df.columns), None)
        date_c  = next((c for c in ["Period_Start", "period_start", "Date", "date"] if c in returns_df.columns), None)
        ret_c   = next((c for c in ["Monthly_Return", "monthly_return", "Return"] if c in returns_df.columns), None)

        if strat_c:
            all_strats = sorted(returns_df[strat_c].dropna().unique().tolist())
            sel_strats = st.multiselect("Strategies", options=all_strats, default=all_strats[:5] if len(all_strats) > 5 else all_strats)
            filtered_ret = returns_df[returns_df[strat_c].isin(sel_strats)] if sel_strats else returns_df
        else:
            filtered_ret = returns_df

        if date_c and ret_c and strat_c:
            filtered_ret = filtered_ret.copy()
            filtered_ret[date_c] = pd.to_datetime(filtered_ret[date_c], errors="coerce")
            filtered_ret = filtered_ret.sort_values(date_c)

            # Cumulative returns
            cum_frames = []
            for strat, grp in filtered_ret.groupby(strat_c):
                grp = grp.sort_values(date_c).copy()
                grp["cumulative"] = (1 + grp[ret_c].fillna(0)).cumprod() - 1
                cum_frames.append(grp)
            if cum_frames:
                cum_df = pd.concat(cum_frames)
                fig_cum = px.line(cum_df, x=date_c, y="cumulative", color=strat_c,
                                  title="Cumulative Returns by Strategy",
                                  labels={date_c: "Date", "cumulative": "Cumulative Return"},
                                  height=400)

                # Regime shading
                rl = research.get("regime_labels")
                if rl is not None and not rl.empty:
                    start_c = next((c for c in ["Trade_Start", "trade_start"] if c in rl.columns), None)
                    end_c   = next((c for c in ["Trade_End", "trade_end"] if c in rl.columns), None)
                    reg_c   = next((c for c in ["Market_Regime", "market_regime"] if c in rl.columns), None)
                    if start_c and end_c and reg_c:
                        reg_colors = {"BULLISH": "rgba(25,135,84,0.1)", "BEARISH": "rgba(220,53,69,0.1)", "NEUTRAL": "rgba(253,126,20,0.08)"}
                        for _, reg_row in rl.iterrows():
                            regime = str(reg_row.get(reg_c, ""))
                            color = reg_colors.get(regime, "rgba(100,100,100,0.05)")
                            try:
                                fig_cum.add_vrect(
                                    x0=pd.to_datetime(reg_row[start_c]),
                                    x1=pd.to_datetime(reg_row[end_c]),
                                    fillcolor=color, line_width=0,
                                    annotation_text=regime, annotation_position="top left",
                                )
                            except Exception:
                                pass

                fig_cum.update_layout(margin=dict(t=40, b=20))
                st.plotly_chart(fig_cum, width='stretch')
        else:
            st.dataframe(filtered_ret, width='stretch')

# ── Gate Analysis ──────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Gate Analysis")
gate_df = research.get("magic_gate_comparison")
if gate_df is None or gate_df.empty:
    st.info("magic_gate_comparison.csv not found.")
else:
    delta_col = next((c for c in ["Delta_CAGR", "delta_cagr", "CAGR_Delta"] if c in gate_df.columns), None)
    cagr_col  = next((c for c in ["CAGR", "cagr"] if c in gate_df.columns), None)
    dd_col    = next((c for c in ["Max_Drawdown", "max_drawdown"] if c in gate_df.columns), None)

    if delta_col:
        harmful = gate_df[pd.to_numeric(gate_df[delta_col], errors="coerce") < -0.05]
        if not harmful.empty:
            names = ", ".join(str(v) for v in harmful.iloc[:, 0].tolist())
            st.warning(f"Gates with CAGR delta < -5%: **{names}**. Historically harmful — consider disabling.")

    st.dataframe(gate_df, width='stretch')

    if cagr_col and dd_col:
        gate_name_col = gate_df.columns[0]
        melted = gate_df[[gate_name_col, cagr_col, dd_col]].melt(
            id_vars=gate_name_col, var_name="Metric", value_name="Value"
        )
        melted["Value"] = pd.to_numeric(melted["Value"], errors="coerce")
        fig_gate = px.bar(melted, x=gate_name_col, y="Value", color="Metric",
                          barmode="group", title="CAGR vs Max Drawdown by Gate Variant",
                          height=320)
        fig_gate.update_layout(margin=dict(t=40, b=20))
        st.plotly_chart(fig_gate, width='stretch')
