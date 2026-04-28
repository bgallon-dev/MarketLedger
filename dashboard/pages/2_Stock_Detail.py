"""Page 2 — Stock Detail: deep-dive into a single ticker."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

_MARKET_LEDGER = Path(__file__).parent.parent.parent
if str(_MARKET_LEDGER) not in sys.path:
    sys.path.insert(0, str(_MARKET_LEDGER))

from dashboard.utils.data_loaders import (
    load_decision_log,
    load_financial_table,
    load_ticker_history,
)
from dashboard.utils.formatting import SIGNAL_COLORS, fmt_currency, fmt_pct, signal_badge_html
from dashboard.brain.discovery import run_all_detectors
from dashboard.brain.proximity import find_analogs, get_current_state
from dashboard.utils.mood import load_mood_vector
from dashboard.utils.refresh import render_refresh_controls

st.set_page_config(page_title="Stock Detail — MarketLedger", layout="wide")
st.title("Stock Detail")
render_refresh_controls()

df = load_decision_log()
if df.empty:
    st.warning("No pipeline output found. Run `py -m main` first.")
    st.stop()

tickers = sorted(df["Ticker"].dropna().unique().tolist()) if "Ticker" in df.columns else []
if not tickers:
    st.warning("No tickers in pipeline output.")
    st.stop()

preselect = st.session_state.get("selected_ticker", tickers[0])
default_idx = tickers.index(preselect) if preselect in tickers else 0
ticker = st.selectbox("Ticker", options=tickers, index=default_idx)

row = df[df["Ticker"] == ticker].iloc[0] if not df[df["Ticker"] == ticker].empty else None
if row is None:
    st.warning(f"{ticker} not found in pipeline output.")
    st.stop()

def _get(col, default=None):
    v = row.get(col, default)
    return default if pd.isna(v) else v

# ── Header ─────────────────────────────────────────────────────────────────────
signal = _get("Investment Signal", "—")
strategy = _get("Strategy", "—")
signal_html = signal_badge_html(signal) if signal != "—" else signal
st.markdown(
    f"**{ticker}** &nbsp; {signal_html} &nbsp; <span style='color:#6c757d'>{strategy}</span>",
    unsafe_allow_html=True,
)

# ── KPI row ────────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
price = _get("Current Price")
fv_base = _get("Fair Value (Base)")
mos = _get("Undervalued %")
conf = _get("Confidence Score")

k1.metric("Current Price", fmt_currency(price))
k2.metric("Fair Value (Base)", fmt_currency(fv_base))
k3.metric("Undervalued %", fmt_pct(mos))
k4.metric("Confidence Score", f"{conf:.0f}/100" if conf is not None else "—")

st.markdown("---")

# ── Main chart + valuation panel ──────────────────────────────────────────────
hist = load_ticker_history(ticker, days=730)
fv_bear = _get("Fair Value (Bear)")
fv_bull = _get("Fair Value (Bull)")

chart_col, val_col = st.columns([3, 2])

with chart_col:
    st.subheader("Price History & Valuation")
    if hist.empty:
        st.info("No price history in DB for this ticker.")
    else:
        # 200-day MA
        hist = hist.sort_values("date")
        hist["ma200"] = hist["close"].rolling(200, min_periods=1).mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist["date"], y=hist["close"],
            name="Price", line=dict(color="#0d6efd", width=1.5),
        ))
        fig.add_trace(go.Scatter(
            x=hist["date"], y=hist["ma200"],
            name="200-day MA", line=dict(color="#6c757d", width=1, dash="dot"),
        ))

        x_range = [hist["date"].min(), hist["date"].max()]

        if fv_bear is not None:
            fig.add_shape(type="line", x0=x_range[0], x1=x_range[1], y0=fv_bear, y1=fv_bear,
                          line=dict(color="#dc3545", width=1.5, dash="dash"))
            fig.add_annotation(x=x_range[1], y=fv_bear, text=f"Bear {fmt_currency(fv_bear)}",
                               showarrow=False, xanchor="right", font=dict(color="#dc3545", size=11))

        if fv_base is not None:
            fig.add_shape(type="line", x0=x_range[0], x1=x_range[1], y0=fv_base, y1=fv_base,
                          line=dict(color="#0dcaf0", width=1.5, dash="dash"))
            fig.add_annotation(x=x_range[1], y=fv_base, text=f"Base {fmt_currency(fv_base)}",
                               showarrow=False, xanchor="right", font=dict(color="#0dcaf0", size=11))

        if fv_bull is not None:
            fig.add_shape(type="line", x0=x_range[0], x1=x_range[1], y0=fv_bull, y1=fv_bull,
                          line=dict(color="#198754", width=1.5, dash="dash"))
            fig.add_annotation(x=x_range[1], y=fv_bull, text=f"Bull {fmt_currency(fv_bull)}",
                               showarrow=False, xanchor="right", font=dict(color="#198754", size=11))

        # Valuation corridor shading
        if fv_bear is not None and fv_bull is not None:
            fig.add_shape(
                type="rect",
                x0=x_range[0], x1=x_range[1],
                y0=fv_bear, y1=fv_bull,
                fillcolor="rgba(25,135,84,0.06)",
                line=dict(width=0),
            )

        fig.update_layout(
            height=380,
            margin=dict(t=10, b=30, l=10, r=10),
            legend=dict(orientation="h", y=-0.15),
            xaxis_title=None,
            yaxis_title="Price ($)",
        )
        st.plotly_chart(fig, width='stretch')

with val_col:
    st.subheader("Valuation Scenarios")
    scenario_data = []
    for label, col_key in [("Bear", "Fair Value (Bear)"), ("Base", "Fair Value (Base)"), ("Bull", "Fair Value (Bull)")]:
        fv = _get(col_key)
        mos_val = (fv / price - 1) * 100 if fv and price else None
        scenario_data.append({"Scenario": label, "Fair Value": fmt_currency(fv), "MOS %": fmt_pct(mos_val)})
    st.table(pd.DataFrame(scenario_data).set_index("Scenario"))

    spread = _get("Scenario Spread %")
    vm = _get("Valuation Method", "—")
    st.caption(f"Method: **{vm}** &nbsp;|&nbsp; Scenario spread: {fmt_pct(spread)}")

    st.markdown("**Trend Projections** *(CAGR extrapolation)*")
    proj_data = []
    for label, col_key in [("6M", "Trend_Proj_6M"), ("1Y", "Trend_Proj_1Y"), ("3Y", "Trend_Proj_3Y")]:
        proj_data.append({"Horizon": label, "Price": fmt_currency(_get(col_key))})
    trend_cagr = _get("Trend_CAGR")
    st.table(pd.DataFrame(proj_data).set_index("Horizon"))
    st.caption(f"Trend CAGR: {fmt_pct(trend_cagr)}")

    if conf is not None:
        conf_int = min(max(int(conf), 0), 100)
        conf_label = _get("Model Confidence", "—")
        st.markdown(f"**Model Confidence**: {conf_label}")
        st.progress(conf_int / 100)

st.markdown("---")

# ── Risk tabs ──────────────────────────────────────────────────────────────────
tab_gates, tab_forensic, tab_tail, tab_momentum, tab_contagion, tab_mood = st.tabs(
    ["Gates", "Forensic", "Tail Risk", "Momentum", "Contagion", "Mood"]
)

with tab_gates:
    gate_map = {
        "RV_Gate_Distress": "Distress",
        "RV_Gate_Tail": "Tail",
        "RV_Gate_Momentum": "Momentum",
        "RV_Gate_Valuation": "Valuation",
    }
    gcols = st.columns(4)
    for i, (col_key, label) in enumerate(gate_map.items()):
        val = _get(col_key, False)
        tripped = bool(val)
        color = "#dc3545" if tripped else "#198754"
        status = "TRIPPED" if tripped else "PASSED"
        gcols[i].markdown(
            f'<div style="background:{color};color:#fff;padding:12px;border-radius:8px;'
            f'text-align:center"><b>{label}</b><br>{status}</div>',
            unsafe_allow_html=True,
        )
    gate_count = _get("RV_Gate_Count", 0)
    st.metric("Total gates tripped", int(gate_count) if gate_count is not None else 0)
    reason = _get("Signal Reason", "—")
    st.caption(f"Signal reason: {reason}")

with tab_forensic:
    z = _get("Altman Z-Score")
    distress = _get("Distress Risk", "Unknown")
    color_map = {"SAFE": "#198754", "GREY ZONE": "#fd7e14", "DISTRESS (Risk)": "#dc3545"}
    d_color = color_map.get(distress, "#6c757d")
    st.markdown(
        f'<span style="background:{d_color};color:#fff;padding:4px 12px;border-radius:6px;'
        f'font-weight:600">{distress}</span>',
        unsafe_allow_html=True,
    )
    st.metric("Altman Z-Score", f"{z:.2f}" if z is not None else "—")
    if z is not None:
        if z > 3:
            st.success("Z > 3: Company is in the safe zone.")
        elif z > 1.81:
            st.warning("1.81 < Z ≤ 3: Grey zone — some financial stress possible.")
        else:
            st.error("Z ≤ 1.81: Distress zone — elevated bankruptcy risk.")

with tab_tail:
    tail_bucket = _get("Tail_Risk", "Unknown")
    skew = _get("Skew_Direction", "—")
    var_adj = _get("VaR_Adjustment")
    risk_summary = _get("Risk_Summary", "—")
    tail_a = _get("Tail_a")
    tail_b = _get("Tail_b")

    tail_color = {"Heavy": "#dc3545", "Moderate": "#fd7e14", "Normal": "#198754", "Light": "#6cc070"}.get(tail_bucket, "#6c757d")
    st.markdown(
        f'<span style="background:{tail_color};color:#fff;padding:4px 12px;border-radius:6px;'
        f'font-weight:600">Tail Risk: {tail_bucket}</span>',
        unsafe_allow_html=True,
    )
    col_t1, col_t2 = st.columns(2)
    col_t1.metric("Skew Direction", skew)
    col_t2.metric("VaR Adjustment", fmt_pct(var_adj) if var_adj is not None else "—")
    st.caption(f"Summary: {risk_summary}")
    if tail_a is not None:
        st.caption(f"Distribution params: a={tail_a:.3f}, b={tail_b:.3f}")

    # Daily returns histogram
    if not hist.empty and len(hist) > 5:
        returns = hist["close"].pct_change().dropna() * 100
        fig_r = px.histogram(returns, nbins=50, title="Daily Returns Distribution (%)",
                             labels={"value": "Return %"}, height=220)
        fig_r.update_layout(showlegend=False, margin=dict(t=30, b=10))
        st.plotly_chart(fig_r, width='stretch')

with tab_momentum:
    mom_price = _get("Momentum_Price")
    mom_ma = _get("Momentum_MA_200")
    mom_above = _get("Momentum_Above_MA")
    mom_gap = _get("Momentum_MA_Gap_Pct")
    mom_regime = _get("RV_Momentum_Regime", "—")

    regime_color = {"BULLISH": "#198754", "BEARISH": "#dc3545", "NEUTRAL": "#fd7e14"}.get(mom_regime, "#6c757d")
    st.markdown(
        f'<span style="background:{regime_color};color:#fff;padding:4px 12px;border-radius:6px;'
        f'font-weight:600">Momentum: {mom_regime}</span>',
        unsafe_allow_html=True,
    )
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Price", fmt_currency(mom_price))
    col_m2.metric("200-day MA", fmt_currency(mom_ma))
    col_m3.metric("MA Gap %", fmt_pct(mom_gap))
    above_text = "Yes ✓" if mom_above else "No ✗"
    st.caption(f"Price above 200-day MA: **{above_text}**")

with tab_contagion:
    c_label = _get("Contagion_Risk_Label", "Insufficient Data")
    c_score = _get("Contagion_Leadership_Score")
    c_rank = _get("Contagion_Disclosure_Rank")
    c_peers = _get("Contagion_Peer_Count")
    c_novel = _get("Contagion_Novel_Risks")
    c_date = _get("Contagion_Filing_Date", "—")

    label_colors = {
        "Leader": "#198754",
        "Follower": "#fd7e14",
        "Isolated": "#6c757d",
        "Insufficient Data": "#adb5bd",
    }
    lbl_color = label_colors.get(c_label, "#adb5bd")
    st.markdown(
        f'<span style="background:{lbl_color};color:#fff;padding:4px 12px;'
        f'border-radius:6px;font-weight:600">Contagion: {c_label}</span>',
        unsafe_allow_html=True,
    )
    st.caption(f"Most recent 10-K: {c_date}")

    col_c1, col_c2, col_c3, col_c4 = st.columns(4)
    col_c1.metric(
        "Leadership Score",
        f"{c_score:.1f}/100" if c_score is not None else "—",
        help="100 = first in sector to disclose a risk cluster. 0 = last.",
    )
    col_c2.metric(
        "Disclosure Rank",
        f"#{int(c_rank)}" if c_rank is not None else "—",
        help="Rank within most recent shared risk cluster (1 = first to disclose).",
    )
    col_c3.metric(
        "Sector Peers",
        int(c_peers) if c_peers is not None else "—",
        help="Peers with overlapping risk language (Jaccard ≥ 0.35).",
    )
    col_c4.metric(
        "Novel Risks",
        int(c_novel) if c_novel is not None else "—",
        help="Estimated count of risk paragraphs unique to this company's 10-K.",
    )

    if c_label == "Insufficient Data":
        st.info(
            "No contagion data. Re-run the pipeline with `--contagion` to fetch "
            "SEC 10-K filings and compute sector propagation scores."
        )
    elif c_label == "Isolated":
        st.warning(
            "No sector peers with overlapping risk language found. "
            "This company may be in a niche industry, or its sector peers "
            "are not in the current pipeline universe."
        )

with tab_mood:
    mood = load_mood_vector(ticker)
    if mood is None:
        st.info(
            "No mood vector computed for this ticker. "
            "Run `py dashboard/experimental/mood_pipeline.py --tickers "
            + ticker
            + " --verbose` to compute it."
        )
    else:
        _DIMS = [
            ("disclosure_pressure",    "pct_disclosure_pressure",    "Disclosure Pressure",
             "Rate of new risk factors added vs retired (YoY Item 1A delta). "
             "High = accumulating risk; low = retiring risk."),
            ("operational_confidence", "pct_operational_confidence", "Operational Confidence",
             "Management forward-looking statement accuracy vs realised results. "
             "High = consistent delivery; low = persistent over-promising."),
            ("cash_flow_coherence",    "pct_cash_flow_coherence",    "Cash Flow Coherence",
             "Slope of FCF coverage of declared distributions over trailing 8 quarters. "
             "High = improving coverage; low = deteriorating coverage."),
            ("narrative_stability",    "pct_narrative_stability",    "Narrative Stability",
             "Semantic similarity between consecutive MD&A sections. "
             "High = stable narrative; low = major rewrite."),
        ]

        pct_vals = [mood.get(pct) for _, pct, _, _ in _DIMS]
        label_vals = [label for _, _, label, _ in _DIMS]

        # Close the polygon by repeating first value/label
        r_vals = pct_vals + [pct_vals[0]]
        theta_vals = label_vals + [label_vals[0]]

        # Fill None with 50 for rendering only; mark missing visually
        r_plot = [v if v is not None else 50.0 for v in r_vals]

        fig_mood = go.Figure(
            go.Scatterpolar(
                r=r_plot,
                theta=theta_vals,
                fill="toself",
                fillcolor="rgba(65, 105, 225, 0.15)",
                line=dict(color="royalblue", width=2),
                name="Peer percentile",
            )
        )
        fig_mood.update_layout(
            polar=dict(
                radialaxis=dict(range=[0, 100], tickfont=dict(size=10)),
                angularaxis=dict(tickfont=dict(size=11)),
            ),
            showlegend=False,
            height=380,
            margin=dict(t=20, b=10, l=20, r=20),
        )
        st.plotly_chart(fig_mood, use_container_width=True)

        # 4-column metric row
        mcols = st.columns(4)
        for i, (raw_col, pct_col, label, tip) in enumerate(_DIMS):
            raw = mood.get(raw_col)
            pct = mood.get(pct_col)
            raw_str = f"{raw:.3f}" if raw is not None else "—"
            pct_str = f"{pct:.0f}th pct" if pct is not None else "—"
            mcols[i].metric(label, pct_str, help=tip)
            mcols[i].caption(f"raw: {raw_str}")

        composite = mood.get("composite_mood")
        filing_date = mood.get("filing_date", "—")
        sector = mood.get("sector", "—")
        bucket = mood.get("size_bucket", "—")
        computed = mood.get("computed_at", "—")

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Composite Mood",
            f"{composite:.0f}/100" if composite is not None else "—",
            help="Mean of available peer-percentile ranks across all four dimensions.",
        )
        c2.metric("Peer Group", f"{sector} / {bucket}")
        c3.metric("Based on 10-K filed", filing_date)
        st.caption(f"Computed at: {computed}")

# ── Signal Tensions (Discovery Chronicle) ─────────────────────────────────────
_SEV_ORDER = {"warning": 0, "caution": 1, "note": 2}
_SEV_ICON = {"warning": "🔴", "caution": "🟡", "note": "⚪"}

@st.cache_data(ttl=3600)
def _load_discovery_entries(ticker: str) -> list:
    from dashboard.utils.db import get_financial_db
    import sqlite3
    conn = get_financial_db()
    try:
        rows = conn.execute(
            """
            SELECT pattern_type, severity, investment_signal, mos_pct,
                   source_type, source_form_type, source_filing_date, excerpt,
                   quant_metric, quant_value, quant_threshold, run_date
            FROM discovery_entries
            WHERE ticker = ?
            ORDER BY run_date DESC, severity ASC
            LIMIT 30
            """,
            (ticker.upper(),),
        ).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []
    finally:
        conn.close()

with st.expander("Signal Tensions", expanded=True):
    tensions = _load_discovery_entries(ticker)
    if not tensions:
        st.caption(
            "No tensions flagged. Run `py dashboard/brain/discovery_pipeline.py` "
            "after contagion fetches filings."
        )
    else:
        seen_patterns = set()
        for t in sorted(tensions, key=lambda x: (_SEV_ORDER.get(x["severity"], 9), x["pattern_type"])):
            key = (t["pattern_type"], t.get("source_filing_date"))
            if key in seen_patterns:
                continue
            seen_patterns.add(key)
            icon = _SEV_ICON.get(t["severity"], "⚪")
            label = t["pattern_type"].replace("_", " ").title()
            st.markdown(f"{icon} **{label}**")
            if t.get("excerpt"):
                st.caption(t["excerpt"])
            meta_parts = [t["source_type"]]
            if t.get("source_form_type"):
                meta_parts.append(t["source_form_type"])
            if t.get("source_filing_date"):
                meta_parts.append(f"filed {t['source_filing_date']}")
            if t.get("investment_signal"):
                meta_parts.append(f"signal: {t['investment_signal']}")
            st.caption("  ·  ".join(meta_parts))
            st.markdown("---")

# ── Historical Analogs (Proximity Engine) ──────────────────────────────────────
@st.cache_data(ttl=3600)
def _load_analogs(ticker: str) -> list:
    from dashboard.utils.db import get_financial_db
    conn = get_financial_db()
    try:
        vec = get_current_state(ticker, conn)
        if vec is None:
            return []
        return find_analogs(vec, conn, exclude_ticker=ticker, k=4)
    except Exception:
        return []
    finally:
        conn.close()

with st.expander("Historical Analogs", expanded=True):
    analogs = _load_analogs(ticker)
    if not analogs:
        st.caption(
            "No analogs indexed. Run `py dashboard/brain/proximity_pipeline.py` "
            "to build the state-vector index."
        )
    else:
        hdr = st.columns([1.5, 1.5, 1.5, 2, 1.5])
        hdr[0].markdown("**Ticker**")
        hdr[1].markdown("**Quarter**")
        hdr[2].markdown("**Similarity**")
        hdr[3].markdown("**12M Return**")
        hdr[4].markdown("**Div Cut?**")
        for a in analogs:
            row_cols = st.columns([1.5, 1.5, 1.5, 2, 1.5])
            row_cols[0].write(a["ticker"])
            row_cols[1].write(a["period"])
            row_cols[2].write(f"{a['similarity']:.2f}")
            fwd = a.get("fwd_return_12m")
            if fwd is not None:
                color = "#198754" if fwd >= 0 else "#dc3545"
                row_cols[3].markdown(
                    f'<span style="color:{color};font-weight:600">{fwd:+.1%}</span>',
                    unsafe_allow_html=True,
                )
            else:
                row_cols[3].write("—")
            dc = a.get("div_cut")
            row_cols[4].write("Yes" if dc == 1 else ("No" if dc == 0 else "—"))
        st.caption(
            "Nearest historical ticker-quarters by structural state (leverage, FCF, "
            "momentum, Altman Z, coverage). Similarity = cosine similarity after z-score normalisation."
        )

# ── Raw valuation inputs ───────────────────────────────────────────────────────
with st.expander("Raw valuation inputs"):
    input_cols = [c for c in df.columns if c.startswith("_Input_")]
    if input_cols:
        input_data = {c.replace("_Input_", ""): _get(c) for c in input_cols}
        st.table(pd.Series(input_data).rename("Value").to_frame())
    else:
        st.info("No _Input_ columns found in pipeline output.")

    strat_params = _get("Strategy Params", "—")
    st.caption(f"Strategy params: {strat_params}")

# ── Financials expander ────────────────────────────────────────────────────────
KEY_METRICS = {
    "income_statement": ["TotalRevenue", "GrossProfit", "EBIT", "NetIncome", "OperatingIncome"],
    "balance_sheet": ["TotalAssets", "TotalLiabilitiesNetMinorityInterest", "StockholdersEquity", "CashAndCashEquivalents"],
    "cash_flow": ["OperatingCashFlow", "FreeCashFlow", "CapitalExpenditure"],
}

with st.expander("Financials"):
    fin_tab = st.selectbox("Table", options=list(KEY_METRICS.keys()))
    fin_df = load_financial_table(ticker, fin_tab)
    if fin_df.empty:
        st.info(f"No {fin_tab} data for {ticker} in DB.")
    else:
        key_rows = [m for m in KEY_METRICS[fin_tab] if m in fin_df.index]
        display_fin = fin_df.loc[key_rows] if key_rows else fin_df.head(20)
        # Sort columns (periods) descending
        try:
            display_fin = display_fin[sorted(display_fin.columns, reverse=True)]
        except Exception:
            pass
        st.dataframe(display_fin, width='stretch')
