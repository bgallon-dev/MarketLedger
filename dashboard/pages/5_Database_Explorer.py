"""Page 5 — Database Explorer: browse raw tickers, price history, financials, and management data."""
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

_MARKET_LEDGER = Path(__file__).parent.parent.parent
if str(_MARKET_LEDGER) not in sys.path:
    sys.path.insert(0, str(_MARKET_LEDGER))

from dashboard.utils.data_loaders import (
    load_all_tickers,
    load_financial_table,
    load_ticker_history,
)
from dashboard.utils.db import get_financial_db
from dashboard.utils.refresh import render_refresh_controls

st.set_page_config(page_title="Database Explorer — MarketLedger", layout="wide")
st.title("Database Explorer")
render_refresh_controls()

tabs = st.tabs([
    "Universe & Health",
    "Ticker Drill-Down",
    "Financial Metric Browser",
    "Management Data",
    "SQL Console",
])

# ══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Universe & Health
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.subheader("Universe Stats")
    all_tickers = load_all_tickers()

    try:
        conn = get_financial_db()
        tickers_with_history = conn.execute(
            "SELECT COUNT(DISTINCT ticker_id) FROM history"
        ).fetchone()[0]

        # Last-update freshness
        freshness = pd.read_sql_query(
            """
            SELECT t.exchange,
                   COUNT(DISTINCT t.id)   AS tickers,
                   MAX(h.date)            AS last_price_date,
                   MIN(h.date)            AS first_price_date
            FROM tickers t
            LEFT JOIN history h ON h.ticker_id = t.id
            GROUP BY t.exchange
            ORDER BY tickers DESC
            """,
            conn,
        )

        # Coverage: % of tickers that have each financial table
        fin_tables = ["balance_sheet", "income_statement", "cash_flow", "financials", "earnings"]
        total_t = conn.execute("SELECT COUNT(*) FROM tickers").fetchone()[0]
        coverage_rows = []
        for tbl in fin_tables:
            try:
                n = conn.execute(
                    f"SELECT COUNT(DISTINCT ticker_id) FROM {tbl}"
                ).fetchone()[0]
                coverage_rows.append({"Table": tbl, "Tickers": n,
                                       "Coverage %": round(100 * n / max(total_t, 1), 1)})
            except Exception:
                pass

        # Stale price report: tickers whose last price is > 30 days old
        stale = pd.read_sql_query(
            """
            SELECT t.symbol, t.exchange, MAX(h.date) AS last_date
            FROM tickers t
            JOIN history h ON h.ticker_id = t.id
            GROUP BY t.id
            HAVING last_date < date('now', '-30 days')
            ORDER BY last_date
            LIMIT 100
            """,
            conn,
        )
        conn.close()
    except Exception as exc:
        st.error(f"DB error: {exc}")
        tickers_with_history = "—"
        freshness = pd.DataFrame()
        coverage_rows = []
        stale = pd.DataFrame()

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Tickers", f"{len(all_tickers):,}" if not all_tickers.empty else "—")
    m2.metric("With Price History", f"{tickers_with_history:,}" if isinstance(tickers_with_history, int) else tickers_with_history)
    m3.metric("Stale (>30 days)", len(stale) if not stale.empty else 0)

    col_l, col_r = st.columns(2)

    with col_l:
        if not all_tickers.empty and "exchange" in all_tickers.columns:
            exch_counts = all_tickers["exchange"].value_counts().reset_index()
            exch_counts.columns = ["Exchange", "Count"]
            fig_exch = px.bar(exch_counts.head(10), x="Exchange", y="Count",
                              title="Tickers by Exchange", height=240)
            fig_exch.update_layout(margin=dict(t=30, b=10))
            st.plotly_chart(fig_exch, width='stretch')

    with col_r:
        if coverage_rows:
            cov_df = pd.DataFrame(coverage_rows)
            fig_cov = px.bar(cov_df, x="Table", y="Coverage %",
                             text="Tickers",
                             title="Financial Table Coverage", height=240,
                             range_y=[0, 100])
            fig_cov.update_layout(margin=dict(t=30, b=10))
            st.plotly_chart(fig_cov, width='stretch')

    if not freshness.empty:
        st.subheader("Price Freshness by Exchange")
        st.dataframe(freshness.reset_index(drop=True), width='stretch')

    if not stale.empty:
        with st.expander(f"Stale tickers — last price > 30 days ago ({len(stale)} shown)"):
            st.dataframe(stale.reset_index(drop=True), width='stretch')


# ══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Ticker Drill-Down
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.subheader("Ticker Search")
    search_query = st.text_input("Search by symbol or name", placeholder="e.g. AAPL or Apple")

    if search_query:
        q = search_query.strip().upper()
        all_tickers_local = load_all_tickers()
        if not all_tickers_local.empty:
            mask = (
                all_tickers_local["symbol"].str.upper().str.contains(q, na=False) |
                all_tickers_local["name"].str.upper().str.contains(q, na=False)
            )
            results = all_tickers_local[mask].head(50)
        else:
            try:
                conn = get_financial_db()
                results = pd.read_sql_query(
                    "SELECT symbol, name, sector, industry, exchange FROM tickers "
                    "WHERE symbol LIKE ? OR name LIKE ? LIMIT 50",
                    conn, params=(f"%{q}%", f"%{q}%"),
                )
                conn.close()
            except Exception:
                results = pd.DataFrame()

        if results.empty:
            st.info(f"No tickers matching '{search_query}'.")
        else:
            st.dataframe(results.reset_index(drop=True), width='stretch')

            ticker_opts = results["symbol"].tolist()
            selected_ticker = st.selectbox("Drill into ticker", options=ticker_opts)

            if selected_ticker:
                st.markdown(f"### {selected_ticker}")

                # History metadata
                try:
                    conn = get_financial_db()
                    meta_row = conn.execute(
                        """
                        SELECT MIN(h.date), MAX(h.date), COUNT(*),
                               ROUND(AVG(h.volume), 0) AS avg_vol
                        FROM history h
                        JOIN tickers t ON h.ticker_id = t.id
                        WHERE t.symbol = ?
                        """,
                        (selected_ticker,),
                    ).fetchone()
                    conn.close()
                    if meta_row and meta_row[2]:
                        hc1, hc2, hc3, hc4 = st.columns(4)
                        hc1.metric("First Date", str(meta_row[0])[:10] if meta_row[0] else "—")
                        hc2.metric("Last Date",  str(meta_row[1])[:10] if meta_row[1] else "—")
                        hc3.metric("History Rows", f"{meta_row[2]:,}")
                        hc4.metric("Avg Daily Volume", f"{int(meta_row[3] or 0):,}")
                except Exception:
                    pass

                # Price + volume chart
                lookback = st.select_slider("History window", options=[90, 180, 365, 730, 1825], value=365,
                                            format_func=lambda d: f"{d//365}y" if d >= 365 else f"{d}d")
                hist = load_ticker_history(selected_ticker, days=lookback)
                if not hist.empty:
                    fig_h = go.Figure()
                    fig_h.add_trace(go.Scatter(x=hist["date"], y=hist["close"],
                                               name="Close", line=dict(color="#4C9BE8")))
                    if "volume" in hist.columns:
                        fig_h.add_trace(go.Bar(x=hist["date"], y=hist["volume"],
                                               name="Volume", yaxis="y2",
                                               marker_color="rgba(150,150,150,0.3)"))
                    fig_h.update_layout(
                        title=f"{selected_ticker} — Price & Volume",
                        yaxis=dict(title="Close ($)"),
                        yaxis2=dict(title="Volume", overlaying="y", side="right",
                                    showgrid=False),
                        height=300,
                        margin=dict(t=35, b=20),
                        legend=dict(orientation="h"),
                    )
                    st.plotly_chart(fig_h, width='stretch')
                else:
                    st.info("No price history for this ticker.")

                # Financial availability + drill-down
                st.markdown("**Financial Data Availability**")
                fin_tables_local = ["balance_sheet", "income_statement", "cash_flow", "financials", "earnings"]
                avail_data = []
                try:
                    conn = get_financial_db()
                    for tbl in fin_tables_local:
                        row_result = conn.execute(
                            f"SELECT COUNT(*), MIN(period), MAX(period) FROM {tbl} f "
                            f"JOIN tickers t ON f.ticker_id = t.id WHERE t.symbol = ?",
                            (selected_ticker,),
                        ).fetchone()
                        avail_data.append({
                            "Table": tbl,
                            "Rows": row_result[0] if row_result else 0,
                            "Earliest Period": str(row_result[1])[:10] if row_result and row_result[1] else "—",
                            "Latest Period":   str(row_result[2])[:10] if row_result and row_result[2] else "—",
                        })
                    conn.close()
                except Exception:
                    pass

                if avail_data:
                    st.dataframe(pd.DataFrame(avail_data).set_index("Table"), width='stretch')

                fin_choice = st.selectbox("View financial table", options=fin_tables_local, key="fin_table_select")
                fin_df = load_financial_table(selected_ticker, fin_choice)
                if fin_df.empty:
                    st.info(f"No {fin_choice} data for {selected_ticker}.")
                else:
                    try:
                        fin_df = fin_df[sorted(fin_df.columns, reverse=True)]
                    except Exception:
                        pass
                    st.dataframe(fin_df, width='stretch', height=400)


# ══════════════════════════════════════════════════════════════════════════════
# Tab 3 — Financial Metric Browser
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader("Financial Metric Browser")
    st.caption("Look up any metric across all tickers for a given period.")

    metric_table = st.selectbox("Table", ["balance_sheet", "income_statement", "cash_flow", "financials"],
                                key="mb_table")

    try:
        conn = get_financial_db()
        metric_opts = [r[0] for r in conn.execute(
            f"SELECT DISTINCT metric FROM {metric_table} ORDER BY metric"
        ).fetchall()]
        period_opts = [r[0] for r in conn.execute(
            f"SELECT DISTINCT period FROM {metric_table} ORDER BY period DESC LIMIT 20"
        ).fetchall()]
        conn.close()
    except Exception:
        metric_opts = []
        period_opts = []

    mc1, mc2 = st.columns(2)
    with mc1:
        selected_metric = st.selectbox("Metric", options=metric_opts, key="mb_metric")
    with mc2:
        selected_period = st.selectbox("Period", options=period_opts, key="mb_period")

    if selected_metric and selected_period:
        try:
            conn = get_financial_db()
            metric_df = pd.read_sql_query(
                f"""
                SELECT t.symbol, t.sector, t.exchange,
                       CAST(f.value AS REAL) AS value
                FROM {metric_table} f
                JOIN tickers t ON f.ticker_id = t.id
                WHERE f.metric = ? AND f.period = ?
                  AND f.value IS NOT NULL
                ORDER BY ABS(CAST(f.value AS REAL)) DESC
                LIMIT 200
                """,
                conn, params=(selected_metric, selected_period),
            )
            conn.close()
        except Exception as exc:
            metric_df = pd.DataFrame()
            st.error(str(exc))

        if metric_df.empty:
            st.info("No data for that metric/period combination.")
        else:
            st.markdown(f"**{len(metric_df)} tickers** with `{selected_metric}` for `{selected_period}`")

            fig_m = px.bar(
                metric_df.head(40),
                x="symbol", y="value",
                color="sector",
                title=f"{selected_metric} — top 40 by magnitude ({selected_period})",
                labels={"symbol": "Ticker", "value": selected_metric},
                height=320,
            )
            fig_m.update_layout(margin=dict(t=35, b=20), xaxis_tickangle=-45)
            st.plotly_chart(fig_m, width='stretch')

            st.dataframe(
                metric_df.reset_index(drop=True),
                column_config={"value": st.column_config.NumberColumn(selected_metric, format="%.2f")},
                width='stretch',
                height=350,
            )


# ══════════════════════════════════════════════════════════════════════════════
# Tab 4 — Management Data
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.subheader("Management Claims & Credibility Scores")

    try:
        conn = get_financial_db()

        # Summary stats
        n_claims = conn.execute("SELECT COUNT(*) FROM management_claims").fetchone()[0]
        n_resolved = conn.execute(
            "SELECT COUNT(*) FROM management_claims WHERE outcome_met IS NOT NULL"
        ).fetchone()[0]
        n_met = conn.execute(
            "SELECT COUNT(*) FROM management_claims WHERE outcome_met = 1"
        ).fetchone()[0]
        n_tickers_scored = conn.execute(
            "SELECT COUNT(DISTINCT ticker) FROM management_claims"
        ).fetchone()[0]
        conn.close()

        mg1, mg2, mg3, mg4 = st.columns(4)
        mg1.metric("Total Claims", f"{n_claims:,}")
        mg2.metric("Resolved", f"{n_resolved:,}")
        mg3.metric("Met", f"{n_met:,}")
        mg4.metric("Tickers Scored", n_tickers_scored)
    except Exception:
        st.info("No management claims data yet. Run the pipeline with `--management` to populate.")
        st.stop()

    # Per-ticker summary
    try:
        conn = get_financial_db()
        summary = pd.read_sql_query(
            """
            SELECT ticker,
                   COUNT(*)                                          AS total_claims,
                   SUM(CASE WHEN outcome_met IS NOT NULL THEN 1 ELSE 0 END) AS resolved,
                   SUM(CASE WHEN outcome_met = 1 THEN 1 ELSE 0 END)         AS met,
                   ROUND(100.0 * SUM(CASE WHEN outcome_met = 1 THEN 1 ELSE 0 END)
                         / NULLIF(SUM(CASE WHEN outcome_met IS NOT NULL THEN 1 ELSE 0 END), 0), 1)
                                                                     AS hit_rate_pct,
                   MIN(filed_date)                                   AS earliest_filing,
                   MAX(filed_date)                                   AS latest_filing
            FROM management_claims
            GROUP BY ticker
            ORDER BY resolved DESC, hit_rate_pct DESC
            """,
            conn,
        )
        conn.close()
    except Exception:
        summary = pd.DataFrame()

    if not summary.empty:
        fig_hr = px.bar(
            summary[summary["resolved"] >= 2].sort_values("hit_rate_pct", ascending=False).head(30),
            x="ticker", y="hit_rate_pct",
            color="hit_rate_pct",
            color_continuous_scale=["#e74c3c", "#f39c12", "#2ecc71"],
            range_color=[0, 100],
            title="Claim Hit Rate % by Ticker (≥2 resolved claims, top 30)",
            labels={"ticker": "Ticker", "hit_rate_pct": "Hit Rate %"},
            height=280,
        )
        fig_hr.update_layout(margin=dict(t=35, b=20), coloraxis_showscale=False)
        st.plotly_chart(fig_hr, width='stretch')

        st.dataframe(
            summary.reset_index(drop=True),
            column_config={
                "hit_rate_pct": st.column_config.ProgressColumn(
                    "Hit Rate %", min_value=0, max_value=100, format="%.1f%%"
                ),
                "total_claims": st.column_config.NumberColumn("Total Claims"),
                "resolved":     st.column_config.NumberColumn("Resolved"),
                "met":          st.column_config.NumberColumn("Met"),
            },
            width='stretch',
            height=350,
        )

    # Individual ticker claim browser
    st.markdown("---")
    st.subheader("Browse Claims for a Ticker")
    claim_ticker = st.text_input("Ticker", placeholder="e.g. TSM", key="claim_ticker").upper().strip()
    if claim_ticker:
        try:
            conn = get_financial_db()
            claims_df = pd.read_sql_query(
                """
                SELECT filed_date, claim_type, claim_value, claim_dir,
                       outcome_met, outcome_checked_date,
                       claim_text
                FROM management_claims
                WHERE ticker = ?
                ORDER BY filed_date DESC, claim_type
                """,
                conn, params=(claim_ticker,),
            )
            conn.close()
        except Exception:
            claims_df = pd.DataFrame()

        if claims_df.empty:
            st.info(f"No claims for {claim_ticker}.")
        else:
            outcome_map = {None: "⏳ Pending", 0: "❌ Missed", 1: "✅ Met", 2: "🚀 Exceeded"}
            claims_df["outcome"] = claims_df["outcome_met"].map(outcome_map)
            st.dataframe(
                claims_df.drop(columns=["outcome_met"]).reset_index(drop=True),
                column_config={
                    "claim_value": st.column_config.NumberColumn("Value", format="%.3f"),
                    "claim_text":  st.column_config.TextColumn("Claim Text", width="large"),
                    "outcome":     st.column_config.TextColumn("Outcome"),
                },
                width='stretch',
                height=400,
            )

    # Claim type breakdown
    st.markdown("---")
    st.subheader("Claims by Type")
    try:
        conn = get_financial_db()
        type_df = pd.read_sql_query(
            """
            SELECT claim_type,
                   COUNT(*) AS total,
                   SUM(CASE WHEN outcome_met IS NOT NULL THEN 1 ELSE 0 END) AS resolved,
                   SUM(CASE WHEN outcome_met = 1 THEN 1 ELSE 0 END)         AS met,
                   ROUND(100.0 * SUM(CASE WHEN outcome_met = 1 THEN 1 ELSE 0 END)
                         / NULLIF(SUM(CASE WHEN outcome_met IS NOT NULL THEN 1 ELSE 0 END), 0), 1)
                                                                             AS hit_rate_pct
            FROM management_claims
            GROUP BY claim_type
            ORDER BY total DESC
            """,
            conn,
        )
        conn.close()
    except Exception:
        type_df = pd.DataFrame()

    if not type_df.empty:
        fig_type = px.bar(type_df, x="claim_type", y=["met", "resolved"],
                          barmode="overlay",
                          title="Claims: Resolved vs Met by Type",
                          labels={"claim_type": "Claim Type", "value": "Count"},
                          height=260)
        fig_type.update_layout(margin=dict(t=35, b=20))
        st.plotly_chart(fig_type, width='stretch')
        st.dataframe(type_df.reset_index(drop=True), width='content')


# ══════════════════════════════════════════════════════════════════════════════
# Tab 5 — SQL Console
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.subheader("SQL Console")
    st.caption(
        "Read-only queries against the financial database. "
        "Available tables: `tickers`, `history`, `balance_sheet`, `income_statement`, "
        "`cash_flow`, `financials`, `earnings`, `management_claims`, `sec_filings`."
    )

    EXAMPLE_QUERIES = {
        "Latest price for all tickers": (
            "SELECT t.symbol, t.exchange, h.date, h.close\n"
            "FROM history h\n"
            "JOIN tickers t ON h.ticker_id = t.id\n"
            "WHERE h.date = (SELECT MAX(h2.date) FROM history h2 WHERE h2.ticker_id = h.ticker_id)\n"
            "ORDER BY t.symbol\n"
            "LIMIT 100"
        ),
        "Tickers missing income statement": (
            "SELECT t.symbol, t.exchange\n"
            "FROM tickers t\n"
            "WHERE t.id NOT IN (SELECT DISTINCT ticker_id FROM income_statement)\n"
            "LIMIT 100"
        ),
        "Management claims hit rate by type": (
            "SELECT claim_type,\n"
            "       COUNT(*) AS total,\n"
            "       SUM(CASE WHEN outcome_met=1 THEN 1 ELSE 0 END) AS met,\n"
            "       ROUND(100.0*SUM(CASE WHEN outcome_met=1 THEN 1 ELSE 0 END)\n"
            "             /NULLIF(SUM(CASE WHEN outcome_met IS NOT NULL THEN 1 ELSE 0 END),0),1) AS hit_pct\n"
            "FROM management_claims\n"
            "GROUP BY claim_type ORDER BY total DESC"
        ),
        "Top 20 revenue (latest period)": (
            "SELECT t.symbol, f.period, CAST(f.value AS REAL)/1e9 AS revenue_B\n"
            "FROM income_statement f\n"
            "JOIN tickers t ON f.ticker_id = t.id\n"
            "WHERE f.metric='TotalRevenue'\n"
            "  AND f.period=(SELECT MAX(f2.period) FROM income_statement f2\n"
            "                WHERE f2.ticker_id=f.ticker_id AND f2.metric='TotalRevenue')\n"
            "ORDER BY revenue_B DESC LIMIT 20"
        ),
        "SEC filings cached": (
            "SELECT t.symbol, s.filed_date, s.accession_no,\n"
            "       LENGTH(s.item_1a_text) AS item1a_chars,\n"
            "       LENGTH(s.item_7_text)  AS item7_chars\n"
            "FROM sec_filings s\n"
            "JOIN tickers t ON s.ticker_id = t.id\n"
            "ORDER BY s.filed_date DESC\n"
            "LIMIT 50"
        ),
    }

    example_choice = st.selectbox("Load example query", ["(custom)"] + list(EXAMPLE_QUERIES.keys()))
    default_sql = EXAMPLE_QUERIES.get(example_choice, "SELECT symbol, name FROM tickers LIMIT 20")

    sql_input = st.text_area("SQL query (SELECT only)", value=default_sql, height=140,
                             key="sql_console_input")

    run_sql = st.button("Run Query", type="primary")

    if run_sql and sql_input.strip():
        clean = sql_input.strip().upper()
        if not clean.startswith("SELECT"):
            st.error("Only SELECT statements are permitted.")
        else:
            try:
                conn = get_financial_db()
                result_df = pd.read_sql_query(sql_input, conn)
                conn.close()
                st.success(f"{len(result_df)} row(s) returned.")
                st.dataframe(result_df, width='stretch', height=450)
            except Exception as exc:
                st.error(f"Query error: {exc}")
