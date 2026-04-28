"""Shared sidebar refresh controls — manual button + optional auto-refresh."""
import time

import streamlit as st


def _fmt_age(ts: float) -> str:
    secs = int(time.time() - ts)
    if secs < 60:
        return f"{secs}s ago"
    if secs < 3600:
        return f"{secs // 60}m ago"
    return f"{secs // 3600}h {(secs % 3600) // 60}m ago"


def render_refresh_controls(default_interval: int = 300) -> None:
    """
    Render data-refresh controls in the sidebar.

    Provides:
    - "Refresh" button — clears all cached data and reruns immediately
    - "Last updated" timestamp
    - Auto-refresh toggle with configurable interval (polls every ≤10 s)

    Call once near the top of each page, inside or outside a sidebar block —
    the function opens its own `st.sidebar` context.
    """
    if "last_refresh_ts" not in st.session_state:
        st.session_state.last_refresh_ts = time.time()

    with st.sidebar:
        st.markdown("---")
        btn_col, age_col = st.columns([1, 2])
        with btn_col:
            refresh_clicked = st.button("↺ Refresh", help="Clear cache and reload all data")
        with age_col:
            st.caption(f"Data: {_fmt_age(st.session_state.last_refresh_ts)}")

        if refresh_clicked:
            st.cache_data.clear()
            st.session_state.last_refresh_ts = time.time()
            st.rerun()

        auto = st.toggle("Auto-refresh", key="auto_refresh_toggle", value=False)
        if auto:
            interval = st.select_slider(
                "Interval",
                options=[60, 120, 300, 600, 1800],
                value=default_interval,
                key="refresh_interval",
                format_func=lambda s: f"{s // 60}m" if s >= 60 else f"{s}s",
            )
            elapsed = time.time() - st.session_state.last_refresh_ts
            remaining = int(interval - elapsed)
            if remaining <= 0:
                st.cache_data.clear()
                st.session_state.last_refresh_ts = time.time()
                st.rerun()
            else:
                st.caption(f"Next refresh in {remaining}s")
                time.sleep(min(remaining, 10))
                st.rerun()
