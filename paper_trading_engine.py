"""
paper_trading_engine.py
-----------------------
Paper trading simulation layer for MarketLedger.

Three components:
  1. PaperPortfolio  — SQLite-backed position and transaction ledger
  2. FillSimulator   — realistic price fills with slippage and spread
  3. ExecutionScheduler — monthly rebalance loop that calls run_pipeline()

Drop this file in the MarketLedger root directory.

Usage:
    python paper_trading_engine.py --run-now          # trigger a rebalance immediately
    python paper_trading_engine.py --status           # print current portfolio
    python paper_trading_engine.py --history          # print transaction log
    python paper_trading_engine.py --outcomes         # print scored closed positions
    python paper_trading_engine.py --schedule         # run the monthly scheduler loop
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
import dataclasses
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from database.database import get_connection as get_market_db_connection

# ── constants ─────────────────────────────────────────────────────────────────

PAPER_DB_PATH = Path(__file__).parent / "database" / "paper_trading.db"
DEFAULT_STARTING_CASH = 100_000.0
DEFAULT_MAX_POSITIONS  = 10
DEFAULT_SLIPPAGE_BPS   = 15       # 0.15% — realistic for mid-cap liquid names
DEFAULT_SPREAD_BPS     = 10       # 0.10% half-spread cost per trade
COMMISSION_PER_TRADE   = 0.0      # commission-free assumed (Alpaca-style)
REGIME_BOOST           = 1.5      # weight multiplier for regime-winner tickers


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATABASE SCHEMA
# ─────────────────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS portfolio_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS positions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker          TEXT    NOT NULL,
    shares          REAL    NOT NULL,
    entry_price     REAL    NOT NULL,
    fill_price      REAL    NOT NULL,         -- after slippage + spread
    entry_date      TEXT    NOT NULL,
    strategy        TEXT,
    signal          TEXT,                     -- Investment Signal at entry
    rv_gate_count   INTEGER,
    rv_distress     TEXT,
    rv_tail         TEXT,
    rv_momentum     TEXT,
    regime          TEXT,
    status          TEXT    NOT NULL DEFAULT 'open',   -- open | closed
    exit_price      REAL,
    exit_date       TEXT,
    pnl_pct         REAL,
    hold_days       INTEGER,
    mos_at_entry    REAL,           -- (fair_value_base - entry_price) / entry_price
    fair_value_bear REAL,           -- Bear scenario FV at entry
    fair_value_base REAL,           -- Base scenario FV at entry
    fair_value_bull REAL,           -- Bull scenario FV at entry
    valuation_method TEXT,          -- DCF | DDM | Dist_Yield | P/S | P/B
    confidence_score REAL,          -- 0-100 model confidence score
    sector          TEXT            -- GICS sector at entry time
);

CREATE TABLE IF NOT EXISTS transactions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT    NOT NULL,
    ticker      TEXT    NOT NULL,
    action      TEXT    NOT NULL,     -- BUY | SELL
    shares      REAL    NOT NULL,
    market_price REAL   NOT NULL,
    fill_price  REAL    NOT NULL,
    slippage_bps REAL   NOT NULL,
    spread_bps  REAL    NOT NULL,
    value       REAL    NOT NULL,
    cash_after  REAL    NOT NULL,
    notes       TEXT
);

CREATE TABLE IF NOT EXISTS rebalance_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT    NOT NULL,
    trade_date      TEXT    NOT NULL,
    regime          TEXT,
    candidates      INTEGER,
    selected        INTEGER,
    buys            INTEGER,
    sells           INTEGER,
    holds           INTEGER,
    portfolio_value REAL,
    cash            REAL,
    notes           TEXT
);
"""


_POSITIONS_MIGRATIONS = [
    "ALTER TABLE positions ADD COLUMN mos_at_entry REAL",
    "ALTER TABLE positions ADD COLUMN fair_value_bear REAL",
    "ALTER TABLE positions ADD COLUMN fair_value_base REAL",
    "ALTER TABLE positions ADD COLUMN fair_value_bull REAL",
    "ALTER TABLE positions ADD COLUMN valuation_method TEXT",
    "ALTER TABLE positions ADD COLUMN confidence_score REAL",
    "ALTER TABLE positions ADD COLUMN sector TEXT",
]


def _get_paper_db() -> sqlite3.Connection:
    PAPER_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(PAPER_DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    # Migrate existing DBs — ignore errors for columns that already exist
    for stmt in _POSITIONS_MIGRATIONS:
        try:
            conn.execute(stmt)
        except sqlite3.OperationalError:
            pass
    conn.commit()
    return conn


# ─────────────────────────────────────────────────────────────────────────────
# 2.  FILL SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FillResult:
    ticker:       str
    market_price: float      # last close from DB
    fill_price:   float      # after slippage + spread
    slippage_bps: float
    spread_bps:   float
    side:         str        # BUY | SELL


class FillSimulator:
    """
    Simulate realistic trade fills using historical close prices from the
    MarketLedger SQLite database.  No network calls required.

    Slippage model: uniform random adverse move between 0 and slippage_bps.
    Spread model:   fixed half-spread cost applied symmetrically to buys and
                    sells (you always pay the spread, never earn it).

    Small-cap penalty: tickers with < min_volume_threshold average daily volume
    receive 2× slippage.  Volume is estimated from the price history table.
    """

    def __init__(
        self,
        slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
        spread_bps:   float = DEFAULT_SPREAD_BPS,
        volume_lookback_days: int = 20,
        min_volume_threshold: int = 500_000,   # avg shares/day
        rng_seed: Optional[int] = None,
    ):
        self.slippage_bps           = slippage_bps
        self.spread_bps             = spread_bps
        self.volume_lookback_days   = volume_lookback_days
        self.min_volume_threshold   = min_volume_threshold
        self._rng = np.random.default_rng(rng_seed)

    def _get_market_price(self, ticker: str, as_of: str) -> Optional[float]:
        """Last close price on or before as_of from the price history."""
        conn = get_market_db_connection()
        try:
            row = conn.execute(
                """
                SELECT h.close FROM history h
                JOIN tickers t ON h.ticker_id = t.id
                WHERE t.symbol = ? AND h.date <= ?
                ORDER BY h.date DESC LIMIT 1
                """,
                (ticker, as_of),
            ).fetchone()
            return float(row[0]) if row else None
        finally:
            conn.close()

    def _get_avg_volume(self, ticker: str, as_of: str) -> float:
        conn = get_market_db_connection()
        cutoff = (
            datetime.strptime(as_of, "%Y-%m-%d")
            - timedelta(days=self.volume_lookback_days)
        ).strftime("%Y-%m-%d")
        try:
            row = conn.execute(
                """
                SELECT AVG(h.volume) FROM history h
                JOIN tickers t ON h.ticker_id = t.id
                WHERE t.symbol = ? AND h.date BETWEEN ? AND ?
                """,
                (ticker, cutoff, as_of),
            ).fetchone()
            return float(row[0]) if row and row[0] else 0.0
        finally:
            conn.close()

    def simulate_fill(
        self,
        ticker:    str,
        side:      str,         # "BUY" or "SELL"
        as_of:     str,         # YYYY-MM-DD
    ) -> Optional[FillResult]:
        market_price = self._get_market_price(ticker, as_of)
        if market_price is None or market_price <= 0:
            return None

        # Volume-based slippage multiplier
        avg_vol = self._get_avg_volume(ticker, as_of)
        slip_mult = 2.0 if (avg_vol > 0 and avg_vol < self.min_volume_threshold) else 1.0

        # Random adverse slip (0 → slippage_bps × mult)
        adverse_bps = float(
            self._rng.uniform(0, self.slippage_bps * slip_mult)
        )
        spread_cost_bps = self.spread_bps

        total_bps = adverse_bps + spread_cost_bps
        direction = 1.0 if side == "BUY" else -1.0   # adverse for buyer = higher; seller = lower
        fill_price = market_price * (1 + direction * total_bps / 10_000)

        return FillResult(
            ticker       = ticker,
            market_price = market_price,
            fill_price   = round(fill_price, 4),
            slippage_bps = round(adverse_bps, 2),
            spread_bps   = round(spread_cost_bps, 2),
            side         = side,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3.  PAPER PORTFOLIO MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class PaperPortfolio:
    """
    SQLite-backed paper portfolio.

    Tracks:
    - Cash balance
    - Open positions (with full entry metadata for outcome scoring)
    - Closed positions with P&L
    - Transaction log
    - Rebalance history
    """

    def __init__(
        self,
        starting_cash:  float = DEFAULT_STARTING_CASH,
        max_positions:  int   = DEFAULT_MAX_POSITIONS,
        fill_simulator: Optional[FillSimulator] = None,
    ):
        self.max_positions  = max_positions
        self.fill_simulator = fill_simulator or FillSimulator()
        self._init_meta(starting_cash)

    # ── meta helpers ──────────────────────────────────────────────────────────

    def _init_meta(self, starting_cash: float) -> None:
        conn = _get_paper_db()
        try:
            existing = conn.execute(
                "SELECT value FROM portfolio_meta WHERE key='cash'"
            ).fetchone()
            if not existing:
                conn.execute(
                    "INSERT INTO portfolio_meta VALUES ('cash', ?)",
                    (str(starting_cash),),
                )
                conn.execute(
                    "INSERT INTO portfolio_meta VALUES ('starting_cash', ?)",
                    (str(starting_cash),),
                )
                conn.commit()
        finally:
            conn.close()

    @property
    def cash(self) -> float:
        conn = _get_paper_db()
        try:
            row = conn.execute(
                "SELECT value FROM portfolio_meta WHERE key='cash'"
            ).fetchone()
            return float(row[0]) if row else 0.0
        finally:
            conn.close()

    def _set_cash(self, conn: sqlite3.Connection, amount: float) -> None:
        conn.execute(
            "UPDATE portfolio_meta SET value=? WHERE key='cash'", (str(amount),)
        )

    # ── open positions ─────────────────────────────────────────────────────────

    def open_positions(self) -> pd.DataFrame:
        conn = _get_paper_db()
        try:
            df = pd.read_sql_query(
                "SELECT * FROM positions WHERE status='open' ORDER BY entry_date",
                conn,
            )
            return df
        finally:
            conn.close()

    def position_tickers(self) -> List[str]:
        pos = self.open_positions()
        return pos["ticker"].tolist() if not pos.empty else []

    def portfolio_value(self, as_of: Optional[str] = None) -> float:
        """Cash + mark-to-market value of open positions."""
        as_of = as_of or date.today().strftime("%Y-%m-%d")
        positions = self.open_positions()
        if positions.empty:
            return self.cash

        total_equity = 0.0
        sim = self.fill_simulator
        for _, row in positions.iterrows():
            px = sim._get_market_price(row["ticker"], as_of)
            if px:
                total_equity += px * row["shares"]
            else:
                total_equity += row["fill_price"] * row["shares"]   # fallback to cost
        return self.cash + total_equity

    # ── execution ─────────────────────────────────────────────────────────────

    def buy(
        self,
        ticker:         str,
        trade_date:     str,
        position_size:  float,            # dollar amount to deploy
        metadata:       Optional[Dict]  = None,
    ) -> bool:
        """
        Open a new position.  Returns True on success.
        metadata: dict with keys strategy, signal, rv_gate_count, etc.
        """
        meta = metadata or {}

        fill = self.fill_simulator.simulate_fill(ticker, "BUY", trade_date)
        if fill is None:
            print(f"  [skip] {ticker}: no price data for {trade_date}")
            return False

        shares = position_size / fill.fill_price
        cost   = shares * fill.fill_price

        if cost > self.cash:
            print(f"  [skip] {ticker}: insufficient cash ({self.cash:.2f} < {cost:.2f})")
            return False

        conn = _get_paper_db()
        try:
            new_cash = self.cash - cost
            conn.execute(
                """
                INSERT INTO positions
                (ticker, shares, entry_price, fill_price, entry_date,
                 strategy, signal, rv_gate_count, rv_distress, rv_tail,
                 rv_momentum, regime, status,
                 mos_at_entry, fair_value_bear, fair_value_base, fair_value_bull,
                 valuation_method, confidence_score, sector)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,'open',?,?,?,?,?,?,?)
                """,
                (
                    ticker, round(shares, 6), fill.market_price, fill.fill_price,
                    trade_date,
                    meta.get("strategy"),
                    meta.get("signal"),
                    meta.get("rv_gate_count"),
                    meta.get("rv_distress"),
                    meta.get("rv_tail"),
                    meta.get("rv_momentum"),
                    meta.get("regime"),
                    meta.get("mos_at_entry"),
                    meta.get("fair_value_bear"),
                    meta.get("fair_value_base"),
                    meta.get("fair_value_bull"),
                    meta.get("valuation_method"),
                    meta.get("confidence_score"),
                    meta.get("sector"),
                ),
            )
            conn.execute(
                """
                INSERT INTO transactions
                (ts, ticker, action, shares, market_price, fill_price,
                 slippage_bps, spread_bps, value, cash_after, notes)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    datetime.utcnow().isoformat(),
                    ticker, "BUY", round(shares, 6),
                    fill.market_price, fill.fill_price,
                    fill.slippage_bps, fill.spread_bps,
                    round(cost, 2), round(new_cash, 2),
                    json.dumps({"size_requested": position_size}),
                ),
            )
            self._set_cash(conn, new_cash)
            conn.commit()
            print(
                f"  BUY  {ticker:6s}  {shares:.2f} sh @ {fill.fill_price:.2f}"
                f"  (slip {fill.slippage_bps:.1f}bps)  cash left: {new_cash:,.2f}"
            )
            return True
        finally:
            conn.close()

    def sell(
        self,
        ticker:     str,
        trade_date: str,
        reason:     str = "rebalance",
    ) -> bool:
        """Close an existing position.  Returns True on success."""
        conn = _get_paper_db()
        try:
            row = conn.execute(
                "SELECT * FROM positions WHERE ticker=? AND status='open' LIMIT 1",
                (ticker,),
            ).fetchone()
            if not row:
                print(f"  [skip] {ticker}: no open position found")
                return False

            fill = self.fill_simulator.simulate_fill(ticker, "SELL", trade_date)
            if fill is None:
                print(f"  [skip] {ticker}: no price data for sell on {trade_date}")
                return False

            proceeds = row["shares"] * fill.fill_price
            entry_dt = datetime.strptime(row["entry_date"], "%Y-%m-%d")
            exit_dt  = datetime.strptime(trade_date, "%Y-%m-%d")
            hold_days = (exit_dt - entry_dt).days
            pnl_pct   = (fill.fill_price / row["fill_price"] - 1) * 100

            new_cash = self.cash + proceeds
            conn.execute(
                """
                UPDATE positions
                SET status='closed', exit_price=?, exit_date=?,
                    pnl_pct=?, hold_days=?
                WHERE id=?
                """,
                (fill.fill_price, trade_date,
                 round(pnl_pct, 4), hold_days, row["id"]),
            )
            conn.execute(
                """
                INSERT INTO transactions
                (ts, ticker, action, shares, market_price, fill_price,
                 slippage_bps, spread_bps, value, cash_after, notes)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    datetime.utcnow().isoformat(),
                    ticker, "SELL", row["shares"],
                    fill.market_price, fill.fill_price,
                    fill.slippage_bps, fill.spread_bps,
                    round(proceeds, 2), round(new_cash, 2),
                    json.dumps({"reason": reason, "pnl_pct": round(pnl_pct, 2)}),
                ),
            )
            self._set_cash(conn, new_cash)
            conn.commit()
            print(
                f"  SELL {ticker:6s}  {row['shares']:.2f} sh @ {fill.fill_price:.2f}"
                f"  P&L {pnl_pct:+.2f}%  hold {hold_days}d  cash: {new_cash:,.2f}"
            )
            return True
        finally:
            conn.close()

    # ── position sizing ────────────────────────────────────────────────────────

    def compute_position_sizes(
        self,
        tickers:        List[str],
        trade_date:     str,
        method:         str = "inverse_vol",   # "equal" | "inverse_vol"
        vol_window:     int = 60,              # trading days for vol estimate
        regime_sources: Optional[set] = None, # tickers from regime-winning strategy
        regime_boost:   float = REGIME_BOOST, # weight multiplier for regime picks
    ) -> Dict[str, float]:
        """
        Compute dollar position sizes for a list of tickers.

        inverse_vol: weight = 1/σ, normalised so total = investable cash.
        Investable cash = 95% of portfolio value (5% buffer).
        Falls back to equal weight if volatility data is unavailable.
        Tickers in regime_sources receive a weight multiplier of regime_boost
        before normalisation, shifting capital toward regime-winner picks.
        """
        investable = self.portfolio_value(trade_date) * 0.95
        per_position = investable / max(len(tickers), 1)

        if method == "equal" or not tickers:
            return {t: per_position for t in tickers}

        # Estimate vol for each ticker from historical returns
        vols: Dict[str, float] = {}
        conn = get_market_db_connection()
        try:
            cutoff = (
                datetime.strptime(trade_date, "%Y-%m-%d")
                - timedelta(days=int(vol_window * 1.5))
            ).strftime("%Y-%m-%d")

            for ticker in tickers:
                rows = conn.execute(
                    """
                    SELECT h.close FROM history h
                    JOIN tickers t ON h.ticker_id = t.id
                    WHERE t.symbol = ? AND h.date BETWEEN ? AND ?
                    ORDER BY h.date
                    """,
                    (ticker, cutoff, trade_date),
                ).fetchall()
                prices = [float(r[0]) for r in rows if r[0]]
                if len(prices) >= 10:
                    rets = np.diff(np.log(prices))
                    vols[ticker] = float(np.std(rets, ddof=1)) * np.sqrt(252)
                # else: omit ticker from vols; vols.get(t) returns None (falsy) below
        finally:
            conn.close()

        # Build weights — fall back to equal weight for missing vols
        valid = {t: v for t, v in vols.items() if v and v > 0}
        if not valid:
            return {t: per_position for t in tickers}

        avg_vol = float(np.mean(list(valid.values())))
        raw_weights = {
            t: (1.0 / vols[t]) if vols.get(t) else (1.0 / avg_vol)
            for t in tickers
        }

        # Apply regime boost before normalisation
        if regime_sources:
            boosted = [t for t in tickers if t in regime_sources and t in raw_weights]
            for t in boosted:
                raw_weights[t] *= regime_boost

        total_w = sum(raw_weights.values())
        return {
            t: round((w / total_w) * investable, 2)
            for t, w in raw_weights.items()
        }


# ─────────────────────────────────────────────────────────────────────────────
# 4.  EXECUTION SCHEDULER
# ─────────────────────────────────────────────────────────────────────────────

class ExecutionScheduler:
    """
    Monthly rebalance loop.

    Each rebalance:
    1. Runs run_pipeline() to get the current selected portfolio
    2. Computes which positions to sell (no longer selected)
    3. Computes which tickers to buy (newly selected)
    4. Sizes positions using inverse-vol weighting
    5. Executes fills via FillSimulator
    6. Logs the rebalance to rebalance_log
    """

    def __init__(
        self,
        portfolio:          PaperPortfolio,
        pipeline_kwargs:    Optional[Dict] = None,
        sizing_method:      str = "inverse_vol",
        min_signals:        Optional[set] = None,
        rebalance_cash:     bool  = False,
        max_cash_pct:       float = 0.05,    # keep this fraction as reserve floor
        min_topup:          float = 500.0,   # skip redistribution if deployable < this
        regime_boost:       float = REGIME_BOOST,
    ):
        self.portfolio       = portfolio
        self.pipeline_kwargs = pipeline_kwargs or {}
        self.regime_boost    = regime_boost
        self.sizing_method   = sizing_method
        # Default: exclude Overvalued (negative MOS). Pass set() to accept all signals.
        self.min_signals = min_signals if min_signals is not None else {
            "Strong Buy", "Speculative Buy", "Hold"
        }
        self.rebalance_cash = rebalance_cash
        self.max_cash_pct   = max_cash_pct
        self.min_topup      = min_topup

    def _run_pipeline(self, trade_date: str) -> pd.DataFrame:
        """Run the MarketLedger pipeline and return the selected portfolio."""
        from main import run_pipeline

        kwargs = {
            "update_data":          False,
            "apply_momentum_filter": True,
            "verbose":              True,
            "strict_contracts":     False,
            "include_rejected":     True,
            "export_decision_log":  False,
            "output_file":          str(
                Path(__file__).parent / "paper_trading_output.csv"
            ),
        }
        kwargs.update(self.pipeline_kwargs)
        result = run_pipeline(**kwargs)
        return result

    def rebalance(self, trade_date: Optional[str] = None) -> Dict:
        """
        Execute one rebalance cycle.  Returns a summary dict.
        """
        trade_date = trade_date or date.today().strftime("%Y-%m-%d")
        print(f"\n{'='*60}")
        print(f"  REBALANCE  {trade_date}")
        print(f"  Portfolio value: {self.portfolio.portfolio_value(trade_date):,.2f}")
        print(f"{'='*60}")

        # Step 1: Run pipeline
        print("\n[1/4] Running pipeline...")
        selected_df = self._run_pipeline(trade_date)

        if selected_df is None or selected_df.empty:
            print("  Pipeline returned no selections. Holding current positions.")
            self._log_rebalance(
                trade_date, regime=None, candidates=0, selected=0,
                buys=0, sells=0, holds=len(self.portfolio.position_tickers()),
                notes="Pipeline returned no selections",
            )
            return {"status": "no_selections", "trade_date": trade_date}

        # Apply signal filter — exclude Overvalued and other unwanted signals
        if self.min_signals and "Investment Signal" in selected_df.columns:
            pre_filter = len(selected_df)
            selected_df = selected_df[
                selected_df["Investment Signal"].isin(self.min_signals)
            ].copy()
            dropped = pre_filter - len(selected_df)
            if dropped:
                print(f"  Signal filter: removed {dropped} ticker(s) outside {sorted(self.min_signals)}")

        # Extract target tickers and metadata
        target_tickers = selected_df["Ticker"].tolist()
        current_tickers = self.portfolio.position_tickers()

        # Detect regime if column present
        regime = None
        if "RV_Momentum_Regime" in selected_df.columns:
            regime = selected_df["RV_Momentum_Regime"].mode().iloc[0] if not selected_df.empty else None

        print(f"\n  Target:  {target_tickers}")
        print(f"  Current: {current_tickers}")

        to_sell = [t for t in current_tickers if t not in target_tickers]
        to_buy  = [t for t in target_tickers  if t not in current_tickers]
        to_hold = [t for t in current_tickers if t in target_tickers]

        print(f"\n  Sell: {to_sell}")
        print(f"  Buy:  {to_buy}")
        print(f"  Hold: {to_hold}")

        # Step 2: Execute sells first (free up cash)
        print("\n[2/4] Executing sells...")
        sells_done = 0
        for ticker in to_sell:
            if self.portfolio.sell(ticker, trade_date, reason="rebalance"):
                sells_done += 1

        # Step 3: Size and buy
        print("\n[3/4] Sizing and executing buys...")
        regime_sources: set = set()
        if "RegimeSource" in selected_df.columns:
            regime_sources = set(
                selected_df[selected_df["RegimeSource"] == True]["Ticker"].tolist()
            )
        sizes = self.portfolio.compute_position_sizes(
            to_buy, trade_date, method=self.sizing_method,
            regime_sources=regime_sources if regime_sources else None,
            regime_boost=self.regime_boost,
        )
        boosted_tickers = [t for t in to_buy if t in regime_sources]
        if boosted_tickers:
            print(f"  [Regime boost {REGIME_BOOST}×] {', '.join(boosted_tickers)}")
        buys_done = 0
        for ticker in to_buy:
            size = sizes.get(ticker, 0)
            if size <= 0:
                continue
            # Pull metadata from pipeline output
            row = selected_df[selected_df["Ticker"] == ticker]
            meta = {}
            if not row.empty:
                r = row.iloc[0]
                current_price = r.get("Current Price") or 0
                fv_base = r.get("Fair Value (Base)") or 0
                mos = (fv_base - current_price) / current_price if current_price > 0 and fv_base > 0 else None
                meta = {
                    "strategy":         r.get("Strategy"),
                    "signal":           r.get("Investment Signal"),
                    "rv_gate_count":    r.get("RV_Gate_Count"),
                    "rv_distress":      r.get("RV_Distress_Bucket"),
                    "rv_tail":          r.get("RV_Tail_Bucket"),
                    "rv_momentum":      r.get("RV_Momentum_Regime"),
                    "regime":           regime,
                    "mos_at_entry":     mos,
                    "fair_value_bear":  r.get("Fair Value (Bear)"),
                    "fair_value_base":  fv_base if fv_base > 0 else None,
                    "fair_value_bull":  r.get("Fair Value (Bull)"),
                    "valuation_method": r.get("Valuation Method"),
                    "confidence_score": r.get("Confidence Score"),
                    "sector":           r.get("Sector"),
                }
            if self.portfolio.buy(ticker, trade_date, size, metadata=meta):
                buys_done += 1

        # Step 3b: Redistribute idle cash into holds (if --rebalance-cash)
        if self.rebalance_cash:
            topups = self._distribute_excess_cash(trade_date, to_hold)
            buys_done += topups

        # Step 4: Log
        pv = self.portfolio.portfolio_value(trade_date)
        print(f"\n[4/4] Rebalance complete.")
        print(f"  Portfolio value: {pv:,.2f}  Cash: {self.portfolio.cash:,.2f}")

        self._log_rebalance(
            trade_date, regime=regime,
            candidates=len(selected_df),
            selected=len(target_tickers),
            buys=buys_done,
            sells=sells_done,
            holds=len(to_hold),
        )

        return {
            "status":     "ok",
            "trade_date": trade_date,
            "buys":       buys_done,
            "sells":      sells_done,
            "holds":      len(to_hold),
            "portfolio_value": pv,
        }

    def _log_rebalance(
        self,
        trade_date: str,
        regime:     Optional[str],
        candidates: int,
        selected:   int,
        buys:       int,
        sells:      int,
        holds:      int,
        notes:      Optional[str] = None,
    ) -> None:
        conn = _get_paper_db()
        try:
            conn.execute(
                """
                INSERT INTO rebalance_log
                (ts, trade_date, regime, candidates, selected,
                 buys, sells, holds, portfolio_value, cash, notes)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    datetime.utcnow().isoformat(),
                    trade_date, regime, candidates, selected,
                    buys, sells, holds,
                    round(self.portfolio.portfolio_value(trade_date), 2),
                    round(self.portfolio.cash, 2),
                    notes,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def run_monthly_loop(
        self,
        check_interval_seconds: int = 3600,   # check every hour
    ) -> None:
        """
        Run indefinitely, triggering a rebalance on the first trading day
        of each month.  Sleeps between checks.

        This is the long-running daemon mode.
        """
        print("Monthly scheduler started.  Press Ctrl+C to stop.")
        last_rebalance_month: Optional[str] = self._last_rebalance_month()

        while True:
            now = datetime.utcnow()
            current_month = now.strftime("%Y-%m")

            if current_month != last_rebalance_month:
                trade_date = self._first_trading_day_of_month(now.year, now.month)
                if now.date() >= datetime.strptime(trade_date, "%Y-%m-%d").date():
                    print(f"\n[scheduler] Triggering rebalance for {current_month}")
                    try:
                        self.rebalance(trade_date)
                        last_rebalance_month = current_month
                    except Exception as exc:
                        print(f"[scheduler] Rebalance failed: {exc}")
                else:
                    print(
                        f"[scheduler] Waiting for first trading day "
                        f"({trade_date}). Sleeping {check_interval_seconds}s..."
                    )
            else:
                print(
                    f"[scheduler] Already rebalanced this month ({current_month}). "
                    f"Sleeping {check_interval_seconds}s..."
                )

            time.sleep(check_interval_seconds)

    def _last_rebalance_month(self) -> Optional[str]:
        conn = _get_paper_db()
        try:
            row = conn.execute(
                "SELECT trade_date FROM rebalance_log ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if row:
                return row[0][:7]   # YYYY-MM
            return None
        finally:
            conn.close()

    @staticmethod
    def _first_trading_day_of_month(year: int, month: int) -> str:
        """Return the first weekday of the given month as YYYY-MM-DD."""
        d = date(year, month, 1)
        while d.weekday() >= 5:   # 5=Sat, 6=Sun
            d += timedelta(days=1)
        return d.strftime("%Y-%m-%d")

    def _inverse_vol_weights(
        self,
        tickers:    List[str],
        trade_date: str,
        vol_window: int = 60,
    ) -> Dict[str, float]:
        """
        Return inverse-volatility weights for tickers, normalised to sum to 1.0.
        Falls back to equal weight for any ticker with insufficient price history.
        """
        if not tickers:
            return {}

        vols: Dict[str, float] = {}
        conn = get_market_db_connection()
        try:
            cutoff = (
                datetime.strptime(trade_date, "%Y-%m-%d")
                - timedelta(days=int(vol_window * 1.5))
            ).strftime("%Y-%m-%d")

            for ticker in tickers:
                rows = conn.execute(
                    """
                    SELECT h.close FROM history h
                    JOIN tickers t ON h.ticker_id = t.id
                    WHERE t.symbol = ? AND h.date BETWEEN ? AND ?
                    ORDER BY h.date
                    """,
                    (ticker, cutoff, trade_date),
                ).fetchall()
                prices = [float(r[0]) for r in rows if r[0]]
                if len(prices) >= 10:
                    rets = np.diff(np.log(prices))
                    vols[ticker] = float(np.std(rets, ddof=1)) * np.sqrt(252)
                # else: omit ticker from vols; vols.get(t) returns None (falsy) below
        finally:
            conn.close()

        valid = {t: v for t, v in vols.items() if v and v > 0}
        avg_vol = float(np.mean(list(valid.values()))) if valid else 1.0
        raw = {t: (1.0 / vols[t]) if vols.get(t) else (1.0 / avg_vol) for t in tickers}
        total = sum(raw.values())
        return {t: w / total for t, w in raw.items()}

    def _distribute_excess_cash(
        self,
        trade_date:   str,
        hold_tickers: List[str],
    ) -> int:
        """
        Top up existing hold positions with cash above the reserve floor.
        Returns the number of successful top-up buy orders placed.
        """
        if not hold_tickers:
            return 0

        pv         = self.portfolio.portfolio_value(trade_date)
        cash       = self.portfolio.cash
        deployable = cash - pv * self.max_cash_pct

        if deployable < self.min_topup:
            return 0

        print(
            f"\n  [cash-rebalance] {cash:,.2f} cash, floor={pv*self.max_cash_pct:,.2f}  "
            f"deploying {deployable:,.2f} into {hold_tickers}"
        )

        if self.sizing_method == "inverse_vol":
            weights = self._inverse_vol_weights(hold_tickers, trade_date)
        else:
            n = len(hold_tickers)
            weights = {t: 1.0 / n for t in hold_tickers}

        done = 0
        # Snapshot open positions once (avoids repeated DB calls inside loop)
        positions = self.portfolio.open_positions()
        for ticker in hold_tickers:
            allocation = round(deployable * weights.get(ticker, 1.0 / len(hold_tickers)), 2)
            if allocation < self.min_topup:
                continue
            # Carry forward existing position metadata so the new lot is attributed correctly
            pos_rows = positions[positions["ticker"] == ticker]
            meta = {}
            if not pos_rows.empty:
                r = pos_rows.iloc[0]
                meta = {
                    "strategy":      r.get("strategy"),
                    "signal":        r.get("signal"),
                    "rv_gate_count": r.get("rv_gate_count"),
                    "rv_distress":   r.get("rv_distress"),
                    "rv_tail":       r.get("rv_tail"),
                    "rv_momentum":   r.get("rv_momentum"),
                    "regime":        r.get("regime"),
                }
            if self.portfolio.buy(ticker, trade_date, allocation, metadata=meta):
                done += 1
        return done


# ─────────────────────────────────────────────────────────────────────────────
# 5.  OUTCOME TRACKER
# ─────────────────────────────────────────────────────────────────────────────

def print_outcome_report() -> None:
    """
    Score closed positions: did the Investment Signal at entry predict
    actual returns?  Print a summary grouped by signal label.
    """
    conn = _get_paper_db()
    try:
        df = pd.read_sql_query(
            "SELECT * FROM positions WHERE status='closed'", conn
        )
    finally:
        conn.close()

    if df.empty:
        print("No closed positions yet.")
        return

    print("\n" + "=" * 70)
    print("  OUTCOME REPORT — closed positions")
    print("=" * 70)

    # By signal
    if "signal" in df.columns and df["signal"].notna().any():
        grouped = df.groupby("signal").agg(
            count      = ("pnl_pct", "count"),
            mean_pnl   = ("pnl_pct", "mean"),
            median_pnl = ("pnl_pct", "median"),
            hit_rate   = ("pnl_pct", lambda x: (x > 0).mean() * 100),
            avg_hold   = ("hold_days", "mean"),
        ).sort_values("mean_pnl", ascending=False)
        print("\nBy Investment Signal at entry:\n")
        print(grouped.round(2).to_string())

    # By strategy
    if "strategy" in df.columns and df["strategy"].notna().any():
        grouped2 = df.groupby("strategy").agg(
            count      = ("pnl_pct", "count"),
            mean_pnl   = ("pnl_pct", "mean"),
            hit_rate   = ("pnl_pct", lambda x: (x > 0).mean() * 100),
        ).sort_values("mean_pnl", ascending=False)
        print("\n\nBy Strategy at entry:\n")
        print(grouped2.round(2).to_string())

    # Overall
    print(f"\n\nOverall: {len(df)} closed positions")
    print(f"  Mean P&L:   {df['pnl_pct'].mean():.2f}%")
    print(f"  Median P&L: {df['pnl_pct'].median():.2f}%")
    print(f"  Hit rate:   {(df['pnl_pct'] > 0).mean()*100:.1f}%")
    print(f"  Avg hold:   {df['hold_days'].mean():.0f} days")


def return_attribution_report(output_dir: str = "research_outputs") -> None:
    """
    Build a return-attribution table from closed positions and write it to
    research_outputs/return_attribution.csv.  Prints a pivot summary grouped
    by signal × MOS bucket.
    """
    conn = _get_paper_db()
    try:
        df = pd.read_sql_query("SELECT * FROM positions WHERE status='closed'", conn)
    finally:
        conn.close()

    if df.empty:
        print("No closed positions for attribution analysis.")
        return

    def _mos_bucket(v) -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "Unknown"
        if v < 0:
            return "<0%"
        if v < 0.20:
            return "0-20%"
        if v < 0.40:
            return "20-40%"
        return ">40%"

    df["mos_bucket"] = df["mos_at_entry"].apply(_mos_bucket)

    export_cols = [
        "ticker", "strategy", "signal", "mos_bucket", "regime", "sector",
        "entry_date", "exit_date", "hold_days", "pnl_pct",
        "mos_at_entry", "fair_value_bear", "fair_value_base", "fair_value_bull",
        "valuation_method", "confidence_score",
    ]
    export_cols = [c for c in export_cols if c in df.columns]
    out_df = df[export_cols].copy()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / "return_attribution.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nReturn attribution written to {out_path} ({len(out_df)} closed positions)")

    # Pivot summary: mean P&L by signal × mos_bucket
    if "signal" in df.columns and df["signal"].notna().any() and "mos_bucket" in df.columns:
        pivot = df.groupby(["signal", "mos_bucket"])["pnl_pct"].agg(
            n="count", mean_pnl="mean", hit_rate=lambda x: (x > 0).mean() * 100
        ).round(2)
        print("\nReturn attribution pivot (signal × MOS bucket):\n")
        print(pivot.to_string())


# ─────────────────────────────────────────────────────────────────────────────
# 6.  CLI
# ─────────────────────────────────────────────────────────────────────────────

def print_status(portfolio: PaperPortfolio) -> None:
    trade_date = date.today().strftime("%Y-%m-%d")
    pv   = portfolio.portfolio_value(trade_date)
    cash = portfolio.cash
    pos  = portfolio.open_positions()

    conn = _get_paper_db()
    try:
        starting = float(
            conn.execute(
                "SELECT value FROM portfolio_meta WHERE key='starting_cash'"
            ).fetchone()[0]
        )
    finally:
        conn.close()

    print(f"\n{'='*60}")
    print(f"  PAPER PORTFOLIO STATUS — {trade_date}")
    print(f"{'='*60}")
    print(f"  Starting cash:   {starting:>12,.2f}")
    print(f"  Portfolio value: {pv:>12,.2f}  ({(pv/starting-1)*100:+.2f}%)")
    print(f"  Cash:            {cash:>12,.2f}  ({cash/pv*100:.1f}% of portfolio)")
    print(f"  Open positions:  {len(pos)}")

    if not pos.empty:
        print("\n  Positions:")
        for _, row in pos.iterrows():
            mkt = portfolio.fill_simulator._get_market_price(row["ticker"], trade_date)
            unrealised = ""
            if mkt:
                unreal = (mkt / row["fill_price"] - 1) * 100
                unrealised = f"  unrealised {unreal:+.1f}%"
            print(
                f"    {row['ticker']:6s}  {row['shares']:.2f} sh "
                f"@ {row['fill_price']:.2f}  in {row['entry_date']}"
                f"  [{row['signal'] or '?'}]{unrealised}"
            )


def print_history() -> None:
    conn = _get_paper_db()
    try:
        df = pd.read_sql_query(
            "SELECT ts, ticker, action, shares, fill_price, value, cash_after "
            "FROM transactions ORDER BY id DESC LIMIT 50",
            conn,
        )
    finally:
        conn.close()

    if df.empty:
        print("No transactions yet.")
        return

    print("\n  Last 50 transactions:")
    print(df.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="MarketLedger paper trading engine")
    parser.add_argument("--run-now",   action="store_true", help="Trigger a rebalance immediately")
    parser.add_argument("--status",    action="store_true", help="Print current portfolio")
    parser.add_argument("--history",   action="store_true", help="Print transaction log")
    parser.add_argument("--outcomes",  action="store_true", help="Print outcome report")
    parser.add_argument("--schedule",  action="store_true", help="Run the monthly scheduler loop")
    parser.add_argument("--cash",      type=float, default=DEFAULT_STARTING_CASH,
                        help="Starting cash for a fresh portfolio")
    parser.add_argument("--date",      type=str, default=None,
                        help="Override trade date (YYYY-MM-DD) for --run-now")
    parser.add_argument("--no-momentum", action="store_true",
                        help="Pass --no-momentum to the pipeline")
    parser.add_argument("--exchange",  type=str, default=None,
                        help="Pass --exchange to the pipeline")
    parser.add_argument("--rebalance-cash", action="store_true",
                        help="Redistribute idle cash (above 5%% reserve) into existing hold positions")
    parser.add_argument("--regime-boost", type=float, default=REGIME_BOOST,
                        help="Weight multiplier for regime-winner tickers (default: %(default)s)")
    args = parser.parse_args()

    portfolio = PaperPortfolio(starting_cash=args.cash)

    pipeline_kwargs: Dict = {}
    if args.no_momentum:
        pipeline_kwargs["apply_momentum_filter"] = False
    if args.exchange:
        pipeline_kwargs["exchange"] = args.exchange

    scheduler = ExecutionScheduler(
        portfolio,
        pipeline_kwargs=pipeline_kwargs,
        rebalance_cash=args.rebalance_cash,
        regime_boost=args.regime_boost,
    )

    if args.status:
        print_status(portfolio)
    elif args.history:
        print_history()
        return_attribution_report()
    elif args.outcomes:
        print_outcome_report()
        return_attribution_report()
    elif args.run_now:
        trade_date = args.date or date.today().strftime("%Y-%m-%d")
        scheduler.rebalance(trade_date)
        print_status(portfolio)
    elif args.schedule:
        scheduler.run_monthly_loop()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
