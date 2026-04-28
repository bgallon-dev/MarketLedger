"""
Regime Transition Early Warning System — forensic module.

Monitors aggregate shifts in SEC filing language across the ticker universe
to detect macro regime transitions before they appear in price data. Reads
from the existing sec_filings SQLite cache populated by forensic/contagion.py.

The key idea: companies begin disclosing different risks, hedging more, and
changing forward-guidance vocabulary 3-6 months before macro transitions show
up in price. Monitoring the filing *ensemble* (not individual tickers) surfaces
this signal.

Calibration is anchored to the 2022 Risk-off transition — the only confirmed
historical event in this dataset. Feature weights are domain priors, not
statistically estimated. Treat the headline score as supplementary input.

Public API:
    RegimeEarlyWarningSystem(db_path, verbose, max_workers) — main class
    run_regime_early_warning(tickers, calibrate, output_dir) -> Dict[str, Any]

CLI:
    py -m forensic.regime_early_warning --calibrate
    py -m forensic.regime_early_warning --score
    py -m forensic.regime_early_warning --score --tickers AAPL MSFT
    py -m forensic.regime_early_warning --prefetch
"""

import argparse
import json
import re
import sqlite3
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────

DB_PATH = Path(__file__).parent.parent / "database" / "financial_data.db"
RESEARCH_DIR = Path(__file__).parent.parent / "research_outputs"

# ── NLP vocabulary ─────────────────────────────────────────────────────────────

_UNCERTAINTY_WORDS = frozenset({
    "uncertain", "uncertainty", "may", "could", "challenging",
    "volatile", "adverse", "headwind", "difficult", "weakness",
    "unpredictable", "concern", "concerns", "cautious",
})

_MACRO_RISK_PHRASES = [
    "inflation",
    "interest rate",
    "recession",
    "credit",
    "liquidity",
    "default",
    "tightening",
    "monetary policy",
    "federal reserve",
    "rate hike",
    "rate increase",
    "rising rates",
    "rate environment",
]

_OPERATIONAL_RISK_PHRASES = [
    "supply chain",
    "inventory",
    "pricing pressure",
    "labor",
    "disruption",
    "shortage",
    "demand",
    "margin",
    "cost",
]

# Feature weights — domain priors tunable after inspecting regime_language_drift.csv.
# macro_risk_density is highest: most predictive for the 2022 transition.
_FEATURE_WEIGHTS: Dict[str, float] = {
    "uncertainty_density": 2.0,
    "macro_risk_density": 2.5,
    "operational_risk_density": 1.5,
    "risk_factor_count": 1.0,
    "text_length_words": 0.5,
    "numeric_density": 0.3,
    "novelty_fraction": 1.0,
}

_FEATURE_KEYS: List[str] = list(_FEATURE_WEIGHTS.keys())


# ── Data model ─────────────────────────────────────────────────────────────────

@dataclass
class FilingFeatures:
    ticker: str
    filed_date: str
    fiscal_year: int
    uncertainty_density: float
    macro_risk_density: float
    operational_risk_density: float
    risk_factor_count: int
    text_length_words: int
    numeric_density: float
    novelty_fraction: float    # nan when no prior filing is available
    item_7_word_count: int
    source: str = "cache"


@dataclass
class CorpusSnapshot:
    cohort_label: str
    as_of_date: str
    n_tickers: int
    n_missing: int
    feature_means: Dict[str, float] = field(default_factory=dict)
    feature_stds: Dict[str, float] = field(default_factory=dict)
    feature_p75: Dict[str, float] = field(default_factory=dict)
    frac_above_2sigma: Dict[str, float] = field(default_factory=dict)
    raw_features: List[FilingFeatures] = field(default_factory=list)  # in-memory only
    computed_at: str = ""


# ── Main class ─────────────────────────────────────────────────────────────────

class RegimeEarlyWarningSystem:
    """
    Aggregate SEC filing language monitor for regime transition detection.

    Usage::

        rew = RegimeEarlyWarningSystem()
        result = rew.score_current()   # uses all cached tickers
        print(result["filing_transition_score"], result["transition_risk_label"])
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        verbose: bool = True,
        max_workers: int = 4,
    ) -> None:
        self.db_path = db_path or DB_PATH
        self.verbose = verbose
        self.max_workers = max_workers
        self._ensure_snapshot_table()

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    # ── SQLite setup ────────────────────────────────────────────────────────────

    def _ensure_snapshot_table(self) -> None:
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS filing_corpus_snapshots (
                    cohort_label      TEXT NOT NULL,
                    as_of_date        TEXT NOT NULL,
                    n_tickers         INTEGER,
                    n_missing         INTEGER,
                    feature_means     TEXT,
                    feature_stds      TEXT,
                    feature_p75       TEXT,
                    frac_above_2sigma TEXT,
                    computed_at       TEXT NOT NULL,
                    PRIMARY KEY (cohort_label, as_of_date)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_corpus_snapshots_date
                    ON filing_corpus_snapshots(as_of_date)
            """)
            conn.commit()
        finally:
            conn.close()

    # ── Ticker helpers ──────────────────────────────────────────────────────────

    def _get_cached_tickers(self, cap: int = 500) -> List[str]:
        """Return all distinct tickers with cached filings, up to cap."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            rows = conn.execute(
                "SELECT DISTINCT ticker FROM sec_filings ORDER BY ticker LIMIT ?",
                (cap,),
            ).fetchall()
        except Exception:
            return []
        finally:
            conn.close()
        return [r[0] for r in rows]

    # ── Filing cache queries ────────────────────────────────────────────────────

    def _load_cached_filing(
        self, ticker: str, as_of_date: str
    ) -> Optional[Dict[str, Any]]:
        """Return the most recent sec_filings row for ticker filed on or before as_of_date."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                """
                SELECT ticker, filed_date, item_1a_text, item_7_text
                FROM sec_filings
                WHERE ticker = ? AND filed_date <= ?
                  AND item_1a_text IS NOT NULL
                ORDER BY filed_date DESC
                LIMIT 1
                """,
                (ticker.upper(), as_of_date),
            ).fetchone()
        except Exception:
            return None
        finally:
            conn.close()
        if row is None:
            return None
        return dict(row)

    def _load_prior_filing_text(
        self, ticker: str, before_date: str
    ) -> Optional[str]:
        """Return item_1a_text of the filing immediately before before_date."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                """
                SELECT item_1a_text
                FROM sec_filings
                WHERE ticker = ? AND filed_date < ?
                  AND item_1a_text IS NOT NULL
                ORDER BY filed_date DESC
                LIMIT 1
                """,
                (ticker.upper(), before_date),
            ).fetchone()
        except Exception:
            return None
        finally:
            conn.close()
        return row["item_1a_text"] if row else None

    # ── Feature extraction ──────────────────────────────────────────────────────

    @staticmethod
    def extract_features(
        item_1a_text: str,
        item_7_text: Optional[str],
        prior_1a_text: Optional[str] = None,
        ticker: str = "",
        filed_date: str = "",
    ) -> FilingFeatures:
        """
        Extract NLP features from a single filing's Item 1A and Item 7 sections.
        Pure function — no DB calls.
        """

        def _clean(t: str) -> str:
            t = re.sub(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]", " ", t)
            return re.sub(r"\s+", " ", t).strip()

        text_1a = _clean(item_1a_text) if item_1a_text else ""
        text_7 = _clean(item_7_text) if item_7_text else ""
        text_lower = text_1a.lower()

        # Tokenize to individual words for single-term matching
        words = re.findall(r"[a-z]+", text_lower)
        total_words = max(1, len(words))

        # Uncertainty density — single words via set membership
        unc_count = sum(1 for w in words if w in _UNCERTAINTY_WORDS)
        uncertainty_density = unc_count / total_words

        # Macro risk density — mix of single words and multi-word phrases
        macro_count = 0
        for phrase in _MACRO_RISK_PHRASES:
            if " " in phrase:
                macro_count += len(re.findall(phrase.replace(" ", r"\s+"), text_lower))
            else:
                macro_count += text_lower.count(phrase)
        macro_risk_density = macro_count / total_words

        # Operational risk density
        op_count = 0
        for phrase in _OPERATIONAL_RISK_PHRASES:
            if " " in phrase:
                op_count += len(re.findall(phrase.replace(" ", r"\s+"), text_lower))
            else:
                op_count += text_lower.count(phrase)
        operational_risk_density = op_count / total_words

        # Risk factor count — numbered list items or all-caps headings
        numbered = len(re.findall(r"^\s*\d+[\.\)]\s", text_1a, re.MULTILINE))
        headed = len(re.findall(r"\n[A-Z][A-Z\s]{10,60}\n", text_1a))
        risk_factor_count = max(numbered, headed)

        # Numeric density — fraction of tokens that are numbers
        num_tokens = len(re.findall(r"\b\d+\.?\d*\b", text_lower))
        numeric_density = num_tokens / total_words

        # Novelty fraction — unique words in current not in prior filing
        novelty_fraction = float("nan")
        if prior_1a_text and prior_1a_text.strip():
            prior_lower = _clean(prior_1a_text).lower()
            current_vocab = set(re.findall(r"[a-z]{4,}", text_lower))
            prior_vocab = set(re.findall(r"[a-z]{4,}", prior_lower))
            if current_vocab:
                novelty_fraction = len(current_vocab - prior_vocab) / len(current_vocab)

        item_7_word_count = len(re.findall(r"[a-z]+", text_7.lower())) if text_7 else 0
        fiscal_year = int(filed_date[:4]) if len(filed_date) >= 4 else 0

        return FilingFeatures(
            ticker=ticker,
            filed_date=filed_date,
            fiscal_year=fiscal_year,
            uncertainty_density=min(uncertainty_density, 1.0),
            macro_risk_density=min(macro_risk_density, 1.0),
            operational_risk_density=min(operational_risk_density, 1.0),
            risk_factor_count=risk_factor_count,
            text_length_words=len(words),
            numeric_density=min(numeric_density, 1.0),
            novelty_fraction=novelty_fraction,
            item_7_word_count=item_7_word_count,
        )

    # ── Corpus loading ──────────────────────────────────────────────────────────

    def load_corpus(
        self,
        tickers: List[str],
        as_of_date: str,
        cohort_label: Optional[str] = None,
        force_refresh: bool = False,
    ) -> CorpusSnapshot:
        """
        Build a CorpusSnapshot from the sec_filings cache.

        Cache-only: does not fetch from EDGAR. Run forensic/run_all.py (or
        --prefetch) first to populate the cache for the desired ticker set.
        """
        label = cohort_label or f"snapshot_{as_of_date}"

        if not force_refresh:
            cached = self._load_snapshot_from_db(label, as_of_date)
            if cached is not None:
                self._log(
                    f"  [rew] Snapshot '{label}' loaded from cache "
                    f"({cached.n_tickers} tickers)"
                )
                return cached

        self._log(
            f"  [rew] Building corpus snapshot '{label}' as_of={as_of_date} "
            f"({len(tickers)} tickers) ..."
        )

        def _process(ticker: str) -> Optional[FilingFeatures]:
            filing = self._load_cached_filing(ticker, as_of_date)
            if filing is None:
                return None
            prior_text = self._load_prior_filing_text(ticker, filing["filed_date"])
            try:
                feat = self.extract_features(
                    item_1a_text=filing["item_1a_text"] or "",
                    item_7_text=filing.get("item_7_text"),
                    prior_1a_text=prior_text,
                    ticker=ticker,
                    filed_date=filing["filed_date"],
                )
                return feat
            except Exception:
                return None

        features: List[FilingFeatures] = []
        n_missing = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(_process, t): t for t in tickers}
            for future in as_completed(futures):
                result = future.result()
                # Discard filings with fewer than 200 words — likely parse failures
                if result is not None and result.text_length_words >= 200:
                    features.append(result)
                else:
                    n_missing += 1

        snapshot = self._aggregate_features(label, as_of_date, features, n_missing)
        self._save_snapshot_to_db(snapshot)
        self._log(
            f"  [rew] Snapshot '{label}': {snapshot.n_tickers} scored, "
            f"{snapshot.n_missing} missing/thin"
        )
        return snapshot

    def _aggregate_features(
        self,
        cohort_label: str,
        as_of_date: str,
        features: List[FilingFeatures],
        n_missing: int,
    ) -> CorpusSnapshot:
        snap = CorpusSnapshot(
            cohort_label=cohort_label,
            as_of_date=as_of_date,
            n_tickers=len(features),
            n_missing=n_missing,
            raw_features=features,
            computed_at=datetime.now().isoformat(),
        )
        if not features:
            return snap

        for key in _FEATURE_KEYS:
            vals = np.array([getattr(f, key) for f in features], dtype=float)
            snap.feature_means[key] = float(np.nanmean(vals))
            snap.feature_stds[key] = float(np.nanstd(vals))
            snap.feature_p75[key] = float(np.nanpercentile(vals, 75))
            snap.frac_above_2sigma[key] = float("nan")  # populated in calibrate_history

        return snap

    # ── Snapshot persistence ────────────────────────────────────────────────────

    def _load_snapshot_from_db(
        self, cohort_label: str, as_of_date: str
    ) -> Optional[CorpusSnapshot]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                """
                SELECT * FROM filing_corpus_snapshots
                WHERE cohort_label = ? AND as_of_date = ?
                """,
                (cohort_label, as_of_date),
            ).fetchone()
        except Exception:
            return None
        finally:
            conn.close()

        if row is None:
            return None
        try:
            return CorpusSnapshot(
                cohort_label=row["cohort_label"],
                as_of_date=row["as_of_date"],
                n_tickers=row["n_tickers"] or 0,
                n_missing=row["n_missing"] or 0,
                feature_means=json.loads(row["feature_means"] or "{}"),
                feature_stds=json.loads(row["feature_stds"] or "{}"),
                feature_p75=json.loads(row["feature_p75"] or "{}"),
                frac_above_2sigma=json.loads(row["frac_above_2sigma"] or "{}"),
                raw_features=[],
                computed_at=row["computed_at"],
            )
        except Exception:
            return None

    def _save_snapshot_to_db(self, snap: CorpusSnapshot) -> None:
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO filing_corpus_snapshots
                    (cohort_label, as_of_date, n_tickers, n_missing,
                     feature_means, feature_stds, feature_p75,
                     frac_above_2sigma, computed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snap.cohort_label,
                    snap.as_of_date,
                    snap.n_tickers,
                    snap.n_missing,
                    json.dumps(snap.feature_means),
                    json.dumps(snap.feature_stds),
                    json.dumps(snap.feature_p75),
                    json.dumps(snap.frac_above_2sigma),
                    snap.computed_at,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    # ── Drift computation ───────────────────────────────────────────────────────

    @staticmethod
    def compute_drift(
        current: CorpusSnapshot, baseline: CorpusSnapshot
    ) -> Dict[str, Any]:
        """
        Compute corpus-level language drift between two snapshots.

        Returns a dict with:
            composite_score        : float 0-100 (50 = no drift from baseline)
            transition_risk_label  : "Elevated" / "Moderate" / "Low"
            feature_z_scores       : {feature: z-score}
            feature_drift_pct      : {feature: % change vs baseline mean}
            top_drivers            : [top 3 feature names by |z-score|]
        """
        z_scores: Dict[str, float] = {}
        drift_pct: Dict[str, float] = {}

        for key in _FEATURE_KEYS:
            c_mean = current.feature_means.get(key, float("nan"))
            b_mean = baseline.feature_means.get(key, float("nan"))
            b_std = baseline.feature_stds.get(key, float("nan"))

            if np.isnan(c_mean) or np.isnan(b_mean):
                z_scores[key] = float("nan")
                drift_pct[key] = float("nan")
                continue

            denom = max(float(b_std) if not np.isnan(b_std) else 0.0, 1e-9)
            z_scores[key] = (c_mean - b_mean) / denom
            drift_pct[key] = (c_mean - b_mean) / max(abs(b_mean), 1e-9) * 100.0

        # Weighted composite z
        weighted_z = 0.0
        total_weight = 0.0
        for key, w in _FEATURE_WEIGHTS.items():
            z = z_scores.get(key, float("nan"))
            if not np.isnan(z):
                weighted_z += w * z
                total_weight += w

        composite_z = weighted_z / total_weight if total_weight > 0.0 else 0.0

        # Map to 0-100: zero drift -> 50, ±5σ -> 0/100
        composite_score = float(np.clip(50.0 + 10.0 * composite_z, 0.0, 100.0))

        if composite_score >= 65:
            risk_label = "Elevated"
        elif composite_score >= 40:
            risk_label = "Moderate"
        else:
            risk_label = "Low"

        valid_z = {k: v for k, v in z_scores.items() if not np.isnan(v)}
        top_drivers = sorted(valid_z, key=lambda k: abs(valid_z[k]), reverse=True)[:3]

        return {
            "composite_score": composite_score,
            "composite_z": composite_z,
            "transition_risk_label": risk_label,
            "feature_z_scores": z_scores,
            "feature_drift_pct": drift_pct,
            "top_drivers": top_drivers,
            "n_current": current.n_tickers,
            "n_baseline": baseline.n_tickers,
            "current_cohort": current.cohort_label,
            "baseline_cohort": baseline.cohort_label,
        }

    # ── Historical calibration ──────────────────────────────────────────────────

    def calibrate_history(
        self,
        tickers: Optional[List[str]] = None,
        start_year: int = 2019,
        end_year: Optional[int] = None,
        output_dir: Optional[Path] = None,
    ) -> pd.DataFrame:
        """
        Build CorpusSnapshots for each fiscal year from start_year to end_year,
        compute drift vs baseline (FY start_year) and vs prior year, and write
        research_outputs/regime_language_drift.csv.

        Inspect that CSV to validate the 2022 transition signature and tune
        _FEATURE_WEIGHTS if needed.
        """
        if tickers is None:
            tickers = self._get_cached_tickers()
        if not tickers:
            self._log(
                "[rew] No cached tickers — run 'py -m forensic.run_all' or "
                "'py -m forensic.regime_early_warning --prefetch' first."
            )
            return pd.DataFrame()

        end_yr = end_year or (datetime.now().year - 1)
        out_dir = output_dir or RESEARCH_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        regime_map = self._load_regime_labels(out_dir)

        # FY N filings typically filed Feb-Apr of N+1; use June 30 of N+1 as cutoff
        cohorts: List[CorpusSnapshot] = []
        for year in range(start_year, end_yr + 1):
            as_of = f"{year + 1}-06-30"
            snap = self.load_corpus(tickers, as_of, cohort_label=f"FY{year}")
            cohorts.append(snap)

        if len(cohorts) < 2:
            self._log("[rew] Need at least 2 cohorts for drift calculation.")
            return pd.DataFrame()

        baseline = cohorts[0]
        rows: List[Dict[str, Any]] = []
        prior: Optional[CorpusSnapshot] = None

        for snap in cohorts:
            row: Dict[str, Any] = {
                "cohort_label": snap.cohort_label,
                "as_of_date": snap.as_of_date,
                "n_tickers": snap.n_tickers,
                "n_missing": snap.n_missing,
            }
            for key in _FEATURE_KEYS:
                row[f"{key}_mean"] = snap.feature_means.get(key, float("nan"))

            drift_base = self.compute_drift(snap, baseline)
            row["drift_vs_baseline_composite"] = drift_base["composite_score"]
            for key in _FEATURE_KEYS:
                row[f"z_vs_baseline_{key}"] = drift_base["feature_z_scores"].get(
                    key, float("nan")
                )

            if prior is not None:
                drift_prior = self.compute_drift(snap, prior)
                row["drift_vs_prior_composite"] = drift_prior["composite_score"]
                for key in _FEATURE_KEYS:
                    row[f"z_vs_prior_{key}"] = drift_prior["feature_z_scores"].get(
                        key, float("nan")
                    )
            else:
                row["drift_vs_prior_composite"] = float("nan")
                for key in _FEATURE_KEYS:
                    row[f"z_vs_prior_{key}"] = float("nan")

            # Join regime label by calendar year (the year the filing describes)
            fy_str = snap.cohort_label.replace("FY", "")
            fy = int(fy_str) if fy_str.isdigit() else None
            row["market_regime"] = regime_map.get(f"{fy}_Market_Regime", "")
            row["rate_regime"] = regime_map.get(f"{fy}_Rate_Regime", "")
            row["regime_label"] = regime_map.get(fy, "")

            rows.append(row)
            prior = snap

        df = pd.DataFrame(rows)
        out_path = out_dir / "regime_language_drift.csv"
        df.to_csv(out_path, index=False)
        self._log(f"[rew] Calibration complete -> {out_path}")
        return df

    def _load_regime_labels(self, research_dir: Path) -> Dict:
        """Read regime_labels.csv into a lookup dict keyed by calendar year."""
        labels_path = research_dir / "regime_labels.csv"
        mapping: Dict = {}
        if not labels_path.exists():
            return mapping
        try:
            df = pd.read_csv(labels_path)
            df.columns = [c.strip() for c in df.columns]
            for _, row in df.iterrows():
                # Calendar_Year column may be "2022" or "2025_YTD"; take first 4 digits
                raw = str(row.get("Calendar_Year", row.get("Year_Bucket", "0")))
                year = int(raw[:4]) if raw[:4].isdigit() else None
                if year is None:
                    continue
                market = str(row.get("Market_Regime", ""))
                rate = str(row.get("Rate_Regime", ""))
                mapping[year] = f"{market} / {rate}"
                mapping[f"{year}_Market_Regime"] = market
                mapping[f"{year}_Rate_Regime"] = rate
        except Exception:
            pass
        return mapping

    # ── Live scoring ────────────────────────────────────────────────────────────

    def score_current(
        self,
        tickers: Optional[List[str]] = None,
        as_of_date: Optional[str] = None,
        baseline_cohort_label: str = "FY2019",
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Score the current filing corpus against a historical baseline.

        Returns a dict with filing_transition_score (0-100) and supporting
        detail suitable for merging into detect_current_regime() output.
        """
        if tickers is None:
            tickers = self._get_cached_tickers()
        as_of = as_of_date or datetime.now().strftime("%Y-%m-%d")
        out_dir = output_dir or RESEARCH_DIR

        current = self.load_corpus(tickers, as_of, cohort_label="current")

        if current.n_tickers == 0:
            self._log(
                "[rew] No filing data — populate sec_filings cache first "
                "(run_all.py or --prefetch)."
            )
            return self._empty_score_result(as_of, baseline_cohort_label, current)

        # Resolve baseline snapshot
        fy_num = int(baseline_cohort_label.replace("FY", "")) if baseline_cohort_label.startswith("FY") else 2019
        baseline_as_of = f"{fy_num + 1}-06-30"
        baseline = self._load_snapshot_from_db(baseline_cohort_label, baseline_as_of)

        if baseline is None:
            self._log(
                f"[rew] Baseline '{baseline_cohort_label}' not cached — "
                "running calibrate_history() ..."
            )
            self.calibrate_history(tickers, output_dir=out_dir)
            baseline = self._load_snapshot_from_db(baseline_cohort_label, baseline_as_of)

        if baseline is None or baseline.n_tickers == 0:
            self._log(f"[rew] Baseline '{baseline_cohort_label}' has no filing data.")
            return self._empty_score_result(as_of, baseline_cohort_label, current)

        drift = self.compute_drift(current, baseline)
        self.generate_report(drift, current, out_dir)

        return {
            "filing_transition_score": drift["composite_score"],
            "transition_risk_label": drift["transition_risk_label"],
            "feature_z_scores": drift["feature_z_scores"],
            "feature_drift_pct": drift["feature_drift_pct"],
            "top_drivers": drift["top_drivers"],
            "composite_z": drift["composite_z"],
            "n_tickers_scored": current.n_tickers,
            "n_tickers_missing": current.n_missing,
            "current_as_of_date": as_of,
            "baseline_cohort": baseline_cohort_label,
        }

    @staticmethod
    def _empty_score_result(
        as_of: str, baseline_cohort_label: str, current: CorpusSnapshot
    ) -> Dict[str, Any]:
        return {
            "filing_transition_score": None,
            "transition_risk_label": "Unknown",
            "feature_z_scores": {},
            "feature_drift_pct": {},
            "top_drivers": [],
            "composite_z": None,
            "n_tickers_scored": current.n_tickers,
            "n_tickers_missing": current.n_missing,
            "current_as_of_date": as_of,
            "baseline_cohort": baseline_cohort_label,
        }

    # ── Report generation ───────────────────────────────────────────────────────

    def generate_report(
        self,
        drift: Dict[str, Any],
        current: CorpusSnapshot,
        output_dir: Optional[Path] = None,
    ) -> None:
        """Write a plain-text transition risk report to regime_ew_report.txt."""
        out_dir = output_dir or RESEARCH_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        score = drift["composite_score"]
        risk_label = drift["transition_risk_label"]
        z_scores = drift["feature_z_scores"]

        lines = [
            "=" * 64,
            "REGIME TRANSITION EARLY WARNING REPORT",
            f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"As-of     : {current.as_of_date}",
            "=" * 64,
            "",
            "HEADLINE SCORE",
            f"  TransitionRisk Score : {score:.1f} / 100",
            f"  Risk Label           : {risk_label}",
            f"  Tickers Scored       : {current.n_tickers}",
            f"  Tickers Missing/Thin : {current.n_missing}",
            f"  Baseline Cohort      : {drift.get('baseline_cohort', 'FY2019')}",
            "",
            "TOP LANGUAGE DRIVERS  (corpus z-scores vs baseline)",
        ]
        for i, key in enumerate(drift.get("top_drivers", []), 1):
            z = z_scores.get(key, float("nan"))
            sign = "+" if z >= 0 else ""
            lines.append(f"  {i}. {key:<34} {sign}{z:.2f}σ")

        lines += [
            "",
            "FEATURE DETAIL",
            f"  {'Feature':<36} {'Current Mean':>14} {'Z-Score':>10}",
            "  " + "-" * 64,
        ]
        for key in _FEATURE_KEYS:
            cur_mean = current.feature_means.get(key, float("nan"))
            z = z_scores.get(key, float("nan"))
            sign = "+" if (not np.isnan(z) and z >= 0) else ""
            z_str = f"{sign}{z:.2f}σ" if not np.isnan(z) else "   n/a"
            lines.append(f"  {key:<36} {cur_mean:>14.5f} {z_str:>10}")

        # Historical context if calibration CSV exists
        drift_csv = out_dir / "regime_language_drift.csv"
        if drift_csv.exists():
            try:
                hist = pd.read_csv(drift_csv)
                lines += [
                    "",
                    "HISTORICAL CONTEXT  (drift_vs_baseline_composite)",
                    "",
                ]
                for _, hrow in hist.iterrows():
                    val = hrow.get("drift_vs_baseline_composite", float("nan"))
                    regime = hrow.get("regime_label", "")
                    cohort = hrow.get("cohort_label", "")
                    if not np.isnan(float(val)):
                        marker = " ◄ RISK-OFF" if "Risk-off" in str(regime) else ""
                        lines.append(f"  {cohort:<12} {float(val):>6.1f}{marker}")
                lines.append(f"  {'current':<12} {score:>6.1f}   ← now")
            except Exception:
                pass

        lines += [
            "",
            "NOTE: Experimental. The 2022 Risk-off is the only confirmed",
            "calibration event. Feature weights are domain priors, not",
            "statistically estimated. Use as supplementary input only.",
            "=" * 64,
        ]

        out_path = out_dir / "regime_ew_report.txt"
        out_path.write_text("\n".join(lines), encoding="utf-8")
        self._log(f"[rew] Report written -> {out_path}")


# ── Convenience wrapper ─────────────────────────────────────────────────────────

def run_regime_early_warning(
    tickers: Optional[List[str]] = None,
    calibrate: bool = False,
    output_dir: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    High-level entry point. Calibrates if needed then scores the current corpus.
    Returns score_current() result dict.
    """
    out = Path(output_dir) if output_dir else RESEARCH_DIR
    rew = RegimeEarlyWarningSystem(verbose=verbose)
    if calibrate:
        rew.calibrate_history(tickers, output_dir=out)
    return rew.score_current(tickers, output_dir=out)


# ── CLI ─────────────────────────────────────────────────────────────────────────

def _prefetch_filings(tickers: List[str], verbose: bool) -> None:
    """Trigger EDGAR fetch for any ticker not yet in the sec_filings cache."""
    try:
        from forensic.contagion import fetch_filing_history
    except ImportError:
        from contagion import fetch_filing_history  # type: ignore[no-redef]

    for i, ticker in enumerate(tickers, 1):
        if verbose:
            print(f"  [{i}/{len(tickers)}] {ticker}")
        try:
            fetch_filing_history(ticker, n_filings=5, verbose=verbose)
        except Exception as exc:
            if verbose:
                print(f"    Error: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m forensic.regime_early_warning",
        description="Regime Transition Early Warning System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Workflow:
              1. Populate filing cache (once, ~30-60 min for large universe):
                   py -m forensic.run_all --no-archaeology --no-credibility

              2. Build historical calibration CSV:
                   py -m forensic.regime_early_warning --calibrate

              3. Score the current corpus:
                   py -m forensic.regime_early_warning --score

              Inspect research_outputs/regime_language_drift.csv to validate
              that FY2020 drift_vs_baseline_composite is elevated vs FY2019,
              then tune _FEATURE_WEIGHTS in the module if needed.
        """),
    )
    parser.add_argument(
        "--tickers", nargs="+", metavar="TICKER",
        help="Ticker subset (default: all tickers in sec_filings cache)",
    )
    parser.add_argument(
        "--calibrate", action="store_true",
        help="Build historical drift CSV from start_year to present",
    )
    parser.add_argument(
        "--score", action="store_true",
        help="Score the current corpus and write regime_ew_report.txt",
    )
    parser.add_argument(
        "--prefetch", action="store_true",
        help="Fetch missing EDGAR filings via contagion before analysis",
    )
    parser.add_argument(
        "--baseline", default="FY2019", metavar="COHORT",
        help="Baseline cohort label for drift (default: FY2019)",
    )
    parser.add_argument(
        "--start-year", type=int, default=2019, metavar="YEAR",
        help="First fiscal year for calibration (default: 2019)",
    )
    parser.add_argument(
        "--output-dir", default=None, metavar="DIR",
        help="Output directory (default: research_outputs/)",
    )
    parser.add_argument("--max-workers", type=int, default=4, metavar="N")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else RESEARCH_DIR
    verbose = not args.quiet
    rew = RegimeEarlyWarningSystem(verbose=verbose, max_workers=args.max_workers)

    tickers: Optional[List[str]] = args.tickers or None
    if tickers is None:
        tickers = rew._get_cached_tickers()
        if not tickers:
            print(
                "[rew] No cached filings found.\n"
                "Run: py -m forensic.run_all --no-archaeology --no-credibility\n"
                "Or:  py -m forensic.regime_early_warning --prefetch --tickers AAPL MSFT ..."
            )
            return

    if args.prefetch:
        print(f"[rew] Prefetching EDGAR filings for {len(tickers)} tickers ...")
        _prefetch_filings(tickers, verbose)

    if args.calibrate:
        print(f"[rew] Running historical calibration from FY{args.start_year} ...")
        df = rew.calibrate_history(
            tickers, start_year=args.start_year, output_dir=out_dir
        )
        if not df.empty:
            cols = ["cohort_label", "n_tickers", "drift_vs_baseline_composite",
                    "market_regime", "rate_regime"]
            show = [c for c in cols if c in df.columns]
            print(df[show].to_string(index=False))

    # Default: score if --score given, or if nothing else was requested
    if args.score or (not args.calibrate and not args.prefetch):
        print(f"[rew] Scoring current corpus ({len(tickers)} tickers) ...")
        result = rew.score_current(
            tickers, baseline_cohort_label=args.baseline, output_dir=out_dir
        )
        score = result.get("filing_transition_score")
        label = result.get("transition_risk_label", "Unknown")
        if score is not None:
            print(f"\n  TransitionRisk Score : {score:.1f}/100 ({label})")
            print(f"  Tickers scored       : {result.get('n_tickers_scored', 0)}")
            drivers = result.get("top_drivers", [])
            if drivers:
                print(f"  Top drivers          : {', '.join(drivers)}")
        else:
            print("\n  Score unavailable — no filing data in cache.")


if __name__ == "__main__":
    main()
