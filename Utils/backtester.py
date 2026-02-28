"""
Point-in-Time Backtester - Avoids Look-Ahead Bias

The key insight: You cannot use 2023 Annual Numbers to buy a stock in Jan 2023,
because those numbers weren't released yet.

Logic:
1. Pick a Trade Date (e.g., 2023-01-01)
2. Apply a "Reporting Lag" (e.g., 90 days) - financial data dated 2022-09-30 is visible on 2023-01-01
3. Filter: Only look at data older than the reporting lag
4. Rank & Pick: Run screening logic on that historical snapshot
5. Calculate Return: Check price change from Trade Date to Exit Date
"""

import pandas as pd
import numpy as np
import json
import os
import time
from copy import deepcopy
from dataclasses import dataclass
from numbers import Real
from datetime import timedelta, datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any, Sequence, cast, Callable
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from database.database import get_connection, query_database, ensure_indices
from Utils.scoring import robust_score
from risk.risk_vector import RiskVectorConfig, attach_risk_vectors
from valuation.valuation_projector import (
    calculate_scenario_valuations,
    check_valuation_sanity,
)

# Define metrics required for all strategies to optimize loading
REQUIRED_METRICS = [
    "EBIT",
    "InvestedCapital",
    "TotalDebt",
    "CashAndCashEquivalents",
    "OrdinarySharesNumber",
    "TotalRevenue",
    "OperatingIncome",
    "ResearchAndDevelopment",
    "OperatingCashFlow",
    "CapitalExpenditure",
    # Added for super_margins strategy
    "NetIncome",
    "StockholdersEquity",
    "GrossProfit",
    # Added for cannibal, piotroski_f_score, rule_of_40 strategies
    "TotalAssets",
    "CurrentAssets",
    "CurrentLiabilities",
    "RepurchaseOfCapitalStock",
    # Added for risk-vector distress calculations
    "WorkingCapital",
    "RetainedEarnings",
    "TotalLiabilitiesNetMinorityInterest",
]

DEFAULT_PROXY_SYMBOLS: Tuple[str, ...] = ("SPY", "QQQ", "IWD", "IWF", "^IRX")
DEFAULT_VALUE_PROXY_CANDIDATES: Tuple[str, ...] = ("IWD", "VTV", "IVE")
DEFAULT_GROWTH_PROXY_CANDIDATES: Tuple[str, ...] = ("IWF", "VUG", "IVW")
DEFAULT_RATE_PROXY_CANDIDATES: Tuple[str, ...] = ("^IRX", "IRX", "^TNX", "TNX")
DEFAULT_MAGIC_GATE_VARIANTS: Tuple[str, ...] = (
    "Magic (raw)",
    "Magic + Distress Gate",
    "Magic + Tail Gate",
    "Magic + Momentum Gate",
    "Magic + Full Risk Vector",
)
DEFAULT_BACKTESTER_CONFIG_FILE = "backtester_config.json"
DEFAULT_BACKTESTER_PROFILE_NAME = "default"

SUPPORTED_STRATEGY_NAMES: Tuple[str, ...] = (
    "magic_formula",
    "moat",
    "fcf_yield",
    "quality",
    "diamonds_in_dirt",
    "cannibal",
    "piotroski_f_score",
    "rule_of_40",
    "super_margins",
)
STRATEGY_ALIASES: Dict[str, str] = {
    "super_margins": "diamonds_in_dirt",
}

DEFAULT_TOURNAMENT_CONTENDERS: List[Dict[str, Any]] = [
    {"name": "Pure Magic Formula", "strategies": ["magic_formula"], "n": 10},
    {"name": "Pure Moat", "strategies": ["moat"], "n": 10},
    {"name": "Pure Diamonds", "strategies": ["diamonds_in_dirt"], "n": 10},
    {"name": "Pure FCF Yield", "strategies": ["fcf_yield"], "n": 10},
    {"name": "Pure Quality", "strategies": ["quality"], "n": 10},
    {"name": "Pure Cannibal", "strategies": ["cannibal"], "n": 10},
    {
        "name": "Pure Piotroski F-Score",
        "strategies": ["piotroski_f_score"],
        "n": 10,
    },
    {"name": "Pure Rule of 40", "strategies": ["rule_of_40"], "n": 10},
    {
        "name": "Fortress (Moat + Diamond)",
        "strategies": ["moat", "diamonds_in_dirt"],
        "n": 5,
    },
    {
        "name": "Greenblatt+ (Magic + Moat)",
        "strategies": ["magic_formula", "moat"],
        "n": 5,
    },
    {
        "name": "Deep Value (Magic + Diamond)",
        "strategies": ["magic_formula", "diamonds_in_dirt"],
        "n": 5,
    },
    {
        "name": "Cash Cow (FCF + Cannibal)",
        "strategies": ["fcf_yield", "cannibal"],
        "n": 5,
    },
    {
        "name": "Growth Quality (R40 + Piotroski)",
        "strategies": ["rule_of_40", "piotroski_f_score"],
        "n": 5,
    },
]

DEFAULT_BACKTESTER_PROFILE: Dict[str, Any] = {
    "universe": {
        "strategy_min_market_cap_mm": 250.0,
        "direct_filter_default_min_market_cap_mm": 200.0,
        "revenue_proxy_multiplier": 0.5,
        "warrant_regex": r"^[A-Z]{4}[WRU]$",
    },
    "execution": {
        "default_top_n": 10,
        "combo_top_n_per_strategy": 5,
        "magic_gate_top_n": 10,
    },
    "strategies": {
        "magic_formula": {},
        "moat": {"min_total_revenue": 100_000_000},
        "fcf_yield": {"require_positive_fcf": True},
        "quality": {"min_roic": 0.05, "max_roic": 1.0},
        "diamonds_in_dirt": {
            "gross_margin_min": 0.40,
            "op_margin_min": 0.15,
            "roe_min": 0.15,
            "debt_to_revenue_max": 3.0,
            "require_positive_earnings_yield": True,
        },
        "cannibal": {
            "min_buyback_pct": 0.03,
            "require_positive_fcf": True,
            "min_market_cap_mm": 200.0,
        },
        "piotroski_f_score": {"min_f_score": 7},
        "rule_of_40": {
            "min_r40_score": 40.0,
            "min_total_revenue": 50_000_000,
        },
    },
    "tournament": {
        "magic_gate_top_n": 10,
        "contenders": deepcopy(DEFAULT_TOURNAMENT_CONTENDERS),
    },
}

STRATEGY_PARAM_KEYS: Dict[str, set] = {
    "magic_formula": set(),
    "moat": {"min_total_revenue"},
    "fcf_yield": {"require_positive_fcf"},
    "quality": {"min_roic", "max_roic"},
    "diamonds_in_dirt": {
        "gross_margin_min",
        "op_margin_min",
        "roe_min",
        "debt_to_revenue_max",
        "require_positive_earnings_yield",
    },
    "cannibal": {"min_buyback_pct", "require_positive_fcf", "min_market_cap_mm"},
    "piotroski_f_score": {"min_f_score"},
    "rule_of_40": {"min_r40_score", "min_total_revenue"},
}


@dataclass(frozen=True)
class RegimeLabelConfig:
    start_date: str = "2022-01-01"
    end_date: str = "2025-06-30"
    quantile_low: float = 0.33
    quantile_high: float = 0.67
    required_proxy_symbols: Tuple[str, ...] = DEFAULT_PROXY_SYMBOLS


@dataclass(frozen=True)
class TournamentOutputConfig:
    output_dir: str = "research_outputs"
    csv_only: bool = True
    rf_symbol: str = "^IRX"
    min_months_per_bucket: int = 3
    preserve_return_delta_floor: float = -0.01
    cash_fill_cost_bps: float = 5.0


@dataclass(frozen=True)
class UniverseProfile:
    strategy_min_market_cap_mm: float = 250.0
    direct_filter_default_min_market_cap_mm: float = 200.0
    revenue_proxy_multiplier: float = 0.5
    warrant_regex: str = r"^[A-Z]{4}[WRU]$"


@dataclass(frozen=True)
class ExecutionProfile:
    default_top_n: int = 10
    combo_top_n_per_strategy: int = 5
    magic_gate_top_n: int = 10


@dataclass(frozen=True)
class StrategyProfile:
    params: Dict[str, Dict[str, Any]]


@dataclass(frozen=True)
class TournamentContender:
    name: str
    strategies: Tuple[str, ...]
    n: int


@dataclass(frozen=True)
class TournamentProfile:
    magic_gate_top_n: int = 10
    contenders: Tuple[TournamentContender, ...] = ()


@dataclass(frozen=True)
class BacktesterProfile:
    universe: UniverseProfile
    execution: ExecutionProfile
    strategies: StrategyProfile
    tournament: TournamentProfile


def _require_real(value: Any, field_name: str) -> float:
    """
    Validate that a value is a non-missing real number and convert it to float.

    Raises:
        TypeError: If the value is missing or not a real numeric type.
    """
    if value is None:
        raise TypeError(f"{field_name} must be a real number, got None")
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be a real number, got bool")
    try:
        missing = pd.isna(value)
    except Exception:
        # Some objects may not support pandas null checks.
        missing = False
    if isinstance(missing, (bool, np.bool_)) and missing:
        raise TypeError(f"{field_name} must be a real number, got missing value")
    if not isinstance(value, Real):
        raise TypeError(
            f"{field_name} must be a real number, got {type(value).__name__}"
        )
    return float(value)


class VectorBacktester:
    """
    Point-in-time backtester that avoids look-ahead bias.

    Usage:
    ------
    >>> bt = VectorBacktester()
    >>> bt.load_data()
    >>> results = bt.run_backtest('2023-01-01', '2024-01-01', strategy='magic_formula')
    """

    def __init__(
        self,
        reporting_lag_days: int = 90,
        price_staleness_days: int = 7,
        config_path: Optional[str] = None,
        profile_name: Optional[str] = None,
    ):
        """
        Initialize the backtester.

        Parameters:
        -----------
        reporting_lag_days : int
            Days between period end and when data becomes available.
            Default 90 days (companies report ~45-90 days after quarter end)
        """
        self.reporting_lag = reporting_lag_days
        self.price_staleness_days = max(0, int(price_staleness_days))
        self.price_matrix: Optional[pd.DataFrame] = None
        self.financial_data: Optional[pd.DataFrame] = None
        self.benchmark_symbol = "QQQ"  # Nasdaq ETF for comparison
        self._last_research_outputs: Optional[Dict[str, pd.DataFrame]] = None
        self._regime_cache: Dict[Tuple[str, int, int], str] = {}
        self._tail_state_cache: Dict[Tuple[str, str, int, int], Dict[str, Any]] = {}
        self._momentum_cache: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
        self._snapshot_cache: Dict[Tuple[str, int], pd.DataFrame] = {}
        self._default_strategy_config_path = str(
            Path(__file__).resolve().parent / DEFAULT_BACKTESTER_CONFIG_FILE
        )
        self._loaded_profile_cache: Dict[Tuple[str, str], BacktesterProfile] = {}
        self._active_profile_name: Optional[str] = None
        self._active_config_path: Optional[str] = None
        self._active_profile = self._profile_from_dict(
            deepcopy(DEFAULT_BACKTESTER_PROFILE)
        )
        self._strategy_registry: Dict[
            str, Callable[[pd.DataFrame, int, Optional[str], Dict[str, Any]], List[str]]
        ] = {
            "magic_formula": self._strategy_magic_formula,
            "moat": self._strategy_moat,
            "fcf_yield": self._strategy_fcf_yield,
            "quality": self._strategy_quality,
            "diamonds_in_dirt": self._strategy_diamonds_in_dirt,
            "cannibal": self._strategy_cannibal,
            "piotroski_f_score": self._strategy_piotroski_f_score,
            "rule_of_40": self._strategy_rule_of_40,
        }
        if profile_name is not None or config_path is not None:
            self.set_strategy_profile(
                profile_name or DEFAULT_BACKTESTER_PROFILE_NAME,
                config_path=config_path,
            )

    def set_strategy_profile(
        self, profile_name: str, config_path: Optional[str] = None
    ) -> None:
        """Set the active strategy profile for this backtester instance."""
        profile = self._resolve_profile(profile_name=profile_name, config_path=config_path)
        self._active_profile = profile
        self._active_profile_name = profile_name
        self._active_config_path = (
            str(Path(config_path))
            if config_path is not None
            else self._default_strategy_config_path
        )

    def get_active_profile(self) -> Dict[str, Any]:
        """Return the active profile as a plain dictionary."""
        return self._profile_to_dict(self._active_profile)

    @staticmethod
    def _require_positive_real(value: Any, field_name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(
                f"{field_name} must be a real number, got {type(value).__name__}"
            )
        out = float(value)
        if out <= 0:
            raise ValueError(f"{field_name} must be > 0, got {value}")
        return out

    @staticmethod
    def _require_non_negative_real(value: Any, field_name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(
                f"{field_name} must be a real number, got {type(value).__name__}"
            )
        out = float(value)
        if out < 0:
            raise ValueError(f"{field_name} must be >= 0, got {value}")
        return out

    @staticmethod
    def _require_positive_int(value: Any, field_name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{field_name} must be an integer, got {type(value).__name__}")
        if value <= 0:
            raise ValueError(f"{field_name} must be > 0, got {value}")
        return int(value)

    def _canonical_strategy_name(self, name: str) -> str:
        key = str(name).strip()
        return STRATEGY_ALIASES.get(key, key)

    def _load_profile_file(self, config_path: str) -> Dict[str, Any]:
        path = Path(config_path)
        if not path.exists():
            raise ValueError(f"Backtester config file does not exist: {config_path}")
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in config file {config_path}: {exc}") from exc
        except OSError as exc:
            raise ValueError(f"Failed to read config file {config_path}: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"Config root must be a JSON object: {config_path}")
        return data

    def _validate_strategy_params(
        self, strategy_name: str, params: Dict[str, Any], source: str
    ) -> None:
        canonical = self._canonical_strategy_name(strategy_name)
        allowed = STRATEGY_PARAM_KEYS.get(canonical)
        if allowed is None:
            raise ValueError(
                f"Unknown strategy '{strategy_name}' in {source}. "
                f"Supported: {sorted(SUPPORTED_STRATEGY_NAMES)}"
            )
        unknown = set(params.keys()) - allowed
        if unknown:
            raise ValueError(
                f"Unknown params for strategy '{strategy_name}' in {source}: "
                f"{sorted(unknown)} (allowed: {sorted(allowed)})"
            )

        bool_fields = {"require_positive_fcf", "require_positive_earnings_yield"}
        for key in bool_fields & set(params.keys()):
            if not isinstance(params[key], bool):
                raise ValueError(
                    f"{source}.{strategy_name}.{key} must be bool, got {type(params[key]).__name__}"
                )

        non_negative_fields = {
            "min_total_revenue",
            "min_market_cap_mm",
            "min_f_score",
            "min_r40_score",
            "min_roic",
            "max_roic",
        }
        for key in non_negative_fields & set(params.keys()):
            self._require_non_negative_real(
                params[key], f"{source}.{strategy_name}.{key}"
            )

        positive_fields = {
            "gross_margin_min",
            "op_margin_min",
            "roe_min",
            "debt_to_revenue_max",
            "min_buyback_pct",
        }
        for key in positive_fields & set(params.keys()):
            self._require_positive_real(params[key], f"{source}.{strategy_name}.{key}")

        if "min_f_score" in params:
            min_f = int(self._require_non_negative_real(params["min_f_score"], f"{source}.{strategy_name}.min_f_score"))
            if min_f > 9:
                raise ValueError(
                    f"{source}.{strategy_name}.min_f_score must be <= 9, got {min_f}"
                )
        if "min_roic" in params and "max_roic" in params:
            min_roic = float(params["min_roic"])
            max_roic = float(params["max_roic"])
            if min_roic > max_roic:
                raise ValueError(
                    f"{source}.{strategy_name}.min_roic must be <= max_roic"
                )

    def _validate_profile_block(self, profile_name: str, block: Dict[str, Any], source: str) -> None:
        allowed_profile_keys = {"universe", "execution", "strategies", "tournament"}
        unknown_profile_keys = set(block.keys()) - allowed_profile_keys
        if unknown_profile_keys:
            raise ValueError(
                f"Unknown keys in profile '{profile_name}' ({source}): {sorted(unknown_profile_keys)}"
            )

        universe = block.get("universe", {})
        if not isinstance(universe, dict):
            raise ValueError(f"profile '{profile_name}'.universe must be an object")
        allowed_universe = {
            "strategy_min_market_cap_mm",
            "direct_filter_default_min_market_cap_mm",
            "revenue_proxy_multiplier",
            "warrant_regex",
        }
        unknown_universe = set(universe.keys()) - allowed_universe
        if unknown_universe:
            raise ValueError(
                f"Unknown universe keys in profile '{profile_name}': {sorted(unknown_universe)}"
            )
        if "strategy_min_market_cap_mm" in universe:
            self._require_non_negative_real(
                universe["strategy_min_market_cap_mm"],
                f"profile '{profile_name}'.universe.strategy_min_market_cap_mm",
            )
        if "direct_filter_default_min_market_cap_mm" in universe:
            self._require_non_negative_real(
                universe["direct_filter_default_min_market_cap_mm"],
                f"profile '{profile_name}'.universe.direct_filter_default_min_market_cap_mm",
            )
        if "revenue_proxy_multiplier" in universe:
            self._require_positive_real(
                universe["revenue_proxy_multiplier"],
                f"profile '{profile_name}'.universe.revenue_proxy_multiplier",
            )
        if "warrant_regex" in universe and not isinstance(universe["warrant_regex"], str):
            raise ValueError(
                f"profile '{profile_name}'.universe.warrant_regex must be a string"
            )

        execution = block.get("execution", {})
        if not isinstance(execution, dict):
            raise ValueError(f"profile '{profile_name}'.execution must be an object")
        allowed_execution = {
            "default_top_n",
            "combo_top_n_per_strategy",
            "magic_gate_top_n",
        }
        unknown_execution = set(execution.keys()) - allowed_execution
        if unknown_execution:
            raise ValueError(
                f"Unknown execution keys in profile '{profile_name}': {sorted(unknown_execution)}"
            )
        for key in allowed_execution & set(execution.keys()):
            self._require_positive_int(
                execution[key], f"profile '{profile_name}'.execution.{key}"
            )

        strategies = block.get("strategies", {})
        if not isinstance(strategies, dict):
            raise ValueError(f"profile '{profile_name}'.strategies must be an object")
        for strategy_name, params in strategies.items():
            if not isinstance(strategy_name, str):
                raise ValueError(
                    f"profile '{profile_name}'.strategies keys must be strings"
                )
            if not isinstance(params, dict):
                raise ValueError(
                    f"profile '{profile_name}'.strategies.{strategy_name} must be an object"
                )
            self._validate_strategy_params(
                strategy_name,
                params,
                source=f"profile '{profile_name}'.strategies",
            )

        tournament = block.get("tournament", {})
        if not isinstance(tournament, dict):
            raise ValueError(f"profile '{profile_name}'.tournament must be an object")
        allowed_tournament = {"magic_gate_top_n", "contenders"}
        unknown_tournament = set(tournament.keys()) - allowed_tournament
        if unknown_tournament:
            raise ValueError(
                f"Unknown tournament keys in profile '{profile_name}': {sorted(unknown_tournament)}"
            )
        if "magic_gate_top_n" in tournament:
            self._require_positive_int(
                tournament["magic_gate_top_n"],
                f"profile '{profile_name}'.tournament.magic_gate_top_n",
            )
        if "contenders" in tournament:
            contenders = tournament["contenders"]
            if not isinstance(contenders, list):
                raise ValueError(
                    f"profile '{profile_name}'.tournament.contenders must be a list"
                )
            for idx, contender in enumerate(contenders):
                if not isinstance(contender, dict):
                    raise ValueError(
                        f"profile '{profile_name}'.tournament.contenders[{idx}] must be an object"
                    )
                allowed_contender = {"name", "strategies", "n"}
                unknown_contender = set(contender.keys()) - allowed_contender
                if unknown_contender:
                    raise ValueError(
                        f"Unknown keys in contender[{idx}] for profile '{profile_name}': {sorted(unknown_contender)}"
                    )
                for req_key in ("name", "strategies", "n"):
                    if req_key not in contender:
                        raise ValueError(
                            f"Missing key '{req_key}' in contender[{idx}] for profile '{profile_name}'"
                        )
                if not isinstance(contender["name"], str) or contender["name"].strip() == "":
                    raise ValueError(
                        f"profile '{profile_name}'.tournament.contenders[{idx}].name must be a non-empty string"
                    )
                if not isinstance(contender["strategies"], list) or len(contender["strategies"]) == 0:
                    raise ValueError(
                        f"profile '{profile_name}'.tournament.contenders[{idx}].strategies must be a non-empty list"
                    )
                for strategy_name in contender["strategies"]:
                    canonical = self._canonical_strategy_name(str(strategy_name))
                    if canonical not in STRATEGY_PARAM_KEYS:
                        raise ValueError(
                            f"Unknown strategy '{strategy_name}' in profile '{profile_name}'."
                        )
                self._require_positive_int(
                    contender["n"],
                    f"profile '{profile_name}'.tournament.contenders[{idx}].n",
                )

    def _validate_profile_schema(self, payload: Dict[str, Any], source: str) -> None:
        allowed_root = {"version", "profiles"}
        unknown_root = set(payload.keys()) - allowed_root
        if unknown_root:
            raise ValueError(f"Unknown root keys in {source}: {sorted(unknown_root)}")
        version = payload.get("version")
        if not isinstance(version, int):
            raise ValueError(f"Config {source} must include integer 'version'")
        if "profiles" not in payload:
            raise ValueError(f"Config {source} is missing required 'profiles' object")
        profiles = payload["profiles"]
        if not isinstance(profiles, dict) or not profiles:
            raise ValueError(f"Config {source} 'profiles' must be a non-empty object")
        for profile_name, block in profiles.items():
            if not isinstance(profile_name, str) or profile_name.strip() == "":
                raise ValueError(f"Config {source} profile names must be non-empty strings")
            if not isinstance(block, dict):
                raise ValueError(f"Profile '{profile_name}' in {source} must be an object")
            self._validate_profile_block(profile_name, block, source)

    def _merge_with_defaults(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        merged = deepcopy(base)
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = self._merge_with_defaults(merged[key], value)
            else:
                merged[key] = deepcopy(value)
        return merged

    def _normalize_profile_dict(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        normalized = deepcopy(profile)

        raw_strategies = normalized.get("strategies", {})
        strategy_out = deepcopy(DEFAULT_BACKTESTER_PROFILE["strategies"])
        for strategy_name, params in raw_strategies.items():
            canonical = self._canonical_strategy_name(strategy_name)
            merged_params = self._merge_with_defaults(
                strategy_out.get(canonical, {}),
                params,
            )
            strategy_out[canonical] = merged_params
        normalized["strategies"] = strategy_out

        contenders = normalized.get("tournament", {}).get("contenders", [])
        for contender in contenders:
            canonical_strategies = [
                self._canonical_strategy_name(name) for name in contender["strategies"]
            ]
            contender["strategies"] = canonical_strategies
        return normalized

    def _profile_from_dict(self, profile: Dict[str, Any]) -> BacktesterProfile:
        normalized = self._normalize_profile_dict(profile)
        universe_cfg = normalized["universe"]
        execution_cfg = normalized["execution"]
        tournament_cfg = normalized["tournament"]

        contenders = tuple(
            TournamentContender(
                name=str(contender["name"]),
                strategies=tuple(str(s) for s in contender["strategies"]),
                n=int(contender["n"]),
            )
            for contender in tournament_cfg.get("contenders", [])
        )

        tournament_magic_gate_top_n = int(
            tournament_cfg.get("magic_gate_top_n", execution_cfg["magic_gate_top_n"])
        )
        return BacktesterProfile(
            universe=UniverseProfile(
                strategy_min_market_cap_mm=float(
                    universe_cfg["strategy_min_market_cap_mm"]
                ),
                direct_filter_default_min_market_cap_mm=float(
                    universe_cfg["direct_filter_default_min_market_cap_mm"]
                ),
                revenue_proxy_multiplier=float(universe_cfg["revenue_proxy_multiplier"]),
                warrant_regex=str(universe_cfg["warrant_regex"]),
            ),
            execution=ExecutionProfile(
                default_top_n=int(execution_cfg["default_top_n"]),
                combo_top_n_per_strategy=int(execution_cfg["combo_top_n_per_strategy"]),
                magic_gate_top_n=int(execution_cfg["magic_gate_top_n"]),
            ),
            strategies=StrategyProfile(
                params={
                    str(name): deepcopy(params)
                    for name, params in normalized["strategies"].items()
                }
            ),
            tournament=TournamentProfile(
                magic_gate_top_n=tournament_magic_gate_top_n,
                contenders=contenders,
            ),
        )

    def _profile_to_dict(self, profile: BacktesterProfile) -> Dict[str, Any]:
        return {
            "universe": {
                "strategy_min_market_cap_mm": float(
                    profile.universe.strategy_min_market_cap_mm
                ),
                "direct_filter_default_min_market_cap_mm": float(
                    profile.universe.direct_filter_default_min_market_cap_mm
                ),
                "revenue_proxy_multiplier": float(profile.universe.revenue_proxy_multiplier),
                "warrant_regex": str(profile.universe.warrant_regex),
            },
            "execution": {
                "default_top_n": int(profile.execution.default_top_n),
                "combo_top_n_per_strategy": int(profile.execution.combo_top_n_per_strategy),
                "magic_gate_top_n": int(profile.execution.magic_gate_top_n),
            },
            "strategies": deepcopy(profile.strategies.params),
            "tournament": {
                "magic_gate_top_n": int(profile.tournament.magic_gate_top_n),
                "contenders": [
                    {
                        "name": contender.name,
                        "strategies": list(contender.strategies),
                        "n": contender.n,
                    }
                    for contender in profile.tournament.contenders
                ],
            },
        }

    def _resolve_profile(
        self,
        profile_name: Optional[str] = None,
        config_path: Optional[str] = None,
    ) -> BacktesterProfile:
        if profile_name is None and config_path is None:
            return self._active_profile

        selected_profile = (
            profile_name or self._active_profile_name or DEFAULT_BACKTESTER_PROFILE_NAME
        )
        if config_path is not None:
            selected_path = config_path
        elif profile_name is not None:
            selected_path = self._default_strategy_config_path
        else:
            selected_path = self._active_config_path or self._default_strategy_config_path
        selected_path = str(Path(selected_path))
        cache_key = (str(Path(selected_path).resolve()), str(selected_profile))
        if cache_key in self._loaded_profile_cache:
            return self._loaded_profile_cache[cache_key]

        payload = self._load_profile_file(selected_path)
        self._validate_profile_schema(payload, source=selected_path)
        profiles = cast(Dict[str, Any], payload["profiles"])
        if selected_profile not in profiles:
            raise ValueError(
                f"Profile '{selected_profile}' not found in {selected_path}. "
                f"Available profiles: {sorted(profiles.keys())}"
            )
        merged = self._merge_with_defaults(
            DEFAULT_BACKTESTER_PROFILE,
            cast(Dict[str, Any], profiles[selected_profile]),
        )
        resolved = self._profile_from_dict(merged)
        self._loaded_profile_cache[cache_key] = resolved
        return resolved

    def _strategy_params(self, profile: BacktesterProfile, strategy_name: str) -> Dict[str, Any]:
        canonical = self._canonical_strategy_name(strategy_name)
        return deepcopy(profile.strategies.params.get(canonical, {}))

    def load_data(self):
        """Load all required data from database."""
        print("Loading data from database...")
        ensure_indices()
        conn = get_connection()

        # Load price history (Long format)
        print("  ...loading price history")
        df_prices = pd.read_sql_query(
            """
            SELECT t.symbol, h.date, h.close
            FROM history h
            JOIN tickers t ON h.ticker_id = t.id
            ORDER BY h.date
        """,
            conn,
        )
        # Force conversion to UTC then strip timezone to ensure naive datetime64[ns]
        df_prices["date"] = pd.to_datetime(
            df_prices["date"], utc=True, errors="coerce"
        ).dt.tz_localize(None)

        print("  ...creating price matrix")
        self.price_matrix = df_prices.pivot(
            index="date", columns="symbol", values="close"
        )
        self.price_matrix = self.price_matrix.sort_index()
        # Keep missingness explicit. Filling across time can leak future prices into the past.

        print(
            f"      Loaded price history for {len(self.price_matrix.columns)} symbols"
        )

        # Close main connection before starting threads (threads open their own)
        conn.close()

        # Load financial tables in parallel
        tables = ["income_statement", "balance_sheet", "cash_flow", "financials"]
        metrics_str = "', '".join(REQUIRED_METRICS)

        dfs = []
        print("  ...loading financials")
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for table in tables:
                sql = f"""
                    SELECT t.symbol, f.metric, f.period, f.value
                    FROM {table} f
                    JOIN tickers t ON f.ticker_id = t.id
                    WHERE f.metric IN ('{metrics_str}')
                """
                futures.append(executor.submit(query_database, sql))

            for future in as_completed(futures):
                try:
                    df = future.result()
                    if not df.empty:
                        df["period"] = pd.to_datetime(
                            df["period"], utc=True, errors="coerce"
                        ).dt.tz_localize(None)
                        # Pivot immediately: index=(symbol, period), columns=metric
                        pivoted = df.pivot_table(
                            index=["symbol", "period"],
                            columns="metric",
                            values="value",
                            aggfunc="last",
                        )
                        dfs.append(pivoted)
                except Exception as e:
                    print(f"Error loading table: {e}")

        if dfs:
            print("  ...merging financial data")
            # Merge all pivoted DFs.
            self.financial_data = pd.concat(dfs, axis=1)
            # Deduplicate columns (if same metric in multiple tables)
            # Pandas 3 removed DataFrame.groupby(axis=1); transpose to group columns.
            self.financial_data = self.financial_data.T.groupby(level=0).last().T
            self.financial_data = self.financial_data.sort_index()
        else:
            self.financial_data = pd.DataFrame()

        # Invalidate derived caches after reload.
        self._regime_cache.clear()
        self._tail_state_cache.clear()
        self._momentum_cache.clear()
        self._snapshot_cache.clear()

        print("Data loading complete.\n")

    def get_valid_snapshot(
        self, trade_date: str, lag_days: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get financial data snapshot as it would have been visible on the trade date.

        Parameters:
        -----------
        trade_date : str
            The date we're "trading" on (e.g., '2023-01-01')
        lag_days : int, optional
            Override default reporting lag

        Returns:
        --------
        pd.DataFrame
            Wide-format DataFrame with symbol as index, metrics as columns
        """
        if lag_days is None:
            lag_days = self.reporting_lag

        cache_key = (str(trade_date), int(lag_days))
        if cache_key in self._snapshot_cache:
            return self._snapshot_cache[cache_key].copy()

        trade_dt = pd.to_datetime(trade_date)
        cutoff_dt = trade_dt - timedelta(days=lag_days)

        if self.financial_data is None or self.financial_data.empty:
            return pd.DataFrame()

        # Filter by period <= cutoff
        # self.financial_data index is (symbol, period)
        valid_mask = self.financial_data.index.get_level_values("period") <= cutoff_dt
        valid_data = self.financial_data[valid_mask]

        if valid_data.empty:
            print(f"WARNING: No financial data available before {cutoff_dt}")
            return pd.DataFrame()

        # Get latest for each symbol
        # Group by symbol and take last (since we sorted by index (symbol, period))
        snapshot = valid_data.groupby(level="symbol").last()
        self._snapshot_cache[cache_key] = snapshot

        return snapshot.copy()

    def enrich_with_yoy_data(
        self, current_df: pd.DataFrame, trade_date: str
    ) -> pd.DataFrame:
        """
        Fetches data from 1 year ago and merges it with the current snapshot
        to allow Year-Over-Year (YoY) calculations.
        """
        # 1. Calculate the date 1 year ago
        curr_dt = pd.to_datetime(trade_date)
        prev_dt = curr_dt - pd.Timedelta(days=365)
        prev_date_str = prev_dt.strftime("%Y-%m-%d")

        # 2. Fetch the snapshot for that previous date
        # We reuse your existing get_valid_snapshot method
        print(f"    (Data) Fetching YoY comparison data from {prev_date_str}...")
        try:
            prev_df = self.get_valid_snapshot(prev_date_str)
        except Exception:
            print(
                "    (Warning) Could not fetch previous year data. YoY metrics will be NaN."
            )
            prev_df = pd.DataFrame()

        # 3. If no previous data, create empty _Prev columns so strategies don't crash
        if prev_df.empty:
            print(
                "    (Warning) No previous year data available. Adding empty _Prev columns."
            )
            for col in current_df.columns:
                current_df[f"{col}_Prev"] = np.nan
            return current_df

        # 4. Rename columns to '_Prev' (e.g., 'TotalRevenue' -> 'TotalRevenue_Prev')
        prev_df = prev_df.add_suffix("_Prev")

        # 5. Merge with current data
        # Left join ensures we keep our current universe of stocks
        combined_df = current_df.join(prev_df, how="left")

        return combined_df

    def _get_asof_price(
        self,
        symbol: str,
        target_dt: pd.Timestamp,
        staleness_days: int,
    ) -> float:
        """Get the last known price on/before target date with a staleness cap."""
        if self.price_matrix is None or symbol not in self.price_matrix.columns:
            return np.nan

        series = self.price_matrix[symbol].dropna()
        if series.empty:
            return np.nan

        # No look-ahead: only allow observations known at or before the target timestamp.
        hist = series[series.index <= target_dt]
        if hist.empty:
            return np.nan

        asof_dt = pd.Timestamp(hist.index[-1])
        assert asof_dt <= target_dt, "as-of pricing must never use future timestamps"
        age_days = (target_dt.normalize() - asof_dt.normalize()).days
        if age_days > int(staleness_days):
            return np.nan

        px = hist.iloc[-1]
        return float(px) if pd.notna(px) else np.nan

    @staticmethod
    def _cash_fill_monthly_return(monthly_rf: float, cash_fill_cost_bps: float) -> float:
        rf = float(monthly_rf) if pd.notna(monthly_rf) else 0.0
        drag = float(cash_fill_cost_bps) / 10000.0
        return rf - drag

    def get_price(self, symbol: str, date: str, tolerance_days: int = 7) -> float:
        """Get closing price using strict as-of semantics (never future prices)."""
        if self.price_matrix is None:
            return np.nan

        target_dt = pd.to_datetime(date)
        if hasattr(target_dt, "tzinfo") and target_dt.tzinfo is not None:
            target_dt = target_dt.tz_localize(None)

        staleness_days = max(0, int(tolerance_days))
        return self._get_asof_price(symbol, target_dt, staleness_days)

    def get_prices_bulk(self, symbols: list, date: str) -> pd.Series:
        """Get prices for multiple symbols using strict as-of semantics."""
        if self.price_matrix is None:
            return pd.Series(np.nan, index=symbols, dtype=float)

        target_dt = pd.to_datetime(date)
        if hasattr(target_dt, "tzinfo") and target_dt.tzinfo is not None:
            target_dt = target_dt.tz_localize(None)
        staleness_days = int(self.price_staleness_days)
        values = [
            self._get_asof_price(str(symbol), target_dt, staleness_days)
            for symbol in symbols
        ]
        return pd.Series(values, index=symbols, dtype=float)

    def filter_universe(
        self,
        df: pd.DataFrame,
        trade_date: str,
        min_mkt_cap_mm: Optional[float] = None,
        profile_name: Optional[str] = None,
        config_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Filters out warrants, illiquid stocks, and micro-caps.

        Parameters:
        -----------
        df : pd.DataFrame
            The financial snapshot
        trade_date : str
            Date to check prices for Market Cap calculation
        min_mkt_cap_mm : int
            Minimum Market Cap in Millions (defaults to active profile config)
        """
        profile = self._resolve_profile(profile_name=profile_name, config_path=config_path)
        return self._filter_universe_with_profile(
            df=df,
            trade_date=trade_date,
            min_mkt_cap_mm=min_mkt_cap_mm,
            profile=profile,
        )

    def _filter_universe_with_profile(
        self,
        df: pd.DataFrame,
        trade_date: str,
        min_mkt_cap_mm: Optional[float],
        profile: BacktesterProfile,
    ) -> pd.DataFrame:
        effective_min_mkt_cap_mm = (
            float(min_mkt_cap_mm)
            if min_mkt_cap_mm is not None
            else float(profile.universe.direct_filter_default_min_market_cap_mm)
        )

        mask_warrants = (
            pd.Index(df.index)
            .astype(str)
            .str.match(profile.universe.warrant_regex, na=False)
        )
        df = df[~mask_warrants]

        if "OrdinarySharesNumber" in df.columns:
            df = df.dropna(subset=["OrdinarySharesNumber"])
            current_prices = self.get_prices_bulk(df.index.tolist(), trade_date)
            df = df.copy()
            df["_CurrentPrice"] = current_prices
            df["MarketCap"] = (
                df["OrdinarySharesNumber"] * df["_CurrentPrice"]
            ) / 1_000_000

            original_count = len(df)
            df = df[df["MarketCap"] > effective_min_mkt_cap_mm]
            print(
                f"    (Universe Filter) Dropped {original_count - len(df)} stocks < "
                f"${effective_min_mkt_cap_mm}M Market Cap"
            )

        elif "TotalRevenue" in df.columns:
            print(
                "    (Universe Filter) 'OrdinarySharesNumber' missing. Using Revenue proxy."
            )
            df = df[
                df["TotalRevenue"]
                > (
                    effective_min_mkt_cap_mm
                    * 1_000_000
                    * float(profile.universe.revenue_proxy_multiplier)
                )
            ]

        return df

    def _apply_strategy_legacy(
        self,
        snapshot: pd.DataFrame,
        strategy: str,
        top_n: int = 10,
        trade_date: Optional[str] = None,
    ) -> list:
        """
        Apply a stock selection strategy to the snapshot.

        Parameters:
        -----------
        snapshot : pd.DataFrame
            Point-in-time financial data
        strategy : str
            Strategy name: 'magic_formula', 'moat', 'fcf_yield', 'quality'
        top_n : int
            Number of stocks to select
        trade_date : str
            Date for universe filtering

        Returns:
        --------
        list
            Selected stock symbols
        """
        # --- NEW: Apply Universe Filter First ---
        if trade_date:
            snapshot = self.filter_universe(snapshot, trade_date, min_mkt_cap_mm=250)
        # ----------------------------------------

        df = snapshot.copy()

        # Ensure MarketCap exists for tie-breaking (fill with 0 if missing)
        if "MarketCap" not in df.columns:
            df["MarketCap"] = 0

        if strategy == "magic_formula":
            # Greenblatt's Magic Formula: High ROIC + High Earnings Yield

            # Calculate EnterpriseValue if missing (EV = Market Cap + Debt - Cash)
            if "EnterpriseValue" not in df.columns:
                ev_cols = ["TotalDebt", "CashAndCashEquivalents", "MarketCap"]
                if all(c in df.columns for c in ev_cols):
                    # Ensure MarketCap is valid for calculation
                    df = df[df["MarketCap"] > 0]
                    # MarketCap is in Millions, convert to raw
                    df["EnterpriseValue"] = (
                        (df["MarketCap"] * 1_000_000)
                        + df["TotalDebt"]
                        - df["CashAndCashEquivalents"]
                    )

            required = ["EBIT", "InvestedCapital", "EnterpriseValue"]
            if not all(col in df.columns for col in required):
                print(
                    f"  Missing columns for magic_formula. Have: {df.columns.tolist()}"
                )
                return []

            df = df.dropna(subset=required)
            df = df[df["InvestedCapital"] > 0]
            df = df[df["EnterpriseValue"] > 0]

            # 1. Calculate the Ratios
            df["ROIC"] = df["EBIT"] / df["InvestedCapital"]
            df["EarningsYield"] = df["EBIT"] / df["EnterpriseValue"]

            # 2. Rank both factors (higher is better)
            df["Quality_Rank"] = robust_score(df["ROIC"], higher_is_better=True)
            df["Value_Rank"] = robust_score(df["EarningsYield"], higher_is_better=True)

            # 3. Combine Ranks
            df["Magic_Score"] = df["Quality_Rank"] + df["Value_Rank"]

            # 4. Pick top N, using MarketCap as a tie-breaker to avoid alphabetical bias
            top_picks = df.nlargest(top_n, ["Magic_Score", "MarketCap"]).index.tolist()

        elif strategy == "moat":
            # High Operating Margin + R&D Intensity
            required = ["OperatingIncome", "TotalRevenue"]
            if not all(col in df.columns for col in required):
                print(f"  Missing columns for moat. Have: {df.columns.tolist()}")
                return []

            df = df.dropna(subset=required)
            df = df[df["TotalRevenue"] > 100_000_000]  # > $100M revenue

            df["Operating_Margin"] = df["OperatingIncome"] / df["TotalRevenue"]

            # R&D is optional
            if "ResearchAndDevelopment" in df.columns:
                df["RD_Intensity"] = (
                    df["ResearchAndDevelopment"].fillna(0) / df["TotalRevenue"]
                )
                df["RD_Score"] = robust_score(df["RD_Intensity"], higher_is_better=True)
            else:
                df["RD_Score"] = 50  # Neutral

            df["Margin_Score"] = robust_score(
                df["Operating_Margin"], higher_is_better=True
            )
            df["Moat_Score"] = (df["Margin_Score"] + df["RD_Score"]) / 2

            top_picks = df.nlargest(top_n, ["Moat_Score", "MarketCap"]).index.tolist()

        elif strategy == "fcf_yield":
            # Free Cash Flow Yield
            required = ["OperatingCashFlow", "CapitalExpenditure"]
            if not all(col in df.columns for col in required):
                print(f"  Missing columns for fcf_yield. Have: {df.columns.tolist()}")
                return []

            df = df.dropna(subset=required)
            df["FreeCashFlow"] = (
                df["OperatingCashFlow"] - df["CapitalExpenditure"].abs()
            )
            df = df[df["FreeCashFlow"] > 0]  # Positive FCF only

            df["FCF_Score"] = robust_score(df["FreeCashFlow"], higher_is_better=True)
            top_picks = df.nlargest(top_n, ["FCF_Score", "MarketCap"]).index.tolist()

        elif strategy == "quality":
            # Simple quality: High ROIC but stricter
            required = ["EBIT", "InvestedCapital"]
            if not all(col in df.columns for col in required):
                return []

            df = df.dropna(subset=required)
            df = df[df["InvestedCapital"] > 0]
            df["ROIC"] = df["EBIT"] / df["InvestedCapital"]

            # Filter for realistic ROIC (avoid data errors > 100%)
            df = df[(df["ROIC"] > 0.05) & (df["ROIC"] < 1.0)]

            df["Quality_Score"] = robust_score(df["ROIC"], higher_is_better=True)
            top_picks = df.nlargest(
                top_n, ["Quality_Score", "MarketCap"]
            ).index.tolist()

        elif strategy in ("super_margins", "diamonds_in_dirt"):
            # -----------------------------------------------------------
            # Strategy: "Diamond in the Dirt"
            # Find high-quality companies (great margins) at cheap prices
            # -----------------------------------------------------------

            required = [
                "NetIncome",
                "StockholdersEquity",
                "TotalRevenue",
                "GrossProfit",
                "OperatingIncome",
                "TotalDebt",
            ]

            # Check availability
            if not all(col in df.columns for col in required):
                print(
                    f"    [Warning] Missing data for strategy: {[c for c in required if c not in df.columns]}"
                )
                return []

            # Clean Zeros
            df = df[(df["TotalRevenue"] > 0) & (df["StockholdersEquity"] > 0)].copy()

            # A. Calculate Metrics
            df["ROE"] = df["NetIncome"] / df["StockholdersEquity"]
            df["Gross_Margin"] = df["GrossProfit"] / df["TotalRevenue"]
            df["Op_Margin"] = df["OperatingIncome"] / df["TotalRevenue"]
            df["Debt_to_Rev"] = df["TotalDebt"] / df["TotalRevenue"]

            # B. The "Diamond" Screen (High Quality, but REALISTIC)
            #    - Gross Margin > 40% (Keep this, it's good for Moats)
            #    - Op Margin > 15% (Lowered from 40% to catch real businesses like AAPL/MSFT)
            #    - ROE > 15% (Increased to ensure efficiency)
            #    - Debt < 3x Revenue (Safety)
            mask_quality = (
                (df["Gross_Margin"] > 0.40)
                & (df["Op_Margin"] > 0.15)  # <--- CRITICAL FIX (Was 0.40)
                & (df["ROE"] > 0.15)
                & (df["Debt_to_Rev"] < 3.0)
            )
            passed = df[mask_quality].copy()

            # Debugging: See how many survived
            if len(passed) < top_n and len(passed) > 0:
                print(
                    f"    (Note) Only {len(passed)} companies met the 'Diamond' criteria."
                )
            elif len(passed) == 0:
                print("    (Note) No companies met the 'Diamond' criteria.")
                return []

            # C. The "Dirt" Sort (Find the CHEAPEST of the Great Companies)
            #    If we just pick High ROE, we buy expensive stocks.
            #    We must sort by EARNINGS YIELD (Operating Income / Market Cap).
            if "MarketCap" in passed.columns and (passed["MarketCap"] > 0).any():
                # Higher Yield = Cheaper Stock
                passed["Earnings_Yield"] = passed["OperatingIncome"] / (
                    passed["MarketCap"] * 1_000_000
                )
                # Filter out negative yields (loss makers that slipped through)
                passed = passed[passed["Earnings_Yield"] > 0]
                top_picks = passed.nlargest(top_n, "Earnings_Yield").index.tolist()
            else:
                # Fallback if no Market Cap data (Rank by ROE)
                top_picks = passed.nlargest(top_n, "ROE").index.tolist()

        # ==============================================================================
        # STRATEGY: THE CANNIBAL (Aggressive Buybacks)
        # Logic: Buy companies reducing share count by >3% AND paying with Free Cash Flow
        # ==============================================================================
        elif strategy == "cannibal":
            # 1. Get YoY Data (Need 'OrdinarySharesNumber_Prev')
            if trade_date is None:
                print("  trade_date is required for cannibal strategy.")
                return []
            df = self.enrich_with_yoy_data(df, trade_date)

            # Calculate FreeCashFlow if not present
            if "FreeCashFlow" not in df.columns:
                if (
                    "OperatingCashFlow" in df.columns
                    and "CapitalExpenditure" in df.columns
                ):
                    df["FreeCashFlow"] = (
                        df["OperatingCashFlow"] - df["CapitalExpenditure"].abs()
                    )
                else:
                    print("  Missing columns for FreeCashFlow calculation.")
                    return []

            # 2. Required Columns - check if YoY data is available
            req = [
                "OrdinarySharesNumber",
                "OrdinarySharesNumber_Prev",
                "FreeCashFlow",
                "MarketCap",
            ]
            if not all(c in df.columns for c in req):
                print(f"  Missing columns for cannibal. Have: {df.columns.tolist()}")
                return []

            # Drop rows where we don't have YoY comparison data
            df = df.dropna(subset=["OrdinarySharesNumber", "OrdinarySharesNumber_Prev"])
            if df.empty:
                print("  No stocks with valid YoY share data for cannibal strategy.")
                return []

            # 3. Calculate Metrics
            # Share Change: (Current - Prev) / Prev. Negative number = Buyback.
            df["Share_Change_Pct"] = (
                df["OrdinarySharesNumber"] - df["OrdinarySharesNumber_Prev"]
            ) / df["OrdinarySharesNumber_Prev"]

            # Buyback Yield: RepurchaseOfCapitalStock (usually negative in CF) / MarketCap
            if "RepurchaseOfCapitalStock" in df.columns:
                df["Buyback_Yield"] = df["RepurchaseOfCapitalStock"].abs() / (
                    df["MarketCap"] * 1_000_000
                )
            else:
                df["Buyback_Yield"] = 0

            # 4. The "Cannibal" Screen
            mask = (
                (df["Share_Change_Pct"] < -0.03)  # Shares reduced by at least 3%
                & (df["FreeCashFlow"] > 0)  # Must be FCF positive
                & (df["MarketCap"] > 200)  # Liquidity Filter ($200M+)
            )
            passed = df[mask].copy()

            # 5. Rank by "Buyback Yield" (Who is eating themselves the fastest?)
            # We want the highest yield (biggest buyback relative to price)
            top_picks = passed.nlargest(top_n, "Buyback_Yield").index.tolist()

        # ==============================================================================
        # STRATEGY: PIOTROSKI F-SCORE (Fundamental Health)
        # Logic: 0-9 Point Score based on Profitability, Leverage, and Efficiency
        # ==============================================================================
        elif strategy == "piotroski_f_score":
            # 1. Get YoY Data for comparisons
            if trade_date is None:
                print("  trade_date is required for piotroski_f_score strategy.")
                return []
            df = self.enrich_with_yoy_data(df, trade_date)

            # Check if we have any valid YoY data
            prev_cols = [c for c in df.columns if c.endswith("_Prev")]
            if not prev_cols or df[prev_cols].isna().all().all():
                print(
                    "  No YoY data available for piotroski_f_score. Skipping YoY criteria."
                )

            # 2. Define Score Components (Start at 0)
            df["F_Score"] = 0

            # --- Profitability (4 Points) ---
            # 1. Positive Net Income
            if "NetIncome" in df.columns:
                df.loc[df["NetIncome"] > 0, "F_Score"] += 1
            # 2. Positive Operating Cash Flow
            if "OperatingCashFlow" in df.columns:
                df.loc[df["OperatingCashFlow"] > 0, "F_Score"] += 1
            # 3. ROA Increasing (Current ROA > Prev ROA)
            # Need Average Assets for strict ROA, but simplified: NetIncome/TotalAssets
            if all(
                c in df.columns
                for c in [
                    "NetIncome",
                    "TotalAssets",
                    "NetIncome_Prev",
                    "TotalAssets_Prev",
                ]
            ):
                df["ROA"] = df["NetIncome"] / df["TotalAssets"]
                df["ROA_Prev"] = df["NetIncome_Prev"] / df["TotalAssets_Prev"]
                df.loc[df["ROA"] > df["ROA_Prev"], "F_Score"] += 1
            # 4. Quality of Earnings (Cash Flow > Net Income)
            if all(c in df.columns for c in ["OperatingCashFlow", "NetIncome"]):
                df.loc[df["OperatingCashFlow"] > df["NetIncome"], "F_Score"] += 1

            # --- Leverage & Liquidity (3 Points) ---
            # 5. Long Term Debt Decreasing (or stable)
            if all(c in df.columns for c in ["TotalDebt", "TotalDebt_Prev"]):
                df.loc[df["TotalDebt"] <= df["TotalDebt_Prev"], "F_Score"] += 1
            # 6. Current Ratio Increasing (Liquidity improving)
            if all(
                c in df.columns
                for c in [
                    "CurrentAssets",
                    "CurrentLiabilities",
                    "CurrentAssets_Prev",
                    "CurrentLiabilities_Prev",
                ]
            ):
                df["Current_Ratio"] = df["CurrentAssets"] / df["CurrentLiabilities"]
                df["Current_Ratio_Prev"] = (
                    df["CurrentAssets_Prev"] / df["CurrentLiabilities_Prev"]
                )
                df.loc[df["Current_Ratio"] > df["Current_Ratio_Prev"], "F_Score"] += 1
            # 7. No Dilution (Shares <= Prev Shares)
            if all(
                c in df.columns
                for c in ["OrdinarySharesNumber", "OrdinarySharesNumber_Prev"]
            ):
                df.loc[
                    df["OrdinarySharesNumber"] <= df["OrdinarySharesNumber_Prev"],
                    "F_Score",
                ] += 1

            # --- Operating Efficiency (2 Points) ---
            # 8. Gross Margin Increasing
            if all(
                c in df.columns
                for c in [
                    "GrossProfit",
                    "TotalRevenue",
                    "GrossProfit_Prev",
                    "TotalRevenue_Prev",
                ]
            ):
                df["Gross_Margin"] = df["GrossProfit"] / df["TotalRevenue"]
                df["Gross_Margin_Prev"] = (
                    df["GrossProfit_Prev"] / df["TotalRevenue_Prev"]
                )
                df.loc[df["Gross_Margin"] > df["Gross_Margin_Prev"], "F_Score"] += 1
            # 9. Asset Turnover Increasing (Revenue / Assets)
            if all(
                c in df.columns
                for c in [
                    "TotalRevenue",
                    "TotalAssets",
                    "TotalRevenue_Prev",
                    "TotalAssets_Prev",
                ]
            ):
                df["Asset_Turnover"] = df["TotalRevenue"] / df["TotalAssets"]
                df["Asset_Turnover_Prev"] = (
                    df["TotalRevenue_Prev"] / df["TotalAssets_Prev"]
                )
                df.loc[df["Asset_Turnover"] > df["Asset_Turnover_Prev"], "F_Score"] += 1

            # 3. Filter: Only Elite Stocks (Score 7, 8, or 9)
            passed = df[df["F_Score"] >= 7].copy()

            # 4. Rank: Break ties with Cheapest Valuation (Earnings Yield)
            if "MarketCap" in passed.columns and "EBIT" in passed.columns:
                passed["Earnings_Yield"] = passed["EBIT"] / (
                    passed["MarketCap"] * 1_000_000
                )
                top_picks = passed.nlargest(top_n, "Earnings_Yield").index.tolist()
            else:
                top_picks = passed.nlargest(top_n, "F_Score").index.tolist()

        # ==============================================================================
        # STRATEGY: RULE OF 40 (SaaS / High Growth)
        # Logic: Revenue Growth % + FCF Margin % > 40
        # ==============================================================================
        elif strategy == "rule_of_40":
            # 1. Get YoY Data
            if trade_date is None:
                print("  trade_date is required for rule_of_40 strategy.")
                return []
            df = self.enrich_with_yoy_data(df, trade_date)

            # Calculate FreeCashFlow if not present
            if "FreeCashFlow" not in df.columns:
                if (
                    "OperatingCashFlow" in df.columns
                    and "CapitalExpenditure" in df.columns
                ):
                    df["FreeCashFlow"] = (
                        df["OperatingCashFlow"] - df["CapitalExpenditure"].abs()
                    )
                else:
                    print("  Missing columns for FreeCashFlow calculation.")
                    return []

            # 2. Calculate Growth & Margin
            # Rev Growth = (Current - Prev) / Prev
            if not all(c in df.columns for c in ["TotalRevenue", "TotalRevenue_Prev"]):
                print("  Missing TotalRevenue columns for rule_of_40.")
                return []

            # Drop rows without YoY revenue data
            df = df.dropna(subset=["TotalRevenue", "TotalRevenue_Prev"])
            if df.empty:
                print(
                    "  No stocks with valid YoY revenue data for rule_of_40 strategy."
                )
                return []

            df["Rev_Growth"] = (df["TotalRevenue"] - df["TotalRevenue_Prev"]) / df[
                "TotalRevenue_Prev"
            ]

            # FCF Margin = FCF / TotalRevenue
            df["FCF_Margin"] = df["FreeCashFlow"] / df["TotalRevenue"]

            # 3. The "Rule of 40" Score
            # Multiply by 100 to get the integer (e.g., 0.20 + 0.20 = 40)
            df["R40_Score"] = (df["Rev_Growth"] + df["FCF_Margin"]) * 100

            # 4. Filter
            # Standard rule is > 40. We can be stricter (> 50) for top picks.
            mask = (df["R40_Score"] > 40) & (
                df["TotalRevenue"] > 50_000_000
            )  # Minimum $50M Revenue to avoid tiny startups
            passed = df[mask].copy()

            # 5. Rank by Score (Highest combo of Growth + Cash Flow)
            top_picks = passed.nlargest(top_n, "R40_Score").index.tolist()

        else:
            # Fallback for existing strategies
            # Ensure we return at least something if specific logic isn't changed
            return snapshot.head(top_n).index.tolist()

        return top_picks

    # Profile-driven strategy dispatch (legacy branch logic remains in _apply_strategy_legacy)
    def apply_strategy(
        self,
        snapshot: pd.DataFrame,
        strategy: str,
        top_n: int = 10,
        trade_date: Optional[str] = None,
        profile_name: Optional[str] = None,
        config_path: Optional[str] = None,
    ) -> list:
        profile = self._resolve_profile(profile_name=profile_name, config_path=config_path)
        effective_top_n = int(top_n) if int(top_n) > 0 else int(profile.execution.default_top_n)
        return self._apply_strategy_with_profile(
            snapshot=snapshot,
            strategy=strategy,
            top_n=effective_top_n,
            trade_date=trade_date,
            profile=profile,
        )

    def _apply_strategy_with_profile(
        self,
        snapshot: pd.DataFrame,
        strategy: str,
        top_n: int,
        trade_date: Optional[str],
        profile: BacktesterProfile,
    ) -> List[str]:
        working = snapshot
        if trade_date:
            working = self._filter_universe_with_profile(
                df=snapshot,
                trade_date=trade_date,
                min_mkt_cap_mm=float(profile.universe.strategy_min_market_cap_mm),
                profile=profile,
            )

        canonical_strategy = self._canonical_strategy_name(strategy)
        handler = self._strategy_registry.get(canonical_strategy)
        if handler is None:
            return working.head(top_n).index.tolist()

        df = working.copy()
        if "MarketCap" not in df.columns:
            df["MarketCap"] = 0
        params = self._strategy_params(profile, canonical_strategy)
        return handler(df, top_n, trade_date, params)

    def _strategy_magic_formula(
        self, df: pd.DataFrame, top_n: int, _trade_date: Optional[str], _params: Dict[str, Any]
    ) -> List[str]:
        if "EnterpriseValue" not in df.columns:
            ev_cols = ["TotalDebt", "CashAndCashEquivalents", "MarketCap"]
            if all(c in df.columns for c in ev_cols):
                df = df[df["MarketCap"] > 0]
                df["EnterpriseValue"] = (
                    (df["MarketCap"] * 1_000_000)
                    + df["TotalDebt"]
                    - df["CashAndCashEquivalents"]
                )

        required = ["EBIT", "InvestedCapital", "EnterpriseValue"]
        if not all(col in df.columns for col in required):
            print(f"  Missing columns for magic_formula. Have: {df.columns.tolist()}")
            return []

        df = df.dropna(subset=required)
        df = df[df["InvestedCapital"] > 0]
        df = df[df["EnterpriseValue"] > 0]
        df["ROIC"] = df["EBIT"] / df["InvestedCapital"]
        df["EarningsYield"] = df["EBIT"] / df["EnterpriseValue"]
        df["Quality_Rank"] = robust_score(df["ROIC"], higher_is_better=True)
        df["Value_Rank"] = robust_score(df["EarningsYield"], higher_is_better=True)
        df["Magic_Score"] = df["Quality_Rank"] + df["Value_Rank"]
        return df.nlargest(top_n, ["Magic_Score", "MarketCap"]).index.tolist()

    def _strategy_moat(
        self, df: pd.DataFrame, top_n: int, _trade_date: Optional[str], params: Dict[str, Any]
    ) -> List[str]:
        required = ["OperatingIncome", "TotalRevenue"]
        if not all(col in df.columns for col in required):
            print(f"  Missing columns for moat. Have: {df.columns.tolist()}")
            return []

        min_total_revenue = float(params.get("min_total_revenue", 100_000_000))
        df = df.dropna(subset=required)
        df = df[df["TotalRevenue"] > min_total_revenue]
        df["Operating_Margin"] = df["OperatingIncome"] / df["TotalRevenue"]

        if "ResearchAndDevelopment" in df.columns:
            df["RD_Intensity"] = df["ResearchAndDevelopment"].fillna(0) / df["TotalRevenue"]
            df["RD_Score"] = robust_score(df["RD_Intensity"], higher_is_better=True)
        else:
            df["RD_Score"] = 50

        df["Margin_Score"] = robust_score(df["Operating_Margin"], higher_is_better=True)
        df["Moat_Score"] = (df["Margin_Score"] + df["RD_Score"]) / 2
        return df.nlargest(top_n, ["Moat_Score", "MarketCap"]).index.tolist()

    def _strategy_fcf_yield(
        self, df: pd.DataFrame, top_n: int, _trade_date: Optional[str], params: Dict[str, Any]
    ) -> List[str]:
        required = ["OperatingCashFlow", "CapitalExpenditure"]
        if not all(col in df.columns for col in required):
            print(f"  Missing columns for fcf_yield. Have: {df.columns.tolist()}")
            return []

        require_positive_fcf = bool(params.get("require_positive_fcf", True))
        df = df.dropna(subset=required)
        df["FreeCashFlow"] = df["OperatingCashFlow"] - df["CapitalExpenditure"].abs()
        if require_positive_fcf:
            df = df[df["FreeCashFlow"] > 0]
        df["FCF_Score"] = robust_score(df["FreeCashFlow"], higher_is_better=True)
        return df.nlargest(top_n, ["FCF_Score", "MarketCap"]).index.tolist()

    def _strategy_quality(
        self, df: pd.DataFrame, top_n: int, _trade_date: Optional[str], params: Dict[str, Any]
    ) -> List[str]:
        required = ["EBIT", "InvestedCapital"]
        if not all(col in df.columns for col in required):
            return []

        min_roic = float(params.get("min_roic", 0.05))
        max_roic = float(params.get("max_roic", 1.0))
        df = df.dropna(subset=required)
        df = df[df["InvestedCapital"] > 0]
        df["ROIC"] = df["EBIT"] / df["InvestedCapital"]
        df = df[(df["ROIC"] > min_roic) & (df["ROIC"] < max_roic)]
        df["Quality_Score"] = robust_score(df["ROIC"], higher_is_better=True)
        return df.nlargest(top_n, ["Quality_Score", "MarketCap"]).index.tolist()

    def _strategy_diamonds_in_dirt(
        self, df: pd.DataFrame, top_n: int, _trade_date: Optional[str], params: Dict[str, Any]
    ) -> List[str]:
        required = [
            "NetIncome",
            "StockholdersEquity",
            "TotalRevenue",
            "GrossProfit",
            "OperatingIncome",
            "TotalDebt",
        ]
        if not all(col in df.columns for col in required):
            print(
                "    [Warning] Missing data for strategy: "
                f"{[c for c in required if c not in df.columns]}"
            )
            return []

        gross_margin_min = float(params.get("gross_margin_min", 0.40))
        op_margin_min = float(params.get("op_margin_min", 0.15))
        roe_min = float(params.get("roe_min", 0.15))
        debt_to_revenue_max = float(params.get("debt_to_revenue_max", 3.0))
        require_positive_earnings_yield = bool(
            params.get("require_positive_earnings_yield", True)
        )

        df = df[(df["TotalRevenue"] > 0) & (df["StockholdersEquity"] > 0)].copy()
        df["ROE"] = df["NetIncome"] / df["StockholdersEquity"]
        df["Gross_Margin"] = df["GrossProfit"] / df["TotalRevenue"]
        df["Op_Margin"] = df["OperatingIncome"] / df["TotalRevenue"]
        df["Debt_to_Rev"] = df["TotalDebt"] / df["TotalRevenue"]

        mask_quality = (
            (df["Gross_Margin"] > gross_margin_min)
            & (df["Op_Margin"] > op_margin_min)
            & (df["ROE"] > roe_min)
            & (df["Debt_to_Rev"] < debt_to_revenue_max)
        )
        passed = df[mask_quality].copy()

        if len(passed) < top_n and len(passed) > 0:
            print(f"    (Note) Only {len(passed)} companies met the 'Diamond' criteria.")
        elif len(passed) == 0:
            print("    (Note) No companies met the 'Diamond' criteria.")
            return []

        if "MarketCap" in passed.columns and (passed["MarketCap"] > 0).any():
            passed["Earnings_Yield"] = passed["OperatingIncome"] / (
                passed["MarketCap"] * 1_000_000
            )
            if require_positive_earnings_yield:
                passed = passed[passed["Earnings_Yield"] > 0]
            return passed.nlargest(top_n, "Earnings_Yield").index.tolist()
        return passed.nlargest(top_n, "ROE").index.tolist()

    def _strategy_cannibal(
        self, df: pd.DataFrame, top_n: int, trade_date: Optional[str], params: Dict[str, Any]
    ) -> List[str]:
        if trade_date is None:
            print("  trade_date is required for cannibal strategy.")
            return []
        df = self.enrich_with_yoy_data(df, trade_date)

        if "FreeCashFlow" not in df.columns:
            if "OperatingCashFlow" in df.columns and "CapitalExpenditure" in df.columns:
                df["FreeCashFlow"] = df["OperatingCashFlow"] - df["CapitalExpenditure"].abs()
            else:
                print("  Missing columns for FreeCashFlow calculation.")
                return []

        req = [
            "OrdinarySharesNumber",
            "OrdinarySharesNumber_Prev",
            "FreeCashFlow",
            "MarketCap",
        ]
        if not all(c in df.columns for c in req):
            print(f"  Missing columns for cannibal. Have: {df.columns.tolist()}")
            return []

        df = df.dropna(subset=["OrdinarySharesNumber", "OrdinarySharesNumber_Prev"])
        if df.empty:
            print("  No stocks with valid YoY share data for cannibal strategy.")
            return []

        min_buyback_pct = float(params.get("min_buyback_pct", 0.03))
        require_positive_fcf = bool(params.get("require_positive_fcf", True))
        min_market_cap_mm = float(params.get("min_market_cap_mm", 200.0))

        df["Share_Change_Pct"] = (
            df["OrdinarySharesNumber"] - df["OrdinarySharesNumber_Prev"]
        ) / df["OrdinarySharesNumber_Prev"]
        if "RepurchaseOfCapitalStock" in df.columns:
            df["Buyback_Yield"] = df["RepurchaseOfCapitalStock"].abs() / (
                df["MarketCap"] * 1_000_000
            )
        else:
            df["Buyback_Yield"] = 0

        mask = df["Share_Change_Pct"] < -min_buyback_pct
        if require_positive_fcf:
            mask = mask & (df["FreeCashFlow"] > 0)
        if min_market_cap_mm > 0:
            mask = mask & (df["MarketCap"] > min_market_cap_mm)
        passed = df[mask].copy()
        return passed.nlargest(top_n, "Buyback_Yield").index.tolist()

    def _strategy_piotroski_f_score(
        self, df: pd.DataFrame, top_n: int, trade_date: Optional[str], params: Dict[str, Any]
    ) -> List[str]:
        if trade_date is None:
            print("  trade_date is required for piotroski_f_score strategy.")
            return []
        df = self.enrich_with_yoy_data(df, trade_date)

        prev_cols = [c for c in df.columns if c.endswith("_Prev")]
        if not prev_cols or df[prev_cols].isna().all().all():
            print("  No YoY data available for piotroski_f_score. Skipping YoY criteria.")

        min_f_score = int(params.get("min_f_score", 7))
        df["F_Score"] = 0

        if "NetIncome" in df.columns:
            df.loc[df["NetIncome"] > 0, "F_Score"] += 1
        if "OperatingCashFlow" in df.columns:
            df.loc[df["OperatingCashFlow"] > 0, "F_Score"] += 1
        if all(
            c in df.columns
            for c in ["NetIncome", "TotalAssets", "NetIncome_Prev", "TotalAssets_Prev"]
        ):
            df["ROA"] = df["NetIncome"] / df["TotalAssets"]
            df["ROA_Prev"] = df["NetIncome_Prev"] / df["TotalAssets_Prev"]
            df.loc[df["ROA"] > df["ROA_Prev"], "F_Score"] += 1
        if all(c in df.columns for c in ["OperatingCashFlow", "NetIncome"]):
            df.loc[df["OperatingCashFlow"] > df["NetIncome"], "F_Score"] += 1

        if all(c in df.columns for c in ["TotalDebt", "TotalDebt_Prev"]):
            df.loc[df["TotalDebt"] <= df["TotalDebt_Prev"], "F_Score"] += 1
        if all(
            c in df.columns
            for c in [
                "CurrentAssets",
                "CurrentLiabilities",
                "CurrentAssets_Prev",
                "CurrentLiabilities_Prev",
            ]
        ):
            df["Current_Ratio"] = df["CurrentAssets"] / df["CurrentLiabilities"]
            df["Current_Ratio_Prev"] = (
                df["CurrentAssets_Prev"] / df["CurrentLiabilities_Prev"]
            )
            df.loc[df["Current_Ratio"] > df["Current_Ratio_Prev"], "F_Score"] += 1
        if all(c in df.columns for c in ["OrdinarySharesNumber", "OrdinarySharesNumber_Prev"]):
            df.loc[
                df["OrdinarySharesNumber"] <= df["OrdinarySharesNumber_Prev"],
                "F_Score",
            ] += 1

        if all(
            c in df.columns
            for c in ["GrossProfit", "TotalRevenue", "GrossProfit_Prev", "TotalRevenue_Prev"]
        ):
            df["Gross_Margin"] = df["GrossProfit"] / df["TotalRevenue"]
            df["Gross_Margin_Prev"] = df["GrossProfit_Prev"] / df["TotalRevenue_Prev"]
            df.loc[df["Gross_Margin"] > df["Gross_Margin_Prev"], "F_Score"] += 1
        if all(
            c in df.columns
            for c in ["TotalRevenue", "TotalAssets", "TotalRevenue_Prev", "TotalAssets_Prev"]
        ):
            df["Asset_Turnover"] = df["TotalRevenue"] / df["TotalAssets"]
            df["Asset_Turnover_Prev"] = (
                df["TotalRevenue_Prev"] / df["TotalAssets_Prev"]
            )
            df.loc[df["Asset_Turnover"] > df["Asset_Turnover_Prev"], "F_Score"] += 1

        passed = df[df["F_Score"] >= min_f_score].copy()
        if "MarketCap" in passed.columns and "EBIT" in passed.columns:
            passed["Earnings_Yield"] = passed["EBIT"] / (passed["MarketCap"] * 1_000_000)
            return passed.nlargest(top_n, "Earnings_Yield").index.tolist()
        return passed.nlargest(top_n, "F_Score").index.tolist()

    def _strategy_rule_of_40(
        self, df: pd.DataFrame, top_n: int, trade_date: Optional[str], params: Dict[str, Any]
    ) -> List[str]:
        if trade_date is None:
            print("  trade_date is required for rule_of_40 strategy.")
            return []
        df = self.enrich_with_yoy_data(df, trade_date)

        if "FreeCashFlow" not in df.columns:
            if "OperatingCashFlow" in df.columns and "CapitalExpenditure" in df.columns:
                df["FreeCashFlow"] = df["OperatingCashFlow"] - df["CapitalExpenditure"].abs()
            else:
                print("  Missing columns for FreeCashFlow calculation.")
                return []

        if not all(c in df.columns for c in ["TotalRevenue", "TotalRevenue_Prev"]):
            print("  Missing TotalRevenue columns for rule_of_40.")
            return []

        df = df.dropna(subset=["TotalRevenue", "TotalRevenue_Prev"])
        if df.empty:
            print("  No stocks with valid YoY revenue data for rule_of_40 strategy.")
            return []

        min_r40_score = float(params.get("min_r40_score", 40.0))
        min_total_revenue = float(params.get("min_total_revenue", 50_000_000))
        df["Rev_Growth"] = (df["TotalRevenue"] - df["TotalRevenue_Prev"]) / df["TotalRevenue_Prev"]
        df["FCF_Margin"] = df["FreeCashFlow"] / df["TotalRevenue"]
        df["R40_Score"] = (df["Rev_Growth"] + df["FCF_Margin"]) * 100

        mask = (df["R40_Score"] > min_r40_score) & (df["TotalRevenue"] > min_total_revenue)
        passed = df[mask].copy()
        return passed.nlargest(top_n, "R40_Score").index.tolist()

    def run_backtest(
        self,
        start_date: str,
        end_date: str,
        strategy: str = "magic_formula",
        top_n: int = 10,
        verbose: bool = True,
        profile_name: Optional[str] = None,
        config_path: Optional[str] = None,
    ) -> dict:
        """
        Run a single backtest period.

        Parameters:
        -----------
        start_date : str
            Entry date (e.g., '2023-01-01')
        end_date : str
            Exit date (e.g., '2024-01-01')
        strategy : str
            Strategy to use
        top_n : int
            Number of stocks to hold
        verbose : bool
            Print detailed output

        Returns:
        --------
        dict
            Results including picks, returns, and benchmark comparison
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"  BACKTEST: {start_date} → {end_date}")
            print(f"  Strategy: {strategy.upper()} | Top {top_n} picks")
            print(f"{'='*60}")

        # 1. Get point-in-time snapshot
        snapshot = self.get_valid_snapshot(start_date)
        if snapshot.empty:
            return {"error": "No data available for this period"}

        if verbose:
            print(f"\n  Data snapshot: {len(snapshot)} companies available")

        # 2. Apply strategy to select stocks
        picks = self.apply_strategy(snapshot, strategy, top_n, trade_date=start_date)

        if not picks:
            return {"error": "No stocks selected by strategy"}

        if verbose:
            print(f"  Selected: {picks}")

        # 3. Get entry and exit prices
        entry_prices = self.get_prices_bulk(picks, start_date)
        exit_prices = self.get_prices_bulk(picks, end_date)

        # 4. Calculate returns
        # FIX: Align prices to symbols using index to avoid mismatch
        results_df = pd.DataFrame(index=picks)
        results_df.index.name = "symbol"
        results_df["entry_price"] = entry_prices
        results_df["exit_price"] = exit_prices
        results_df = results_df.reset_index()

        results_df["return"] = (
            results_df["exit_price"] - results_df["entry_price"]
        ) / results_df["entry_price"]

        # Remove stocks with missing prices
        valid_results = results_df.dropna()

        if valid_results.empty:
            return {"error": "No valid price data for selected stocks"}

        # Portfolio return (equal weighted)
        portfolio_return = valid_results["return"].mean()

        # 5. Benchmark comparison (QQQ or SPY)
        benchmark_entry = self.get_price(self.benchmark_symbol, start_date)
        benchmark_exit = self.get_price(self.benchmark_symbol, end_date)

        if pd.notna(benchmark_entry) and pd.notna(benchmark_exit):
            benchmark_return = (benchmark_exit - benchmark_entry) / benchmark_entry
        else:
            benchmark_return = np.nan

        # 6. Alpha (excess return vs benchmark)
        alpha = (
            portfolio_return - benchmark_return
            if pd.notna(benchmark_return)
            else np.nan
        )

        if verbose:
            print(f"\n  RESULTS:")
            print(f"  {'-'*40}")
            print(valid_results.to_string(index=False))
            print(f"  {'-'*40}")
            print(f"  Portfolio Return: {portfolio_return*100:+.2f}%")
            print(
                f"  Benchmark ({self.benchmark_symbol}): {benchmark_return*100:+.2f}%"
                if pd.notna(benchmark_return)
                else f"  Benchmark: N/A"
            )
            print(f"  Alpha: {alpha*100:+.2f}%" if pd.notna(alpha) else "  Alpha: N/A")

        return {
            "start_date": start_date,
            "end_date": end_date,
            "strategy": strategy,
            "picks": picks,
            "results_df": valid_results,
            "portfolio_return": portfolio_return,
            "benchmark_return": benchmark_return,
            "alpha": alpha,
            "num_stocks": len(valid_results),
        }

    def run_multi_period_backtest(
        self,
        periods: list,
        strategy: str = "magic_formula",
        top_n: int = 10,
        show_progress: bool = True,
        profile_name: Optional[str] = None,
        config_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Run backtest across multiple periods.

        Parameters:
        -----------
        periods : list of tuples
            List of (start_date, end_date) tuples
        strategy : str
            Strategy to use
        top_n : int
            Number of stocks to hold

        Returns:
        --------
        pd.DataFrame
            Summary of all period results
        """
        all_results = []

        loop = tqdm(
            periods,
            desc=f"Backtest {strategy}",
            unit="period",
            disable=not show_progress,
        )
        for start, end in loop:
            if profile_name is None and config_path is None:
                result = self.run_backtest(start, end, strategy, top_n, verbose=True)
            else:
                result = self.run_backtest(
                    start,
                    end,
                    strategy,
                    top_n,
                    verbose=True,
                    profile_name=profile_name,
                    config_path=config_path,
                )

            if "error" not in result:
                all_results.append(
                    {
                        "Period": f"{start} → {end}",
                        "Strategy_Return": result["portfolio_return"],
                        "Benchmark_Return": result["benchmark_return"],
                        "Alpha": result["alpha"],
                        "Num_Stocks": result["num_stocks"],
                    }
                )
            loop.set_postfix({"valid": len(all_results)})

        if not all_results:
            print("No valid results from any period.")
            return pd.DataFrame()

        summary = pd.DataFrame(all_results)

        # Calculate cumulative/average performance
        print(f"\n{'='*60}")
        print("  MULTI-PERIOD SUMMARY")
        print(f"{'='*60}")
        print(summary.to_string(index=False))

        avg_return = summary["Strategy_Return"].mean()
        avg_benchmark = summary["Benchmark_Return"].mean()
        avg_alpha = summary["Alpha"].mean()

        print(f"\n  Average Strategy Return: {avg_return*100:+.2f}%")
        print(f"  Average Benchmark Return: {avg_benchmark*100:+.2f}%")
        print(f"  Average Alpha: {avg_alpha*100:+.2f}%")

        return summary

    def calculate_portfolio_return(
        self, symbols: list, start_date: str, end_date: str
    ) -> dict:
        """
        Calculate portfolio return for a list of symbols (equal-weighted).

        Parameters:
        -----------
        symbols : list
            List of stock symbols
        start_date : str
            Entry date
        end_date : str
            Exit date

        Returns:
        --------
        dict
            Dictionary with portfolio_return and individual results
        """
        if not symbols:
            return {"portfolio_return": np.nan, "results_df": pd.DataFrame()}

        entry_prices = self.get_prices_bulk(symbols, start_date)
        exit_prices = self.get_prices_bulk(symbols, end_date)

        results_df = pd.DataFrame(index=symbols)
        results_df.index.name = "symbol"
        results_df["entry_price"] = entry_prices
        results_df["exit_price"] = exit_prices
        results_df = results_df.reset_index()

        results_df["return"] = (
            results_df["exit_price"] - results_df["entry_price"]
        ) / results_df["entry_price"]

        valid_results = results_df.dropna()

        if valid_results.empty:
            return {"portfolio_return": np.nan, "results_df": valid_results}

        portfolio_return = valid_results["return"].mean()

        return {"portfolio_return": portfolio_return, "results_df": valid_results}

    def run_combo_backtest(
        self,
        start_date: str,
        end_date: str,
        strategies: list,
        top_n_per_strat: int = 5,
        verbose: bool = True,
        profile_name: Optional[str] = None,
        config_path: Optional[str] = None,
    ) -> dict:
        """
        Runs a backtest combining multiple strategies (e.g., 5 picks from Moat + 5 from Magic Formula).
        """
        # 1. Get Snapshot
        snapshot = self.get_valid_snapshot(start_date)

        profile = self._resolve_profile(profile_name=profile_name, config_path=config_path)

        # 2. Run Universe Filter ONCE (Efficiency)
        snapshot = self._filter_universe_with_profile(
            snapshot,
            start_date,
            min_mkt_cap_mm=float(profile.universe.strategy_min_market_cap_mm),
            profile=profile,
        )

        # 3. Collect Picks from each Strategy
        combined_picks = set()  # Use a set to handle duplicates automatically

        if verbose:
            print(f"\n  [Combo] Testing Mix: {strategies} on {start_date}")

        effective_top_n_per_strat = (
            int(top_n_per_strat)
            if int(top_n_per_strat) > 0
            else int(profile.execution.combo_top_n_per_strategy)
        )
        per_strategy_pick_counts: Dict[str, int] = {}
        for strat in strategies:
            picks = self._apply_strategy_with_profile(
                snapshot=snapshot,
                strategy=strat,
                top_n=effective_top_n_per_strat,
                trade_date=start_date,
                profile=profile,
            )
            per_strategy_pick_counts[strat] = len(picks)
            if verbose:
                print(f"    -> {strat} contributed: {picks}")
            combined_picks.update(picks)

        final_portfolio = list(combined_picks)

        # 4. Calculate Returns
        # Note: If a stock is picked by BOTH strategies, it is just held once in this simple model.
        result = self.calculate_portfolio_return(final_portfolio, start_date, end_date)
        valid_results = cast(pd.DataFrame, result["results_df"])
        selected_count = int(len(final_portfolio))
        priced_count = int(len(valid_results))

        if selected_count == 0:
            status = "no_selections"
        elif priced_count == 0:
            status = "no_valid_prices"
        else:
            status = "ok"

        return {
            "period": f"{start_date} -> {end_date}",
            "strategy": " + ".join(strategies),
            "return": result["portfolio_return"],
            "picks": final_portfolio,
            "selected_count": selected_count,
            "picks_count": priced_count,
            "status": status,
            "per_strategy_pick_counts": per_strategy_pick_counts,
        }

    def _generate_monthly_periods(
        self, start_date: str, end_date: str, holding_months: int = 1
    ) -> List[Tuple[str, str]]:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        if start >= end:
            return []

        entry_dates = pd.date_range(start=start, end=end, freq="MS")
        periods: List[Tuple[str, str]] = []
        for entry_dt in entry_dates:
            exit_dt = entry_dt + pd.DateOffset(months=holding_months)
            if exit_dt > end:
                break
            periods.append(
                (entry_dt.strftime("%Y-%m-%d"), exit_dt.strftime("%Y-%m-%d"))
            )

        if not periods:
            exit_dt = min(start + pd.DateOffset(months=holding_months), end)
            if exit_dt > start:
                periods.append(
                    (start.strftime("%Y-%m-%d"), exit_dt.strftime("%Y-%m-%d"))
                )
        return periods

    def _validate_required_proxies(self, required_symbols: Sequence[str]) -> None:
        if self.price_matrix is None or self.price_matrix.empty:
            raise ValueError("Price matrix is empty. Call load_data() before running research.")

        missing = [sym for sym in required_symbols if sym not in self.price_matrix.columns]
        if missing:
            raise ValueError(
                "Missing required proxy symbols in local DB: "
                f"{missing}. Ingest them first with: "
                'py -m data.data SPY QQQ IWD IWF "^IRX" --exchange CUSTOM'
            )

        no_data = [
            sym
            for sym in required_symbols
            if self.price_matrix[sym].dropna().empty
        ]
        if no_data:
            raise ValueError(
                "Required proxy symbols have no usable history: "
                f"{no_data}. Re-ingest with: "
                'py -m data.data SPY QQQ IWD IWF "^IRX" --exchange CUSTOM'
            )

    def _symbol_has_history(self, symbol: str) -> bool:
        if self.price_matrix is None or self.price_matrix.empty:
            return False
        if symbol not in self.price_matrix.columns:
            return False
        return not self.price_matrix[symbol].dropna().empty

    def _first_available_proxy(self, candidates: Sequence[str]) -> Optional[str]:
        for symbol in candidates:
            if self._symbol_has_history(symbol):
                return symbol
        return None

    def _resolve_tournament_proxies(self) -> Dict[str, Optional[str]]:
        return {
            "market": self._first_available_proxy(("SPY",)),
            "growth": self._first_available_proxy(("QQQ",)),
            "value": self._first_available_proxy(DEFAULT_VALUE_PROXY_CANDIDATES),
            "growth_style": self._first_available_proxy(DEFAULT_GROWTH_PROXY_CANDIDATES),
            "rate": self._first_available_proxy(DEFAULT_RATE_PROXY_CANDIDATES),
        }

    @staticmethod
    def _normalize_rate_quote_to_annual(value: Any) -> float:
        if value is None or pd.isna(value):
            return np.nan
        raw = float(value)
        abs_raw = abs(raw)
        # Handle mixed quote conventions:
        # - ^IRX style: 5.0 -> 0.05 (divide by 100)
        # - ^TNX style: 45.0 -> 0.045 (divide by 1000)
        # - Already decimal: 0.05 -> 0.05
        if abs_raw >= 20.0:
            return raw / 1000.0
        if abs_raw >= 1.0:
            return raw / 100.0
        return raw

    def _build_year_buckets(self, start_date: str, end_date: str) -> pd.DataFrame:
        if self.price_matrix is None or self.price_matrix.empty:
            return pd.DataFrame(
                columns=[
                    "Year_Bucket",
                    "Calendar_Year",
                    "Bucket_Start",
                    "Bucket_End",
                    "Trade_Start",
                    "Trade_End",
                    "Is_YTD",
                ]
            )

        idx = self.price_matrix.index.sort_values().unique()
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        rows: List[Dict[str, Any]] = []

        for year in range(start.year, end.year + 1):
            cal_start = pd.Timestamp(year=year, month=1, day=1)
            cal_end = pd.Timestamp(year=year, month=12, day=31)
            bucket_start = max(start, cal_start)
            bucket_end = min(end, cal_end)
            if bucket_start > bucket_end:
                continue

            tradable = idx[(idx >= bucket_start) & (idx <= bucket_end)]
            if len(tradable) == 0:
                continue

            is_full_year = bucket_start == cal_start and bucket_end == cal_end
            year_bucket = str(year) if is_full_year else f"{year}_YTD"
            rows.append(
                {
                    "Year_Bucket": year_bucket,
                    "Calendar_Year": int(year),
                    "Bucket_Start": bucket_start.normalize(),
                    "Bucket_End": bucket_end.normalize(),
                    "Trade_Start": pd.Timestamp(tradable[0]).normalize(),
                    "Trade_End": pd.Timestamp(tradable[-1]).normalize(),
                    "Is_YTD": bool(not is_full_year),
                }
            )

        out = pd.DataFrame(rows)
        if out.empty:
            return out
        return out.sort_values(["Calendar_Year"]).reset_index(drop=True)

    def _period_return_for_symbol(
        self, symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> float:
        if self.price_matrix is None or symbol not in self.price_matrix.columns:
            return np.nan
        series = self.price_matrix[symbol]
        window = series[(series.index >= start_date) & (series.index <= end_date)].dropna()
        if len(window) < 2:
            return np.nan
        start_px = float(window.iloc[0])
        end_px = float(window.iloc[-1])
        if start_px <= 0:
            return np.nan
        return (end_px / start_px) - 1.0

    def _compute_year_proxy_features(
        self,
        year_buckets: pd.DataFrame,
        proxy_symbols: Optional[Dict[str, Optional[str]]] = None,
    ) -> pd.DataFrame:
        if year_buckets.empty:
            return pd.DataFrame()

        proxies = proxy_symbols or self._resolve_tournament_proxies()
        market_symbol = proxies.get("market")
        growth_symbol = proxies.get("growth")
        value_symbol = proxies.get("value")
        growth_style_symbol = proxies.get("growth_style")
        rate_symbol = proxies.get("rate")

        rows: List[Dict[str, Any]] = []
        for _, row in year_buckets.iterrows():
            trade_start = pd.to_datetime(row["Trade_Start"])
            trade_end = pd.to_datetime(row["Trade_End"])

            spy_return = (
                self._period_return_for_symbol(market_symbol, trade_start, trade_end)
                if market_symbol
                else np.nan
            )
            qqq_return = (
                self._period_return_for_symbol(growth_symbol, trade_start, trade_end)
                if growth_symbol
                else np.nan
            )
            iwd_return = (
                self._period_return_for_symbol(value_symbol, trade_start, trade_end)
                if value_symbol
                else np.nan
            )
            iwf_return = (
                self._period_return_for_symbol(growth_style_symbol, trade_start, trade_end)
                if growth_style_symbol
                else np.nan
            )

            irx_mean_annual = np.nan
            if self.price_matrix is not None and rate_symbol and rate_symbol in self.price_matrix.columns:
                irx_window = self.price_matrix[rate_symbol]
                irx_window = irx_window[
                    (irx_window.index >= trade_start) & (irx_window.index <= trade_end)
                ].dropna()
                if not irx_window.empty:
                    irx_mean_annual = self._normalize_rate_quote_to_annual(
                        float(irx_window.mean())
                    )

            rows.append(
                {
                    "Year_Bucket": row["Year_Bucket"],
                    "Calendar_Year": int(row["Calendar_Year"]),
                    "Trade_Start": trade_start.normalize(),
                    "Trade_End": trade_end.normalize(),
                    "spy_return": spy_return,
                    "qqq_return": qqq_return,
                    "iwd_return": iwd_return,
                    "iwf_return": iwf_return,
                    "growth_minus_value": (
                        iwf_return - iwd_return
                        if pd.notna(iwf_return) and pd.notna(iwd_return)
                        else np.nan
                    ),
                    "value_minus_growth": (
                        iwd_return - iwf_return
                        if pd.notna(iwf_return) and pd.notna(iwd_return)
                        else np.nan
                    ),
                    "irx_mean_annual": irx_mean_annual,
                    "Market_Proxy_Symbol": market_symbol,
                    "Growth_Proxy_Symbol": growth_symbol,
                    "Value_Proxy_Symbol": value_symbol,
                    "GrowthStyle_Proxy_Symbol": growth_style_symbol,
                    "Rate_Proxy_Symbol": rate_symbol,
                }
            )

        return pd.DataFrame(rows).sort_values(["Calendar_Year"]).reset_index(drop=True)

    def _label_yearly_regimes(
        self,
        features_df: pd.DataFrame,
        quantile_low: float = 0.33,
        quantile_high: float = 0.67,
    ) -> pd.DataFrame:
        if features_df.empty:
            out = features_df.copy()
            out["Market_Regime"] = []
            out["Value_Rotation"] = []
            out["Rate_Regime"] = []
            return out

        out = features_df.copy()

        def _quantile(series: pd.Series, q: float) -> float:
            valid = series.dropna()
            if valid.empty:
                return np.nan
            return float(valid.quantile(q))

        spy_q_low = _quantile(out["spy_return"], quantile_low)
        spy_q_high = _quantile(out["spy_return"], quantile_high)
        growth_q_high = _quantile(out["growth_minus_value"], quantile_high)
        value_q_high = _quantile(out["value_minus_growth"], quantile_high)
        irx_q_low = _quantile(out["irx_mean_annual"], quantile_low)
        irx_q_high = _quantile(out["irx_mean_annual"], quantile_high)

        market_labels: List[str] = []
        value_rotation_flags: List[bool] = []
        rate_labels: List[str] = []

        for _, row in out.iterrows():
            spy_ret = row.get("spy_return")
            growth_minus_value = row.get("growth_minus_value")
            value_minus_growth = row.get("value_minus_growth")
            irx = row.get("irx_mean_annual")

            market_regime = "Neutral"
            if pd.notna(spy_ret) and pd.notna(spy_q_low) and spy_ret <= spy_q_low:
                market_regime = "Risk-off"
            elif (
                pd.notna(spy_ret)
                and pd.notna(spy_q_high)
                and pd.notna(growth_minus_value)
                and pd.notna(growth_q_high)
                and spy_ret >= spy_q_high
                and growth_minus_value >= growth_q_high
            ):
                market_regime = "Bull growth"
            market_labels.append(market_regime)

            value_rotation = bool(
                pd.notna(value_minus_growth)
                and pd.notna(value_q_high)
                and value_minus_growth >= value_q_high
            )
            value_rotation_flags.append(value_rotation)

            rate_regime = "Mid rate"
            if pd.notna(irx) and pd.notna(irx_q_high) and irx >= irx_q_high:
                rate_regime = "High rate"
            elif pd.notna(irx) and pd.notna(irx_q_low) and irx <= irx_q_low:
                rate_regime = "Low rate"
            rate_labels.append(rate_regime)

        out["Market_Regime"] = market_labels
        out["Value_Rotation"] = value_rotation_flags
        out["Rate_Regime"] = rate_labels
        return out

    def build_yearly_regime_labels(
        self,
        config: Optional[RegimeLabelConfig] = None,
        proxy_symbols: Optional[Dict[str, Optional[str]]] = None,
    ) -> pd.DataFrame:
        cfg = config or RegimeLabelConfig()
        proxies = proxy_symbols or self._resolve_tournament_proxies()
        required_core: List[str] = []
        if proxies.get("market"):
            required_core.append(cast(str, proxies["market"]))
        if proxies.get("growth"):
            required_core.append(cast(str, proxies["growth"]))
        if len(required_core) < 2:
            raise ValueError(
                "Missing required core market proxies (SPY/QQQ). "
                "Ingest them with: py -m data.data SPY QQQ --exchange CUSTOM"
            )
        self._validate_required_proxies(required_core)
        year_buckets = self._build_year_buckets(cfg.start_date, cfg.end_date)
        features = self._compute_year_proxy_features(
            year_buckets,
            proxy_symbols=proxies,
        )
        return self._label_yearly_regimes(
            features,
            quantile_low=cfg.quantile_low,
            quantile_high=cfg.quantile_high,
        )

    def _resolve_year_bucket(self, date: str, year_buckets: pd.DataFrame) -> str:
        if year_buckets.empty:
            return str(pd.to_datetime(date).year)

        target = pd.to_datetime(date).normalize()
        for _, row in year_buckets.iterrows():
            start = pd.to_datetime(row["Trade_Start"]).normalize()
            end = pd.to_datetime(row["Trade_End"]).normalize()
            if start <= target <= end:
                return str(row["Year_Bucket"])
        return str(target.year)

    def _compute_monthly_risk_free_map(
        self,
        periods: Sequence[Tuple[str, str]],
        rf_symbol: Optional[str] = "^IRX",
    ) -> Dict[Tuple[str, str], float]:
        rf_map: Dict[Tuple[str, str], float] = {}
        if (
            self.price_matrix is None
            or self.price_matrix.empty
            or not rf_symbol
            or rf_symbol not in self.price_matrix.columns
        ):
            for start, end in periods:
                rf_map[(str(start), str(end))] = 0.0
            return rf_map

        rf_series = self.price_matrix[rf_symbol]
        for start, end in periods:
            start_dt = pd.to_datetime(start)
            end_dt = pd.to_datetime(end)
            window = rf_series[(rf_series.index >= start_dt) & (rf_series.index <= end_dt)]
            annual = np.nan
            if not window.dropna().empty:
                annual = self._normalize_rate_quote_to_annual(float(window.dropna().mean()))

            if pd.notna(annual) and annual > -1.0:
                monthly_rf = float((1.0 + annual) ** (1.0 / 12.0) - 1.0)
            else:
                monthly_rf = 0.0
            rf_map[(str(start), str(end))] = monthly_rf
        return rf_map

    @staticmethod
    def _compute_return_summary(
        monthly_returns: pd.Series,
        year_buckets: Optional[pd.Series] = None,
        monthly_rf: Optional[pd.Series] = None,
    ) -> Dict[str, float]:
        rets = monthly_returns.fillna(0.0).astype(float)
        n_months = int(len(rets))
        if n_months == 0:
            return {
                "Months": 0,
                "Total_Return": np.nan,
                "CAGR": np.nan,
                "Volatility": np.nan,
                "Sharpe": np.nan,
                "Max_Drawdown": np.nan,
                "Worst_Month": np.nan,
                "Worst_Year": np.nan,
                "Mean_Monthly_Return": np.nan,
            }

        equity = (1.0 + rets).cumprod()
        final_equity = float(equity.iloc[-1])
        total_return = final_equity - 1.0
        cagr = np.nan
        if final_equity > 0:
            cagr = float(final_equity ** (12.0 / n_months) - 1.0)

        vol = 0.0
        std_monthly = float(rets.std(ddof=1)) if n_months > 1 else 0.0
        if n_months > 1:
            vol = float(std_monthly * np.sqrt(12.0))

        if monthly_rf is None:
            rf = pd.Series(0.0, index=rets.index, dtype=float)
        else:
            rf = monthly_rf.reindex(rets.index).fillna(0.0).astype(float)
        excess = rets - rf
        sharpe = np.nan
        if n_months > 1 and std_monthly > 0:
            sharpe = float((excess.mean() / std_monthly) * np.sqrt(12.0))

        running_peak = equity.cummax()
        drawdowns = (equity / running_peak) - 1.0
        max_drawdown = float(drawdowns.min())
        worst_month = float(rets.min())

        if year_buckets is None:
            year_labels = pd.Series("All", index=rets.index, dtype=object)
        else:
            year_labels = (
                year_buckets.reindex(rets.index).fillna("Unknown").astype(str)
            )
        yearly_returns = (1.0 + rets).groupby(year_labels).prod() - 1.0
        worst_year = float(yearly_returns.min()) if not yearly_returns.empty else np.nan

        return {
            "Months": float(n_months),
            "Total_Return": float(total_return),
            "CAGR": float(cagr) if pd.notna(cagr) else np.nan,
            "Volatility": float(vol),
            "Sharpe": float(sharpe) if pd.notna(sharpe) else np.nan,
            "Max_Drawdown": max_drawdown,
            "Worst_Month": worst_month,
            "Worst_Year": worst_year,
            "Mean_Monthly_Return": float(rets.mean()),
        }

    def compute_strategy_risk_metrics(
        self,
        monthly_df: pd.DataFrame,
        rf_symbol: str = "^IRX",
    ) -> pd.DataFrame:
        if monthly_df.empty:
            return pd.DataFrame()
        if "Strategy" not in monthly_df.columns or "Monthly_Return" not in monthly_df.columns:
            raise ValueError("monthly_df must contain Strategy and Monthly_Return columns.")

        panel = monthly_df.copy()
        if "Monthly_RF" not in panel.columns:
            panel["Monthly_RF"] = 0.0
            if {"Period_Start", "Period_End"}.issubset(panel.columns):
                unique_periods = [
                    (str(start), str(end))
                    for start, end in panel[["Period_Start", "Period_End"]]
                    .drop_duplicates()
                    .itertuples(index=False, name=None)
                ]
                rf_map = self._compute_monthly_risk_free_map(unique_periods, rf_symbol=rf_symbol)
                panel["Monthly_RF"] = panel.apply(
                    lambda row: rf_map.get(
                        (str(row["Period_Start"]), str(row["Period_End"])),
                        0.0,
                    ),
                    axis=1,
                )
        else:
            panel["Monthly_RF"] = panel["Monthly_RF"].fillna(0.0)

        records: List[Dict[str, Any]] = []
        grouped = panel.groupby("Strategy")
        for strategy, grp in grouped:
            grp = grp.sort_values(["Period_Start", "Period_End"], na_position="last")
            years = grp["Year_Bucket"] if "Year_Bucket" in grp.columns else None
            summary = self._compute_return_summary(
                grp["Monthly_Return"],
                year_buckets=years,
                monthly_rf=grp["Monthly_RF"],
            )
            records.append({"Strategy": strategy, **summary})

        out = pd.DataFrame(records)
        if out.empty:
            return out
        out["Months"] = out["Months"].astype(int)
        return out.sort_values(
            ["Sharpe", "CAGR", "Max_Drawdown", "Strategy"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)

    def compute_regime_strategy_metrics(
        self,
        monthly_df: pd.DataFrame,
        regime_labels_df: pd.DataFrame,
        min_months: int = 3,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if monthly_df.empty:
            return (pd.DataFrame(), pd.DataFrame())

        panel = monthly_df.copy()
        merge_cols = ["Year_Bucket", "Market_Regime", "Value_Rotation", "Rate_Regime"]
        if (
            "Year_Bucket" in panel.columns
            and not regime_labels_df.empty
            and all(col in regime_labels_df.columns for col in merge_cols)
            and not all(col in panel.columns for col in merge_cols[1:])
        ):
            panel = panel.merge(
                regime_labels_df[merge_cols].drop_duplicates(),
                on="Year_Bucket",
                how="left",
            )

        if "Market_Regime" not in panel.columns:
            panel["Market_Regime"] = "Neutral"
        if "Value_Rotation" not in panel.columns:
            panel["Value_Rotation"] = False
        if "Rate_Regime" not in panel.columns:
            panel["Rate_Regime"] = "Mid rate"
        if "Monthly_RF" not in panel.columns:
            panel["Monthly_RF"] = 0.0

        bucket_filters = {
            "Bull growth": panel["Market_Regime"] == "Bull growth",
            "Risk-off": panel["Market_Regime"] == "Risk-off",
            "Value rotation": panel["Value_Rotation"] == True,
            "High rate": panel["Rate_Regime"] == "High rate",
            "Low rate": panel["Rate_Regime"] == "Low rate",
        }

        rows: List[Dict[str, Any]] = []
        for bucket_name, mask in bucket_filters.items():
            bucket_df = panel[mask].copy()
            if bucket_df.empty:
                continue
            for strategy, grp in bucket_df.groupby("Strategy"):
                grp = grp.sort_values(["Period_Start", "Period_End"], na_position="last")
                years = grp["Year_Bucket"] if "Year_Bucket" in grp.columns else None
                summary = self._compute_return_summary(
                    grp["Monthly_Return"],
                    year_buckets=years,
                    monthly_rf=grp["Monthly_RF"],
                )
                row = {"Regime_Bucket": bucket_name, "Strategy": strategy, **summary}
                row["insufficient_data"] = bool(int(summary["Months"]) < int(min_months))
                rows.append(row)

        all_metrics = pd.DataFrame(rows)
        if all_metrics.empty:
            return (all_metrics, pd.DataFrame())

        all_metrics["Z_Sharpe"] = np.nan
        all_metrics["Z_CAGR"] = np.nan
        all_metrics["Z_Max_Drawdown"] = np.nan
        all_metrics["Score"] = np.nan

        def _zscore(values: pd.Series) -> pd.Series:
            std = float(values.std(ddof=0))
            if pd.isna(std) or std == 0.0:
                return pd.Series(0.0, index=values.index, dtype=float)
            return (values - float(values.mean())) / std

        for _, grp in all_metrics.groupby("Regime_Bucket"):
            eligible_idx = grp.index[
                (~grp["insufficient_data"])
                & grp["Sharpe"].notna()
                & grp["CAGR"].notna()
                & grp["Max_Drawdown"].notna()
            ]
            if len(eligible_idx) == 0:
                continue

            sharpe_z = _zscore(all_metrics.loc[eligible_idx, "Sharpe"])
            cagr_z = _zscore(all_metrics.loc[eligible_idx, "CAGR"])
            dd_z = _zscore(all_metrics.loc[eligible_idx, "Max_Drawdown"])
            score = 0.5 * sharpe_z + 0.3 * cagr_z + 0.2 * dd_z

            all_metrics.loc[eligible_idx, "Z_Sharpe"] = sharpe_z
            all_metrics.loc[eligible_idx, "Z_CAGR"] = cagr_z
            all_metrics.loc[eligible_idx, "Z_Max_Drawdown"] = dd_z
            all_metrics.loc[eligible_idx, "Score"] = score

        winners: List[Dict[str, Any]] = []
        for _, grp in all_metrics.groupby("Regime_Bucket"):
            eligible = grp[
                (~grp["insufficient_data"])
                & grp["Score"].notna()
            ].copy()
            if eligible.empty:
                continue
            eligible = eligible.sort_values(
                ["Score", "Sharpe", "CAGR", "Max_Drawdown", "Strategy"],
                ascending=[False, False, False, False, True],
            )
            winner_row = {str(k): v for k, v in eligible.iloc[0].to_dict().items()}
            winners.append(winner_row)

        winners_df = pd.DataFrame(winners)
        all_metrics = all_metrics.sort_values(
            ["Regime_Bucket", "Score", "Sharpe", "CAGR", "Max_Drawdown", "Strategy"],
            ascending=[True, False, False, False, False, True],
        ).reset_index(drop=True)
        return (all_metrics, winners_df)

    def _evaluate_magic_gate_variants_for_period(
        self,
        start_date: str,
        end_date: str,
        regime: str,
        snapshot: Optional[pd.DataFrame] = None,
        top_n: int = 10,
        profile_name: Optional[str] = None,
        config_path: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        empty_result = {
            name: {
                "monthly_return": np.nan,
                "selected_count": 0,
                "priced_count": 0,
                "candidate_count": 0,
                "status": "no_candidates",
                "is_cash_fill": True,
            }
            for name in DEFAULT_MAGIC_GATE_VARIANTS
        }

        if snapshot is None:
            snapshot = self.get_valid_snapshot(start_date)
        if snapshot is None or snapshot.empty:
            return empty_result

        panel = self._build_asset_panel_for_period(
            start_date=start_date,
            end_date=end_date,
            strategy="magic_formula",
            top_n=top_n,
            regime=regime,
            risk_config=RiskVectorConfig(),
            snapshot=snapshot,
            momentum_table=None,
            use_cached_tail=True,
            profile_name=profile_name,
            config_path=config_path,
        )
        if panel.empty:
            return empty_result

        candidates = int(panel["symbol"].nunique()) if "symbol" in panel.columns else int(len(panel))
        variant_masks: Dict[str, pd.Series] = {
            "Magic (raw)": pd.Series(True, index=panel.index),
            "Magic + Distress Gate": ~panel["RV_Gate_Distress"].fillna(False).astype(bool),
            "Magic + Tail Gate": ~panel["RV_Gate_Tail"].fillna(False).astype(bool),
            "Magic + Momentum Gate": ~panel["RV_Gate_Momentum"].fillna(False).astype(bool),
            "Magic + Full Risk Vector": panel["RV_Gate_Count"].fillna(0).astype(float) == 0.0,
        }

        results: Dict[str, Dict[str, Any]] = {}
        for name, mask in variant_masks.items():
            selected = panel[mask].copy()
            selected_count = (
                int(selected["symbol"].nunique()) if "symbol" in selected.columns else int(len(selected))
            )
            valid = selected.dropna(subset=["return"])
            priced_count = int(len(valid))
            if valid.empty:
                monthly_return = np.nan
                status = "no_valid_prices" if selected_count > 0 else "no_survivors"
                is_cash_fill = True
            else:
                monthly_return = float(valid["return"].mean())
                status = "ok"
                is_cash_fill = False
            results[name] = {
                "monthly_return": monthly_return,
                "selected_count": selected_count,
                "priced_count": priced_count,
                "candidate_count": candidates,
                "status": status,
                "is_cash_fill": is_cash_fill,
            }
        return results

    def _build_magic_gate_comparison(
        self,
        strategy_metrics: pd.DataFrame,
        preserve_return_delta_floor: float = -0.01,
    ) -> pd.DataFrame:
        if strategy_metrics.empty or "Strategy" not in strategy_metrics.columns:
            return pd.DataFrame()

        variants = list(DEFAULT_MAGIC_GATE_VARIANTS)
        subset = strategy_metrics[strategy_metrics["Strategy"].isin(variants)].copy()
        if subset.empty:
            return subset

        raw = subset[subset["Strategy"] == "Magic (raw)"]
        if raw.empty:
            return pd.DataFrame()
        raw_row = raw.iloc[0]

        rows: List[Dict[str, Any]] = []
        for _, row in subset.iterrows():
            cagr = row.get("CAGR", np.nan)
            max_dd = row.get("Max_Drawdown", np.nan)
            raw_cagr = raw_row.get("CAGR", np.nan)
            raw_max_dd = raw_row.get("Max_Drawdown", np.nan)

            delta_cagr = np.nan
            if pd.notna(cagr) and pd.notna(raw_cagr):
                delta_cagr = float(cagr - raw_cagr)

            delta_max_dd = np.nan
            if pd.notna(max_dd) and pd.notna(raw_max_dd):
                delta_max_dd = float(max_dd - raw_max_dd)

            preserve_return_flag = bool(
                pd.notna(delta_cagr) and float(delta_cagr) >= float(preserve_return_delta_floor)
            )
            improve_drawdown_flag = bool(
                pd.notna(delta_max_dd) and float(delta_max_dd) > 0.0
            )

            rows.append(
                {
                    "Strategy": row["Strategy"],
                    "CAGR": cagr,
                    "Max_Drawdown": max_dd,
                    "delta_cagr": delta_cagr,
                    "delta_max_drawdown": delta_max_dd,
                    "preserve_return_flag": preserve_return_flag,
                    "improve_drawdown_flag": improve_drawdown_flag,
                    "publishable_flag": bool(preserve_return_flag and improve_drawdown_flag),
                }
            )

        out = pd.DataFrame(rows)
        if out.empty:
            return out
        return out.sort_values(["Strategy"]).reset_index(drop=True)

    def run_regime_tournament(
        self,
        start_date: str = "2022-01-01",
        end_date: str = "2025-06-30",
        output_dir: str = "research_outputs",
        csv_only: bool = True,
        show_progress: bool = True,
        cash_fill_cost_bps: Optional[float] = None,
        profile_name: Optional[str] = None,
        config_path: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        if self.price_matrix is None or self.financial_data is None:
            self.load_data()

        profile = self._resolve_profile(profile_name=profile_name, config_path=config_path)
        regime_cfg = RegimeLabelConfig(start_date=start_date, end_date=end_date)
        output_cfg = TournamentOutputConfig(output_dir=output_dir, csv_only=csv_only)
        effective_cash_fill_cost_bps = (
            float(output_cfg.cash_fill_cost_bps)
            if cash_fill_cost_bps is None
            else float(cash_fill_cost_bps)
        )
        proxy_map = self._resolve_tournament_proxies()
        self._validate_required_proxies(["SPY", "QQQ"])
        if proxy_map.get("value") is None or proxy_map.get("growth_style") is None:
            print(
                "[regime] warning: style proxies unavailable "
                f"(value={proxy_map.get('value')}, growth={proxy_map.get('growth_style')}). "
                "Value rotation labels will default to False where style returns are missing."
            )
        if proxy_map.get("rate") is None:
            print(
                "[regime] warning: no rate proxy available; "
                "Monthly_RF defaults to 0 and rate regime defaults to Mid rate."
            )

        regime_labels = self.build_yearly_regime_labels(
            regime_cfg,
            proxy_symbols=proxy_map,
        )
        if regime_labels.empty:
            raise ValueError("Failed to build yearly regime labels for the requested date range.")

        periods = self._generate_monthly_periods(
            regime_cfg.start_date,
            regime_cfg.end_date,
            holding_months=1,
        )
        if not periods:
            raise ValueError("No monthly periods available for the requested date range.")

        rf_symbol = proxy_map.get("rate") or output_cfg.rf_symbol
        rf_map = self._compute_monthly_risk_free_map(periods, rf_symbol=rf_symbol)
        regime_lookup = {
            str(row["Year_Bucket"]): row for _, row in regime_labels.iterrows()
        }
        contenders = [
            {
                "name": contender.name,
                "strategies": list(contender.strategies),
                "n": int(contender.n),
            }
            for contender in profile.tournament.contenders
        ]
        gate_top_n = int(profile.tournament.magic_gate_top_n)

        rows: List[Dict[str, Any]] = []
        period_iter = tqdm(
            periods,
            desc="Regime tournament",
            unit="period",
            disable=not show_progress,
        )
        for period_start, period_end in period_iter:
            year_bucket = self._resolve_year_bucket(period_start, regime_labels)
            regime_row = regime_lookup.get(year_bucket, {})
            market_regime = regime_row.get("Market_Regime", "Neutral")
            value_rotation = bool(regime_row.get("Value_Rotation", False))
            rate_regime = regime_row.get("Rate_Regime", "Mid rate")
            monthly_rf = float(rf_map.get((str(period_start), str(period_end)), 0.0))

            base_meta = {
                "Period_Start": str(period_start),
                "Period_End": str(period_end),
                "Year_Bucket": str(year_bucket),
                "Market_Regime": str(market_regime),
                "Value_Rotation": value_rotation,
                "Rate_Regime": str(rate_regime),
                "Monthly_RF": monthly_rf,
            }

            for contender in contenders:
                if profile_name is None and config_path is None:
                    res = self.run_combo_backtest(
                        period_start,
                        period_end,
                        contender["strategies"],
                        top_n_per_strat=contender["n"],
                        verbose=False,
                    )
                else:
                    res = self.run_combo_backtest(
                        period_start,
                        period_end,
                        contender["strategies"],
                        top_n_per_strat=contender["n"],
                        verbose=False,
                        profile_name=profile_name,
                        config_path=config_path,
                    )
                raw_return = res.get("return", np.nan)
                is_cash_fill = bool(pd.isna(raw_return))
                if is_cash_fill:
                    monthly_return = self._cash_fill_monthly_return(
                        monthly_rf, effective_cash_fill_cost_bps
                    )
                    raw_strategy_return = np.nan
                else:
                    monthly_return = float(raw_return)
                    raw_strategy_return = float(raw_return)
                rows.append(
                    {
                        **base_meta,
                        "Strategy": contender["name"],
                        "Monthly_Return": monthly_return,
                        "Raw_Strategy_Return": raw_strategy_return,
                        "Is_Cash_Fill": is_cash_fill,
                        "Selected_Count": int(res.get("selected_count", 0)),
                        "Priced_Count": int(res.get("picks_count", 0)),
                        "Candidate_Count": int(res.get("selected_count", 0)),
                        "Status": str(res.get("status", "unknown")),
                        "Source": "contender",
                    }
                )

            snapshot = self.get_valid_snapshot(period_start)
            if profile_name is None and config_path is None:
                gate_results = self._evaluate_magic_gate_variants_for_period(
                    start_date=period_start,
                    end_date=period_end,
                    regime=str(market_regime),
                    snapshot=snapshot,
                    top_n=gate_top_n,
                )
            else:
                gate_results = self._evaluate_magic_gate_variants_for_period(
                    start_date=period_start,
                    end_date=period_end,
                    regime=str(market_regime),
                    snapshot=snapshot,
                    top_n=gate_top_n,
                    profile_name=profile_name,
                    config_path=config_path,
                )
            for strategy_name, gate_result in gate_results.items():
                raw_return = gate_result.get("monthly_return", np.nan)
                is_cash_fill = bool(gate_result.get("is_cash_fill", pd.isna(raw_return)))
                if is_cash_fill or pd.isna(raw_return):
                    monthly_return = self._cash_fill_monthly_return(
                        monthly_rf, effective_cash_fill_cost_bps
                    )
                    raw_strategy_return = np.nan
                    is_cash_fill = True
                else:
                    monthly_return = float(raw_return)
                    raw_strategy_return = float(raw_return)
                rows.append(
                    {
                        **base_meta,
                        "Strategy": strategy_name,
                        "Monthly_Return": monthly_return,
                        "Raw_Strategy_Return": raw_strategy_return,
                        "Is_Cash_Fill": is_cash_fill,
                        "Selected_Count": int(gate_result["selected_count"]),
                        "Priced_Count": int(gate_result["priced_count"]),
                        "Candidate_Count": int(gate_result["candidate_count"]),
                        "Status": str(gate_result["status"]),
                        "Source": "magic_gate",
                    }
                )

        tournament_monthly = pd.DataFrame(rows)
        if not tournament_monthly.empty:
            tournament_monthly = tournament_monthly.sort_values(
                ["Period_Start", "Strategy"]
            ).reset_index(drop=True)

        strategy_metrics = self.compute_strategy_risk_metrics(
            tournament_monthly,
            rf_symbol=cast(str, rf_symbol) if rf_symbol else "^IRX",
        )
        regime_metrics, regime_winners = self.compute_regime_strategy_metrics(
            tournament_monthly,
            regime_labels,
            min_months=output_cfg.min_months_per_bucket,
        )
        magic_gate_comparison = self._build_magic_gate_comparison(
            strategy_metrics,
            preserve_return_delta_floor=output_cfg.preserve_return_delta_floor,
        )

        outputs = {
            "regime_labels": regime_labels,
            "tournament_monthly_returns": tournament_monthly,
            "strategy_risk_metrics": strategy_metrics,
            "regime_strategy_metrics": regime_metrics,
            "regime_winners": regime_winners,
            "magic_gate_comparison": magic_gate_comparison,
        }

        out_dir = Path(output_cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        file_map = {
            "regime_labels": "regime_labels.csv",
            "tournament_monthly_returns": "tournament_monthly_returns.csv",
            "strategy_risk_metrics": "strategy_risk_metrics.csv",
            "regime_strategy_metrics": "regime_strategy_metrics.csv",
            "regime_winners": "regime_winners.csv",
            "magic_gate_comparison": "magic_gate_comparison.csv",
        }
        for key, filename in file_map.items():
            outputs[key].to_csv(out_dir / filename, index=False)

        return outputs

    def _classify_regime(
        self,
        date: str,
        return_window_days: int = 252,
        vol_window_days: int = 63,
    ) -> str:
        cache_key = (str(date), int(return_window_days), int(vol_window_days))
        if cache_key in self._regime_cache:
            return self._regime_cache[cache_key]

        if self.price_matrix is None or self.benchmark_symbol not in self.price_matrix:
            self._regime_cache[cache_key] = "NEUTRAL"
            return "NEUTRAL"

        target_dt = pd.to_datetime(date)
        bench = self.price_matrix[self.benchmark_symbol].dropna()
        bench = bench[bench.index <= target_dt]
        if len(bench) < 20:
            self._regime_cache[cache_key] = "NEUTRAL"
            return "NEUTRAL"

        lookback = bench.iloc[-min(return_window_days, len(bench)) :]
        trailing_return = (
            (lookback.iloc[-1] / lookback.iloc[0] - 1) if len(lookback) > 1 else 0
        )
        drawdown = (lookback.iloc[-1] / lookback.max() - 1) if len(lookback) > 0 else 0

        recent_rets = bench.pct_change().dropna().iloc[-min(vol_window_days, len(bench)) :]
        realized_vol = (
            float(recent_rets.std() * np.sqrt(252)) if len(recent_rets) > 1 else 0.0
        )

        if drawdown <= -0.20 or trailing_return <= -0.15:
            regime = "BEAR"
        elif drawdown <= -0.10 or realized_vol >= 0.35:
            regime = "STRESS"
        elif trailing_return >= 0.15 and drawdown > -0.10:
            regime = "BULL"
        else:
            regime = "NEUTRAL"

        self._regime_cache[cache_key] = regime
        return regime

    def _compute_momentum_state(
        self,
        symbol: str,
        date: str,
        ma_period: int = 200,
        precomputed: Optional[Dict[Tuple[str, str], Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        if precomputed is not None:
            pre_key = (symbol, str(date))
            if pre_key in precomputed:
                return dict(precomputed[pre_key])

        cache_key = (str(symbol), str(date), int(ma_period))
        if cache_key in self._momentum_cache:
            return dict(self._momentum_cache[cache_key])

        if self.price_matrix is None or symbol not in self.price_matrix.columns:
            result = {"above_ma": None, "ma_gap_pct": None, "price": np.nan, "ma": np.nan}
            self._momentum_cache[cache_key] = result
            return dict(result)

        target_dt = pd.to_datetime(date)
        series = self.price_matrix[symbol].dropna()
        series = series[series.index <= target_dt]
        if series.empty:
            result = {"above_ma": None, "ma_gap_pct": None, "price": np.nan, "ma": np.nan}
            self._momentum_cache[cache_key] = result
            return dict(result)

        price = float(series.iloc[-1])
        if len(series) < ma_period:
            result = {"above_ma": None, "ma_gap_pct": None, "price": price, "ma": np.nan}
            self._momentum_cache[cache_key] = result
            return dict(result)

        ma = float(series.iloc[-ma_period:].mean())
        if ma == 0:
            result = {"above_ma": None, "ma_gap_pct": None, "price": price, "ma": ma}
            self._momentum_cache[cache_key] = result
            return dict(result)
        gap = ((price - ma) / ma) * 100.0
        result = {"above_ma": price > ma, "ma_gap_pct": gap, "price": price, "ma": ma}
        self._momentum_cache[cache_key] = result
        return dict(result)

    def _compute_tail_state(
        self,
        symbol: str,
        date: str,
        lookback_days: int = 252,
        min_obs: int = 80,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        cache_key = (str(symbol), str(date), int(lookback_days), int(min_obs))
        if use_cache and cache_key in self._tail_state_cache:
            return dict(self._tail_state_cache[cache_key])

        if self.price_matrix is None or symbol not in self.price_matrix.columns:
            result = {"tail_bucket": "Unknown", "tail_a": np.nan, "tail_b": np.nan}
            if use_cache:
                self._tail_state_cache[cache_key] = result
            return dict(result)

        target_dt = pd.to_datetime(date)
        series = self.price_matrix[symbol].dropna()
        series = series[series.index <= target_dt]
        if len(series) < min_obs:
            result = {"tail_bucket": "Unknown", "tail_a": np.nan, "tail_b": np.nan}
            if use_cache:
                self._tail_state_cache[cache_key] = result
            return dict(result)

        returns = series.pct_change().dropna().iloc[-lookback_days:]
        if len(returns) < min_obs:
            result = {"tail_bucket": "Unknown", "tail_a": np.nan, "tail_b": np.nan}
            if use_cache:
                self._tail_state_cache[cache_key] = result
            return dict(result)

        kurt_fisher = _require_real(returns.kurtosis(), "kurtosis")
        skewness = _require_real(returns.skew(), "skewness")

        if kurt_fisher > 2.0:
            bucket = "Heavy"
        elif kurt_fisher < 0.0:
            bucket = "Thin"
        else:
            bucket = "Normal"

        # Lightweight shape proxies for reporting continuity.
        tail_a = 1.0 - np.clip(skewness, -1.5, 1.5) * 0.2
        tail_b = 1.0 + np.clip(skewness, -1.5, 1.5) * 0.2
        result = {"tail_bucket": bucket, "tail_a": tail_a, "tail_b": tail_b}
        if use_cache:
            self._tail_state_cache[cache_key] = result
        return dict(result)

    def _precompute_regimes(
        self, period_starts: List[str], use_precomputed_regimes: bool = True
    ) -> Dict[str, str]:
        if not use_precomputed_regimes:
            return {}
        return {start: self._classify_regime(start) for start in period_starts}

    def _precompute_momentum_states(
        self,
        symbols: List[str],
        period_starts: List[str],
        ma_period: int = 200,
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:
        table: Dict[Tuple[str, str], Dict[str, Any]] = {}
        if self.price_matrix is None or not symbols or not period_starts:
            return table

        cols = [s for s in symbols if s in self.price_matrix.columns]
        if not cols:
            return table

        prices = self.price_matrix[cols].sort_index()
        ma = prices.rolling(ma_period, min_periods=ma_period).mean()
        index = prices.index

        for date_str in period_starts:
            target = pd.to_datetime(date_str)
            pos = index.searchsorted(target, side="right") - 1
            if pos < 0:
                continue
            px_row = prices.iloc[pos]
            ma_row = ma.iloc[pos]
            for symbol in cols:
                px = px_row.get(symbol, np.nan)
                ma_val = ma_row.get(symbol, np.nan)
                if pd.isna(px):
                    state = {
                        "above_ma": None,
                        "ma_gap_pct": None,
                        "price": np.nan,
                        "ma": np.nan,
                    }
                elif pd.isna(ma_val) or ma_val == 0:
                    state = {
                        "above_ma": None,
                        "ma_gap_pct": None,
                        "price": float(px),
                        "ma": np.nan if pd.isna(ma_val) else float(ma_val),
                    }
                else:
                    gap = ((float(px) - float(ma_val)) / float(ma_val)) * 100.0
                    state = {
                        "above_ma": bool(float(px) > float(ma_val)),
                        "ma_gap_pct": gap,
                        "price": float(px),
                        "ma": float(ma_val),
                    }
                table[(symbol, date_str)] = state
        return table

    def _compute_altman_state(
        self, row: pd.Series, current_price: float
    ) -> Tuple[float, str]:
        total_assets = row.get("TotalAssets")
        total_liab = row.get("TotalLiabilitiesNetMinorityInterest")
        working_capital = row.get("WorkingCapital")
        retained_earnings = row.get("RetainedEarnings")
        ebit = row.get("EBIT")
        revenue = row.get("TotalRevenue")
        shares = row.get("OrdinarySharesNumber")

        fields = [
            total_assets,
            total_liab,
            working_capital,
            retained_earnings,
            ebit,
            revenue,
            shares,
        ]
        if any(pd.isna(v) for v in fields):
            return (np.nan, "Unknown")
        total_assets_f = _require_real(total_assets, "TotalAssets")
        total_liab_f = _require_real(
            total_liab, "TotalLiabilitiesNetMinorityInterest"
        )
        working_capital_f = _require_real(working_capital, "WorkingCapital")
        retained_earnings_f = _require_real(retained_earnings, "RetainedEarnings")
        ebit_f = _require_real(ebit, "EBIT")
        revenue_f = _require_real(revenue, "TotalRevenue")
        shares_f = _require_real(shares, "OrdinarySharesNumber")

        if total_assets_f <= 0 or total_liab_f <= 0:
            return (np.nan, "Unknown")
        if pd.isna(current_price) or current_price <= 0:
            return (np.nan, "Unknown")

        market_cap = float(current_price) * shares_f
        z_score = (
            1.2 * (working_capital_f / total_assets_f)
            + 1.4 * (retained_earnings_f / total_assets_f)
            + 3.3 * (ebit_f / total_assets_f)
            + 0.6 * (market_cap / total_liab_f)
            + 1.0 * (revenue_f / total_assets_f)
        )
        if z_score < 1.81:
            bucket = "DISTRESS (Risk)"
        elif z_score < 2.99:
            bucket = "GREY ZONE"
        else:
            bucket = "SAFE"
        return (z_score, bucket)

    def _compute_valuation_state(
        self, row: pd.Series, strategy: str, current_price: float
    ) -> Dict[str, Any]:
        shares = row.get("OrdinarySharesNumber")
        if pd.isna(current_price) or current_price <= 0 or pd.isna(shares) or shares <= 0:
            return {
                "Fair Value (Bear)": np.nan,
                "Fair Value (Base)": np.nan,
                "Fair Value": np.nan,
                "Valuation Sanity": "Unknown",
            }

        operating_cf = row.get("OperatingCashFlow")
        capex = row.get("CapitalExpenditure")
        revenue = row.get("TotalRevenue")
        equity = row.get("StockholdersEquity")

        fcf = 0.0
        if pd.notna(operating_cf) and pd.notna(capex):
            fcf = float(operating_cf) - abs(float(capex))

        sales = float(revenue) if pd.notna(revenue) else 0.0
        book = float(equity) if pd.notna(equity) else 0.0

        fcf_per_share = fcf / float(shares)
        sales_per_share = sales / float(shares) if sales > 0 else 0.0
        book_per_share = book / float(shares) if book > 0 else 0.0

        scenario = calculate_scenario_valuations(
            fcf_per_share=fcf_per_share,
            sales_per_share=sales_per_share,
            book_value_per_share=book_per_share,
            strategy=strategy,
        )
        fair_value = float(scenario["prob_weighted_value"])
        sanity_passed, sanity_reason = check_valuation_sanity(
            fair_value=fair_value,
            current_price=float(current_price),
        )

        return {
            "Fair Value (Bear)": float(scenario["bear_value"]),
            "Fair Value (Base)": float(scenario["base_value"]),
            "Fair Value": fair_value,
            "Valuation Sanity": "Passed" if sanity_passed else sanity_reason,
        }

    def _build_asset_panel_for_period(
        self,
        start_date: str,
        end_date: str,
        strategy: str,
        top_n: int,
        regime: str,
        risk_config: RiskVectorConfig,
        snapshot: Optional[pd.DataFrame] = None,
        momentum_table: Optional[Dict[Tuple[str, str], Dict[str, Any]]] = None,
        use_cached_tail: bool = True,
        profile_name: Optional[str] = None,
        config_path: Optional[str] = None,
    ) -> pd.DataFrame:
        if snapshot is None:
            snapshot = self.get_valid_snapshot(start_date)
        if snapshot.empty:
            return pd.DataFrame()

        if profile_name is None and config_path is None:
            picks = self.apply_strategy(snapshot, strategy, top_n, trade_date=start_date)
        else:
            picks = self.apply_strategy(
                snapshot,
                strategy,
                top_n,
                trade_date=start_date,
                profile_name=profile_name,
                config_path=config_path,
            )
        if not picks:
            return pd.DataFrame()

        entry_prices = self.get_prices_bulk(picks, start_date)
        exit_prices = self.get_prices_bulk(picks, end_date)
        rows: List[Dict[str, Any]] = []

        for symbol in picks:
            if symbol not in snapshot.index:
                continue
            row = snapshot.loc[symbol]
            entry_px = entry_prices.get(symbol, np.nan)
            exit_px = exit_prices.get(symbol, np.nan)
            ret = (
                (exit_px - entry_px) / entry_px
                if pd.notna(entry_px) and pd.notna(exit_px) and entry_px > 0
                else np.nan
            )

            momentum = self._compute_momentum_state(
                symbol, start_date, precomputed=momentum_table
            )
            tail = self._compute_tail_state(
                symbol, start_date, use_cache=use_cached_tail
            )
            altman_z, distress = self._compute_altman_state(row, float(entry_px))
            valuation = self._compute_valuation_state(row, strategy, float(entry_px))

            rows.append(
                {
                    "symbol": symbol,
                    "strategy": strategy,
                    "start_date": start_date,
                    "end_date": end_date,
                    "regime": regime,
                    "entry_price": entry_px,
                    "exit_price": exit_px,
                    "return": ret,
                    "Altman Z-Score": altman_z,
                    "Distress Risk": distress,
                    "Tail_Risk": tail["tail_bucket"],
                    "Tail_a": tail["tail_a"],
                    "Tail_b": tail["tail_b"],
                    "Momentum_Above_MA": momentum["above_ma"],
                    "Momentum_MA_Gap_Pct": momentum["ma_gap_pct"],
                    "Momentum_Price": momentum["price"],
                    "Momentum_MA_200": momentum["ma"],
                    "Current Price": entry_px,
                    "Fair Value (Bear)": valuation["Fair Value (Bear)"],
                    "Fair Value (Base)": valuation["Fair Value (Base)"],
                    "Fair Value": valuation["Fair Value"],
                    "Valuation Sanity": valuation["Valuation Sanity"],
                }
            )

        if not rows:
            return pd.DataFrame()

        panel = pd.DataFrame(rows)
        panel = attach_risk_vectors(panel, config=risk_config, include_signal=True)
        return panel

    def run_research_backtest(
        self,
        start_date: str,
        end_date: str,
        strategy: str = "magic_formula",
        top_n: int = 10,
        holding_months: int = 1,
        risk_config: Optional[RiskVectorConfig] = None,
        verbose: bool = True,
        show_progress: bool = True,
        max_workers: Optional[int] = None,
        use_cached_tail: bool = True,
        use_precomputed_regimes: bool = True,
        profile_name: Optional[str] = None,
        config_path: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Run monthly rolling research backtests and return query-ready panels.
        """
        if self.price_matrix is None or self.financial_data is None:
            self.load_data()

        periods = self._generate_monthly_periods(start_date, end_date, holding_months)
        if not periods:
            empty = pd.DataFrame()
            self._last_research_outputs = {
                "asset_panel": empty,
                "portfolio_panel": empty,
                "equity_panel": empty,
            }
            return self._last_research_outputs

        cfg = risk_config or RiskVectorConfig()
        workers = (
            int(max_workers)
            if max_workers is not None and int(max_workers) > 0
            else min(8, os.cpu_count() or 4)
        )
        asset_frames: List[pd.DataFrame] = []
        portfolio_rows: List[Dict[str, Any]] = []
        equity_rows: List[Dict[str, Any]] = []
        stage_times: Dict[str, float] = {}
        run_start_ts = time.perf_counter()

        period_starts = [start for start, _ in periods]
        all_symbols = (
            self.price_matrix.columns.tolist() if self.price_matrix is not None else []
        )

        if verbose:
            print(
                f"[research] periods={len(periods)} workers={workers} cached_tail={use_cached_tail} precomputed_regimes={use_precomputed_regimes}"
            )
            if workers >= 8:
                print(
                    "[research] warning: high worker counts can increase SQLite contention."
                )

        t0 = time.perf_counter()
        snapshot_map: Dict[str, pd.DataFrame] = {}
        if workers > 1 and len(period_starts) > 1:
            with ThreadPoolExecutor(max_workers=min(workers, len(period_starts))) as executor:
                futures = {
                    executor.submit(self.get_valid_snapshot, start): start
                    for start in period_starts
                }
                for future in as_completed(futures):
                    snapshot_map[futures[future]] = future.result()
        else:
            for start in period_starts:
                snapshot_map[start] = self.get_valid_snapshot(start)
        stage_times["snapshot_build"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        regime_map = self._precompute_regimes(
            period_starts, use_precomputed_regimes=use_precomputed_regimes
        )
        stage_times["regime_precompute"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        momentum_table = self._precompute_momentum_states(all_symbols, period_starts)
        stage_times["momentum_precompute"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        benchmark_return_map: Dict[Tuple[str, str], float] = {}
        for period_start, period_end in periods:
            benchmark_entry = self.get_price(self.benchmark_symbol, period_start)
            benchmark_exit = self.get_price(self.benchmark_symbol, period_end)
            if (
                pd.notna(benchmark_entry)
                and pd.notna(benchmark_exit)
                and benchmark_entry > 0
            ):
                benchmark_return_map[(period_start, period_end)] = (
                    benchmark_exit - benchmark_entry
                ) / benchmark_entry
            else:
                benchmark_return_map[(period_start, period_end)] = np.nan
        stage_times["benchmark_returns"] = time.perf_counter() - t0

        def _build_period_panel(period: Tuple[str, str]) -> Tuple[str, str, str, pd.DataFrame]:
            p_start, p_end = period
            regime = regime_map.get(p_start) or self._classify_regime(p_start)
            snap = snapshot_map.get(p_start, pd.DataFrame())
            panel = self._build_asset_panel_for_period(
                start_date=p_start,
                end_date=p_end,
                strategy=strategy,
                top_n=top_n,
                regime=regime,
                risk_config=cfg,
                snapshot=snap,
                momentum_table=momentum_table,
                use_cached_tail=use_cached_tail,
                profile_name=profile_name,
                config_path=config_path,
            )
            return (p_start, p_end, regime, panel)

        t0 = time.perf_counter()
        period_panel_map: Dict[Tuple[str, str], Tuple[str, pd.DataFrame]] = {}
        if workers > 1 and len(periods) > 1:
            with ThreadPoolExecutor(max_workers=min(workers, len(periods))) as executor:
                futures = {executor.submit(_build_period_panel, p): p for p in periods}
                for future in as_completed(futures):
                    p_start, p_end, regime, panel = future.result()
                    period_panel_map[(p_start, p_end)] = (regime, panel)
        else:
            for period in periods:
                p_start, p_end, regime, panel = _build_period_panel(period)
                period_panel_map[(p_start, p_end)] = (regime, panel)
        stage_times["asset_panel_build"] = time.perf_counter() - t0

        progress = tqdm(
            total=len(periods) * 2,
            desc=f"Research {strategy}",
            unit="period",
            disable=not show_progress,
        )

        for track in ("ungated", "gated"):
            equity = 1.0
            peak = 1.0
            for period_start, period_end in periods:
                regime, base_panel = period_panel_map.get(
                    (period_start, period_end), ("NEUTRAL", pd.DataFrame())
                )

                if not base_panel.empty:
                    asset_panel = base_panel.copy()
                    asset_panel["track"] = track
                    if track == "gated":
                        asset_panel["SelectedForPortfolio"] = (
                            asset_panel["RV_Gate_Count"] == 0
                        )
                    else:
                        asset_panel["SelectedForPortfolio"] = True
                    asset_frames.append(asset_panel)

                    selected = asset_panel[asset_panel["SelectedForPortfolio"] == True]
                    portfolio_return = (
                        selected["return"].mean() if not selected.empty else np.nan
                    )
                    selected_count = int(selected["symbol"].nunique())
                    candidate_count = int(asset_panel["symbol"].nunique())
                    gate_trip_rate = float((asset_panel["RV_Gate_Count"] > 0).mean())
                else:
                    portfolio_return = np.nan
                    selected_count = 0
                    candidate_count = 0
                    gate_trip_rate = np.nan

                benchmark_return = benchmark_return_map.get(
                    (period_start, period_end), np.nan
                )
                alpha = (
                    portfolio_return - benchmark_return
                    if pd.notna(portfolio_return) and pd.notna(benchmark_return)
                    else np.nan
                )

                portfolio_rows.append(
                    {
                        "track": track,
                        "strategy": strategy,
                        "period_start": period_start,
                        "period_end": period_end,
                        "regime": regime,
                        "candidate_count": candidate_count,
                        "selected_count": selected_count,
                        "portfolio_return": portfolio_return,
                        "benchmark_return": benchmark_return,
                        "alpha": alpha,
                        "gate_trip_rate": gate_trip_rate,
                    }
                )

                if pd.notna(portfolio_return):
                    equity *= 1.0 + float(portfolio_return)
                peak = max(peak, equity)
                drawdown = (equity / peak) - 1 if peak > 0 else 0.0
                equity_rows.append(
                    {
                        "track": track,
                        "strategy": strategy,
                        "period_start": period_start,
                        "period_end": period_end,
                        "regime": regime,
                        "equity": equity,
                        "drawdown": drawdown,
                        "portfolio_return": portfolio_return,
                    }
                )

                if verbose:
                    print(
                        f"[research] {track} {period_start}->{period_end}: return={portfolio_return:+.2%}"
                        if pd.notna(portfolio_return)
                        else f"[research] {track} {period_start}->{period_end}: return=N/A"
                    )
                progress.set_postfix({"track": track, "regime": regime})
                progress.update(1)

        progress.close()

        asset_panel_df = (
            pd.concat(asset_frames, ignore_index=True) if asset_frames else pd.DataFrame()
        )
        portfolio_panel_df = pd.DataFrame(portfolio_rows)
        equity_panel_df = pd.DataFrame(equity_rows)

        outputs = {
            "asset_panel": asset_panel_df,
            "portfolio_panel": portfolio_panel_df,
            "equity_panel": equity_panel_df,
        }
        self._last_research_outputs = outputs

        stage_times["total"] = time.perf_counter() - run_start_ts
        if verbose:
            stage_msg = ", ".join(f"{k}={v:.2f}s" for k, v in stage_times.items())
            print(f"[research] timings: {stage_msg}")

        return outputs

    def query_signal_stability(
        self, asset_panel: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Aggregate signal performance across regimes."""
        if asset_panel is None:
            asset_panel = (
                self._last_research_outputs.get("asset_panel")
                if self._last_research_outputs
                else None
            )
        if asset_panel is None or asset_panel.empty:
            return pd.DataFrame()

        panel = asset_panel.dropna(subset=["return"]).copy()
        if panel.empty:
            return pd.DataFrame()

        summary = (
            panel.groupby(["track", "strategy", "regime", "Investment Signal"])
            .agg(
                observations=("symbol", "count"),
                mean_return=("return", "mean"),
                median_return=("return", "median"),
                std_return=("return", "std"),
                hit_rate=("return", lambda x: float((x > 0).mean())),
                gate_trip_rate=("RV_Gate_Count", lambda x: float((x > 0).mean())),
            )
            .reset_index()
            .sort_values(["track", "strategy", "regime", "observations"], ascending=[True, True, True, False])
        )
        return summary

    def query_gate_trip_rates(
        self, asset_panel: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Report gate trip counts/rates by track, strategy, and regime."""
        if asset_panel is None:
            asset_panel = (
                self._last_research_outputs.get("asset_panel")
                if self._last_research_outputs
                else None
            )
        if asset_panel is None or asset_panel.empty:
            return pd.DataFrame()

        gate_cols = [
            "RV_Gate_Distress",
            "RV_Gate_Tail",
            "RV_Gate_Momentum",
            "RV_Gate_Valuation",
        ]
        rows: List[Dict[str, Any]] = []
        grouped = asset_panel.groupby(["track", "strategy", "regime"])
        for (track, strategy, regime), grp in grouped:
            total = len(grp)
            if total == 0:
                continue
            for gate_col in gate_cols:
                trips = int(grp[gate_col].fillna(False).astype(bool).sum())
                rows.append(
                    {
                        "track": track,
                        "strategy": strategy,
                        "regime": regime,
                        "gate": gate_col.replace("RV_Gate_", "").lower(),
                        "trips": trips,
                        "candidates": total,
                        "trip_rate": trips / total,
                    }
                )
        return pd.DataFrame(rows)

    def query_drawdown_failures(
        self,
        equity_panel: Optional[pd.DataFrame] = None,
        threshold: float = -0.15,
    ) -> pd.DataFrame:
        """Return drawdown episodes where drawdown breaches the threshold."""
        if equity_panel is None:
            equity_panel = (
                self._last_research_outputs.get("equity_panel")
                if self._last_research_outputs
                else None
            )
        if equity_panel is None or equity_panel.empty:
            return pd.DataFrame()

        events: List[Dict[str, Any]] = []
        for (track, strategy), grp in equity_panel.groupby(["track", "strategy"]):
            grp = grp.sort_values("period_end").reset_index(drop=True)
            in_event = False
            start_idx = -1
            trough_idx = -1
            trough_dd = 0.0

            for idx_i in range(len(grp)):
                row = grp.iloc[idx_i]
                dd = float(row["drawdown"]) if pd.notna(row["drawdown"]) else 0.0
                if not in_event and dd <= threshold:
                    in_event = True
                    start_idx = idx_i
                    trough_idx = idx_i
                    trough_dd = dd
                if in_event and dd < trough_dd:
                    trough_dd = dd
                    trough_idx = idx_i
                if in_event and dd > threshold:
                    start_row = grp.iloc[start_idx]
                    trough_row = grp.iloc[trough_idx]
                    recovery_date = row["period_end"]
                    end_date = (
                        grp.iloc[idx_i - 1]["period_end"]
                        if idx_i > 0
                        else row["period_end"]
                    )
                    events.append(
                        {
                            "track": track,
                            "strategy": strategy,
                            "start_date": start_row["period_start"],
                            "regime_at_start": start_row["regime"],
                            "trough_date": trough_row["period_end"],
                            "min_drawdown": trough_dd,
                            "end_date": end_date,
                            "recovery_date": recovery_date,
                            "duration_periods": int(idx_i - start_idx),
                        }
                    )
                    in_event = False

            if in_event:
                start_row = grp.iloc[start_idx]
                trough_row = grp.iloc[trough_idx]
                events.append(
                    {
                        "track": track,
                        "strategy": strategy,
                        "start_date": start_row["period_start"],
                        "regime_at_start": start_row["regime"],
                        "trough_date": trough_row["period_end"],
                        "min_drawdown": trough_dd,
                        "end_date": grp.iloc[-1]["period_end"],
                        "recovery_date": None,
                        "duration_periods": int(len(grp) - start_idx),
                    }
                )

        return pd.DataFrame(events)

    def export_research_results(
        self, results: Dict[str, pd.DataFrame], output_dir: str
    ) -> Dict[str, str]:
        """Export research result panels to CSV files."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        exported: Dict[str, str] = {}
        for name, df in results.items():
            file_path = out_dir / f"{name}.csv"
            df.to_csv(file_path, index=False)
            exported[name] = str(file_path)
        return exported


def _get_period_benchmark_return(
    bt: VectorBacktester, symbol: str, start_date: str, end_date: str
) -> float:
    entry = bt.get_price(symbol, start_date)
    exit_ = bt.get_price(symbol, end_date)
    if pd.notna(entry) and pd.notna(exit_) and entry > 0:
        return float((exit_ - entry) / entry)
    return np.nan


def _format_pct_or_na(value: float, width: int = 10) -> str:
    if pd.isna(value):
        return f"{'N/A':>{width}}"
    return f"{value * 100:>{width}.2f}%"


def _format_pick_count(value: Optional[int], width: int = 5) -> str:
    if value is None:
        return f"{'-':>{width}}"
    return f"{int(value):>{width}d}"


def _nanmean(values: List[float]) -> float:
    valid = [float(v) for v in values if pd.notna(v)]
    if not valid:
        return np.nan
    return float(np.mean(valid))


def _format_tournament_row(
    name: str,
    returns: List[float],
    picks: List[Optional[int]],
    avg_return: Optional[float] = None,
) -> str:
    avg_val = _nanmean(returns) if avg_return is None else avg_return
    return (
        f"{name:<30} | {_format_pct_or_na(returns[0])} | {_format_pick_count(picks[0])} | "
        f"{_format_pct_or_na(returns[1])} | {_format_pick_count(picks[1])} | "
        f"{_format_pct_or_na(returns[2])} | {_format_pick_count(picks[2])} | "
        f"{_format_pct_or_na(avg_val)}"
    )


def _investigate_magic_formula_weakness(
    bt: VectorBacktester,
    start_date: str,
    end_date: str,
    top_n: int = 10,
    profile_name: Optional[str] = None,
    config_path: Optional[str] = None,
) -> None:
    """Print focused diagnostics for Magic Formula in a specific period."""
    print(f"\n{'='*110}")
    print(f"  MAGIC FORMULA DIAGNOSTIC ({start_date} -> {end_date})")
    print(f"{'='*110}")

    snapshot = bt.get_valid_snapshot(start_date)
    if snapshot.empty:
        print("  No point-in-time snapshot available for the diagnostic window.")
        return

    profile = bt._resolve_profile(profile_name=profile_name, config_path=config_path)

    initial_count = int(len(snapshot))
    universe_df = bt.filter_universe(
        snapshot.copy(),
        start_date,
        min_mkt_cap_mm=float(profile.universe.strategy_min_market_cap_mm),
        profile_name=profile_name,
        config_path=config_path,
    )
    universe_count = int(len(universe_df))

    df = universe_df.copy()
    if "MarketCap" not in df.columns:
        df["MarketCap"] = 0.0

    market_cap_positive_count = int((df["MarketCap"] > 0).sum())

    if "EnterpriseValue" not in df.columns:
        ev_cols = ["TotalDebt", "CashAndCashEquivalents", "MarketCap"]
        if all(c in df.columns for c in ev_cols):
            df = df[df["MarketCap"] > 0].copy()
            df["EnterpriseValue"] = (
                (df["MarketCap"] * 1_000_000)
                + df["TotalDebt"]
                - df["CashAndCashEquivalents"]
            )

    required = ["EBIT", "InvestedCapital", "EnterpriseValue"]
    missing_required = [c for c in required if c not in df.columns]
    if missing_required:
        print(
            f"  Missing required columns for Magic Formula in this window: {missing_required}"
        )
        print(
            f"  Attrition so far: snapshot={initial_count}, post_universe_filter={universe_count}"
        )
        return

    eligible = df.dropna(subset=required).copy()
    post_dropna_count = int(len(eligible))

    eligible = eligible[eligible["InvestedCapital"] > 0].copy()
    post_invested_capital_count = int(len(eligible))

    eligible = eligible[eligible["EnterpriseValue"] > 0].copy()
    post_enterprise_value_count = int(len(eligible))

    if eligible.empty:
        print("  No eligible names remained after Magic Formula filters.")
        print(
            "  Counts: "
            f"snapshot={initial_count}, "
            f"post_universe_filter={universe_count}, "
            f"marketcap_gt0={market_cap_positive_count}, "
            f"post_required_non_null={post_dropna_count}, "
            f"post_invested_capital_filter={post_invested_capital_count}, "
            f"post_enterprise_value_filter={post_enterprise_value_count}"
        )
        return

    eligible["ROIC"] = eligible["EBIT"] / eligible["InvestedCapital"]
    eligible["EarningsYield"] = eligible["EBIT"] / eligible["EnterpriseValue"]
    eligible["Quality_Rank"] = robust_score(eligible["ROIC"], higher_is_better=True)
    eligible["Value_Rank"] = robust_score(
        eligible["EarningsYield"], higher_is_better=True
    )
    eligible["Magic_Score"] = eligible["Quality_Rank"] + eligible["Value_Rank"]

    picks = eligible.nlargest(top_n, ["Magic_Score", "MarketCap"]).index.tolist()
    pick_returns = bt.calculate_portfolio_return(picks, start_date, end_date)
    pick_results = cast(pd.DataFrame, pick_returns["results_df"])
    priced_count = int(len(pick_results))
    if len(picks) == 0:
        status = "no_selections"
    elif priced_count == 0:
        status = "no_valid_prices"
    else:
        status = "ok"
    portfolio_return = pick_returns["portfolio_return"]
    qqq_return = _get_period_benchmark_return(bt, "QQQ", start_date, end_date)
    spy_return = _get_period_benchmark_return(bt, "SPY", start_date, end_date)

    print("  Universe attrition:")
    print(
        "    "
        f"snapshot={initial_count}, "
        f"post_universe_filter={universe_count}, "
        f"marketcap_gt0={market_cap_positive_count}, "
        f"post_required_non_null={post_dropna_count}, "
        f"post_invested_capital_filter={post_invested_capital_count}, "
        f"post_enterprise_value_filter={post_enterprise_value_count}"
    )
    print(
        "  Selection summary: "
        f"picked={len(picks)} | priced={priced_count} | status={status}"
    )
    print(f"  Picks: {picks if picks else 'None'}")
    print(
        "  Returns: "
        f"Magic={_format_pct_or_na(portfolio_return, width=8).strip()} | "
        f"QQQ={_format_pct_or_na(qqq_return, width=8).strip()} | "
        f"SPY={_format_pct_or_na(spy_return, width=8).strip()}"
    )
    if pd.notna(portfolio_return) and pd.notna(qqq_return):
        print(f"  Alpha vs QQQ: {(portfolio_return - qqq_return) * 100:+.2f}%")
    if pd.notna(portfolio_return) and pd.notna(spy_return):
        print(f"  Alpha vs SPY: {(portfolio_return - spy_return) * 100:+.2f}%")

    if not pick_results.empty:
        ranked = pick_results.sort_values("return")
        print("  Per-pick returns (worst -> best):")
        for _, row in ranked.iterrows():
            symbol = row.get("symbol", "")
            ret = row.get("return", np.nan)
            print(f"    {symbol:<8} {_format_pct_or_na(ret, width=8).strip()}")
    else:
        print("  No valid per-pick price pairs available for diagnostic ranking.")


def _default_tournament_contenders() -> List[Dict[str, Any]]:
    return deepcopy(DEFAULT_TOURNAMENT_CONTENDERS)


def run_strategy_tournament(
    show_progress: bool = True,
    verbose_combo: bool = False,
    profile_name: Optional[str] = None,
    config_path: Optional[str] = None,
):
    """Compatibility wrapper that delegates to the regime-aware tournament path."""
    bt = VectorBacktester(
        reporting_lag_days=90,
        profile_name=profile_name,
        config_path=config_path,
    )
    bt.load_data()
    if verbose_combo:
        print("[tournament] verbose_combo is ignored in regime-aware tournament mode.")
    return bt.run_regime_tournament(
        show_progress=show_progress,
        profile_name=profile_name,
        config_path=config_path,
    )


if __name__ == "__main__":
    run_strategy_tournament()
