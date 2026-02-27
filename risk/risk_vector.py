"""Unified risk vector model and helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import pandas as pd


UNKNOWN = "Unknown"


@dataclass
class RiskVectorConfig:
    """Configuration for risk-vector gate behavior."""

    version: str = "v1"
    distress_safe_bucket: str = "SAFE"
    tail_heavy_bucket: str = "Heavy"
    enable_distress_gate: bool = True
    enable_tail_gate: bool = True
    enable_momentum_gate: bool = True
    enable_valuation_gate: bool = True


@dataclass
class RiskVector:
    """Canonical per-asset risk profile."""

    version: str
    altman_z_score: Optional[float]
    distress_bucket: str
    tail_bucket: str
    tail_a: Optional[float]
    tail_b: Optional[float]
    momentum_above_ma200: Optional[bool]
    momentum_ma_gap_pct: Optional[float]
    momentum_regime: str
    valuation_sanity_passed: Optional[bool]
    valuation_sanity_ratio: Optional[float]
    valuation_sanity_reason: str
    gate_distress: bool
    gate_tail: bool
    gate_momentum: bool
    gate_valuation: bool
    gate_count: int


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    text = str(value).strip()
    if text == "":
        return None
    if text.endswith("%"):
        text = text[:-1]
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _to_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _normalize_distress_bucket(raw: Any) -> str:
    if raw is None:
        return UNKNOWN
    text = str(raw).strip()
    if text == "" or text.lower() == "nan":
        return UNKNOWN
    upper = text.upper()
    if "SAFE" in upper:
        return "SAFE"
    if "GREY" in upper:
        return "GREY ZONE"
    if "DISTRESS" in upper:
        return "DISTRESS (Risk)"
    return UNKNOWN


def _normalize_tail_bucket(raw: Any) -> str:
    if raw is None:
        return UNKNOWN
    text = str(raw).strip()
    if text == "" or text.lower() == "nan":
        return UNKNOWN
    low = text.lower()
    if low.startswith("heavy"):
        return "Heavy"
    if low.startswith("normal"):
        return "Normal"
    if low.startswith("thin"):
        return "Thin"
    return UNKNOWN


def _classify_momentum_regime(
    above_ma: Optional[bool], gap_pct: Optional[float]
) -> str:
    if above_ma is None:
        return UNKNOWN
    if gap_pct is None:
        return "BULLISH" if above_ma else "BEARISH"
    if above_ma and gap_pct >= 3:
        return "BULLISH"
    if (not above_ma) and gap_pct <= -3:
        return "BEARISH"
    return "NEUTRAL"


def _valuation_sanity_from_row(row: pd.Series) -> Tuple[Optional[bool], str]:
    raw_status = row.get("Valuation Sanity")
    if raw_status is None:
        return (None, UNKNOWN)
    try:
        if pd.isna(raw_status):
            return (None, UNKNOWN)
    except Exception:
        pass
    text = str(raw_status).strip()
    if text == "":
        return (None, UNKNOWN)
    if text.lower() == "passed":
        return (True, "Passed")
    return (False, text)


def _valuation_ratio_from_row(row: pd.Series) -> Optional[float]:
    fair = _to_float(row.get("Fair Value"))
    if fair is None:
        fair = _to_float(row.get("Fair Value (Base)"))
    price = _to_float(row.get("Current Price"))
    if price is None:
        price = _to_float(row.get("Price"))
    if fair is None or price is None or price <= 0:
        return None
    return fair / price


def build_risk_vector(
    row: pd.Series, config: Optional[RiskVectorConfig] = None
) -> RiskVector:
    """Build a typed risk vector from a mixed source row."""
    config = config or RiskVectorConfig()

    altman_z = _to_float(row.get("Altman Z-Score"))
    distress_bucket = _normalize_distress_bucket(row.get("Distress Risk"))
    tail_bucket = _normalize_tail_bucket(row.get("Tail_Risk"))
    tail_a = _to_float(row.get("Tail_a"))
    if tail_a is None:
        tail_a = _to_float(row.get("a_shape"))
    tail_b = _to_float(row.get("Tail_b"))
    if tail_b is None:
        tail_b = _to_float(row.get("b_shape"))

    above_ma = _to_bool(row.get("Momentum_Above_MA"))
    if above_ma is None:
        above_ma = _to_bool(row.get("Above_MA"))
    ma_gap_pct = _to_float(row.get("Momentum_MA_Gap_Pct"))
    if ma_gap_pct is None:
        px = _to_float(row.get("Momentum_Price"))
        ma = _to_float(row.get("Momentum_MA_200"))
        if px is not None and ma is not None and ma != 0:
            ma_gap_pct = ((px - ma) / ma) * 100.0
    momentum_regime = _classify_momentum_regime(above_ma, ma_gap_pct)

    valuation_passed, valuation_reason = _valuation_sanity_from_row(row)
    valuation_ratio = _valuation_ratio_from_row(row)

    gate_distress = config.enable_distress_gate and (
        distress_bucket != config.distress_safe_bucket
    )
    gate_tail = config.enable_tail_gate and (tail_bucket == config.tail_heavy_bucket)
    gate_momentum = config.enable_momentum_gate and (above_ma is not True)
    gate_valuation = config.enable_valuation_gate and (valuation_passed is not True)
    gate_count = int(gate_distress) + int(gate_tail) + int(gate_momentum) + int(
        gate_valuation
    )

    return RiskVector(
        version=config.version,
        altman_z_score=altman_z,
        distress_bucket=distress_bucket,
        tail_bucket=tail_bucket,
        tail_a=tail_a,
        tail_b=tail_b,
        momentum_above_ma200=above_ma,
        momentum_ma_gap_pct=ma_gap_pct,
        momentum_regime=momentum_regime,
        valuation_sanity_passed=valuation_passed,
        valuation_sanity_ratio=valuation_ratio,
        valuation_sanity_reason=valuation_reason,
        gate_distress=gate_distress,
        gate_tail=gate_tail,
        gate_momentum=gate_momentum,
        gate_valuation=gate_valuation,
        gate_count=gate_count,
    )


def gate_stage_and_reason(rv: RiskVector) -> Tuple[str, str]:
    """Map gate state to a deterministic decision stage and reason."""
    if rv.gate_distress:
        return ("distress", f"Distress gate tripped ({rv.distress_bucket})")
    if rv.gate_valuation:
        return ("valuation_sanity", f"Valuation sanity gate tripped ({rv.valuation_sanity_reason})")
    if rv.gate_tail:
        return ("tail_risk", f"Tail risk gate tripped ({rv.tail_bucket})")
    if rv.gate_momentum:
        return ("momentum", "Momentum gate tripped (not above MA200)")
    return ("selected", "")


def derive_investment_signal_from_vector(
    rv: RiskVector,
    current_price: Optional[float],
    bear_fv: Optional[float],
    base_fv: Optional[float],
) -> Tuple[str, str]:
    """Derive compatible investment signal from vector components."""
    if rv.gate_valuation:
        return ("Needs Review", "Failed valuation sanity checks")
    if rv.gate_distress:
        return ("Avoid", f"Bankruptcy risk: {rv.distress_bucket}")
    if rv.gate_tail:
        return ("Caution", "Heavy tail risk - high volatility")
    if rv.gate_momentum:
        return ("Caution", "Momentum gate tripped")

    if (
        current_price is None
        or base_fv is None
        or current_price <= 0
        or base_fv <= 0
    ):
        return ("Needs Review", "Invalid price data")

    base_mos = (base_fv - current_price) / current_price
    bear_protected = (
        bear_fv is not None and current_price > 0 and bear_fv >= current_price
    )

    if bear_protected and base_mos >= 0.20:
        return ("Strong Buy", f"Bear-protected, {base_mos*100:.0f}% MOS")
    if base_mos >= 0.20:
        downside = (
            ((current_price - bear_fv) / current_price * 100)
            if bear_fv is not None
            else 0
        )
        return (
            "Speculative Buy",
            f"{base_mos*100:.0f}% MOS, {downside:.0f}% bear downside",
        )
    if base_mos > 0:
        return ("Hold", f"Modest {base_mos*100:.0f}% MOS")
    return ("Overvalued", f"Negative MOS ({base_mos*100:.0f}%)")


def risk_vector_to_columns(rv: RiskVector, prefix: str = "RV_") -> dict[str, Any]:
    """Flatten a RiskVector into deterministic DataFrame columns."""
    return {
        f"{prefix}Version": rv.version,
        f"{prefix}Altman_Z": rv.altman_z_score,
        f"{prefix}Distress_Bucket": rv.distress_bucket,
        f"{prefix}Tail_Bucket": rv.tail_bucket,
        f"{prefix}Tail_a": rv.tail_a,
        f"{prefix}Tail_b": rv.tail_b,
        f"{prefix}Momentum_Above_MA200": rv.momentum_above_ma200,
        f"{prefix}Momentum_MA_Gap_Pct": rv.momentum_ma_gap_pct,
        f"{prefix}Momentum_Regime": rv.momentum_regime,
        f"{prefix}Valuation_Sanity_Passed": rv.valuation_sanity_passed,
        f"{prefix}Valuation_Sanity_Ratio": rv.valuation_sanity_ratio,
        f"{prefix}Valuation_Sanity_Reason": rv.valuation_sanity_reason,
        f"{prefix}Gate_Distress": rv.gate_distress,
        f"{prefix}Gate_Tail": rv.gate_tail,
        f"{prefix}Gate_Momentum": rv.gate_momentum,
        f"{prefix}Gate_Valuation": rv.gate_valuation,
        f"{prefix}Gate_Count": rv.gate_count,
    }


def attach_risk_vectors(
    df: pd.DataFrame,
    config: Optional[RiskVectorConfig] = None,
    include_signal: bool = True,
) -> pd.DataFrame:
    """Attach `RV_*` columns and optional derived investment signals."""
    if df.empty:
        return df.copy()

    out = df.copy()
    cfg = config or RiskVectorConfig()
    records = []

    for _, row in out.iterrows():
        rv = build_risk_vector(row, cfg)
        rec = risk_vector_to_columns(rv)
        if include_signal:
            current_price = _to_float(row.get("Current Price"))
            if current_price is None:
                current_price = _to_float(row.get("Price"))
            bear_fv = _to_float(row.get("Fair Value (Bear)"))
            base_fv = _to_float(row.get("Fair Value (Base)"))
            if base_fv is None:
                base_fv = _to_float(row.get("Fair Value"))
            signal, reason = derive_investment_signal_from_vector(
                rv=rv,
                current_price=current_price,
                bear_fv=bear_fv,
                base_fv=base_fv,
            )
            rec["Investment Signal"] = signal
            rec["Signal Reason"] = reason
        records.append(rec)

    rv_df = pd.DataFrame(records, index=out.index)
    for col in rv_df.columns:
        out[col] = rv_df[col]
    return out

