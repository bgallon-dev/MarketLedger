import math

SIGNAL_COLORS: dict[str, str] = {
    "Strong Buy":      "#198754",
    "Speculative Buy": "#6cc070",
    "Hold":            "#adb5bd",
    "Caution":         "#fd7e14",
    "Avoid":           "#dc3545",
    "Overvalued":      "#dc3545",
    "Needs Review":    "#ffc107",
}

SIGNAL_ORDER = [
    "Strong Buy", "Speculative Buy", "Hold",
    "Caution", "Needs Review", "Avoid", "Overvalued",
]


def fmt_pct(v, decimals: int = 1) -> str:
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "—"
        return f"{float(v):.{decimals}f}%"
    except (TypeError, ValueError):
        return "—"


def fmt_currency(v) -> str:
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "—"
        return f"${float(v):,.2f}"
    except (TypeError, ValueError):
        return "—"


def signal_badge_html(signal: str) -> str:
    color = SIGNAL_COLORS.get(signal, "#6c757d")
    return (
        f'<span style="background:{color};color:#fff;padding:2px 8px;'
        f'border-radius:4px;font-size:0.85em;font-weight:600">{signal}</span>'
    )
