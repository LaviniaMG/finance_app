"""Number and label formatters for the Streamlit UI."""
from config.settings import APP_CURRENCY_SYMBOL


def fmt_number(value: float | None, unit: int = 1_000, decimals: int = 1) -> str:
    """Formats a number for display (default: thousands with 1 decimal)."""
    if value is None:
        return "—"
    try:
        v = float(value) / unit
        return f"{v:,.{decimals}f}"
    except (TypeError, ValueError):
        return "—"


def fmt_currency(value: float | None, unit: int = 1_000, decimals: int = 1) -> str:
    """Formats a currency value with symbol."""
    if value is None:
        return "—"
    try:
        v = float(value) / unit
        return f"{APP_CURRENCY_SYMBOL}{v:,.{decimals}f}k"
    except (TypeError, ValueError):
        return "—"


def fmt_pct(value: float | None, decimals: int = 1) -> str:
    """Formats a percentage (input already as %, e.g., 5.2 → '5.2%')."""
    if value is None:
        return "—"
    try:
        sign = "+" if float(value) > 0 else ""
        return f"{sign}{float(value):.{decimals}f}%"
    except (TypeError, ValueError):
        return "—"


def fmt_variance(value: float | None, unit: int = 1_000, decimals: int = 1) -> str:
    """Formats a variance with sign and color hint."""
    if value is None:
        return "—"
    try:
        v = float(value) / unit
        sign = "+" if v > 0 else ""
        return f"{sign}{v:,.{decimals}f}"
    except (TypeError, ValueError):
        return "—"


MONTH_NAMES_RO = {
    1: "Ian", 2: "Feb", 3: "Mar", 4: "Apr",
    5: "Mai", 6: "Iun", 7: "Iul", 8: "Aug",
    9: "Sep", 10: "Oct", 11: "Noi", 12: "Dec",
}

MONTH_NAMES_RO_FULL = {
    1: "Ianuarie", 2: "Februarie", 3: "Martie", 4: "Aprilie",
    5: "Mai", 6: "Iunie", 7: "Iulie", 8: "August",
    9: "Septembrie", 10: "Octombrie", 11: "Noiembrie", 12: "Decembrie",
}


def period_label(period_id: int, fiscal_year: int | None = None) -> str:
    name = MONTH_NAMES_RO.get(int(period_id), f"P{period_id}")
    if fiscal_year:
        return f"{name} {fiscal_year}"
    return name


CONFIDENCE_COLORS = {
    "High": "🟢",
    "Medium": "🟡",
    "Low": "🔴",
}

STATUS_COLORS = {
    "Aprobat": "🟢",
    "Draft": "🔵",
    "In review": "🟡",
    "Respins": "🔴",
    "Submitted": "🟡",
}


def confidence_badge(level: str) -> str:
    icon = CONFIDENCE_COLORS.get(level, "⚪")
    return f"{icon} {level}"


def status_badge(status: str) -> str:
    icon = STATUS_COLORS.get(status, "⚪")
    return f"{icon} {status}"
