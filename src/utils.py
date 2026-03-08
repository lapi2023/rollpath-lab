# src/utils.py

def get_window_and_ppy(invest_val: int, invest_unit: str, data_freq: str) -> tuple[int, int]:
    """Compute (window length, periods-per-year) from data frequency and investment horizon."""
    if data_freq == "daily":
        ppy = 252
        if invest_unit == "days":
            window = invest_val
        elif invest_unit == "months":
            window = int(invest_val * 21)
        elif invest_unit == "years":
            window = invest_val * 252
    elif data_freq == "monthly":
        ppy = 12
        if invest_unit == "days":
            raise ValueError("月次データに対して日単位の投資期間は指定できません")
        elif invest_unit == "months":
            window = invest_val
        elif invest_unit == "years":
            window = invest_val * 12
    elif data_freq == "yearly":
        ppy = 1
        if invest_unit in ["days", "months"]:
            raise ValueError("年次データに対して日/月単位の投資期間は指定できません")
        elif invest_unit == "years":
            window = invest_val
    else:
        raise ValueError(f"Unknown DATA_FREQ: {data_freq}")

    return window, ppy

def format_number_kmg(num: float) -> str:
    """Format a number using compact financial units.

    Examples:
        950 -> '950'
        12_300 -> '12.3K'
        4_560_000 -> '4.56M'
        7_890_000_000 -> '7.89B'
        1_230_000_000_000 -> '1.23T'

    Notes:
        - Uses K/M/B/T for thousand/million/billion/trillion.
        - Keeps up to ~3 significant digits via general format.
    """
    try:
        x = float(num)
    except Exception:
        return str(num)
    ax = abs(x)
    if ax >= 1_000_000_000_000:
        return f"{x / 1_000_000_000_000:g}T"
    if ax >= 1_000_000_000:
        return f"{x / 1_000_000_000:g}B"
    if ax >= 1_000_000:
        return f"{x / 1_000_000:g}M"
    if ax >= 1_000:
        return f"{x / 1_000:g}K"
    # Plain (no scientific) for small numbers
    return f"{x:g}"


def format_integer_commas(num: float | int) -> str:
    """Format as an integer with thousands separators (comma)."""
    try:
        return f"{int(round(float(num))):,}"
    except Exception:
        return str(num)
