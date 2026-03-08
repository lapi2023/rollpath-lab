# src/metrics.py
from __future__ import annotations
from typing import Dict
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

# -----------------------------
# Helpers
# -----------------------------
def _ensure_1d_float32(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 1:
        x = x.reshape(-1)
    return x.astype(np.float32, copy=False)

def _rolling_prices_from_returns(returns: np.ndarray) -> np.ndarray:
    """Cumprod of (1+r), float32."""
    r = _ensure_1d_float32(returns)
    P = np.cumprod(1.0 + r, dtype=np.float32)  # P[0] = 1+r0
    return P

def _batched_path_extrema_draws_stream(Pn_windows: np.ndarray, *, batch_size: int = 256):
    """
    Memory-safe path-extrema computation by streaming over the time axis.
    Inputs
    ------
    Pn_windows : np.ndarray, shape (nW, W), float32 (view is fine)
        Normalized price paths (start=1).
    Returns
    -------
    path_min, path_max, max_up, max_dd : float32 arrays of shape (nW,)
    """
    nW, W = Pn_windows.shape
    path_min = np.empty((nW,), dtype=np.float32)
    path_max = np.empty((nW,), dtype=np.float32)
    max_up   = np.empty((nW,), dtype=np.float32)
    max_dd   = np.empty((nW,), dtype=np.float32)

    b = max(1, int(batch_size))
    for i in range(0, nW, b):
        j = min(i + b, nW)
        v = Pn_windows[i:j, :]  # (B, W) view

        # Initialize with first column
        col0 = v[:, 0]
        cur_min = col0.copy()
        cur_max = col0.copy()
        pmin = col0.copy()
        pmax = col0.copy()
        up = np.zeros_like(col0, dtype=np.float32)
        dd = np.zeros_like(col0, dtype=np.float32)  # drawdown is <= 0

        # Stream over columns 1..W-1
        for k in range(1, W):
            col = v[:, k]
            cur_min = np.minimum(cur_min, col)
            cur_max = np.maximum(cur_max, col)
            pmin = np.minimum(pmin, col)
            pmax = np.maximum(pmax, col)
            up = np.maximum(up, (col / cur_min) - 1.0)
            dd = np.minimum(dd, (col / cur_max) - 1.0)

        path_min[i:j] = pmin
        path_max[i:j] = pmax
        max_up[i:j]   = up
        max_dd[i:j]   = dd

    return path_min, path_max, max_up, max_dd


def calculate_rolling_metrics(
    returns: np.ndarray,
    window: int,
    ppy: int,
    risk_free_annual: float,
    style: str,
    dca_amount: float,
    initial_capital: float,
    dates: np.ndarray,
    dca_interval: str,
    *,
    batch_size: int = 256,
) -> Dict[str, np.ndarray]:
    """
    Rolling-window core metrics.

    IMPORTANT:
    - For Lump Sum: path statistics are based on normalized price paths (start=1).
    - For DCA: Path Min/Max are REDEFINED to use portfolio VALUE within the window,
      i.e., V_t = Pn_t * (I0 + amount * sum_{k=0..t} mask[k] / Pn_k), strictly > 0.
      (mask at t=0 contributes immediately since contributions are at period start.)
    """
    r = _ensure_1d_float32(returns)
    N = r.shape[0]
    if window <= 0 or N < window:
        empty = np.array([], dtype=np.float32)
        return {
            'ret_window': empty, 'cagr': empty, 'risk': empty, 'sharpe': empty,
            'path_min': empty, 'path_max': empty, 'max_drawup': empty, 'max_drawdown': empty,
            # aliases / legacy
            'Return': empty, 'CAGR': empty, 'Risk': empty, 'Sharpe': empty,
            'PathMin': empty, 'PathMax': empty, 'Path_Min': empty, 'Path_Max': empty,
            'MaxDrawup': empty, 'Max_Drawup': empty, 'MaxDD': empty, 'Max_DD': empty,
            # DCA/Lump Sum additions
            'Final_Value': empty, 'CAGR_Simple': empty,
        }

    # Price path, rolling windows
    P = _rolling_prices_from_returns(r)  # (N,)
    Pw = sliding_window_view(P, window_shape=window).astype(np.float32, copy=False)  # (nW, W)
    nW = Pw.shape[0]

    start = Pw[:, [0]]
    Pn = Pw / np.maximum(start, 1e-12)  # normalized start=1

    # Total return over window
    ret_window = (Pw[:, -1] / np.maximum(Pw[:, 0], 1e-12)) - 1.0

    # CAGR (annualized)
    years = window / float(max(ppy, 1))
    cagr = np.where(
        years > 0.0,
        np.power(np.maximum(1.0 + ret_window, 1.0e-12), 1.0 / years) - 1.0,
        0.0
    ).astype(np.float32)

    # Risk (ann. vol)
    Rw = sliding_window_view(r, window_shape=window).astype(np.float32, copy=False)  # (nW, W)
    if window > 1:
        mean_w = Rw.mean(axis=1)
        var_w = ((Rw - mean_w[:, None]) ** 2).sum(axis=1) / (window - 1)
        std_w = np.sqrt(np.maximum(var_w, 0.0))
    else:
        std_w = np.zeros((nW,), dtype=np.float32)
    risk = std_w * np.sqrt(float(max(ppy, 1)))

    # Sharpe (approx using CAGR vs RF)
    excess = cagr - float(risk_free_annual)
    sharpe = np.divide(excess, np.where(risk > 1e-12, risk, 1.0),
                       out=np.zeros_like(excess), where=risk > 1e-12)

    # Default path stats (normalized prices)
    path_min, path_max, max_up, max_dd = _batched_path_extrema_draws_stream(Pn, batch_size=batch_size)

    # ---- helper: interval to step ----
    def _interval_to_step(_ppy: int, _interval: str | None) -> int:
        inter = (str(_interval or "every_period")).lower()
        if inter in ("", "every_period"): return 1
        if inter == "weekly":     return max(1, int(round((_ppy or 252) / 52)))
        if inter == "monthly":    return max(1, int(round((_ppy or 252) / 12)))
        if inter == "quarterly":  return max(1, int(round((_ppy or 252) / 4)))
        if inter == "yearly":     return max(1, int(round((_ppy or 252) / 1)))
        return 1

    style_norm = (style or "").lower()

    # --- VALUE path (for DCA path_min/max, and Final_Value) ---
    final_value = None
    cagr_simple = None

    if style_norm == "dca":
        step = _interval_to_step(int(ppy), dca_interval)
        mask = np.zeros((window,), dtype=np.float32)
        upto = max(0, window - 1)
        mask[0:upto:step] = 1.0  # include t=0 contribution (period start)
        eps = 1e-12
        inv_Pn = 1.0 / np.clip(Pn, eps, None)           # (nW, W)
        C = inv_Pn * mask[None, :]                      # (nW, W)
        C_prefix = np.cumsum(C, axis=1)                 # (nW, W) <-- NO SHIFT
        V = Pn * (float(initial_capital) + float(dca_amount) * C_prefix)
        V = np.clip(V, eps, None)

        # Path Min/Max are VALUE-based in DCA mode
        path_min = V.min(axis=1).astype(np.float32, copy=False)
        path_max = V.max(axis=1).astype(np.float32, copy=False)

        # Final value (per window)
        final_value = V[:, -1].astype(np.float32, copy=False)

        # Principal and Simple CAGR (per window; principal is constant across windows)
        n_contrib = int(mask.sum())
        principal = float(initial_capital) + float(dca_amount) * n_contrib
        principal = max(principal, eps)
        cagr_simple = (np.power(np.maximum(final_value / principal, 1.0e-12), 1.0 / years) - 1.0).astype(np.float32)

    else:
        # Lump Sum: Final value = initial_effective * (Pw[:, -1]/Pw[:, 0])
        init_eff = float(initial_capital) if (initial_capital and initial_capital > 0) else 1.0
        final_value = (init_eff * (Pw[:, -1] / np.maximum(Pw[:, 0], 1e-12))).astype(np.float32, copy=False)
        # Simple CAGR (same as cagr in lump-sum)
        cagr_simple = cagr.astype(np.float32, copy=False)

    result = {
        'ret_window': ret_window.astype(np.float32, copy=False),
        'cagr': cagr.astype(np.float32, copy=False),
        'risk': risk.astype(np.float32, copy=False),
        'sharpe': sharpe.astype(np.float32, copy=False),
        'path_min': path_min.astype(np.float32, copy=False),
        'path_max': path_max.astype(np.float32, copy=False),
        'max_drawup': max_up.astype(np.float32, copy=False),
        'max_drawdown': max_dd.astype(np.float32, copy=False),
        # additions
        'Final_Value': final_value.astype(np.float32, copy=False) if isinstance(final_value, np.ndarray) else np.array([], dtype=np.float32),
        'CAGR_Simple': cagr_simple.astype(np.float32, copy=False) if isinstance(cagr_simple, np.ndarray) else np.array([], dtype=np.float32),
    }

    # Compatibility aliases expected by visualizer
    result.update({
        'Return': result['ret_window'],
        'CAGR': result['cagr'],
        'Risk': result['risk'],
        'Sharpe': result['sharpe'],
        'PathMin': result['path_min'],
        'PathMax': result['path_max'],
        'Path_Min': result['path_min'],
        'Path_Max': result['path_max'],
        'MaxDrawup': result['max_drawup'],
        'Max_Drawup': result['max_drawup'],
        'MaxDD': result['max_drawdown'],
        'Max_DD': result['max_drawdown'],
    })

    _ = (dates, dca_amount, initial_capital, dca_interval)
    return result