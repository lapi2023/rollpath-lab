from __future__ import annotations
from typing import Dict
try:
    import cupy as cp
    from cupy import numpy as cnp
    from cupy.lib.stride_tricks import sliding_window_view as cp_sliding_window_view
    _HAS_CUPY = True
except Exception:
    cp = None; cnp = None; cp_sliding_window_view = None; _HAS_CUPY = False
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as np_sliding_window_view

def _xp(engine: str):
    if engine == 'cupy' and _HAS_CUPY:
        return cnp, cp_sliding_window_view
    return np, np_sliding_window_view

def calculate_rolling_metrics(
    returns: np.ndarray,
    window: int,
    ppy: int,
    risk_free_annual: float,
    style: str,
    dca_amount: float,
    initial_capital: float,
    dates,
    dca_interval: str,
    *,
    batch_size: int = 256,
    engine: str = 'numpy',
) -> Dict[str, np.ndarray]:
    import numpy as _np

    def _max_dd_with_idx_1d(y: _np.ndarray):
        peak = float(y[0]);
        peak_i = 0
        best = 0.0;
        i0 = i1 = 0
        for k in range(1, y.size):
            x = float(y[k])
            if x > peak:
                peak = x;
                peak_i = k
            dd = (x / max(peak, 1e-12)) - 1.0
            if dd < best:
                best = dd;
                i0 = peak_i;
                i1 = k
        return best, i0, i1

    def _max_du_1d(y: _np.ndarray):
        trough = float(y[0]);
        best = 0.0
        for k in range(1, y.size):
            x = float(y[k])
            if x < trough:
                trough = x
            up = (x / max(trough, 1e-12)) - 1.0
            if up > best:
                best = up
        return best


    xp, swv = _xp(engine)
    r = xp.asarray(returns, dtype=xp.float32).reshape(-1)
    N = int(r.shape[0])
    if window <= 0 or N < window:
        empty = np.array([], dtype=np.float32)
        return {
            'ret_window': empty, 'cagr': empty, 'risk': empty, 'sharpe': empty,
            'path_min': empty, 'path_max': empty, 'max_drawup': empty, 'max_drawdown': empty,
            'Final_Value': empty, 'CAGR_Simple': empty,
            'Return': empty, 'CAGR': empty, 'Risk': empty, 'Sharpe': empty,
            'PathMin': empty, 'PathMax': empty, 'Path_Min': empty, 'Path_Max': empty,
            'MaxDrawup': empty, 'Max_Drawup': empty, 'MaxDD': empty, 'Max_DD': empty,
        }
    r_safe = xp.clip(r, -0.999999, None)
    logP = xp.cumsum(xp.log1p(r_safe.astype(xp.float64)), dtype=xp.float64)
    Lw = swv(logP, window_shape=window).astype(xp.float64, copy=False)
    nW = int(Lw.shape[0])
    Pn = xp.exp(Lw - Lw[:, [0]]).astype(xp.float32, copy=False)
    ret_window = xp.exp(Lw[:, -1] - Lw[:, 0]).astype(xp.float32, copy=False) - 1.0
    Pn = xp.nan_to_num(Pn, nan=1.0, posinf=1e30, neginf=1e-30)
    ret_window = xp.nan_to_num(ret_window, nan=0.0, posinf=1e6, neginf=-1.0)
    years = window / float(max(ppy, 1))
    cagr = xp.where(years > 0.0,
                    xp.power(xp.maximum(1.0 + ret_window, 1.0e-12), 1.0 / years) - 1.0,
                    0.0).astype(xp.float32)
    Rw = swv(r_safe, window_shape=window).astype(xp.float32, copy=False)
    if window > 1:
        mean_w = Rw.mean(axis=1)
        var_w = ((Rw - mean_w[:, None]) ** 2).sum(axis=1) / (window - 1)
        std_w = xp.sqrt(xp.maximum(var_w, 0.0))
    else:
        std_w = xp.zeros((nW,), dtype=xp.float32)
    risk = std_w * xp.sqrt(float(max(ppy, 1)))
    excess = cagr - float(risk_free_annual)
    sharpe = xp.divide(excess, xp.where(risk > 1e-12, risk, 1.0))
    B = int(max(1, batch_size))
    path_min = xp.empty((nW,), dtype=xp.float32)
    path_max = xp.empty((nW,), dtype=xp.float32)
    max_up   = xp.empty((nW,), dtype=xp.float32)

    # Allocate output container
    max_up_nonoverlap = _np.zeros((nW,), dtype=_np.float32)
    B2 = int(max(1, batch_size))
    for i in range(0, nW, B2):
        j = min(i + B2, nW)
        v = Pn[i:j, :]  # xp array
        # bring to CPU as numpy for per-row loop (works for both numpy/cupy engine)
        v_np = v.get().astype(_np.float64, copy=False) if (_HAS_CUPY and hasattr(v, 'get')) else _np.asarray(v,
                                                                                                             dtype=_np.float64)

        for r in range(v_np.shape[0]):
            y = v_np[r, :]
            _, i0, i1 = _max_dd_with_idx_1d(y)
            upL = _max_du_1d(y[:max(0, i0)]) if i0 > 0 else 0.0
            upR = _max_du_1d(y[min(y.size, i1 + 1):]) if (i1 + 1) < y.size else 0.0
            max_up_nonoverlap[i + r] = float(upR if upR >= upL else upL)

    max_dd   = xp.empty((nW,), dtype=xp.float32)
    for i in range(0, nW, B):
        j = min(i + B, nW)
        v = Pn[i:j, :]
        col0 = v[:, 0]
        cur_min = col0.copy()
        cur_max = col0.copy()
        pmin = col0.copy()
        pmax = col0.copy()
        up = xp.zeros_like(col0, dtype=xp.float32)
        dd = xp.zeros_like(col0, dtype=xp.float32)
        for k in range(1, window):
            col = v[:, k]
            cur_min = xp.minimum(cur_min, col)
            cur_max = xp.maximum(cur_max, col)
            pmin = xp.minimum(pmin, col)
            pmax = xp.maximum(pmax, col)
            up = xp.maximum(up, (col / xp.maximum(cur_min, 1e-12)) - 1.0)
            dd = xp.minimum(dd, (col / xp.maximum(cur_max, 1e-12)) - 1.0)
        path_min[i:j] = pmin
        path_max[i:j] = pmax
        max_up[i:j]   = up
        max_dd[i:j]   = dd
    style_norm = (style or '').lower()
    if style_norm == 'dca':
        inter = (dca_interval or 'every_period').lower()
        def step(ppy_: int, inter_: str) -> int:
            if inter_ in ('', 'every_period'): return 1
            if inter_ == 'weekly':    return max(1, int(round((ppy_ or 252)/52)))
            if inter_ == 'monthly':   return max(1, int(round((ppy_ or 252)/12)))
            if inter_ == 'quarterly': return max(1, int(round((ppy_ or 252)/4)))
            if inter_ == 'yearly':    return max(1, int(round((ppy_ or 252)/1)))
            return 1
        st = step(int(ppy), inter)
        mask = xp.zeros((window,), dtype=xp.float32)
        upto = max(0, window - 1)
        mask[0:upto:st] = 1.0
        inv_Pn = 1.0 / xp.clip(Pn, 1e-12, None)
        C = inv_Pn * mask[None, :]
        C_prefix = xp.cumsum(C, axis=1)
        V = Pn * (float(initial_capital) + float(dca_amount) * C_prefix)
        V = xp.clip(V, 1.0e-12, None)
        path_min = V.min(axis=1).astype(xp.float32)
        path_max = V.max(axis=1).astype(xp.float32)
        final_value = V[:, -1].astype(xp.float32)
        n_contrib = int(mask.sum())
        principal = max(float(initial_capital) + float(dca_amount) * n_contrib, 1.0e-12)
        cagr_simple = (xp.power(xp.maximum(final_value / principal, 1.0e-12), 1.0 / years) - 1.0).astype(xp.float32)
    else:
        init_eff = float(initial_capital) if (initial_capital and initial_capital > 0) else 1.0
        final_value = (init_eff * (1.0 + ret_window)).astype(xp.float32)
        cagr_simple = cagr.astype(xp.float32)
    def to_np(a):
        if _HAS_CUPY and hasattr(a, 'get'):
            return a.get().astype(np.float32, copy=False)
        return np.asarray(a, dtype=np.float32)
    out = {
        'ret_window': to_np(ret_window),
        'cagr': to_np(cagr),
        'risk': to_np(risk),
        'sharpe': to_np(sharpe),
        'path_min': to_np(path_min),
        'path_max': to_np(path_max),
        'max_drawup': _np.asarray(max_up_nonoverlap, dtype=_np.float32),
        'max_drawdown': to_np(max_dd),
        'Final_Value': to_np(final_value),
        'CAGR_Simple': to_np(cagr_simple),
    }
    out.update({
        'Return': out['ret_window'], 'CAGR': out['cagr'], 'Risk': out['risk'], 'Sharpe': out['sharpe'],
        'PathMin': out['path_min'], 'PathMax': out['path_max'],
        'Path_Min': out['path_min'], 'Path_Max': out['path_max'],
        'MaxDrawup': out['max_drawup'], 'Max_Drawup': out['max_drawup'],
        'MaxDD': out['max_drawdown'], 'Max_DD': out['max_drawdown'],
    })
    return out
