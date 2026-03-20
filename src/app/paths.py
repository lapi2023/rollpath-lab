# src/app/paths.py
from __future__ import annotations
from typing import List, Tuple
import numpy as np
import pandas as pd

def rolling_typical_paths_from_returns(returns: np.ndarray, window: int, *, sample_cap: int = 5000) -> tuple[np.ndarray, np.ndarray]:
    """Mean & median of normalized (start=1) price paths across rolling windows."""
    r = returns.astype(np.float32)
    prices = np.cumprod(1.0 + np.insert(r, 0, 0.0), dtype=np.float32)[1:]
    if len(prices) < window:
        return np.array([]), np.array([])
    N = prices.shape[0]
    nW = N - window + 1
    if nW <= 0:
        return np.array([]), np.array([])
    starts = np.arange(nW, dtype=np.int64)
    if nW > sample_cap:
        sel = np.linspace(0, nW - 1, num=sample_cap, dtype=int)
        starts = starts[sel]
        nW = starts.shape[0]
    W = window
    B = 256
    sum_cols = np.zeros((W,), dtype=np.float64)
    stack_list = []
    total_rows = 0
    for i in range(0, nW, B):
        j = min(i + B, nW)
        s_batch = starts[i:j]
        idx2d = s_batch[:, None] + np.arange(W, dtype=np.int64)[None, :]
        Pw = prices[idx2d]
        Pn = Pw / np.maximum(Pw[:, [0]], 1e-12)
        sum_cols += Pn.sum(axis=0)
        total_rows += Pn.shape[0]
        stack_list.append(Pn.astype(np.float32))
    mean_path = (sum_cols / float(total_rows)).astype(np.float32)
    Pn_all = np.concatenate(stack_list, axis=0)
    median_path = np.median(Pn_all, axis=0).astype(np.float32)
    return mean_path, median_path

def _infer_ppy_from_freq(freq: str) -> int:
    f = (freq or "").lower()
    if f in ("", "daily", "day"):
        return 252
    if f in ("monthly", "month"):
        return 12
    if f in ("yearly", "year"):
        return 1
    return 252

def _interval_to_step(ppy: int, dca_interval: str | None) -> int:
    inter = (str(dca_interval or "every_period")).lower()
    if inter in ("", "every_period"): return 1
    if inter == "weekly":   return max(1, int(round((ppy or 252) / 52)))
    if inter == "monthly":  return max(1, int(round((ppy or 252) / 12)))
    if inter == "quarterly":return max(1, int(round((ppy or 252) / 4)))
    if inter == "yearly":   return max(1, int(round((ppy or 252) / 1)))
    return 1

def build_value_paths_for_windows(
    returns: np.ndarray,
    dates: np.ndarray,
    window: int,
    style: str,
    amount: float,
    initial_cap: float,
    dca_interval: str,
    *,
    sample_cap: int = 2000,
    batch_size: int = 128,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build per-window VALUE paths (small batches) and return:
    - mean_value_path (W,), median_value_path (W,),
    - sampled_starts indices

    NOTE:
    - In DCA mode the internal computation uses normalized prices (Pn, start=1).
      The VALUE path formula uses 'shares' = initial/P0 + sum(amount / P_t),
      and Value_t = P_t * shares. Since P0 == 1 by construction in each window,
      this reduces to shares = initial + sum(amount / Pn_t), Value_t = Pn_t * shares.
    - This function returns VALUE paths only; 'principal' series for DCA is
      constructed in `build_representative_paths`.
    """
    r = returns.astype(np.float32)
    P = np.cumprod(1.0 + np.insert(r, 0, 0.0), dtype=np.float32)[1:]
    N = P.shape[0]
    nW = N - window + 1
    if nW <= 0:
        return np.array([]), np.array([]), np.array([])
    starts_all = np.arange(nW, dtype=np.int64)
    if nW > sample_cap:
        sel = np.linspace(0, nW - 1, num=sample_cap, dtype=int)
        starts = starts_all[sel]
    else:
        starts = starts_all
    W = window
    ppy = _infer_ppy_from_freq("daily")
    step = _interval_to_step(ppy, dca_interval)
    mask = np.zeros((W,), dtype=np.float32)
    upto = max(0, W - 1)              # contributions occur from step 0 to W-2
    mask[0:upto:step] = 1.0

    init_eff = float(initial_cap if initial_cap and initial_cap > 0 else 1.0)
    M = len(starts)
    values_all = np.zeros((M, W), dtype=np.float32)
    for i in range(0, M, batch_size):
        j = min(i + batch_size, M)
        s_batch = starts[i:j]
        idx2d = s_batch[:, None] + np.arange(W, dtype=np.int64)[None, :]
        Pw = P[idx2d]
        Pn = Pw / np.maximum(Pw[:, [0]], 1e-12)

        if (style or "").lower() == "dca":
            # shares_t = initial + sum(amount / Pn_k)
            eps = 1e-12
            inv_Pn = 1.0 / np.clip(Pn, eps, None)
            shares_cum = np.cumsum(inv_Pn * mask[None, :], axis=1)
            shares = float(initial_cap) + float(amount) * shares_cum
            V = Pn * shares
            values_all[i:j, :] = V
        else:
            V = Pn * init_eff
            values_all[i:j, :] = V

    mean_value_path = values_all.mean(axis=0)
    median_value_path = np.median(values_all, axis=0)
    return mean_value_path, median_value_path, starts

def build_representative_paths(
    returns: np.ndarray,
    dates: pd.Series,
    window: int,
    style: str,
    amount: float,
    initial_cap: float,
    dca_interval: str,
    final_values: np.ndarray,
    labels: Tuple[str, ...] = ("Max", "P75", "Med", "P25", "Min"),
) -> List[dict]:
    """
    Pick windows by final wealth (Max, 75th, Med, 25th, Min),
    then build full VALUE paths (with calendar dates).

    IMPORTANT (DCA):
    - From now on, the 'principal' series returned by this function is the
      **invested capital in currency units** (not shares):
        principal_t = initial_cap + amount * (# of contributions up to t).
    - VALUE is computed via "shares" internally:
        shares_t = initial_cap/P0 + sum(amount / P_t); with normalized P (start=1),
        shares_t = initial_cap + sum(amount / Pn_t).
      Then Value_t = Pn_t * shares_t.
    - 'contrib_mask' indicates contribution steps (for stacked bars/CSV).

    This aligns the CSV/plots semantics with visualizer.py expectations where
    'Principal' means invested capital (dollars), and Profit = Value - Principal.
    """
    assert final_values.size > 0
    order = np.argsort(final_values)
    idx_min = int(order[0])
    idx_max = int(order[-1])
    q75 = np.percentile(final_values, 75.0)
    q50 = np.percentile(final_values, 50.0)
    q25 = np.percentile(final_values, 25.0)
    idx_p75 = int(np.argmin(np.abs(final_values - q75)))
    idx_p50 = int(np.argmin(np.abs(final_values - q50)))
    idx_p25 = int(np.argmin(np.abs(final_values - q25)))
    pick_map = {"Max": idx_max, "P75": idx_p75, "Med": idx_p50, "P25": idx_p25, "Min": idx_min}

    out = []
    r = returns.astype(np.float32)
    P = np.cumprod(1.0 + np.insert(r, 0, 0.0), dtype=np.float32)[1:]
    N = P.shape[0]
    ppy = _infer_ppy_from_freq("daily")
    step = _interval_to_step(ppy, dca_interval)
    W = window

    # Contribution schedule (start-of-period contributions)
    base_mask = np.zeros((W,), dtype=bool)
    upto = max(0, W - 1)
    base_mask[0:upto:step] = True

    init_eff = float(initial_cap if initial_cap and initial_cap > 0 else 1.0)

    for lab in labels:
        if lab not in pick_map:
            continue
        start_idx = int(pick_map[lab])
        end_idx = start_idx + W
        if end_idx > N:
            continue

        Pw = P[start_idx:end_idx]
        Pn = Pw / max(Pw[0], 1e-12)

        if (style or "").lower() == "dca":
            eps = 1e-12
            inv_Pn = 1.0 / np.clip(Pn, eps, None)

            # --- Shares path (internal) ---
            # shares_t = initial_cap + amount * cumsum(1/Pn) at contribution steps
            shares_cum = np.cumsum(inv_Pn * base_mask.astype(np.float32))
            shares = (float(initial_cap) + float(amount) * shares_cum).astype(np.float32)

            # --- Value path in currency units ---
            V = (Pn * shares).astype(np.float32)

            # --- Principal (invested capital in currency units) ---
            contrib_count = np.cumsum(base_mask.astype(np.float32))
            principal_dollars = (float(initial_cap) + float(amount) * contrib_count).astype(np.float32)

            cmask = base_mask.copy()
        else:
            # Lump sum: invested capital stays constant at 'initial_cap' for the window.
            principal_dollars = np.full(W, float(initial_cap), dtype=np.float32)
            V = (Pn * (float(initial_cap) if (initial_cap and initial_cap > 0) else 1.0)).astype(np.float32)
            cmask = None

        dts = pd.to_datetime(dates.iloc[start_idx:end_idx])
        out.append(
            {
                "label": lab,
                "dates": dts,
                "values": V,
                "principal": principal_dollars,  # now "invested dollars"
                "contrib_mask": cmask,
            }
        )
    return out