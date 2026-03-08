# src/dca_metrics.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

__all__ = ["enrich_results_with_dca_metrics", "compute_dca_metrics_for_one", "DCAConfig"]

@dataclass(frozen=True)
class DCAConfig:
    """
    計算に必要なDCA設定をとりまとめるデータクラス
    - window: ローリング・ウィンドウの長さ（期間数）
    - ppy: 1年あたり期間数（daily=252, monthly=12, yearly=1 など）
    - amount: 1回あたり拠出額
    - initial_cap: 初期投資額
    - dca_interval: "every_period" | "weekly" | "monthly" | "quarterly" | "yearly" | None
    """
    window: int
    ppy: int
    amount: float
    initial_cap: float
    dca_interval: Optional[str] = None

def _interval_to_step(ppy: int, dca_interval: Optional[str]) -> int:
    inter = (dca_interval or "every_period").lower()
    if inter in ("", "every_period"): return 1
    if inter == "weekly":     return max(1, int(round(ppy / 52)))
    if inter == "monthly":    return max(1, int(round(ppy / 12)))
    if inter == "quarterly":  return max(1, int(round(ppy / 4)))
    if inter == "yearly":     return max(1, int(round(ppy / 1)))
    return 1

def _make_contribution_mask(W: int, step: int) -> np.ndarray:
    """
    長さWのブール配列（ウィンドウ相対の拠出スケジュール）を作成。
    仕様：
      - 拠出は各期の「開始時」に行う前提。
      - 最終インデックス（W-1）は拠出しない（期末は成長のみ）。
    """
    mask = np.zeros(W, dtype=bool)
    upto = max(0, W - 1)
    mask[0:upto:step] = True   # 0, step, 2*step, … < W-1
    mask[-1] = False
    return mask

def compute_dca_metrics_for_one(price_series: pd.Series, cfg: DCAConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    1銘柄（ポートフォリオ列）の価格系列に対し、全ローリング・ウィンドウで以下を計算：
      - Final_Value（最終価値）
      - CAGR_Simple（単純CAGR = 総元本ベースの年率）
    返り値は shape=(N-W+1,) の2配列。N<W なら空配列を返します。
    """
    p = price_series.to_numpy(dtype=float)
    N = p.shape[0]
    W = int(cfg.window)
    if N < W or W <= 1:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    # 1/価格（“開始時に1通貨拠出→期末価格までの複利”の係数に使う）
    inv_p = 1.0 / p

    # DCA拠出のマスク（長さW, 最終点は拠出なし）
    step = _interval_to_step(cfg.ppy, cfg.dca_interval)
    mask = _make_contribution_mask(W, step).astype(float)

    # Σ inv_p[s+k] * mask[k] を sについてベクトル化（畳み込み）
    # conv_inv_p[s] はウィンドウ s..s+W-1 内の「期首拠出の逆価格」の合計
    conv_inv_p = np.convolve(inv_p, mask, mode="valid")  # 長さ N-W+1

    # ウィンドウ期末/期首の価格
    p_end   = p[W-1:]            # 期末価格 (N-W+1)
    p_start = p[:N - W + 1]      # 期首価格 (N-W+1)
    A = p_end / p_start          # 期首→期末の倍率

    # Final_Value:
    #   初期投資 initial_cap は期首に投じて A 倍に成長
    #   各拠出 amount は「開始時」に投じ、期末では p_end * Σ(1/p) 倍
    FV = cfg.initial_cap * A + cfg.amount * (p_end * conv_inv_p)  # (N-W+1,)

    # 単純CAGR（総元本ベース、W/ppy 年）
    years = W / float(cfg.ppy if cfg.ppy > 0 else 1)
    num_contribs = float(mask.sum())
    total_principal = cfg.initial_cap + cfg.amount * num_contribs
    if years > 0 and total_principal > 0:
        CAGR_simple = np.power(FV / total_principal, 1.0 / years) - 1.0
    else:
        CAGR_simple = np.full_like(FV, np.nan, dtype=float)

    return FV, CAGR_simple

def enrich_results_with_dca_metrics(
    results: Dict[str, dict],
    prices_df: pd.DataFrame,
    portfolio_names: List[str],
    window: int,
    ppy: int,
    amount: float,
    initial_cap: float,
    dca_interval: Optional[str] = None,
) -> None:
    """
    既存の results[portfolio] に対し、DCA の
      - "Final_Value": ndarray (各ローリング・ウィンドウの最終価値)
      - "CAGR_Simple": ndarray (各ローリング・ウィンドウの単純CAGR)
    をインプレースで追記します（戻り値は None）。
    - prices_df の列は portfolio_names と一致している想定（無ければスキップ）
    """
    cfg = DCAConfig(
        window=window, ppy=ppy, amount=amount, initial_cap=initial_cap, dca_interval=dca_interval
    )

    for name in portfolio_names:
        if name not in prices_df.columns:
            # 列名が一致しない場合はスキップ（可視化側でフォールバックします）
            continue
        series = prices_df[name].astype(float)
        fv, cagr_s = compute_dca_metrics_for_one(series, cfg)
        results.setdefault(name, {})
        results[name]["Final_Value"]  = fv
        results[name]["CAGR_Simple"]  = cagr_s

    # 明示的に None を返す（インプレース更新のため）
    return None