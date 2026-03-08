# src/data_loader.py
# -*- coding: utf-8 -*-
"""
Data loader that strictly respects filenames/patterns provided via settings.PORTFOLIOS.

Policy (very important)
-----------------------
- DO NOT guess or remap names beyond what caller explicitly provides.
- Use EXACT filename if given and exists; otherwise raise FileNotFoundError.
- If a wildcard pattern (* ? []) is provided, resolve by glob and choose the
  newest (by mtime) among matches.
- If 'patterns' (list[str]) is provided, evaluate in order:
    for pat in patterns:
        - if pat contains wildcard: glob and choose newest among matches; return if any
        - else: treat as exact filename; return if exists
  If none match → raise.

CSV Robustness
--------------
- Accept price column names in this priority:
    'Adj Close' > 'Adjusted Close' > 'Price' > 'Close' > (other plausible column)
- Accept **headerless 2-column CSV** (interpreted as 'Date', 'Price').
- Parse 'Date' (with potential tz offsets like -05:00/-04:00), normalize to
  UTC-naive, and **floor to date (00:00)**.
- Return pandas.DataFrame indexed by Date.

Resampling / Missing
--------------------
- Frequency:
    daily : no change
    weekly: 'W-FRI' (last)
    monthly: 'ME' (month-end, last)
    otherwise: passed to pandas.resample(rule).last()
- Missing:
    'ffill' / 'bfill' / 'both'('fbfill') / none

Public API
----------
- load_data(data_dir, portfolios, start_date=None, end_date=None, freq='daily', missing='ffill', verbose=True)
- read_price_csv(path, name=None, verbose=False)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import pandas as pd

# ============================================================================
# Helpers: Date & Price column handling
# ============================================================================


def _normalize_dates_robust(s: pd.Series) -> pd.Series:
    """
    混在日付（YYYY-MM-DD / YYYY/M/D / YYYY.M.D / 全角記号や各種ダッシュなど）を頑健に正規化→Datetime
    - まず区切りをハイフンに正規化
    - pandas.to_datetime(..., utc=True) でUTC取込み
    - tzを除去してnaive化し、日単位に丸める（00:00:00）
    """
    s = s.astype(str).str.strip()
    # 全角や各種ダッシュ、スラッシュ/ドットをまとめて '-' に
    trans = {
        ord("／"): "-",
        ord("．"): "-",
        ord("－"): "-",
        0x2010: "-",
        0x2011: "-",
        0x2012: "-",
        0x2013: "-",
        0x2014: "-",
        0x2212: "-",
        ord("/"): "-",
        ord("."): "-",
    }
    s = s.str.translate(trans)

    # to_datetime（UTCで取り込み）
    try:
        dt = pd.to_datetime(s, errors="coerce", utc=True)
    except TypeError:
        # 古いpandas互換
        dt = pd.to_datetime(s, errors="coerce")
        if dt.dt.tz is None:
            dt = dt.dt.tz_localize("UTC")

    # UTC-aware → naive（tz除去）→ 日単位に丸め
    return dt.dt.tz_convert(None).dt.floor("D")


def _detect_price_column(columns: List[str]) -> str:
    """
    価格列候補の自動検出。
    優先順：'Adj Close' > 'Adjusted Close' > 'Price' > 'Close' > その他（OHLC/出来高以外で最初に見つかった列）
    """
    priority = ("Adj Close", "Adjusted Close", "Price", "Close")
    for cand in priority:
        if cand in columns:
            return cand

    # 大文字小文字無視のフォールバック
    lc = {c.lower(): c for c in columns}
    for cand in ("adj close", "adjusted close", "price", "close"):
        if cand in lc:
            return lc[cand]

    # 除外列をのぞいて最初に見つかった“それっぽい”列を返す
    exclude = {"open", "high", "low", "volume", "dividends", "stock splits"}
    for c in columns:
        if c.lower() not in exclude:
            return c

    # 最後の砦：末尾の列
    return columns[-1]


# ============================================================================
# Resolver: strictly follow provided filenames/patterns
# ============================================================================


def _contains_wildcard(s: str) -> bool:
    return any(ch in s for ch in "*?[]")


def resolve_by_spec(
    data_dir: Path,
    file_or_pattern: str,
    prefer_csv: bool = True,
    verbose: bool = True,
) -> Path:
    """
    Resolve a single 'file' spec that can be:
      - exact filename (no wildcard) → must exist; else FileNotFoundError
      - wildcard pattern            → glob under data_dir; choose newest (mtime)
    """
    data_dir = Path(data_dir)
    s = str(file_or_pattern)
    if not _contains_wildcard(s):
        # exact filename
        p = data_dir / s
        if p.exists():
            if verbose:
                print(f"[data] Using exact file: {p.name}")
            return p
        raise FileNotFoundError(f"Data file not found: {p}")
    else:
        # wildcard pattern
        cands = list(data_dir.glob(s))
        if not cands:
            raise FileNotFoundError(f"No files matched pattern: {data_dir / s}")

        def _score(path: Path):
            score = 1 if (prefer_csv and path.suffix.lower() == ".csv") else 0
            return (score, path.stat().st_mtime)

        best = max(cands, key=_score)
        if verbose:
            n = len(cands)
            print(f"[data] Resolved pattern '{s}' → '{best.name}'  (matches={n})")
        return best


def resolve_from_patterns_in_order(
    data_dir: Path,
    patterns: List[str],
    prefer_csv: bool = True,
    verbose: bool = True,
) -> Path:
    """
    Evaluate patterns in the given order. For each pattern:
      - if wildcard → choose newest among matches and RETURN if any
      - if exact    → return if exists
    If none matched, raise FileNotFoundError.
    """
    tried: List[str] = []
    for pat in patterns:
        tried.append(pat)
        s = str(pat)
        if _contains_wildcard(s):
            cands = list(Path(data_dir).glob(s))
            if cands:

                def _score(path: Path):
                    score = 1 if (prefer_csv and path.suffix.lower() == ".csv") else 0
                    return (score, path.stat().st_mtime)

                best = max(cands, key=_score)
                if verbose:
                    print(
                        f"[data] Resolved pattern '{s}' → '{best.name}'  (matches={len(cands)})"
                    )
                return best
        else:
            p = Path(data_dir) / s
            if p.exists():
                if verbose:
                    print(f"[data] Using exact file: {p.name}")
                return p

    # none matched
    tried_str = ", ".join([str(Path(data_dir) / t) for t in tried])
    raise FileNotFoundError(
        f"No files matched any of the provided patterns. Tried: {tried_str}"
    )


# ============================================================================
# Public API
# ============================================================================

PortfolioType = Union[
    Mapping[
        str, str | Mapping[str, Any]
    ],  # {"SPXL": "file.csv"} or {"SPXL": {"file": "...", "column": "..."}}
    List[Union[str, Dict[str, Any]]],
    Tuple[Union[str, Dict[str, Any]], ...],
]


def _as_list(x: Optional[Any]) -> Optional[List[str]]:
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return [str(v) for v in x]
    return [str(x)]


def _infer_name_from_spec(spec: Mapping[str, Any], fallback: str = "series") -> str:
    # priority: name > file > glob > patterns[0] > fallback
    if "name" in spec and spec["name"]:
        return str(spec["name"])
    if "file" in spec and spec["file"]:
        return Path(str(spec["file"])).stem
    if "glob" in spec and spec["glob"]:
        return Path(str(spec["glob"])).stem
    pats = _as_list(spec.get("patterns"))
    if pats:
        return Path(pats[0]).stem
    return fallback


def _normalize_portfolios(portfolios: PortfolioType) -> List[Dict[str, Any]]:
    """
    Normalize portfolio definitions into a canonical list of entries:
      [{"name": <logical>, "file": <filename_or_pattern>, "column": None|str, "patterns": None|list[str]}]

    Accepted inputs:
      - {"SPXL": "^spxl_simulated_d_TR_*%cost.csv"}              # string → file
      - {"SPXL": {"file": "^spxl_simulated_d_TR_*%cost.csv"}}
      - {"SPXL": {"glob": "^spxl_simulated_d_TR_*%cost.csv"}}    # alias of file
      - {"SPXL": {"patterns": ["A*.csv", "B*.csv"]}}             # pattern list (ordered)
      - ["^spxl_simulated_d_TR_*%cost.csv", {"file":"^sso_sim.csv","name":"SSO"}]
    """
    normalized: List[Dict[str, Any]] = []

    def _norm_one(
        default_name: Optional[str],
        spec: Union[str, Mapping[str, Any]],
    ) -> Dict[str, Any]:
        if isinstance(spec, str):
            logical = default_name if default_name else Path(spec).stem
            return {"name": logical, "file": spec, "column": None, "patterns": None}

        if not isinstance(spec, Mapping):
            raise TypeError(f"Unexpected portfolio spec type: {type(spec)}")

        name = spec.get("name", default_name)
        file_val = spec.get("file") or spec.get("glob")
        patterns_val = _as_list(spec.get("patterns"))
        column_val = spec.get("column")

        # If file/glob is not provided, use first of patterns as representative 'file'
        if not file_val and patterns_val:
            file_val = patterns_val[0]

        if not file_val:
            raise ValueError(
                "Portfolio entry must have 'file' (or 'glob') or 'patterns'."
            )

        logical = (
            name
            if name
            else _infer_name_from_spec({"file": file_val, "patterns": patterns_val})
        )
        return {
            "name": logical,
            "file": str(file_val),
            "column": column_val,
            "patterns": patterns_val,
        }

    if isinstance(portfolios, dict):
        for logical, spec in portfolios.items():
            normalized.append(_norm_one(logical, spec))
    elif isinstance(portfolios, (list, tuple)):
        for spec in portfolios:
            normalized.append(_norm_one(None, spec))
    else:
        raise TypeError(f"Unsupported portfolios type: {type(portfolios)}")

    return normalized


def _resample(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Resample to the desired frequency:
      - 'daily'   : no change
      - 'weekly'  : W-FRI (last)
      - 'monthly' : ME (month-end, last)
      - other     : passed to pandas rule (e.g., 'QE', 'YS', etc.)
    """
    f = (freq or "").lower()
    if f in ("", "d", "day", "daily"):
        return df
    if f in ("w", "week", "weekly"):
        return df.resample("W-FRI").last()
    if f in ("m", "mon", "month", "monthly"):
        return df.resample("ME").last()  # ← 月末に統一（Mは将来非推奨）
    # custom pandas rule
    return df.resample(freq).last()


def _apply_missing(df: pd.DataFrame, how: str) -> pd.DataFrame:
    """
    Fill missing values if requested.
      - 'ffill' : forward fill
      - 'bfill' : back fill
      - 'both'  : ffill then bfill
      - other/None : no fill
    """
    h = (how or "").lower()
    if h == "ffill":
        return df.ffill()
    if h == "bfill":
        return df.bfill()
    if h in ("fbfill", "both"):
        return df.ffill().bfill()
    return df


def read_price_csv(
    p: Path, name: Optional[str] = None, verbose: bool = False
) -> pd.DataFrame:
    """
    価格CSVを読み込み、以下の仕様で DataFrame を返します（Date を index にセット）:
      - ヘッダ有り: 'Date' 列 + 価格列（'Adj Close' / 'Adjusted Close' / 'Price' / 'Close' / その他）
      - ヘッダ無し2列: 1列目=Date, 2列目=Price と解釈
      - 'Date' は UTC-aware で取り込み → tz 除去 → '日単位に丸め'
      - 価格列は numeric に変換（coerce）し、Date/価格ともに NaN を除去
    """
    df = pd.read_csv(p)

    # ヘッダ無し2列CSV対応（例: "YYYY-MM-DD ... , 123.45"）
    if "Date" not in df.columns:
        if df.shape[1] == 2:
            df = df.rename(columns={df.columns[0]: "Date", df.columns[1]: "Price"})
        else:
            raise ValueError(
                f"'Date' column not found in {p.name}. Found: {df.columns.tolist()}"
            )

    price_col = _detect_price_column(list(df.columns))

    # 混在日付を頑健に正規化（UTC→naive→日単位）
    df["Date"] = _normalize_dates_robust(df["Date"])

    # 数値化
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")

    # 欠損除去 & ソート
    df = (
        df.dropna(subset=["Date", price_col]).sort_values("Date").reset_index(drop=True)
    )

    # 出力列名は logical 名（name 引数）にする
    series_name = name if name else p.stem
    out = (
        df[["Date", price_col]]
        .rename(columns={price_col: series_name})
        .set_index("Date")
    )

    if verbose and not out.empty:
        print(
            f"[data] Loaded {p.name}: rows={len(out)}, col='{series_name}', "
            f"first={out.index.min().date()} last={out.index.max().date()}"
        )
    return out


def load_data(
    data_dir: Union[str, Path],
    portfolios: PortfolioType,
    start_date: Optional[Union[str, pd.Timestamp]] = None,
    end_date: Optional[Union[str, pd.Timestamp]] = None,
    freq: str = "daily",
    missing: str = "ffill",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load and merge multiple series specified in `portfolios`.
    STRICTLY adhere to the filenames/patterns provided by the caller.
    """
    data_dir = Path(data_dir)
    entries = _normalize_portfolios(portfolios)

    frames: List[pd.DataFrame] = []

    for ent in entries:
        logical = ent["name"]
        file_or_pattern = ent["file"]
        price_col = ent.get("column")  # 任意：ロード後の列名を上書きしたい時に使用
        patterns_list = ent.get("patterns")

        # Resolve path strictly by the provided spec
        if patterns_list:
            path = resolve_from_patterns_in_order(
                data_dir=data_dir,
                patterns=patterns_list,
                verbose=verbose,
            )
        else:
            path = resolve_by_spec(
                data_dir=data_dir,
                file_or_pattern=file_or_pattern,
                verbose=verbose,
            )

        # Load CSV
        df = read_price_csv(path, name=logical, verbose=verbose)

        # Optional column rename（ほとんど不要だが、logical を明示上書きしたい場合のみ）
        if price_col and logical in df.columns and price_col != logical:
            df = df.rename(columns={logical: price_col})
            logical = price_col

        frames.append(df)

    if not frames:
        raise ValueError("No portfolio entries to load.")

    # Merge all series (outer join to keep union of dates), then trim/fill/resample
    merged = pd.concat(frames, axis=1, join="outer").sort_index()

    # Trim to [start, end]
    if start_date is not None:
        start_ts = pd.to_datetime(start_date)
        merged = merged.loc[merged.index >= start_ts]
    if end_date is not None:
        end_ts = pd.to_datetime(end_date)
        merged = merged.loc[merged.index <= end_ts]

    # Missing handling
    merged = _apply_missing(merged, missing)

    # Resample
    merged = _resample(merged, freq)

    # Final trim (safe-guard)
    if start_date is not None:
        start_ts = pd.to_datetime(start_date)
        merged = merged.loc[merged.index >= start_ts]
    if end_date is not None:
        end_ts = pd.to_datetime(end_date)
        merged = merged.loc[merged.index <= end_ts]

    # if verbose and not merged.empty:
    #     print(f"Data Period: {merged.index.min().date()} to {merged.index.max().date()}")

    return merged
