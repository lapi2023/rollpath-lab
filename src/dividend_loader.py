# src/dividend_loader.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class DividendInputPaths:
    """Paths to input files required for dividend & risk-free processing."""
    sp500_dividend_monthly_csv: Path
    tbill_1920_1934_monthly_csv: Path
    tb3ms_1934_now_monthly_csv: Path


# --- Internal helpers ---------------------------------------------------------

_MONTH_3 = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

# “二文字”表記（ファイル末尾の歴史パートに混在）
#   例: 'De311899' = 1899-12-31 / 'Fe281871' = 1871-02-28
_MONTH_2 = {
    "ja": 1, "fe": 2, "ma": 3, "ap": 4, "my": 5, "jn": 6, "ju": 6,  # 'ju' は異表記を 6 月に吸収
    "jl": 7, "au": 8, "se": 9, "oc": 10, "no": 11, "de": 12,
}


def _parse_dividend_csv_with_century_boundary(div_file: Path) -> pd.DataFrame:
    """
    Parse the S&P 500 dividend yield monthly CSV (1871-…-2026) whose 'Date' column
    mixes multiple formats and *two-digit years*.

    Strategy:
      1) For 'dd-Mon-yy' rows, detect the single boundary where yy increases
         (e.g., 00 -> 99) in file order, and map rows *above* the boundary to 20yy
         and *below* to 19yy.  => prevents 1926 being mistaken for 2026.
      2) For 'De311899' style rows, parse deterministically as YYYY-MM-DD.
      3) For each YearMonth, keep the *last day in that month* (max ParsedDate).

    Returns:
      DataFrame[YearMonth (str 'YYYY-MM'), DivYield (float as fraction, e.g., 0.0114)]
    """
    df = pd.read_csv(div_file)
    if "Date" not in df.columns or "Value" not in df.columns:
        raise ValueError(f"Unexpected columns in {div_file.name}: {df.columns.tolist()}")

    df["row_id"] = range(len(df))
    dcol = df["Date"].astype(str).str.strip()

    # pattern1: dd-Mon-yy  (e.g., '31-Jan-08')
    p1_mask = dcol.str.match(r"^\d{1,2}-[A-Za-z]{3}-\d{2}$")
    p1 = df.loc[p1_mask].copy()
    if not p1.empty:
        p1["day"] = p1["Date"].str.extract(r"^(\d{1,2})-")[0].astype(int)
        p1["mon"] = p1["Date"].str.extract(r"^\d{1,2}-([A-Za-z]{3})-")[0].str.lower().map(_MONTH_3)
        p1["yy"] = p1["Date"].str.extract(r"-(\d{2})$")[0].astype(int)

        # detect the *first* increase in yy (00.. → ..99) in file order
        yy = p1["yy"].to_numpy()
        rid = p1["row_id"].to_numpy()
        boundary_row: Optional[int] = None
        for i in range(1, len(yy)):
            if yy[i] > yy[i - 1]:  # increase appears once around ...-00 -> ...-99
                boundary_row = rid[i]
                break

        def map_year(row) -> int:
            if boundary_row is None:
                # Fallback (should not happen with this file)
                return 2000 + int(row["yy"]) if int(row["yy"]) <= 30 else 1900 + int(row["yy"])
            # 上側（新しいブロック）→ 2000年代, 下側（歴史ブロック）→ 1900年代
            return (2000 + int(row["yy"])) if (int(row["row_id"]) < boundary_row) else (1900 + int(row["yy"]))

        p1["year"] = p1.apply(map_year, axis=1)
        p1["ParsedDate"] = pd.to_datetime(dict(year=p1["year"], month=p1["mon"], day=p1["day"]), errors="coerce")

    # pattern2: De311899 / Fe281871 ...  (two-letter month + dd + yyyy)
    p2_mask = ~p1_mask
    p2 = df.loc[p2_mask].copy()
    if not p2.empty:
        m = p2["Date"].str.extract(r"^([A-Z][a-z])(\d{1,2})(\d{4})$")
        valid = ~m.isna().any(axis=1)
        p2 = p2.loc[valid].copy()
        p2["mon"] = m.loc[valid, 0].str.lower().map(_MONTH_2)
        p2["day"] = m.loc[valid, 1].astype(int)
        p2["year"] = m.loc[valid, 2].astype(int)
        p2["ParsedDate"] = pd.to_datetime(dict(year=p2["year"], month=p2["mon"], day=p2["day"]), errors="coerce")

    parsed = pd.concat([p1, p2], ignore_index=True, sort=False)
    parsed = parsed.dropna(subset=["ParsedDate"]).copy()

    parsed["YearMonth"] = parsed["ParsedDate"].dt.to_period("M").astype(str)
    parsed["DivYield"] = parsed["Value"].astype(str).str.replace("%", "", regex=False).astype(float) / 100.0

    # 月内の重複は「月末に最も近い日」（= ParsedDate の最大）を採用
    parsed = parsed.sort_values("ParsedDate").drop_duplicates(subset=["YearMonth"], keep="last")

    return parsed[["YearMonth", "DivYield"]]


def _load_risk_free_monthly(
    tbill_1920_1934_csv: Path,
    tb3ms_1934_now_csv: Path
) -> pd.DataFrame:
    """
    Make a continuous monthly risk-free series (as annualized fraction) from 1920-01 onward.

    tbill_1920_1934_csv:
      columns: observation_date, M1329AUSM193NNBR  (monthly %, 1920-1934)
    tb3ms_1934_now_csv:
      columns: observation_date, TB3MS  (monthly %, 1934-…)
    """
    early = pd.read_csv(tbill_1920_1934_csv)
    if not {"observation_date", "M1329AUSM193NNBR"} <= set(early.columns):
        raise ValueError(f"Unexpected columns in {tbill_1920_1934_csv.name}")
    early = early.rename(columns={"M1329AUSM193NNBR": "value"}).copy()
    early["YearMonth"] = pd.to_datetime(early["observation_date"]).dt.to_period("M").astype(str)
    early["RiskFree"] = early["value"].astype(float) / 100.0
    early = early[["YearMonth", "RiskFree"]]

    late = pd.read_csv(tb3ms_1934_now_csv)
    if not {"observation_date", "TB3MS"} <= set(late.columns):
        raise ValueError(f"Unexpected columns in {tb3ms_1934_now_csv.name}")
    late["YearMonth"] = pd.to_datetime(late["observation_date"]).dt.to_period("M").astype(str)
    late["RiskFree"] = late["TB3MS"].astype(float) / 100.0
    late = late[["YearMonth", "RiskFree"]]

    # stitch: prefer 'late' when overlapping, else use 'early'
    rf = pd.concat([early, late], ignore_index=True)
    rf = rf.sort_values("YearMonth").drop_duplicates(subset=["YearMonth"], keep="last").reset_index(drop=True)
    return rf  # ['YearMonth', 'RiskFree']


def load_monthly_dividend_and_riskfree(paths: DividendInputPaths) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parameters:
      paths: DividendInputPaths

    Returns:
      div_m: DataFrame['YearMonth','DivYield']  (annualized fraction)
      rf_m : DataFrame['YearMonth','RiskFree']  (annualized fraction)
    """
    div_m = _parse_dividend_csv_with_century_boundary(paths.sp500_dividend_monthly_csv)
    rf_m = _load_risk_free_monthly(paths.tbill_1920_1934_monthly_csv, paths.tb3ms_1934_now_monthly_csv)
    return div_m, rf_m


def to_daily_series_from_monthly(monthly_df: pd.DataFrame, key: str, daily_dates: pd.Series) -> pd.Series:
    """
    Expand a monthly series to daily by mapping each day to the value of its calendar month.

    Parameters:
      monthly_df: DataFrame['YearMonth', key]
      key: 'DivYield' or 'RiskFree'
      daily_dates: pd.Series of daily timestamps or strings

    Returns:
      pd.Series aligned with daily_dates index
    """
    m = monthly_df.set_index("YearMonth")[key].copy()
    ym = pd.to_datetime(daily_dates, errors="coerce").dt.to_period("M").astype(str)

    out = ym.map(m)
    # 欠損が出た場合は前方→後方埋めで連続化
    out = out.ffill().bfill()
    out.index = daily_dates.index
    return out
