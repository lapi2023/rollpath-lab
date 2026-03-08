# src/comparator.py
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl


@dataclass
class CompareResult:
    symbol: str
    final_divergence_pct: float
    mean_abs_divergence_pct: float
    merged_df: pl.DataFrame  # Date, actual_idx, sim_idx, divergence_pct


def _parse_date_any(s: pl.Expr) -> pl.Expr:
    # "2008/11/5" や "2008-11-05" などに対応
    s2 = s.str.replace_all("-", "/")
    return pl.coalesce(
        [
            s2.str.strptime(pl.Date, format="%Y/%m/%d", strict=False),
            s2.str.strptime(pl.Date, format="%Y/%-m/%-d", strict=False),
            s2.str.to_date(strict=False),
        ]
    )


def find_latest_by_filename_or_mtime(data_dir: Path, glob_pattern: str) -> Path:
    """
    spxl_us_d_20081105-20260227.csv のようなファイル名から end(YYYYMMDD) を読んで最大を選ぶ。
    取れない場合は更新時刻で最大を選ぶ。
    """
    files = sorted(data_dir.glob(glob_pattern))
    if not files:
        raise FileNotFoundError(f"No files matched: {data_dir / glob_pattern}")

    # 例: spxl_us_d_20081105-20260227.csv
    rx = re.compile(r".*_(\d{8})-(\d{8})\.csv$", re.IGNORECASE)

    scored = []
    for p in files:
        m = rx.match(p.name)
        if m:
            end = m.group(2)
            scored.append((end, p))
        else:
            scored.append(("", p))

    # end日が取れたものがあればそれ優先
    with_end = [(e, p) for e, p in scored if e]
    if with_end:
        return max(with_end, key=lambda t: t[0])[1]

    # fallback: mtime
    return max(files, key=lambda p: p.stat().st_mtime)


def load_actual_close(filepath: Path) -> pl.DataFrame:
    """
    入力:
      Date,Close
      2008/11/5,3.57719
    出力:
      Date (pl.Date), Close (float)
    """
    df = pl.read_csv(filepath, schema_overrides={"Date": pl.String, "Close": pl.String})
    df = (
        df.with_columns(
            [
                _parse_date_any(pl.col("Date")).alias("Date"),
                pl.col("Close")
                .str.replace_all(",", "")
                .cast(pl.Float64, strict=False)
                .alias("Close"),
            ]
        )
        .drop_nulls("Date")
        .sort("Date")
    )
    return df


def load_simulated_price(filepath: Path) -> pl.DataFrame:
    """
    入力:
      Date,Price
      1920-01-01,....
    """
    df = pl.read_csv(filepath, schema_overrides={"Date": pl.String, "Price": pl.String})
    df = (
        df.with_columns(
            [
                _parse_date_any(pl.col("Date")).alias("Date"),
                pl.col("Price")
                .str.replace_all(",", "")
                .cast(pl.Float64, strict=False)
                .alias("Price"),
            ]
        )
        .drop_nulls("Date")
        .sort("Date")
    )
    return df


def normalize_to_one(
    df: pl.DataFrame, price_col: str, out_col: str = "idx"
) -> pl.DataFrame:
    first_val = df.select(pl.col(price_col).first()).item()
    return df.with_columns((pl.col(price_col) / first_val).alias(out_col))


def align_and_compute_divergence(
    actual_df: pl.DataFrame, sim_df: pl.DataFrame, symbol: str
) -> CompareResult:
    """
    actual_df: Date, Close
    sim_df:    Date, Price
    それぞれ start を合わせ、Date で inner join し、指数化・乖離率を算出
    """
    start_date = actual_df.select(pl.col("Date").min()).item()
    sim_df2 = sim_df.filter(pl.col("Date") >= start_date)

    a = normalize_to_one(actual_df, "Close", "actual_idx").select(
        ["Date", "actual_idx"]
    )
    s = normalize_to_one(sim_df2, "Price", "sim_idx").select(["Date", "sim_idx"])

    merged = a.join(s, on="Date", how="inner").sort("Date")
    merged = merged.with_columns(
        ((pl.col("sim_idx") / pl.col("actual_idx") - 1.0) * 100.0).alias(
            "divergence_pct"
        )
    )

    final_div = merged.select(pl.col("divergence_pct").tail(1)).item()
    mae = merged.select(pl.col("divergence_pct").abs().mean()).item()

    return CompareResult(
        symbol=symbol,
        final_divergence_pct=float(final_div),
        mean_abs_divergence_pct=float(mae),
        merged_df=merged,
    )


# ====== 最適化用（コストを変えて“同じモデル式”でシミュレーションを再生成） ======


def build_base_merged_df(data_dir: Path, raw_dir: Path, spx_file: Path) -> pl.DataFrame:
    """
    generate_leveraged_etf.py と同等の結合（SPX日次 + 金利/月次 + 配当/月次）
    返り値: Date, SPX_Return, DaysDiff, RiskFreeRate, DivYield
    """
    # SPX
    df_spx = pl.read_csv(
        spx_file, schema_overrides={"Date": pl.String, "Price": pl.String}
    )
    parsed_date = _parse_date_any(pl.col("Date"))
    df_spx = (
        df_spx.with_columns(
            [
                parsed_date.alias("Date"),
                pl.col("Price")
                .str.replace_all(",", "")
                .cast(pl.Float64, strict=False)
                .alias("Price"),
            ]
        )
        .drop_nulls("Date")
        .sort("Date")
    )

    df_spx = df_spx.with_columns(
        [
            (pl.col("Price") / pl.col("Price").shift(1) - 1.0).alias("SPX_Return"),
            (pl.col("Date") - pl.col("Date").shift(1))
            .dt.total_days()
            .alias("DaysDiff"),
        ]
    ).with_columns(
        [
            pl.col("DaysDiff").fill_null(1.0),
            pl.col("Date").dt.strftime("%Y-%m").alias("YearMonth"),
        ]
    )

    # Rate（月次）
    rate1_files = list(raw_dir.glob("*Yields on Short-Term*.csv"))
    rate2_files = list(raw_dir.glob("*3-Month Treasury Bill*.csv"))
    if not rate1_files or not rate2_files:
        raise FileNotFoundError("raw に金利CSVが見つかりません。")

    df_rate1 = pl.read_csv(rate1_files[0])
    df_rate2 = pl.read_csv(rate2_files[0])
    df_rate1 = df_rate1.rename(
        {df_rate1.columns[0]: "Date", df_rate1.columns[1]: "Rate"}
    )
    df_rate2 = df_rate2.rename(
        {df_rate2.columns[0]: "Date", df_rate2.columns[1]: "Rate"}
    )
    df_rate = pl.concat([df_rate1, df_rate2], how="vertical_relaxed")

    df_rate = (
        df_rate.with_columns(
            _parse_date_any(pl.col("Date").cast(pl.String)).alias("Date")
        )
        .with_columns(
            pl.col("Date").dt.strftime("%Y-%m").alias("YearMonth"),
            (pl.col("Rate").cast(pl.Float64, strict=False) / 100.0).alias(
                "RiskFreeRate"
            ),
        )
        .drop_nulls("YearMonth")
        .unique(subset=["YearMonth"], keep="last")
    )

    # Dividend（月次）
    div_files = list(raw_dir.glob("*Dividend Yield*.csv"))
    if not div_files:
        raise FileNotFoundError("raw に配当利回りCSVが見つかりません。")

    def parse_div_yearmonth(d_str: str) -> str | None:
        if not isinstance(d_str, str):
            return None
        d_str = d_str.strip()
        m1 = re.match(r"^\d{1,2}-([A-Za-z]{3})-(\d{2})$", d_str)
        if m1:
            month_str = m1.group(1)[:3].lower()
            year_2d = int(m1.group(2))
            year = 2000 + year_2d if year_2d <= 30 else 1900 + year_2d
            month_map = {
                "jan": 1,
                "feb": 2,
                "mar": 3,
                "apr": 4,
                "may": 5,
                "jun": 6,
                "jul": 7,
                "aug": 8,
                "sep": 9,
                "oct": 10,
                "nov": 11,
                "dec": 12,
            }
            month = month_map.get(month_str, 1)
            return f"{year:04d}-{month:02d}"

        m2 = re.match(r"^([A-Z][a-z])\d{1,2}(\d{4})$", d_str)
        if m2:
            month_str = m2.group(1).lower()
            year = int(m2.group(2))
            month_map = {
                "ja": 1,
                "fe": 2,
                "ma": 3,
                "ap": 4,
                "my": 5,
                "jn": 6,
                "jl": 7,
                "au": 8,
                "se": 9,
                "oc": 10,
                "no": 11,
                "de": 12,
            }
            month = month_map.get(month_str, 1)
            return f"{year:04d}-{month:02d}"
        return None

    df_div = pl.read_csv(
        div_files[0], schema_overrides={"Date": pl.String, "Value": pl.String}
    )
    df_div = df_div.rename({df_div.columns[0]: "Date", df_div.columns[1]: "Value"})
    df_div = (
        df_div.with_columns(
            pl.col("Date")
            .map_elements(parse_div_yearmonth, return_dtype=pl.String)
            .alias("YearMonth")
        )
        .with_columns(
            (
                pl.col("Value").str.replace("%", "").cast(pl.Float64, strict=False)
                / 100.0
            ).alias("DivYield")
        )
        .drop_nulls("YearMonth")
        .unique(subset=["YearMonth"], keep="last")
    )

    # Merge
    df = df_spx.join(
        df_rate.select(["YearMonth", "RiskFreeRate"]), on="YearMonth", how="left"
    )
    df = df.join(df_div.select(["YearMonth", "DivYield"]), on="YearMonth", how="left")
    df = df.with_columns(
        [
            pl.col("RiskFreeRate")
            .fill_null(strategy="forward")
            .fill_null(strategy="backward"),
            pl.col("DivYield")
            .fill_null(strategy="forward")
            .fill_null(strategy="backward"),
        ]
    )

    return (
        df.select(["Date", "SPX_Return", "DaysDiff", "RiskFreeRate", "DivYield"])
        .drop_nulls("Date")
        .sort("Date")
    )


def simulate_index_from_base(
    base_df: pl.DataFrame, start_date: date, leverage: int, cost: float
) -> pl.DataFrame:
    """
    base_df: Date, SPX_Return, DaysDiff, RiskFreeRate, DivYield
    指数（初日=1）を返す: Date, sim_idx
    """
    df = base_df.filter(pl.col("Date") >= start_date).sort("Date")
    if df.height < 2:
        raise ValueError("シミュレーション期間が短すぎます（データが足りません）。")

    # レバレッジETFのモデル式（既存 generate_leveraged_etf.py と同等）
    rf_mult = leverage - 1  # 3x -> 2, 2x -> 1
    df = df.with_columns(
        (
            pl.col("SPX_Return") * leverage
            + (pl.col("DivYield") * leverage - pl.col("RiskFreeRate") * rf_mult)
            * (pl.col("DaysDiff") / 365.0)
            - cost * (pl.col("DaysDiff") / 365.0)
        ).alias("lev_return")
    )

    # 初日は return=0 扱い
    df = df.with_columns(
        pl.when(pl.col("Date") == pl.col("Date").first())
        .then(0.0)
        .otherwise(pl.col("lev_return"))
        .fill_null(0.0)
        .alias("lev_return_adj")
    )

    # idx = cumprod(1+r)
    df = df.with_columns(((pl.col("lev_return_adj") + 1.0).cum_prod()).alias("sim_idx"))
    # 初日を1に揃える（cumprodでもなるが念のため）
    first_val = df.select(pl.col("sim_idx").first()).item()
    df = df.with_columns((pl.col("sim_idx") / first_val).alias("sim_idx"))

    return df.select(["Date", "sim_idx"])


def grid_search_optimal_cost(
    base_df: pl.DataFrame,
    actual_df: pl.DataFrame,
    leverage: int,
    cost_min: float = 0.0,
    cost_max: float = 0.05,
    step: float = 0.0005,
) -> dict:
    """
    目的1: abs(final divergence) 最小
    目的2: mean(abs(daily divergence)) 最小
    """
    start_date = actual_df.select(pl.col("Date").min()).item()
    actual_idx_df = normalize_to_one(actual_df, "Close", "actual_idx").select(
        ["Date", "actual_idx"]
    )

    costs = np.arange(cost_min, cost_max + 1e-12, step)
    best_final = {"cost": None, "abs_final_div": float("inf"), "final_div": None}
    best_mae = {"cost": None, "mae": float("inf")}

    # 速度のため、日付 join の軸は固定し、costごとに sim_idx を作って join する
    for c in costs:
        sim_idx_df = simulate_index_from_base(base_df, start_date, leverage, float(c))
        merged = actual_idx_df.join(sim_idx_df, on="Date", how="inner")
        if merged.height == 0:
            continue
        div = (merged["sim_idx"] / merged["actual_idx"] - 1.0) * 100.0
        final_div = float(div[-1])
        abs_final = abs(final_div)
        mae = float(np.mean(np.abs(div)))

        if abs_final < best_final["abs_final_div"]:
            best_final = {
                "cost": float(c),
                "abs_final_div": abs_final,
                "final_div": final_div,
            }
        if mae < best_mae["mae"]:
            best_mae = {"cost": float(c), "mae": mae}

    return {"best_by_final": best_final, "best_by_mae": best_mae}
