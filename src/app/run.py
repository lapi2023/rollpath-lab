from __future__ import annotations

import concurrent.futures
import os
import re
import time
import unicodedata
from typing import Dict

import numpy as np
import pandas as pd
import polars as pl
import psutil
from rich.console import Console

from src import settings
from src.data_loader import load_data
from src.dca_metrics import enrich_results_with_dca_metrics
from src.metrics_accel import calculate_rolling_metrics as calc_metrics_accel
from src.portfolio import calculate_portfolio_returns
from src.utils import get_window_and_ppy
from src.utils_perf import (
    choose_engine,
    human_hw_summary,
    max_batch_and_sample,
    pin_blas_threads,
    worker_initializer,
)
from src.visualizer import (
    print_summary_table,
    print_win_rate_table,
    save_charts_and_tables,
)

from .cash import inject_cash_if_needed
from .tax import print_tax_assumptions, rolling_tax_window_sum


def _safe_str(s: str) -> str:
    s_clean = unicodedata.normalize("NFKC", str(s))
    s_clean = re.sub(r"[^A-Za-z0-9\._ \-]+", "_", s_clean)
    s_clean = re.sub(r"\s+", "_", s_clean)
    s_clean = re.sub(r"_+", "_", s_clean).strip("_")
    return s_clean


def _safe_datestr(x):
    try:
        return x.date().isoformat()
    except AttributeError:
        return str(x)


def _from_settings(name: str, default):
    return getattr(settings, name, default)


# Windows 'spawn' safe worker: compute metrics for one window


def _metrics_task(returns_np, window, ppy, args, batch_size_eff, engine_eff) -> dict:
    from src import settings

    return calc_metrics_accel(
        returns=returns_np,
        window=int(window),
        ppy=int(ppy),
        risk_free_annual=settings.RISK_FREE_RATE,
        style=str(args.style),
        dca_amount=float(args.amount),
        initial_capital=float(args.initial),
        dates=None,
        dca_interval=str(args.dca_interval),
        batch_size=int(batch_size_eff),
        engine=str(engine_eff),
    )


def run_analysis(args) -> None:
    console = Console(width=200)

    # perf knobs: CLI > settings > built-in
    engine = getattr(args, "engine", None) or _from_settings("PERF_ENGINE", "auto")
    workers = getattr(args, "workers", None)
    if workers in (None, 0):
        workers = int(_from_settings("PERF_WORKERS", 8))
    max_ram_gb = getattr(args, "max_ram_gb", None)
    if max_ram_gb in (None, 0.0):
        max_ram_gb = float(_from_settings("PERF_MAX_RAM_GB", 0.0))
    batch_size_cli = getattr(args, "batch_size", None)
    if batch_size_cli in (None, 0):
        batch_size_cli = int(_from_settings("PERF_BATCH_SIZE", 256))
    render_plots = not bool(
        getattr(args, "no_plots", _from_settings("PERF_NO_PLOTS", False))
    )

    # decide effective engine BEFORE logging
    engine_eff = choose_engine(engine)

    # Pin BLAS threads
    pin_blas_threads(1)

    start_time = time.perf_counter()
    proc = psutil.Process(os.getpid())
    mem_peak_mb = 0.0

    cpu_cores = psutil.cpu_count(logical=True)
    ram_gb = psutil.virtual_memory().total / (1024**3)
    console.print(
        "[bold magenta]System Resources:[/] "
        + str(cpu_cores)
        + " Logical Cores "
        + f"{ram_gb:.1f}"
        + " GB RAM"
    )
    console.print("[bold magenta]Perf Pack:[/] " + human_hw_summary())
    console.print("-" * 80)

    cfg_msg = (
        "[bold cyan]Config:[/] Freq="
        + str(args.freq)
        + ", Rebalance="
        + str(args.rebalance)
        + ", Missing="
        + str(args.missing)
        + ", Style="
        + str(args.style)
        + ", DCA_Interval="
        + str(args.dca_interval)
        + ", TaxRate="
        + f"{args.tax_rate:.4f}"
        + ", CASH="
        + str(args.cash_mode)
        + ", Engine="
        + str(engine)
        + ", EngineEff="
        + str(engine_eff)
        + ", Workers="
        + str(workers)
        + ", MaxRAMGB="
        + f"{max_ram_gb:.2f}"
        + ", Batch="
        + str(batch_size_cli)
        + ", Plots="
        + ("ON" if render_plots else "OFF")
    )
    console.print(cfg_msg)
    print_tax_assumptions(
        console, tax_rate=float(args.tax_rate), rebalance=str(args.rebalance)
    )

    # Load data
    prices_pd = load_data(
        settings.DATA_DIR,
        settings.SERIES_SPECS,
        settings.START_DATE,
        settings.END_DATE,
        args.freq,
        args.missing,
    ).sort_index()
    returns_pd = (
        prices_pd.pct_change(fill_method=None)
        .dropna(how="all")
        .reset_index()
        .rename(columns={"Date": "Date"})
    )
    ret_cols = {c: f"Return_{c}" for c in returns_pd.columns if c != "Date"}
    returns_pd = returns_pd.rename(columns=ret_cols)
    returns_pd = inject_cash_if_needed(
        returns_pd,
        settings.PORTFOLIOS,
        args.freq,
        args.cash_mode,
        float(args.cash_fixed_rate),
        console,
    )

    merged_df = pl.from_pandas(returns_pd)
    actual_start = merged_df.select(pl.col("Date").min()).item()
    actual_end = merged_df.select(pl.col("Date").max()).item()
    console.print(
        "[bold cyan]Data Period:[/] "
        + str(actual_start)
        + " to "
        + str(actual_end)
        + "\n"
    )

    # Portfolio returns
    port_returns_df, tax_reports = calculate_portfolio_returns(
        merged_df,
        settings.PORTFOLIOS,
        str(args.rebalance),
        tax_rate=float(args.tax_rate),
    )
    portfolio_names = list(settings.PORTFOLIOS.keys())
    returns_dict = {
        p: port_returns_df.get_column(p).to_numpy() for p in portfolio_names
    }

    # Effective perf knobs
    batch_size_eff, sample_cap_eff = max_batch_and_sample(
        max_ram_gb, default_batch=batch_size_cli, default_sample_cap=4000
    )

    # Each rolling horizon independently
    for val in args.vals:
        window, ppy = get_window_and_ppy(int(val), str(args.unit), str(args.freq))
        console.print("=" * 80)
        head = (
            "[bold cyan]Rolling Window:[/] "
            + str(val)
            + " "
            + str(args.unit)
            + " (Window: "
            + str(window)
            + " periods) "
            + "[bold yellow]"
            + str(args.style).replace("_", " ")
            + "[/]"
        )
        console.print(head)
        console.print("=" * 80 + "\n")

        # per-window timing/memory baseline
        t_w0 = time.perf_counter()
        rss0_mb = proc.memory_info().rss / (1024 * 1024)

        # Compute metrics for THIS window
        results: Dict[str, dict] = {}
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers, initializer=worker_initializer
        ) as executor:
            futures = {}
            for p_name in portfolio_names:
                fut = executor.submit(
                    _metrics_task,
                    returns_dict[p_name],
                    int(window),
                    int(ppy),
                    args,
                    int(batch_size_eff),
                    str(engine_eff),
                )
                futures[fut] = p_name
            for fut in concurrent.futures.as_completed(futures):
                results[futures[fut]] = fut.result()

        # DCA enrich (optional)
        if str(args.style).lower() == "dca":
            enrich_results_with_dca_metrics(
                results=results,
                prices_df=prices_pd,
                portfolio_names=portfolio_names,
                window=window,
                ppy=ppy,
                amount=float(args.amount),
                initial_cap=float(args.initial),
                dca_interval=str(args.dca_interval),
            )

        # Typical VALUE paths (Mean/Median) from log-domain with sampling
        typical_value_mean: Dict[str, np.ndarray] = {}
        typical_value_median: Dict[str, np.ndarray] = {}
        date_arr = returns_pd["Date"].to_numpy()
        for p in portfolio_names:
            r = returns_dict[p].astype(np.float32, copy=False).reshape(-1)
            N = int(r.shape[0])
            if N < window:
                continue
            r_clip = np.clip(r, -0.999999, None)
            logP = np.cumsum(np.log1p(r_clip), dtype=np.float64)
            nW = N - window + 1
            m = int(min(nW, max(50, sample_cap_eff)))
            if m < 1:
                continue
            idxs = np.linspace(0, nW - 1, num=m, dtype=int)
            paths = np.empty((m, window), dtype=np.float32)
            for i, s in enumerate(idxs):
                seg = logP[s : s + window]
                paths[i, :] = np.exp(seg - seg[0])
            init_eff = (
                float(args.initial) if (args.initial and args.initial > 0) else 1.0
            )
            typical_value_mean[p] = (paths.mean(axis=0) * init_eff).astype(
                np.float32, copy=False
            )
            typical_value_median[p] = (np.median(paths, axis=0) * init_eff).astype(
                np.float32, copy=False
            )

        # Representative percentile paths (Min/P25/Med/P75/Max)
        #
        # IMPORTANT:
        # - In DCA mode we must plot VALUE paths in currency units (portfolio value),
        #   NOT normalized multiples. We therefore select representative windows
        #   using per-window Final_Value and rebuild full value paths including
        #   the contribution schedule.
        representative_paths: Dict[str, list] = {}
        rep_labels = ("Min", "P25", "Med", "P75", "Max")
        for p in portfolio_names:
            r = returns_dict[p].astype(np.float32, copy=False).reshape(-1)
            N = int(r.shape[0])
            if N < window:
                continue

            # DCA: use Final_Value computed by rolling metrics to pick windows,
            # then build value paths with contributions.
            if str(args.style).lower() == "dca":
                fv = results.get(p, {}).get("Final_Value", None)
                fv = (
                    np.asarray(fv, dtype=float)
                    if fv is not None
                    else np.asarray([], dtype=float)
                )
                if fv.size > 0:
                    from src.app.paths import build_representative_paths as _build_rep

                    representative_paths[p] = _build_rep(
                        returns=r,
                        dates=pd.Series(returns_pd["Date"]),
                        window=int(window),
                        style=str(args.style),
                        amount=float(args.amount),
                        initial_cap=float(args.initial),
                        dca_interval=str(args.dca_interval),
                        final_values=fv,
                        labels=rep_labels,
                    )
                    continue

            # Lump Sum (or fallback): percentile picks based on normalized end value.
            r_clip = np.clip(r, -0.999999, None)
            logP = np.cumsum(np.log1p(r_clip), dtype=np.float64)
            final_log = logP[window - 1 :] - logP[: N - window + 1]
            final_val = np.exp(final_log).astype(np.float64, copy=False)

            rep_list = []
            nW = final_val.shape[0]
            if nW <= 0:
                continue
            order = np.argsort(final_val)
            q_map = {"Min": 0.00, "P25": 0.25, "Med": 0.50, "P75": 0.75, "Max": 1.00}
            for lab in rep_labels:
                q = q_map[lab]
                k = int(round(q * (nW - 1)))
                s = int(order[k])
                seg = logP[s : s + window]
                path_norm = np.exp(seg - seg[0]).astype(np.float32, copy=False)
                init_eff = (
                    float(args.initial) if (args.initial and args.initial > 0) else 1.0
                )
                values = (init_eff * path_norm).astype(np.float32, copy=False)
                principal = np.full((window,), init_eff, dtype=np.float32)
                dates = date_arr[s : s + window]
                rep_list.append(
                    {
                        "label": lab,
                        "dates": dates,
                        "values": values,
                        "principal": principal,
                    }
                )
            representative_paths[p] = rep_list

        # Console tables

        print_summary_table(
            console,
            portfolio_names,
            results,
            val,
            str(args.unit),
            str(args.style),
            float(args.amount),
            float(args.initial),
            window,
            ppy=ppy,
            dca_interval=str(args.dca_interval),
        )
        print_win_rate_table(console, portfolio_names, results)

        # Optional: rolling tax summary

        try:
            roll_tax_sums = None

            # tax_reports が dict（ポートフォリオ別）なら、dict[str, ndarray] を作る
            if isinstance(tax_reports, dict):
                roll_tax_sums = {
                    p: rolling_tax_window_sum(pd.Series(tax_reports[p]), window)
                    for p in portfolio_names
                    if p in tax_reports
                }
            else:
                # tax_reports が単一系列なら、キーを固定して dict に包む（必要に応じてキー名変更）
                roll_tax_sums = {
                    "tax": rolling_tax_window_sum(pd.Series(tax_reports), window)
                }

        except Exception:
            roll_tax_sums = None

        if render_plots:
            # Per-window output subfolder
            group = "_".join([_safe_str(p) for p in portfolio_names])
            sub = (
                f"{group}__{_safe_datestr(actual_start)}_to_{_safe_datestr(actual_end)}_"
                f"{val}{str(args.unit)}_{str(args.freq)}_reb-{str(args.rebalance)}_"
                f"tax-rate_{float(args.tax_rate):.1f}_{str(args.style)}"
            )
            if str(args.style).lower() == "dca":
                sub += f"_init_{int(args.initial)}_amount_{int(args.amount)}"
            output_dir_win = settings.OUTPUT_DIR / sub
            output_dir_win.mkdir(parents=True, exist_ok=True)
            console.print(
                f"saving charts and tables to: {str(output_dir_win.resolve())}"
            )

            save_charts_and_tables(
                output_dir_win,
                portfolio_names,
                results,
                val,
                str(args.unit),
                actual_start,
                actual_end,
                str(args.freq),
                str(args.rebalance),
                str(args.missing),
                str(args.style),
                float(args.amount),
                float(args.initial),
                window,
                prices_pd,
                dca_interval=str(args.dca_interval),
                ppy=ppy,
                tax_reports=tax_reports,
                roll_paths_mean=None,
                roll_paths_median=None,
                roll_tax_sums=roll_tax_sums,
                typical_value_mean=typical_value_mean,
                typical_value_median=typical_value_median,
                representative_paths=representative_paths,
            )

        # per-window metrics
        rss1_mb = proc.memory_info().rss / (1024 * 1024)
        elapsed_w = time.perf_counter() - t_w0
        mem_peak_mb = max(mem_peak_mb, rss1_mb)
        console.print(
            f"[green][Window Metrics][/green] {val} {args.unit}: elapsed={elapsed_w:.3f}s, "
            f"rss_now={rss1_mb:.1f}MB, rss_delta={rss1_mb - rss0_mb:+.1f}MB, engine={engine_eff}, workers={workers}, batch={batch_size_eff}"
        )

    # overall metrics (after loop)
    memory_mb = proc.memory_info().rss / (1024 * 1024)
    elapsed = time.perf_counter() - start_time
    console.print("-" * 80)
    console.print(
        "[bold magenta]Performance:[/] Finished in "
        + f"{elapsed:.3f}"
        + " seconds "
        + "Current RSS: "
        + f"{memory_mb:.1f}"
        + " MB "
        + "Peak RSS (observed): "
        + f"{mem_peak_mb:.1f}"
        + " MB"
    )
    console.print("-" * 80)
