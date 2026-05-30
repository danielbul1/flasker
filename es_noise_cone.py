#!/usr/bin/env python3
"""
ES Noise-Cone Breakout Analysis
---------------------------------
Builds a dynamic intraday corridor for each hour of the trading day:

    upper_cone[day D, hour H] = open_D × (1 + mean(cum_max_up_to_H) over last N days)
    lower_cone[day D, hour H] = open_D × (1 - mean(cum_max_dn_to_H) over last N days)

A breakout occurs when today's running high/low exceeds the cone.
The forward return from breakout bar to day close measures signal quality.

Parameters optimised exhaustively via walk-forward Sharpe:
  - N  : lookback window (days)
  - th : threshold multiplier (cone × th before triggering)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from itertools import product

from es_analysis import (load_data, build_daily, add_atr, MACRO_DATES,
                          RTH_OPEN_BUCKET, RTH_CLOSE_BUCKET)


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 — CUMULATIVE EXCURSION PER (DAY, HOUR)
# ──────────────────────────────────────────────────────────────────────────────

def build_cum_excursion(df: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """
    For every bar, compute the cumulative max upside / downside excursion
    from the daily open UP TO AND INCLUDING this bar (i.e., the running
    high/low for the day so far).

    Returns df with extra columns:
      cum_up  = (running_high_today - daily_open) / daily_open
      cum_dn  = (daily_open - running_low_today)  / daily_open
    """
    rows = []
    for d, grp in df.groupby("date"):
        if d not in daily.index:
            continue
        op  = daily.loc[d, "daily_open"]
        cls = daily.loc[d, "rth_close"]   # forward return measured to RTH close
        if pd.isna(op) or pd.isna(cls):
            continue

        # Only use RTH bars for the cone (9:30 AM → 4:00 PM)
        grp = grp[(grp["time_bucket"] >= RTH_OPEN_BUCKET) &
                  (grp["time_bucket"] <= RTH_CLOSE_BUCKET)].sort_index()
        if grp.empty:
            continue

        cum_up = ((grp["high"] - op) / op).cummax()
        cum_dn = ((op - grp["low"])  / op).cummax()
        tmp = grp.copy()
        tmp["cum_up"]      = cum_up.values
        tmp["cum_dn"]      = cum_dn.values
        tmp["daily_open"]  = op
        tmp["daily_close"] = cls
        rows.append(tmp)

    return pd.concat(rows).sort_index()


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 — CONE BOUNDARIES (NO LOOK-AHEAD)
# ──────────────────────────────────────────────────────────────────────────────

def build_cone(excursion_df: pd.DataFrame,
               n: int,
               exclude_macro: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build pivot tables (days × hours) for cone upper and lower boundaries.

    cone_up[D, H]  = open_D × (1 + rolling_N_mean of cum_up at hour H, prior days)
    cone_dn[D, H]  = open_D × (1 - rolling_N_mean of cum_dn at hour H, prior days)

    Macro days are NaN-masked before rolling so they don't pollute the mean,
    but the output contains ALL days (cone can still be computed for macro days
    using prior non-macro history).

    Returns (cone_up_prices, cone_dn_prices) — both indexed by date, columns = hours.
    """
    # Pivot over ALL days so the index is complete
    # Use time_bucket (minutes from midnight) as the column — works for any resolution
    piv_up = (excursion_df.groupby(["date", "time_bucket"])["cum_up"]
              .mean().unstack("time_bucket"))
    piv_dn = (excursion_df.groupby(["date", "time_bucket"])["cum_dn"]
              .mean().unstack("time_bucket"))

    # Mask macro days with NaN so they don't contribute to rolling mean
    if exclude_macro:
        macro_idx = [d for d in piv_up.index if d in MACRO_DATES]
        piv_up.loc[macro_idx] = np.nan
        piv_dn.loc[macro_idx] = np.nan

    daily_opens = excursion_df.groupby("date")["daily_open"].first()

    # Rolling mean of PRIOR N days (shift(1) = strict no look-ahead)
    min_p = max(1, n // 2)
    mean_up = piv_up.shift(1).rolling(n, min_periods=min_p).mean()
    mean_dn = piv_dn.shift(1).rolling(n, min_periods=min_p).mean()

    # Convert to absolute price levels using TODAY's open (aligned on index)
    opens_aligned = daily_opens.reindex(mean_up.index)
    cone_up_prices = mean_up.multiply(opens_aligned, axis=0).add(opens_aligned, axis=0)
    cone_dn_prices = opens_aligned.to_frame("_").values - mean_dn.multiply(opens_aligned, axis=0).values
    cone_dn_prices = pd.DataFrame(cone_dn_prices,
                                  index=mean_dn.index,
                                  columns=mean_dn.columns)

    return cone_up_prices, cone_dn_prices


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 — FIND BREAKOUTS
# ──────────────────────────────────────────────────────────────────────────────

def find_breakouts(excursion_df: pd.DataFrame,
                   cone_up: pd.DataFrame,
                   cone_dn: pd.DataFrame,
                   threshold: float = 1.0,
                   exclude_macro: bool = True) -> pd.DataFrame:
    """
    For each day, find the FIRST bar where the running high/low exceeds
    the cone boundary × threshold.

    threshold > 1.0 requires a stronger breakout (e.g. 1.05 = 5% beyond cone).
    threshold < 1.0 triggers earlier (inside the cone — not recommended).

    Returns DataFrame:
      date, hour, direction ('up'/'dn'), bar_close,
      daily_close, fwd_return (to close), fwd_return_direction (>0 = continuation)
    """
    records = []

    for d, grp in excursion_df.groupby("date"):
        if d not in cone_up.index:
            continue
        if exclude_macro and d in MACRO_DATES:
            continue

        grp = grp.sort_index()
        op  = grp["daily_open"].iloc[0]
        cls = grp["daily_close"].iloc[0]

        triggered_up = triggered_dn = False

        for _, row in grp.iterrows():
            tb = int(row["time_bucket"])

            # UPSIDE breakout
            if not triggered_up and tb in cone_up.columns:
                c_up = cone_up.loc[d, tb]
                if pd.notna(c_up) and row["high"] > c_up * threshold:
                    fwd = (cls - row["close"]) / row["close"]
                    records.append(dict(date=d, time_bucket=tb, direction="up",
                                        bar_close=row["close"], daily_close=cls,
                                        fwd_return=fwd, cone_level=c_up))
                    triggered_up = True

            # DOWNSIDE breakout
            if not triggered_dn and tb in cone_dn.columns:
                c_dn = cone_dn.loc[d, tb]
                if pd.notna(c_dn) and row["low"] < c_dn / threshold:
                    fwd = (row["close"] - cls) / row["close"]
                    records.append(dict(date=d, time_bucket=tb, direction="dn",
                                        bar_close=row["close"], daily_close=cls,
                                        fwd_return=fwd, cone_level=c_dn))
                    triggered_dn = True

            if triggered_up and triggered_dn:
                break

    return pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4 — EVALUATE A (N, THRESHOLD) PAIR
# ──────────────────────────────────────────────────────────────────────────────

def sharpe(returns: pd.Series) -> float:
    if len(returns) < 5 or returns.std() == 0:
        return np.nan
    return returns.mean() / returns.std() * np.sqrt(252)  # annualised


def evaluate(excursion_df: pd.DataFrame,
             n: int,
             threshold: float = 1.0,
             train_frac: float = 0.70,
             exclude_macro: bool = True) -> dict:
    """
    Walk-forward evaluation: build cone on all days (rolling, no look-ahead),
    find breakouts, split by time → report test-set metrics.
    """
    cone_up, cone_dn = build_cone(excursion_df, n, exclude_macro)
    bk = find_breakouts(excursion_df, cone_up, cone_dn, threshold, exclude_macro)

    if bk.empty:
        return dict(n=n, th=threshold, n_signals=0,
                    sharpe=np.nan, hit_rate=np.nan, mean_fwd=np.nan)

    all_dates = sorted(bk["date"].unique())
    split_idx = int(len(all_dates) * train_frac)
    if split_idx >= len(all_dates):
        return dict(n=n, th=threshold, n_signals=0,
                    sharpe=np.nan, hit_rate=np.nan, mean_fwd=np.nan)

    test_dates = set(all_dates[split_idx:])
    test_bk = bk[bk["date"].isin(test_dates)]

    if len(test_bk) < 10:
        return dict(n=n, th=threshold, n_signals=len(test_bk),
                    sharpe=np.nan, hit_rate=np.nan, mean_fwd=np.nan)

    fwd = test_bk["fwd_return"]
    return dict(
        n         = n,
        th        = threshold,
        n_signals = len(test_bk),
        sharpe    = sharpe(fwd),
        hit_rate  = (fwd > 0).mean(),
        mean_fwd  = fwd.mean(),
    )


# ──────────────────────────────────────────────────────────────────────────────
# STEP 5 — GRID SEARCH OVER ALL (N, THRESHOLD) PAIRS
# ──────────────────────────────────────────────────────────────────────────────

def grid_search(excursion_df: pd.DataFrame,
                n_range: range = range(2, 61),
                thresholds: list[float] = None,
                exclude_macro: bool = True,
                verbose: bool = True) -> pd.DataFrame:
    """
    Test every (N, threshold) combination.
    Returns sorted DataFrame of results.
    """
    if thresholds is None:
        thresholds = [1.0, 1.01, 1.02, 1.05]

    results = []
    total = len(n_range) * len(thresholds)
    done  = 0

    for n, th in product(n_range, thresholds):
        row = evaluate(excursion_df, n, th, exclude_macro=exclude_macro)
        results.append(row)
        done += 1
        if verbose and done % len(thresholds) == 0:
            best_so_far = max((r["sharpe"] for r in results if not np.isnan(r.get("sharpe", np.nan))),
                              default=np.nan)
            print(f"  [{done:3d}/{total}] N={n}  best Sharpe so far={best_so_far:.3f}")

    df_res = pd.DataFrame(results).sort_values("sharpe", ascending=False)
    return df_res


# ──────────────────────────────────────────────────────────────────────────────
# VISUALISATION
# ──────────────────────────────────────────────────────────────────────────────

def plot_grid(grid: pd.DataFrame,
              best_n: int,
              best_th: float,
              out: str = "es_cone_grid.png") -> None:

    fig = plt.figure(figsize=(17, 10))
    fig.suptitle("ES Noise-Cone — Grid Search Results (walk-forward, macro filtered)",
                 fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.38)

    thresholds = sorted(grid["th"].unique())
    colors     = ["royalblue", "darkorange", "green", "red"]

    # ── 1. Sharpe vs N (per threshold) ───────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    for th, col in zip(thresholds, colors):
        sub = grid[grid["th"] == th].sort_values("n")
        ax1.plot(sub["n"], sub["sharpe"], "-", color=col, lw=1.2, label=f"th={th}")
    ax1.axvline(best_n, color="black", ls="--", lw=1, label=f"best N={best_n}")
    ax1.set(xlabel="Lookback N (days)", ylabel="Walk-forward Sharpe",
            title="Sharpe vs N")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.25)

    # ── 2. Hit-rate vs N ─────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    for th, col in zip(thresholds, colors):
        sub = grid[grid["th"] == th].sort_values("n")
        ax2.plot(sub["n"], sub["hit_rate"], "-", color=col, lw=1.2, label=f"th={th}")
    ax2.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax2.axvline(best_n, color="black", ls="--", lw=1)
    ax2.set(xlabel="Lookback N (days)", ylabel="Hit rate (fwd > 0)",
            title="Hit Rate vs N")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.25)

    # ── 3. N signals vs N ────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    for th, col in zip(thresholds, colors):
        sub = grid[grid["th"] == th].sort_values("n")
        ax3.plot(sub["n"], sub["n_signals"], "-", color=col, lw=1.2, label=f"th={th}")
    ax3.set(xlabel="Lookback N (days)", ylabel="Signals (test set)",
            title="Signal count vs N")
    ax3.legend(fontsize=8); ax3.grid(alpha=0.25)

    # ── 4. Heatmap: Sharpe (N × threshold) ──────────────────────────────────
    ax4 = fig.add_subplot(gs[1, :2])
    heat = grid.pivot_table(index="n", columns="th", values="sharpe")
    im = ax4.imshow(heat.values, aspect="auto", cmap="RdYlGn", origin="lower",
                    vmin=-0.5, vmax=1.5)
    ax4.set_xticks(range(len(heat.columns)))
    ax4.set_xticklabels([f"{v:.2f}" for v in heat.columns], fontsize=8)
    ax4.set_yticks(range(0, len(heat.index), 5))
    ax4.set_yticklabels(heat.index[::5], fontsize=8)
    ax4.set(xlabel="Threshold", ylabel="N (days)", title="Sharpe Heatmap (N × threshold)")
    plt.colorbar(im, ax=ax4)

    # ── 5. Top 10 combos ─────────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    top = grid.dropna(subset=["sharpe"]).head(10)
    labels = [f"N={int(r.n)} th={r.th:.2f}" for _, r in top.iterrows()]
    ax5.barh(labels[::-1], top["sharpe"].values[::-1], color="steelblue", alpha=0.8)
    ax5.axvline(0, color="black", lw=0.8)
    ax5.set(xlabel="Walk-forward Sharpe", title="Top 10 (N, threshold) combos")
    ax5.grid(alpha=0.25)

    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Grid search chart saved → {out}")


def plot_cone_day(excursion_df: pd.DataFrame,
                  cone_up: pd.DataFrame,
                  cone_dn: pd.DataFrame,
                  date_sample=None,
                  out: str = "es_cone_day.png") -> None:
    """Plot cone + price path for a sample of days."""
    dates = excursion_df["date"].unique()
    if date_sample is None:
        rng = np.random.default_rng(42)
        date_sample = sorted(rng.choice(dates, size=min(6, len(dates)), replace=False))

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    fig.suptitle("Noise-Cone vs Actual Price Path (sample days)", fontsize=12)

    for ax, d in zip(axes.flat, date_sample):
        if d not in cone_up.index:
            ax.set_visible(False)
            continue
        grp = excursion_df[excursion_df["date"] == d].sort_values("hour")
        buckets = grp["time_bucket"].values
        price   = grp["close"].values
        hi      = grp["high"].values
        lo      = grp["low"].values

        up_vals = [cone_up.loc[d, tb] if tb in cone_up.columns else np.nan for tb in buckets]
        dn_vals = [cone_dn.loc[d, tb] if tb in cone_dn.columns else np.nan for tb in buckets]

        ax.fill_between(buckets, dn_vals, up_vals, alpha=0.15, color="blue", label="Cone")
        ax.plot(buckets, up_vals, "b--", lw=0.8)
        ax.plot(buckets, dn_vals, "b--", lw=0.8)
        ax.plot(buckets, price,   "k-",  lw=1.2, label="Close")
        ax.plot(buckets, hi,      "g.",  ms=3)
        ax.plot(buckets, lo,      "r.",  ms=3)
        ax.set_title(str(d), fontsize=9)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Day chart saved → {out}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def run(symbol: str = "ES=F",
        interval: str = "5m",
        period: str = "60d",
        csv_path: str | None = None,
        n_range: range = range(2, 31),
        thresholds: list[float] = None,
        exclude_macro: bool = True) -> dict:

    if thresholds is None:
        thresholds = [1.0, 1.01, 1.02, 1.05]

    print("── Loading data ──────────────────────────────────")
    df = load_data(symbol=symbol, period=period, interval=interval, csv_path=csv_path)
    print(f"  {len(df):,} bars ({interval})  |  {df['date'].min()} → {df['date'].max()}")

    print("── Building daily stats ──────────────────────────")
    daily = build_daily(df)
    n_macro = daily["is_macro"].sum()
    print(f"  {len(daily)} days  |  {n_macro} macro days {'(excluded from cone fit)' if exclude_macro else ''}")

    print("── Computing cumulative excursions ───────────────")
    exc_df = build_cum_excursion(df, daily)
    print(f"  {len(exc_df):,} bars with excursion data")

    print(f"── Grid search: N={list(n_range)[0]}–{list(n_range)[-1]}, "
          f"thresholds={thresholds} ──")
    grid = grid_search(exc_df, n_range=n_range, thresholds=thresholds,
                       exclude_macro=exclude_macro, verbose=True)

    valid = grid.dropna(subset=["sharpe"])
    if valid.empty:
        print("  No valid results — not enough data for test set.")
        return {}

    best = valid.iloc[0]
    best_n  = int(best["n"])
    best_th = float(best["th"])
    print(f"\n── Best combination ──────────────────────────────")
    print(f"  N={best_n}, threshold={best_th:.2f}")
    print(f"  Sharpe={best['sharpe']:.3f}  |  Hit rate={best['hit_rate']:.2%}  "
          f"|  Signals={int(best['n_signals'])}")

    print("\n── Top 10 ────────────────────────────────────────")
    print(valid.head(10).to_string(index=False,
          float_format="{:.3f}".format))

    print("\n── Plotting ──────────────────────────────────────")
    plot_grid(grid, best_n, best_th)

    # Plot cone for best N on sample days
    cone_up, cone_dn = build_cone(exc_df, best_n, exclude_macro)
    plot_cone_day(exc_df, cone_up, cone_dn)

    return dict(grid=grid, best_n=best_n, best_th=best_th,
                exc_df=exc_df, cone_up=cone_up, cone_dn=cone_dn)


if __name__ == "__main__":
    run()
