#!/usr/bin/env python3
"""
ES Excursion & Reversion Analysis
----------------------------------
For each normalized excursion level X (relative to ATR):
  - P(this is the HOD / LOD of the day)
  - Reversion to close  (mean, median, std)
  - Reversion to opposite extreme (mean)
  - Sample size n per bin

Macro days (FOMC, CPI, NFP) are filtered by default.
ATR window N is chosen by minimising walk-forward Brier score.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yfinance as yf
from zoneinfo import ZoneInfo
from datetime import date, timedelta

NY_TZ = ZoneInfo("America/New_York")

# ──────────────────────────────────────────────────────────────────────────────
# MACRO DATES
# ──────────────────────────────────────────────────────────────────────────────

# FOMC decision days (day the statement is released, ~2:00 PM ET)
FOMC_DATES = {
    # 2023
    date(2023,  2,  1), date(2023,  3, 22), date(2023,  5,  3),
    date(2023,  6, 14), date(2023,  7, 26), date(2023,  9, 20),
    date(2023, 11,  1), date(2023, 12, 13),
    # 2024
    date(2024,  1, 31), date(2024,  3, 20), date(2024,  5,  1),
    date(2024,  6, 12), date(2024,  7, 31), date(2024,  9, 18),
    date(2024, 11,  7), date(2024, 12, 18),
    # 2025
    date(2025,  1, 29), date(2025,  3, 19), date(2025,  5,  7),
    date(2025,  6, 18), date(2025,  7, 30),
}

# CPI release dates (BLS, 8:30 AM ET)
CPI_DATES = {
    # 2023
    date(2023,  1, 12), date(2023,  2, 14), date(2023,  3, 14),
    date(2023,  4, 12), date(2023,  5, 10), date(2023,  6, 13),
    date(2023,  7, 12), date(2023,  8, 10), date(2023,  9, 13),
    date(2023, 10, 12), date(2023, 11, 14), date(2023, 12, 12),
    # 2024
    date(2024,  1, 11), date(2024,  2, 13), date(2024,  3, 12),
    date(2024,  4, 10), date(2024,  5, 15), date(2024,  6, 12),
    date(2024,  7, 11), date(2024,  8, 14), date(2024,  9, 11),
    date(2024, 10, 10), date(2024, 11, 13), date(2024, 12, 11),
    # 2025
    date(2025,  1, 15), date(2025,  2, 12), date(2025,  3, 12),
    date(2025,  4, 10), date(2025,  5, 13), date(2025,  6, 11),
}


def _first_friday(year: int, month: int) -> date:
    d = date(year, month, 1)
    return d + timedelta(days=(4 - d.weekday()) % 7)


NFP_DATES = {
    _first_friday(y, m)
    for y in range(2023, 2026)
    for m in range(1, 13)
    if date(y, m, 1) <= date.today()
}

MACRO_DATES: set[date] = FOMC_DATES | CPI_DATES | NFP_DATES


# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────

def load_data(symbol: str = "ES=F",
              period: str = "730d",
              interval: str = "1h",
              csv_path: str | None = None) -> pd.DataFrame:
    """
    Load hourly data and convert index to America/New_York.
    Returns DataFrame with columns: open, high, low, close, volume, date, hour.
    """
    if csv_path:
        df = pd.read_csv(csv_path, parse_dates=["Datetime"], index_col="Datetime")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
    else:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval, auto_adjust=True)
        if df.empty:
            raise ValueError(f"yfinance returned no data for {symbol!r}")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")

    # ── Critical: convert to NY time ──────────────────────────────────────────
    df.index = df.index.tz_convert("America/New_York")
    df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].copy()

    df["date"] = df.index.date
    df["hour"] = df.index.hour
    return df


# ──────────────────────────────────────────────────────────────────────────────
# DAILY STATS
# ──────────────────────────────────────────────────────────────────────────────

def build_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate hourly bars to daily level.
    daily_open  = open of the 00:00 NY bar (first bar at midnight).
    daily_high  = max(high) over all bars that day.
    daily_low   = min(low)  over all bars that day.
    daily_close = last close of the day.
    range       = daily_high - daily_low.
    is_macro    = True if date is in MACRO_DATES.
    """
    midnight = (
        df[df["hour"] == 0]
        .groupby("date")["open"]
        .first()
        .rename("daily_open")
    )

    ohlc = df.groupby("date").agg(
        daily_high  = ("high",  "max"),
        daily_low   = ("low",   "min"),
        daily_close = ("close", "last"),
    )

    daily = ohlc.join(midnight, how="inner")   # drop days without a 00:00 bar
    daily["range"]    = daily["daily_high"] - daily["daily_low"]
    daily["is_macro"] = pd.Series(
        {d: d in MACRO_DATES for d in daily.index}, dtype=bool
    )
    daily["is_up"] = daily["daily_close"] > daily["daily_open"]
    return daily


# ──────────────────────────────────────────────────────────────────────────────
# ATR
# ──────────────────────────────────────────────────────────────────────────────

def add_atr(daily: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Add column atr_<n> = rolling(n) mean of *previous* days' ranges.
    shift(1) ensures strictly no look-ahead.
    """
    daily = daily.copy()
    daily[f"atr_{n}"] = (
        daily["range"]
        .shift(1)
        .rolling(n, min_periods=max(1, n // 2))
        .mean()
    )
    return daily


# ──────────────────────────────────────────────────────────────────────────────
# PROBABILITY TABLE
# ──────────────────────────────────────────────────────────────────────────────

def compute_table(daily: pd.DataFrame,
                  n: int,
                  bins: np.ndarray | None = None,
                  exclude_macro: bool = True) -> pd.DataFrame:
    """
    For each bin level X (in units of ATR):

    UP direction:
      - reached_up  : days where norm_high >= X
      - p_extreme   : P(norm_high < X + bin_step | reached X)
                      = fraction of reached days where price stayed below X+step
      - rev_close   : mean/median/std of (norm_close - X) for reached days
                        negative = mean reversion back toward open
      - rev_opp     : mean of (X - norm_low) for reached days
                        positive = how far below X the low went (max reversion)

    DOWN direction: symmetric, using norm_low.
    """
    d = add_atr(daily, n)
    if exclude_macro:
        d = d[~d["is_macro"]]
    d = d.dropna(subset=[f"atr_{n}"])

    atr_col = f"atr_{n}"
    d["norm_high"]  = (d["daily_high"]  - d["daily_open"]) / d[atr_col]
    d["norm_low"]   = (d["daily_low"]   - d["daily_open"]) / d[atr_col]
    d["norm_close"] = (d["daily_close"] - d["daily_open"]) / d[atr_col]

    if bins is None:
        bins = np.round(np.arange(0.10, 3.01, 0.10), 2)

    bin_step = round(bins[1] - bins[0], 4) if len(bins) > 1 else 0.10
    rows = []

    for x in bins:
        x = round(float(x), 4)
        next_x = round(x + bin_step, 4)

        for direction in ("up", "down"):
            if direction == "up":
                reached = d[d["norm_high"] >= x]
                # Price stopped within this bin (didn't exceed next_x)
                p_extreme = (reached["norm_high"] < next_x).mean() if len(reached) else np.nan
                rev_close = reached["norm_close"] - x
                # How far the price came back from x toward the open (favorable for shorts)
                rev_opp   = x - reached["norm_low"]
            else:
                reached = d[d["norm_low"] <= -x]
                p_extreme = (reached["norm_low"] > -next_x).mean() if len(reached) else np.nan
                rev_close = reached["norm_close"] - (-x)
                rev_opp   = reached["norm_high"] - (-x)

            nn = len(reached)
            if nn < 5:
                continue

            rows.append(dict(
                x           = x,
                direction   = direction,
                n           = nn,
                reliable    = nn >= 20,
                p_extreme   = p_extreme,
                # reversion to close
                rev_close_mean   = rev_close.mean(),
                rev_close_median = rev_close.median(),
                rev_close_std    = rev_close.std(),
                # reversion to opposite extreme
                rev_opp_mean     = rev_opp.mean(),
                rev_opp_median   = rev_opp.median(),
            ))

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# BRIER SCORE / N OPTIMISATION
# ──────────────────────────────────────────────────────────────────────────────

def brier_for_n(daily: pd.DataFrame,
                n: int,
                exclude_macro: bool = True,
                train_frac: float = 0.70) -> float:
    """
    Walk-forward Brier score for ATR(n).

    For every test day and every possible intraday excursion level up to the
    actual daily high, we predict P(that level is the HOD) using the training
    empirical distribution, then score against the actual binary outcome.
    """
    d = add_atr(daily, n)
    if exclude_macro:
        d = d[~d["is_macro"]]
    d = d.dropna(subset=[f"atr_{n}"])
    if len(d) < 40:
        return np.nan

    split   = int(len(d) * train_frac)
    train   = d.iloc[:split]
    test    = d.iloc[split:]
    if len(test) < 15:
        return np.nan

    atr_col = f"atr_{n}"
    norm_h_train = (train["daily_high"] - train["daily_open"]) / train[atr_col]

    levels = np.round(np.arange(0.10, 3.01, 0.10), 2)
    scores = []

    for _, row in test.iterrows():
        if pd.isna(row[atr_col]) or row[atr_col] == 0:
            continue
        actual_norm_high = (row["daily_high"] - row["daily_open"]) / row[atr_col]

        for x in levels:
            if x > actual_norm_high + 0.05:
                break    # once we pass the actual high there's nothing left to score
            x_next = round(x + 0.10, 2)
            reached_train = norm_h_train[norm_h_train >= x]
            if len(reached_train) < 8:
                continue
            p_pred  = (reached_train < x_next).mean()
            outcome = 1 if actual_norm_high < x_next else 0
            scores.append((p_pred - outcome) ** 2)

    return float(np.mean(scores)) if scores else np.nan


def optimise_n(daily: pd.DataFrame,
               n_range: range = range(1, 31),
               exclude_macro: bool = True,
               verbose: bool = True) -> pd.Series:
    """Test every N in n_range; return Series(Brier score, index=N)."""
    results = {}
    for n in n_range:
        score = brier_for_n(daily, n, exclude_macro=exclude_macro)
        results[n] = score
        if verbose:
            tag = f"{score:.5f}" if not np.isnan(score) else "n/a"
            print(f"  N={n:2d} → Brier={tag}")
    return pd.Series(results, name="brier")


# ──────────────────────────────────────────────────────────────────────────────
# VISUALISATION
# ──────────────────────────────────────────────────────────────────────────────

def plot_results(table: pd.DataFrame,
                 n: int,
                 brier: pd.Series | None = None,
                 out: str = "es_analysis.png") -> None:

    fig = plt.figure(figsize=(17, 11))
    fig.suptitle(
        f"ES Excursion & Reversion  —  ATR({n}) normalised  |  macro filtered",
        fontsize=13, fontweight="bold"
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    up = table[table["direction"] == "up"].set_index("x")
    dn = table[table["direction"] == "down"].set_index("x")

    # ── 1. P(extreme) ────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(up.index, up["p_extreme"], "g-o", ms=3, label="P(HOD)")
    ax1.plot(dn.index, dn["p_extreme"], "r-o", ms=3, label="P(LOD)")
    ax1.axhline(0.5, color="gray", ls="--", alpha=0.5, lw=0.8)
    ax1.set(xlabel=f"Excursion (× ATR{n})", ylabel="Probability",
            title="P(this level is the HOD / LOD)")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.25)

    # ── 2. Sample size ───────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    w = 0.04
    ax2.bar(up.index, up["n"], width=w, color="green", alpha=0.55, label="Up")
    ax2.bar(dn.index + w, dn["n"], width=w, color="red",   alpha=0.55, label="Down")
    ax2.axhline(20, color="orange", ls="--", lw=0.9, label="n=20 threshold")
    ax2.set(xlabel=f"Excursion (× ATR{n})", ylabel="Days (n)",
            title="Sample size per bin")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.25)

    # ── 3. Brier score vs N ──────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    if brier is not None and not brier.dropna().empty:
        b = brier.dropna()
        ax3.plot(b.index, b.values, "o-", color="purple", ms=4)
        best = int(b.idxmin())
        ax3.axvline(best, color="orange", ls="--", lw=1.2, label=f"Best N={best}")
        ax3.set(xlabel="ATR window N (days)", ylabel="Brier score (↓ better)",
                title="N optimisation (walk-forward)")
        ax3.legend(fontsize=8); ax3.grid(alpha=0.25)
    else:
        ax3.text(0.5, 0.5, "Run with optimise=True\nto see Brier curve",
                 ha="center", va="center", transform=ax3.transAxes, fontsize=10)

    # ── 4. Reversion to close ────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(up.index, up["rev_close_mean"], "g-",   label="Up (mean)")
    ax4.plot(up.index, up["rev_close_median"], "g--", lw=0.8, label="Up (median)")
    ax4.fill_between(up.index,
                     up["rev_close_mean"] - up["rev_close_std"],
                     up["rev_close_mean"] + up["rev_close_std"],
                     alpha=0.15, color="green")
    ax4.plot(dn.index, dn["rev_close_mean"], "r-",   label="Down (mean)")
    ax4.plot(dn.index, dn["rev_close_median"], "r--", lw=0.8, label="Down (median)")
    ax4.fill_between(dn.index,
                     dn["rev_close_mean"] - dn["rev_close_std"],
                     dn["rev_close_mean"] + dn["rev_close_std"],
                     alpha=0.15, color="red")
    ax4.axhline(0, color="black", lw=0.6, alpha=0.5)
    ax4.set(xlabel=f"Excursion (× ATR{n})", ylabel="Reversion (normalised)",
            title="Reversion to close (mean ± 1σ)")
    ax4.legend(fontsize=7); ax4.grid(alpha=0.25)

    # ── 5. Reversion to opposite extreme ────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(up.index, up["rev_opp_mean"],   "g-",  label="Up → LOD mean")
    ax5.plot(up.index, up["rev_opp_median"], "g--", lw=0.8, label="Up → LOD median")
    ax5.plot(dn.index, dn["rev_opp_mean"],   "r-",  label="Down → HOD mean")
    ax5.plot(dn.index, dn["rev_opp_median"], "r--", lw=0.8, label="Down → HOD median")
    ax5.axhline(0, color="black", lw=0.6, alpha=0.5)
    ax5.set(xlabel=f"Excursion (× ATR{n})", ylabel="Reversion (normalised)",
            title="Reversion to opposite extreme")
    ax5.legend(fontsize=7); ax5.grid(alpha=0.25)

    # ── 6. Up vs Down asymmetry ──────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    common_x = up.index.intersection(dn.index)
    asym = up.loc[common_x, "p_extreme"].values - dn.loc[common_x, "p_extreme"].values
    ax6.bar(common_x, asym, width=0.07,
            color=["green" if v >= 0 else "red" for v in asym], alpha=0.7)
    ax6.axhline(0, color="black", lw=0.8)
    ax6.set(xlabel=f"Excursion (× ATR{n})", ylabel="P(HOD) − P(LOD)",
            title="Up / Down asymmetry")
    ax6.grid(alpha=0.25)

    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Chart saved → {out}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def run(symbol: str = "ES=F",
        period: str = "730d",
        csv_path: str | None = None,
        optimise: bool = True,
        n_override: int | None = None,
        exclude_macro: bool = True,
        n_range: range = range(1, 31)) -> dict:
    """
    Full pipeline.  Returns dict with keys: daily, table, brier, best_n.
    """
    print("── Loading data ──────────────────────────────────")
    df = load_data(symbol=symbol, period=period, csv_path=csv_path)
    print(f"  {len(df):,} hourly bars  |  {df['date'].min()} → {df['date'].max()}")

    print("── Building daily stats ──────────────────────────")
    daily = build_daily(df)

    # Validate 00:00 coverage
    n_days      = len(daily)
    n_macro     = daily["is_macro"].sum()
    n_midnight  = df[df["hour"] == 0]["date"].nunique()
    print(f"  {n_days} trading days  |  {n_macro} macro days  |  "
          f"{n_midnight} days have 00:00 bar")
    if n_midnight < n_days * 0.70:
        print("  WARNING: fewer than 70% of days have a 00:00 bar — "
              "check data source / symbol.")

    brier = None
    if optimise and n_override is None:
        print(f"── Optimising N in {list(n_range)} ──────────────────")
        brier = optimise_n(daily, n_range=n_range, exclude_macro=exclude_macro)
        valid = brier.dropna()
        if valid.empty:
            print("  Not enough data to optimise N; defaulting to N=5")
            best_n = 5
        else:
            best_n = int(valid.idxmin())
            print(f"  ▶ Best N = {best_n}  (Brier = {valid[best_n]:.5f})")
    else:
        best_n = n_override if n_override else 5

    print(f"── Computing probability table  ATR({best_n}) ──────────")
    table = compute_table(daily, n=best_n, exclude_macro=exclude_macro)

    # Print summary tables
    for direction, label in [("up", "Upside (HOD)"), ("down", "Downside (LOD)")]:
        sub = table[table["direction"] == direction][
            ["x", "n", "reliable", "p_extreme",
             "rev_close_mean", "rev_close_median", "rev_close_std",
             "rev_opp_mean"]
        ].set_index("x")
        sub.columns = ["n", "ok?", "P(extreme)",
                       "Rev→Close μ", "Rev→Close med", "Rev→Close σ",
                       "Rev→OppExt μ"]
        print(f"\n{label}")
        print(sub.to_string(float_format="{:.3f}".format))

    print("\n── Plotting ──────────────────────────────────────")
    plot_results(table, best_n, brier)

    return dict(daily=daily, table=table, brier=brier, best_n=best_n)


if __name__ == "__main__":
    run(optimise=True, exclude_macro=True)
