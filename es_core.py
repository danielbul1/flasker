#!/usr/bin/env python3
"""
ES Core Analysis
----------------
Two questions, answered with uncertainty:

  1. P(HOD | price reached X from open)  — with 90% bootstrap CI
  2. Reversion fan (P10/P25/P50/P75/P90) — from X back toward open

Both normalised by ATR(N), N chosen by walk-forward Brier score.
Macro days excluded. Up/down computed separately.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from es_analysis import load_data, build_daily, add_atr, optimise_n, MACRO_DATES


# ──────────────────────────────────────────────────────────────────────────────
# NORMALISED DAILY SERIES
# ──────────────────────────────────────────────────────────────────────────────

def make_series(daily: pd.DataFrame, n: int,
                exclude_macro: bool = True) -> pd.DataFrame:
    """
    Returns a clean DataFrame with one row per day:
      norm_high  = (daily_high  - daily_open) / ATR(n)
      norm_low   = (daily_low   - daily_open) / ATR(n)   ← negative
      norm_close = (daily_close - daily_open) / ATR(n)
    """
    d = add_atr(daily, n)
    if exclude_macro:
        d = d[~d["is_macro"]]
    d = d.dropna(subset=[f"atr_{n}"])
    atr = d[f"atr_{n}"]

    out = pd.DataFrame(index=d.index)
    out["norm_high"]  = (d["daily_high"]  - d["daily_open"]) / atr
    out["norm_low"]   = (d["daily_low"]   - d["daily_open"]) / atr
    out["norm_close"] = (d["daily_close"] - d["daily_open"]) / atr
    return out.dropna()


# ──────────────────────────────────────────────────────────────────────────────
# P(HOD / LOD) WITH BOOTSTRAP CI
# ──────────────────────────────────────────────────────────────────────────────

def p_extreme_table(series: pd.DataFrame,
                    direction: str = "up",
                    x_grid: np.ndarray | None = None,
                    n_boot: int = 1000) -> pd.DataFrame:
    """
    For each level x in x_grid:

      UP:   among days where norm_high >= x,
            P(norm_high < x + dx)  →  probability that x is the HOD
            reversion = norm_close - x

      DOWN: symmetric using norm_low.

    Returns DataFrame with columns:
      x, n, p, ci_lo, ci_hi,
      rev_p10, rev_p25, rev_p50, rev_p75, rev_p90
    """
    if x_grid is None:
        x_grid = np.round(np.arange(0.10, 3.01, 0.10), 2)

    dx = float(round(x_grid[1] - x_grid[0], 4)) if len(x_grid) > 1 else 0.10

    if direction == "up":
        extremes = series["norm_high"].values
        closes   = series["norm_close"].values
    else:
        extremes = -series["norm_low"].values      # flip sign → positive
        closes   = -series["norm_close"].values    # flip sign → positive

    rng   = np.random.default_rng(42)
    N     = len(extremes)
    rows  = []

    for x in x_grid:
        mask    = extremes >= x
        n_reach = mask.sum()
        if n_reach < 10:
            continue

        sub_ext = extremes[mask]
        sub_cls = closes[mask]

        # Point estimate: fraction that stopped within [x, x+dx)
        p_obs = float((sub_ext < x + dx).mean())

        # Bootstrap CI (resample from ALL days, not just reached)
        boot_p = []
        for _ in range(n_boot):
            idx     = rng.integers(0, N, size=N)
            s_ext   = extremes[idx]
            s_mask  = s_ext >= x
            if s_mask.sum() < 5:
                continue
            boot_p.append(float((s_ext[s_mask] < x + dx).mean()))

        ci_lo = float(np.percentile(boot_p, 5))  if boot_p else np.nan
        ci_hi = float(np.percentile(boot_p, 95)) if boot_p else np.nan

        # Reversion fan: how far close moved back from x
        rev = sub_cls - x   # negative = price closed below x (came back)
        rows.append(dict(
            x      = x,
            n      = n_reach,
            p      = p_obs,
            ci_lo  = ci_lo,
            ci_hi  = ci_hi,
            rev_p10 = float(np.percentile(rev, 10)),
            rev_p25 = float(np.percentile(rev, 25)),
            rev_p50 = float(np.percentile(rev, 50)),
            rev_p75 = float(np.percentile(rev, 75)),
            rev_p90 = float(np.percentile(rev, 90)),
        ))

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# PLOT
# ──────────────────────────────────────────────────────────────────────────────

def plot(up: pd.DataFrame, dn: pd.DataFrame,
         n: int, out: str = "es_core.png") -> None:

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        f"ES Excursion Analysis  —  ATR({n}) normalised  |  macro filtered",
        fontsize=13, fontweight="bold"
    )

    ALPHA_BAND = 0.18
    MIN_N      = 20        # shade bins with n < MIN_N

    for row_idx, (df, label, sign) in enumerate([
        (up, "Upside  (HOD)", +1),
        (dn, "Downside (LOD)", -1),
    ]):
        x = df["x"].values

        # ── Left: P(extreme) with CI ─────────────────────────────────────────
        ax = axes[row_idx, 0]
        color = "steelblue" if sign == 1 else "tomato"

        ax.fill_between(x, df["ci_lo"], df["ci_hi"],
                        alpha=ALPHA_BAND, color=color, label="90% CI")
        ax.plot(x, df["p"], color=color, lw=2, label="P(extreme)")
        ax.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.7)

        # Shade unreliable bins
        for _, r in df[df["n"] < MIN_N].iterrows():
            ax.axvspan(r["x"] - 0.05, r["x"] + 0.05,
                       color="yellow", alpha=0.3, lw=0)

        ax.set_xlabel(f"Excursion (× ATR{n})")
        ax.set_ylabel("Probability")
        ax.set_title(f"{label}  —  P(this is the extreme)")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25)

        # Sample-size ticks on twin axis
        ax2 = ax.twinx()
        ax2.bar(x, df["n"], width=0.07, alpha=0.15, color="gray")
        ax2.axhline(MIN_N, color="orange", ls=":", lw=0.9, label=f"n={MIN_N}")
        ax2.set_ylabel("n (days)", fontsize=8, color="gray")
        ax2.legend(fontsize=8, loc="upper right")

        # ── Right: Reversion fan ─────────────────────────────────────────────
        ax = axes[row_idx, 1]

        ax.fill_between(x, df["rev_p10"], df["rev_p90"],
                        alpha=ALPHA_BAND, color=color, label="P10–P90")
        ax.fill_between(x, df["rev_p25"], df["rev_p75"],
                        alpha=ALPHA_BAND * 2, color=color, label="P25–P75")
        ax.plot(x, df["rev_p50"], color=color, lw=2, label="Median")
        ax.axhline(0, color="black", lw=0.8, alpha=0.5)

        ax.set_xlabel(f"Excursion (× ATR{n})")
        ax.set_ylabel("Reversion (× ATR) to close")
        ax.set_title(f"{label}  —  Reversion fan")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")


# ──────────────────────────────────────────────────────────────────────────────
# PRINT TABLE
# ──────────────────────────────────────────────────────────────────────────────

def print_table(df: pd.DataFrame, label: str) -> None:
    print(f"\n── {label} ──")
    disp = df[["x", "n", "p", "ci_lo", "ci_hi",
               "rev_p25", "rev_p50", "rev_p75"]].copy()
    disp.columns = ["x", "n", "P(extreme)", "CI lo", "CI hi",
                    "Rev P25", "Rev P50", "Rev P75"]
    disp["ok"] = df["n"].apply(lambda v: "✓" if v >= 20 else "·")
    print(disp.to_string(index=False, float_format="{:.3f}".format))


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def run(symbol: str = "ES=F",
        period: str = "730d",
        csv_path: str | None = None,
        exclude_macro: bool = True,
        n_boot: int = 1000) -> dict:

    print("── Loading data ──────────────────────────────────")
    df = load_data(symbol=symbol, period=period, csv_path=csv_path)
    print(f"  {len(df):,} bars  |  {df['date'].min()} → {df['date'].max()}")

    print("── Building daily stats ──────────────────────────")
    daily = build_daily(df, open_bucket=0)   # 00:00 NY open
    n_macro = daily["is_macro"].sum()
    clean   = len(daily) - n_macro
    print(f"  {len(daily)} days  |  {n_macro} macro removed  |  {clean} clean days")

    print("── Choosing ATR window N (walk-forward Brier) ────")
    brier = optimise_n(daily, n_range=range(1, 31),
                       exclude_macro=exclude_macro, verbose=True)
    valid = brier.dropna()
    best_n = int(valid.idxmin())
    print(f"  ▶ Best N = {best_n}  (Brier = {valid[best_n]:.5f})")

    print("── Building normalised series ────────────────────")
    series = make_series(daily, best_n, exclude_macro=exclude_macro)
    print(f"  {len(series)} days after ATR warmup")

    print(f"── Computing P(extreme) + reversion fan  "
          f"(bootstrap n={n_boot}) ──")
    up = p_extreme_table(series, direction="up",  n_boot=n_boot)
    dn = p_extreme_table(series, direction="down", n_boot=n_boot)

    print_table(up, "Upside  (HOD)")
    print_table(dn, "Downside (LOD)")

    print("\n── Plotting ──────────────────────────────────────")
    plot(up, dn, best_n)

    return dict(series=series, up=up, dn=dn, best_n=best_n, brier=brier)


if __name__ == "__main__":
    run()
