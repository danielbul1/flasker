#!/usr/bin/env python3
"""
ES Core — Conditional Distribution Tool
-----------------------------------------
Foundation: P(outcome | normalized_excursion)

Given that price has moved X from today's open (normalised by ATR),
what is the empirical distribution of what happens by the close?

Two outputs:
  1. P(price continues past X) with 90% bootstrap CI
  2. Reversion fan: P10 / P25 / P50 / P75 / P90 of (close - X)

Real-time lookup:
  python es_core.py --price 5100 --open 5080 --atr 35.5
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from es_analysis import load_data, build_daily, add_atr, optimise_n


# ─── BUILD NORMALISED SERIES ──────────────────────────────────────────────────

def build_series(daily: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    One row per clean (non-macro) day with enough ATR history.
    norm_high  = (high  - open) / ATR(n)
    norm_low   = (low   - open) / ATR(n)   ← negative
    norm_close = (close - open) / ATR(n)
    """
    d = add_atr(daily, n)
    d = d[~d["is_macro"]].dropna(subset=[f"atr_{n}"])
    atr = d[f"atr_{n}"]
    s = pd.DataFrame(index=d.index)
    s["norm_high"]  = (d["daily_high"]  - d["daily_open"]) / atr
    s["norm_low"]   = (d["daily_low"]   - d["daily_open"]) / atr
    s["norm_close"] = (d["daily_close"] - d["daily_open"]) / atr
    return s.dropna()


# ─── CONDITIONAL DISTRIBUTION ────────────────────────────────────────────────

def conditional_dist(series: pd.DataFrame,
                     direction: str = "up",
                     x_grid: np.ndarray | None = None,
                     n_boot: int = 2000) -> pd.DataFrame:
    """
    For each x in x_grid, given price reached x:

    UP direction:
      reached = days where norm_high >= x
      p_continues = P(norm_high >= x + dx | reached)   ← prob it goes FURTHER
      reversion   = norm_close - x                      ← negative = came back

    DOWN direction: symmetric (using -norm_low).

    Returns DataFrame indexed by x with:
      n, p_cont, ci_lo, ci_hi,
      rev_p10, rev_p25, rev_p50, rev_p75, rev_p90
    """
    if x_grid is None:
        x_grid = np.round(np.arange(0.10, 3.01, 0.10), 2)
    dx = float(round(x_grid[1] - x_grid[0], 4)) if len(x_grid) > 1 else 0.10

    if direction == "up":
        extremes = series["norm_high"].values
        closes   = series["norm_close"].values
    else:
        extremes = -series["norm_low"].values
        closes   = -series["norm_close"].values

    rng  = np.random.default_rng(42)
    N    = len(extremes)
    rows = []

    for x in x_grid:
        mask = extremes >= x
        n    = int(mask.sum())
        if n < 10:
            continue

        # P(continues past x+dx | reached x)
        p_cont = float((extremes[mask] >= x + dx).mean())

        # Bootstrap CI — resample full dataset, re-condition on reaching x
        boot = []
        for _ in range(n_boot):
            idx   = rng.integers(0, N, size=N)
            s_ext = extremes[idx]
            s_msk = s_ext >= x
            if s_msk.sum() < 5:
                continue
            boot.append(float((s_ext[s_msk] >= x + dx).mean()))

        ci_lo = float(np.percentile(boot, 5))  if boot else np.nan
        ci_hi = float(np.percentile(boot, 95)) if boot else np.nan

        # Reversion distribution
        rev = closes[mask] - x
        rows.append(dict(
            x       = x,
            n       = n,
            p_cont  = p_cont,          # P(price goes further than x)
            p_stop  = 1.0 - p_cont,    # P(x is near the extreme)
            ci_lo   = ci_lo,
            ci_hi   = ci_hi,
            rev_p10 = float(np.percentile(rev, 10)),
            rev_p25 = float(np.percentile(rev, 25)),
            rev_p50 = float(np.percentile(rev, 50)),
            rev_p75 = float(np.percentile(rev, 75)),
            rev_p90 = float(np.percentile(rev, 90)),
        ))

    return pd.DataFrame(rows).set_index("x")


# ─── REAL-TIME LOOKUP ────────────────────────────────────────────────────────

def lookup(dist: pd.DataFrame, norm_x: float) -> dict:
    """
    Given current normalised excursion norm_x,
    find the nearest row in dist and return a summary.
    """
    if dist.empty:
        return {}
    idx   = dist.index.values
    nearest = idx[np.argmin(np.abs(idx - norm_x))]
    row   = dist.loc[nearest]
    return dict(
        x         = nearest,
        n         = int(row["n"]),
        p_stop    = row["p_stop"],
        p_cont    = row["p_cont"],
        ci_lo     = row["ci_lo"],
        ci_hi     = row["ci_hi"],
        rev_p25   = row["rev_p25"],
        rev_p50   = row["rev_p50"],
        rev_p75   = row["rev_p75"],
    )


# ─── PLOT ─────────────────────────────────────────────────────────────────────

def plot(up: pd.DataFrame, dn: pd.DataFrame, n: int,
         out: str = "es_core.png") -> None:

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        f"ES — Conditional Distribution  |  ATR({n}) normalised  |  macro filtered",
        fontsize=13, fontweight="bold"
    )

    for row_i, (df, lbl, col) in enumerate([
        (up, "Upside  (price above open)", "steelblue"),
        (dn, "Downside (price below open)", "tomato"),
    ]):
        x = df.index.values

        # ── P(continues) with CI ──────────────────────────────────────────
        ax = axes[row_i, 0]
        ax.fill_between(x, df["ci_lo"], df["ci_hi"],
                        alpha=0.20, color=col, label="90% CI")
        ax.plot(x, df["p_cont"], color=col, lw=2.2, label="P(continues)")
        ax.plot(x, df["p_stop"], color=col, lw=1.2, ls="--", alpha=0.7,
                label="P(stops here)")
        ax.axhline(0.5, color="gray", ls=":", lw=0.8)

        # Shade unreliable bins (n < 20)
        for xi, ni in zip(df.index, df["n"]):
            if ni < 20:
                ax.axvspan(xi - 0.05, xi + 0.05,
                           color="gold", alpha=0.25, lw=0)

        # n on twin axis
        ax2 = ax.twinx()
        ax2.bar(x, df["n"], width=0.07, alpha=0.12, color="gray")
        ax2.set_ylabel("n", fontsize=8, color="gray")
        ax2.axhline(20, color="orange", ls=":", lw=0.8)

        ax.set_xlabel(f"Excursion (× ATR {n})")
        ax.set_ylabel("Probability")
        ax.set_title(f"{lbl}  —  P(continues / stops)")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.22)

        # ── Reversion fan ─────────────────────────────────────────────────
        ax = axes[row_i, 1]
        ax.fill_between(x, df["rev_p10"], df["rev_p90"],
                        alpha=0.15, color=col, label="P10–P90")
        ax.fill_between(x, df["rev_p25"], df["rev_p75"],
                        alpha=0.28, color=col, label="P25–P75")
        ax.plot(x, df["rev_p50"], color=col, lw=2.2, label="Median")
        ax.axhline(0, color="black", lw=0.8, alpha=0.4)

        ax.set_xlabel(f"Excursion (× ATR {n})")
        ax.set_ylabel("(close − x)  in ATR units")
        ax.set_title(f"{lbl}  —  Reversion fan to close")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.22)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def run(symbol: str = "ES=F",
        period: str = "730d",
        csv_path: str | None = None,
        n_boot: int = 2000) -> dict:

    print("── Loading ───────────────────────────────────────")
    df    = load_data(symbol=symbol, period=period, csv_path=csv_path)
    daily = build_daily(df, open_bucket=0)
    print(f"  {len(daily)} days  |  {daily['is_macro'].sum()} macro removed")

    print("── Choosing N ────────────────────────────────────")
    brier = optimise_n(daily, n_range=range(1, 31), verbose=True)
    n     = int(brier.dropna().idxmin())
    print(f"  ▶ N = {n}")

    print("── Building series ───────────────────────────────")
    series = build_series(daily, n)
    print(f"  {len(series)} clean days")

    print(f"── Computing conditional distributions  "
          f"(bootstrap={n_boot}) ──")
    up = conditional_dist(series, direction="up",  n_boot=n_boot)
    dn = conditional_dist(series, direction="down", n_boot=n_boot)

    print("\n── Upside ────────────────────────────────────────")
    print(up[["n","p_cont","p_stop","ci_lo","ci_hi",
              "rev_p25","rev_p50","rev_p75"]].to_string(float_format="{:.3f}".format))

    print("\n── Downside ──────────────────────────────────────")
    print(dn[["n","p_cont","p_stop","ci_lo","ci_hi",
              "rev_p25","rev_p50","rev_p75"]].to_string(float_format="{:.3f}".format))

    print("\n── Plotting ──────────────────────────────────────")
    plot(up, dn, n)

    return dict(up=up, dn=dn, series=series, n=n, brier=brier)


# ─── CLI LOOKUP ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--price", type=float, help="Current price")
    parser.add_argument("--open",  type=float, help="Today's open (00:00 NY)")
    parser.add_argument("--atr",   type=float, help="ATR(16) in points (optional)")
    args = parser.parse_args()

    result = run()

    if args.price and args.open:
        move = args.price - args.open
        if args.atr:
            atr_val = args.atr
        else:
            # Use last ATR from the series
            daily_tmp = build_daily(load_data(), open_bucket=0)
            daily_tmp = add_atr(daily_tmp, result["n"])
            atr_val   = float(daily_tmp[f"atr_{result['n']}"].dropna().iloc[-1])
            print(f"\n  ATR({result['n']}) = {atr_val:.1f} pts  (last available)")

        norm_x    = move / atr_val
        direction = "up" if norm_x >= 0 else "down"
        dist      = result["up"] if direction == "up" else result["dn"]
        res       = lookup(dist, abs(norm_x))

        print(f"\n{'─'*50}")
        print(f"  Price: {args.price}  |  Open: {args.open}  |  ATR: {atr_val:.1f}")
        print(f"  Move:  {move:+.1f} pts  →  {norm_x:+.2f} × ATR  ({direction})")
        print(f"{'─'*50}")
        print(f"  Nearest bin:   x = {res['x']:.2f} × ATR  (n={res['n']} days)")
        print(f"  P(stops here): {res['p_stop']:.1%}   CI: [{res['ci_lo']:.1%} – {res['ci_hi']:.1%}]")
        print(f"  P(continues):  {res['p_cont']:.1%}")
        print(f"  Reversion to close:  P25={res['rev_p25']:+.2f}  P50={res['rev_p50']:+.2f}  P75={res['rev_p75']:+.2f}  (× ATR)")
        print(f"{'─'*50}")
