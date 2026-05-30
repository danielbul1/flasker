#!/usr/bin/env python3
"""
ES Historical Data Downloader — Interactive Brokers
----------------------------------------------------
Downloads 5-minute ES futures bars via ib_insync and saves to CSV.
Loops backward in 30-day chunks to bypass IB's per-request limits.

Requirements:
  - TWS or IB Gateway must be running and logged in
  - pip install ib_insync

Ports:
  - TWS live:    7496
  - TWS paper:   7497  ← default
  - IB Gateway live:  4001
  - IB Gateway paper: 4002

Usage:
  python es_ib_download.py                    # downloads 2 years, saves to es_data_5m.csv
  python es_ib_download.py --port 4002        # IB Gateway paper
  python es_ib_download.py --years 1          # 1 year only
  python es_ib_download.py --bar 1 mins       # 1-minute bars instead
"""

import sys
import time
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

NY_TZ = ZoneInfo("America/New_York")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--host",  default="127.0.0.1")
    p.add_argument("--port",  type=int, default=7497, help="TWS paper=7497, live=7496, GW paper=4002, live=4001")
    p.add_argument("--client-id", type=int, default=10)
    p.add_argument("--years", type=float, default=2.0, help="Years of history to download")
    p.add_argument("--bar",   default="5 mins", help="Bar size: '1 min', '5 mins', '15 mins', '1 hour'")
    p.add_argument("--chunk", type=int, default=30, help="Days per request")
    p.add_argument("--out",   default="es_data_5m.csv", help="Output CSV file")
    p.add_argument("--rth",   action="store_true", help="RTH only (default: full session)")
    return p.parse_args()


def connect(host, port, client_id):
    from ib_insync import IB
    ib = IB()
    print(f"Connecting to IB on {host}:{port} (clientId={client_id})...")
    ib.connect(host, port, clientId=client_id, timeout=20)
    print(f"  Connected. Account: {ib.wrapper.accounts}")
    return ib


def get_contract(ib):
    from ib_insync import ContFuture
    contract = ContFuture("ES", "CME", currency="USD")
    details = ib.qualifyContracts(contract)
    if not details:
        raise ValueError("Could not qualify ES ContFuture — check TWS connection and permissions.")
    print(f"  Contract: {contract.symbol} {contract.lastTradeDateOrContractMonth}")
    return contract


def download_chunk(ib, contract, end_dt: datetime, duration_days: int,
                   bar_size: str, use_rth: bool) -> pd.DataFrame:
    from ib_insync import util
    end_str = end_dt.strftime("%Y%m%d %H:%M:%S") + " ET"
    bars = ib.reqHistoricalData(
        contract,
        endDateTime=end_str,
        durationStr=f"{duration_days} D",
        barSizeSetting=bar_size,
        whatToShow="TRADES",
        useRTH=use_rth,
        formatDate=1,
        keepUpToDate=False,
    )
    if not bars:
        return pd.DataFrame()

    df = util.df(bars)
    df = df.rename(columns={"date": "Datetime", "open": "open", "high": "high",
                             "low": "low", "close": "close", "volume": "volume"})
    df = df[["Datetime", "open", "high", "low", "close", "volume"]].copy()

    # Parse datetime and convert to NY timezone
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    if df["Datetime"].dt.tz is None:
        df["Datetime"] = df["Datetime"].dt.tz_localize("America/New_York")
    else:
        df["Datetime"] = df["Datetime"].dt.tz_convert("America/New_York")

    return df


def main():
    args = parse_args()
    out_path = Path(args.out)

    # Load existing data if resuming
    existing = pd.DataFrame()
    if out_path.exists():
        existing = pd.read_csv(out_path, parse_dates=["Datetime"])
        existing["Datetime"] = pd.to_datetime(existing["Datetime"], utc=True).dt.tz_convert("America/New_York")
        print(f"Resuming — {len(existing)} rows already in {out_path}")

    ib = connect(args.host, args.port, args.client_id)

    try:
        contract = get_contract(ib)

        now_ny  = datetime.now(NY_TZ)
        end_dt  = now_ny
        start_dt = now_ny - timedelta(days=int(args.years * 365))

        print(f"\nDownloading {args.bar} bars from {start_dt.date()} to {end_dt.date()}")
        print(f"  Chunk size: {args.chunk} days | RTH only: {args.use_rth if hasattr(args,'use_rth') else args.rth}")
        print()

        all_chunks = []
        chunk_end = end_dt

        while chunk_end > start_dt:
            chunk_start = chunk_end - timedelta(days=args.chunk)
            if chunk_start < start_dt:
                chunk_start = start_dt

            # Skip if we already have this data
            if not existing.empty:
                chunk_end_naive = chunk_end.replace(tzinfo=None)
                chunk_start_naive = chunk_start.replace(tzinfo=None)
                existing_dt = existing["Datetime"].dt.tz_localize(None) if existing["Datetime"].dt.tz else existing["Datetime"]
                already_have = existing_dt.between(chunk_start_naive, chunk_end_naive)
                if already_have.sum() > 10:
                    print(f"  {chunk_start.date()} → {chunk_end.date()}  SKIP (already downloaded)")
                    chunk_end = chunk_start - timedelta(minutes=5)
                    continue

            actual_days = (chunk_end - chunk_start).days
            print(f"  {chunk_start.date()} → {chunk_end.date()} ({actual_days}d)...", end="", flush=True)

            for attempt in range(4):
                try:
                    df_chunk = download_chunk(ib, contract, chunk_end,
                                              actual_days, args.bar, args.rth)
                    break
                except Exception as e:
                    wait = 2 ** (attempt + 1)
                    print(f" retry in {wait}s ({e})...", end="", flush=True)
                    time.sleep(wait)
            else:
                print(" FAILED — skipping chunk")
                chunk_end = chunk_start - timedelta(minutes=5)
                continue

            if df_chunk.empty:
                print(" no data")
            else:
                all_chunks.append(df_chunk)
                print(f" {len(df_chunk)} bars")

                # Save progress after every chunk
                combined = pd.concat([existing] + all_chunks, ignore_index=True)
                combined = combined.drop_duplicates(subset=["Datetime"]).sort_values("Datetime")
                combined.to_csv(out_path, index=False)

            chunk_end = chunk_start - timedelta(minutes=5)
            time.sleep(0.5)  # respect IB pacing

        # Final save
        if all_chunks:
            combined = pd.concat([existing] + all_chunks, ignore_index=True)
            combined = combined.drop_duplicates(subset=["Datetime"]).sort_values("Datetime")
            combined.to_csv(out_path, index=False)
            print(f"\nDone — {len(combined):,} total bars saved to {out_path}")
            print(f"  Range: {combined['Datetime'].min()} → {combined['Datetime'].max()}")
        else:
            print("\nNo new data downloaded.")

    finally:
        ib.disconnect()
        print("Disconnected from IB.")


if __name__ == "__main__":
    main()
