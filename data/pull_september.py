"""
pull_september.py
=================
Pulls 2026 MLB regular-season Statcast data (2026-09-01 .. 2026-09-30),
one day at a time, and saves each day to data/raw/statcast_2026-09-DD.parquet.

Resume-safe: days whose parquet file already exists are skipped.

Usage:
    python data/pull_september.py              # full September
    python data/pull_september.py --test       # single-day test (2026-09-01)
"""

import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from pybaseball import statcast

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent      # ~/workspace/gameplan-dashboard
RAW_DIR = BASE_DIR / "data" / "raw"

START = date(2026, 9, 1)
END = date(2026, 9, 30)

MAX_RETRIES = 3
RETRY_SLEEP_S = 10

# Lean column list for the downstream aggregates. Any of these missing from the
# pybaseball output is skipped with a warning (see _select_columns).
KEEP_COLS = [
    "game_date", "game_pk", "at_bat_number", "pitch_number",
    "batter", "pitcher", "player_name",
    "pitch_type", "release_speed", "effective_speed",
    "release_spin_rate", "spin_axis", "pfx_x", "pfx_z",
    "release_pos_x", "release_pos_z",
    "stand", "p_throws",
    "events", "description", "zone", "plate_x", "plate_z",
    "estimated_woba_using_speedangle", "woba_value", "woba_denom",
    "launch_speed", "launch_angle", "game_type",
]


def _pull_day(day: date) -> pd.DataFrame:
    """Pull one day of Statcast data, retrying on failure."""
    s = day.strftime("%Y-%m-%d")
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            df = statcast(start_dt=s, end_dt=s)
            return df
        except Exception as e:  # noqa: BLE001 - network flakiness is expected
            last_err = e
            print(f"  attempt {attempt}/{MAX_RETRIES} failed for {s}: {e!r}",
                  flush=True)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP_S)
    raise RuntimeError(f"Failed to pull {s} after {MAX_RETRIES} attempts: {last_err!r}")


def _select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only KEEP_COLS that exist; warn about any missing ones."""
    missing = [c for c in KEEP_COLS if c not in df.columns]
    if missing:
        print(f"  WARNING: expected columns missing from statcast output: {missing}",
              flush=True)
    return df[[c for c in KEEP_COLS if c in df.columns]]


def run(days):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    pulled = 0
    for day in days:
        out_path = RAW_DIR / f"statcast_{day.strftime('%Y-%m-%d')}.parquet"
        if out_path.exists():
            print(f"{day}  -> SKIP (file exists)", flush=True)
            continue
        print(f"{day}  -> pulling...", flush=True)
        df = _pull_day(day)
        if df is None or df.empty:
            print(f"{day}  -> no rows returned, saving empty marker", flush=True)
            df = pd.DataFrame(columns=KEEP_COLS)
        df = _select_columns(df)
        df.to_parquet(out_path, index=False)
        total_rows += len(df)
        pulled += 1
        print(f"{day}  -> saved {len(df):>6,} rows to {out_path.name}", flush=True)
    print(f"\nDone: {pulled} day(s) pulled, {total_rows:,} rows this run.", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
                        help="pull only 2026-09-01 as a column/schema check")
    args = parser.parse_args()

    days = [START] if args.test else [
        START + timedelta(days=i) for i in range((END - START).days + 1)
    ]
    run(days)


if __name__ == "__main__":
    sys.exit(main())
