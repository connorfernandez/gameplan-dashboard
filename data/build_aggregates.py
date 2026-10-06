"""
build_aggregates.py
===================
Reads data/raw/statcast_2026-09-*.parquet (built by pull_september.py) and
writes 8 lean aggregate tables to data/processed/ for the Streamlit dashboard.

Tables (see module docstring of each builder for exact semantics):
  1. hitter_pitch.csv    (batter_id, batter_name, pitch_type, p_throws, n, whiff_pct, xwoba, slg)
  2. pitcher_pitch.csv   (pitcher_id, pitcher_name, pitch_type, stand, usage_pct, avg_velo, avg_spin, avg_pfx_x, avg_pfx_z, stuff_lite)
  3. zone_hitter.csv     (batter_id, batter_name, zone, n, xwoba)
  4. zone_pitcher.csv    (pitcher_id, pitcher_name, zone, pitch_pct)
  5. hitter_game.csv     (batter_id, batter_name, game_date, pa, woba)
  6. hitter_stand.csv    (batter_id, p_throws, stand)
  7. league_pitch.csv    (pitch_type, lg_xwoba, lg_n)
  8. players.csv         (player_id, player_name, role, hand)

Usage:
    python data/build_aggregates.py

Notes:
  - Only regular-season rows (game_type == 'R') are kept.
  - In this pybaseball output, player_name is the PITCHER's name (format
    "Last, First"; verified: 552 distinct player_names == 552 distinct
    pitcher ids, and one game's 9 player_names == its 9 pitchers).
    Canonical display names ("First Last") come from the free MLB Stats API
    (https://statsapi.mlb.com/api/v1/people/<mlbam_id> -> 'fullName') for both
    batters and pitchers; player_name is used as the cross-check for pitchers.
    pybaseball.playerid_reverse_lookup was deliberately NOT used: on first run
    it prompts to download the Chadwick register, which blocks non-interactive
    execution.
  - stuff_lite is a *lite* Stuff+ approximation, documented in the builder.
"""

import sys
import time
import urllib.request
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROC_DIR = BASE_DIR / "data" / "processed"

SWINGS = {"swinging_strike", "swinging_strike_blocked", "foul", "foul_tip", "hit_into_play"}
WHIFFS = {"swinging_strike", "swinging_strike_blocked"}

# events -> total bases for SLG
BASES = {"single": 1, "double": 2, "triple": 3, "home_run": 4}
# events that count as an at-bat
AB_EVENTS = {
    "single", "double", "triple", "home_run",
    "field_out", "grounded_into_double_play", "force_out",
    "fielders_choice_out", "double_play", "flyout", "lineout",
    "pop_out", "strikeout", "strikeout_double_play",
}


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def load_raw() -> pd.DataFrame:
    files = sorted(RAW_DIR.glob("statcast_2026-09-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No raw files in {RAW_DIR}; run pull_september.py first.")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"Loaded {len(df):,} rows from {len(files)} day-files.")

    # Regular season only (September pulls should all be 'R' anyway).
    n_before = len(df)
    df = df[df["game_type"] == "R"].copy()
    print(f"game_type filter: kept {len(df):,} / {n_before:,} rows (dropped {n_before - len(df):,}).")

    # Normalize dtypes for the numeric columns we aggregate on.
    for c in ["estimated_woba_using_speedangle", "woba_value", "woba_denom",
              "release_speed", "release_spin_rate", "pfx_x", "pfx_z",
              "launch_speed", "launch_angle", "plate_x", "plate_z", "zone"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["zone"] = pd.to_numeric(df["zone"], errors="coerce")
    return df


def pa_end_rows(df: pd.DataFrame) -> pd.DataFrame:
    """One row per plate appearance: the last pitch of each (game_pk, at_bat_number)."""
    key = ["game_pk", "at_bat_number"]
    sub = df.dropna(subset=key + ["pitch_number"]).copy()
    sub = sub.sort_values(key + ["pitch_number"])
    return sub.drop_duplicates(subset=key, keep="last")


# ----------------------------------------------------------------------------
# Name resolution
# ----------------------------------------------------------------------------
def _statsapi_name(pid: int) -> str | None:
    """Fetch a player's fullName from the MLB Stats API (no key needed)."""
    url = f"https://statsapi.mlb.com/api/v1/people/{int(pid)}"
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            data = json.loads(r.read().decode("utf-8"))
        people = data.get("people", [])
        if people:
            return people[0].get("fullName")
    except Exception as e:  # noqa: BLE001
        print(f"    statsapi fallback failed for id {pid}: {e!r}")
    return None


def build_name_map(ids: pd.Series) -> dict:
    """
    Map MLBAM id -> display name via the free MLB Stats API
    (https://statsapi.mlb.com/api/v1/people/<id> -> 'fullName').
    No API key needed. ~0.15s polite sleep between requests.
    """
    ids = sorted({int(i) for i in ids.dropna().unique()})
    name_map: dict[int, str] = {}
    print(f"Resolving {len(ids):,} player names via statsapi.mlb.com...")
    for j, pid in enumerate(ids):
        name = _statsapi_name(pid)
        if name:
            name_map[pid] = name
        if (j + 1) % 100 == 0:
            print(f"  {j + 1}/{len(ids)}...", flush=True)
        time.sleep(0.15)
    print(f"  resolved {len(name_map):,} / {len(ids):,}.")
    return name_map


# ----------------------------------------------------------------------------
# Table builders
# ----------------------------------------------------------------------------
def build_hitter_pitch(df: pd.DataFrame, pa_end: pd.DataFrame, batter_names: pd.Series) -> pd.DataFrame:
    """
    Per (batter, pitch_type, pitcher-handedness): volume, whiff%, xwOBA, SLG.
    SLG uses PA-end rows only (a PA's result is attributed to its final pitch).
    """
    sub = df.dropna(subset=["batter", "pitch_type", "p_throws"]).copy()

    # --- whiff% ---
    sub["is_swing"] = sub["description"].isin(SWINGS)
    sub["is_whiff"] = sub["description"].isin(WHIFFS)
    g = sub.groupby(["batter", "pitch_type", "p_throws"], observed=True)
    n = g.size().rename("n")
    swings = g["is_swing"].sum()
    whiffs = g["is_whiff"].sum()
    whiff_pct = 100 * whiffs / swings.replace(0, np.nan)
    xwoba = g["estimated_woba_using_speedangle"].mean()  # NaN-aware mean

    # --- SLG from PA-end rows ---
    pe = pa_end.dropna(subset=["batter", "pitch_type", "p_throws"]).copy()
    pe["tb"] = pe["events"].map(BASES).fillna(0).astype(int)
    pe["is_ab"] = pe["events"].isin(AB_EVENTS)
    pg = pe.groupby(["batter", "pitch_type", "p_throws"], observed=True)
    tb = pg["tb"].sum()
    ab = pg["is_ab"].sum()
    slg = (tb / ab.replace(0, np.nan)).fillna(0.0)  # NaN -> 0 when AB == 0

    out = pd.DataFrame({"n": n, "whiff_pct": whiff_pct, "xwoba": xwoba, "slg": slg}).reset_index()
    # Groups with pitches but no PA-ending pitch have no SLG rows -> AB = 0 -> slg = 0.
    out["slg"] = out["slg"].fillna(0.0)
    out = out.rename(columns={"batter": "batter_id"})
    out["batter_name"] = out["batter_id"].map(batter_names)
    return out[["batter_id", "batter_name", "pitch_type", "p_throws",
                "n", "whiff_pct", "xwoba", "slg"]]


def build_pitcher_pitch(df: pd.DataFrame, pitcher_names: dict) -> pd.DataFrame:
    """
    Per (pitcher, pitch_type, batter-stand): usage%, pitch-shape averages, stuff_lite.

    stuff_lite (LITE Stuff+ approximation -- documented simplification):
      * For each pitch_type, compute the league mean/std of release_speed,
        release_spin_rate, and per-pitch movement = sqrt(pfx_x^2 + pfx_z^2),
        over all individual pitches of that type in September.
      * For each (pitcher, pitch_type, stand) group, z-score its average velo /
        spin / movement (movement from the group's avg pfx_x, pfx_z) against
        the league pitch-type distribution.
      * stuff_lite = clip(100 + 12 * mean(z_velo, z_spin, z_move), 50, 150),
        so ~100 is league average for the pitch type, ~1 SD = 12 points.
      * This ignores release point, extension, spin axis/efficiency, and
        pitch-to-pitch variance -- a real Stuff+ model would include those.
    """
    sub = df.dropna(subset=["pitcher", "pitch_type", "stand"]).copy()
    sub["movement"] = np.sqrt(sub["pfx_x"] ** 2 + sub["pfx_z"] ** 2)

    # League per-pitch-type distributions (over individual pitches).
    lg = sub.groupby("pitch_type", observed=True).agg(
        lg_velo_mean=("release_speed", "mean"), lg_velo_std=("release_speed", "std"),
        lg_spin_mean=("release_spin_rate", "mean"), lg_spin_std=("release_spin_rate", "std"),
        lg_move_mean=("movement", "mean"), lg_move_std=("movement", "std"),
    )

    g = sub.groupby(["pitcher", "pitch_type", "stand"], observed=True)
    n = g.size().rename("n")
    avg_velo = g["release_speed"].mean()
    avg_spin = g["release_spin_rate"].mean()
    avg_pfx_x = g["pfx_x"].mean()
    avg_pfx_z = g["pfx_z"].mean()
    avg_move = np.sqrt(avg_pfx_x ** 2 + avg_pfx_z ** 2)

    out = pd.DataFrame({
        "n": n, "avg_velo": avg_velo, "avg_spin": avg_spin,
        "avg_pfx_x": avg_pfx_x, "avg_pfx_z": avg_pfx_z, "avg_move": avg_move,
    }).reset_index()

    # usage% vs the pitcher's total pitches against that stand.
    totals = out.groupby(["pitcher", "stand"], observed=True)["n"].transform("sum")
    out["usage_pct"] = 100 * out["n"] / totals

    # z-scores vs league pitch-type distributions.
    out = out.merge(lg, left_on="pitch_type", right_index=True, how="left")

    def _z(val, mean, std):
        std = std.replace(0, np.nan)
        return (val - mean) / std

    z_velo = _z(out["avg_velo"], out["lg_velo_mean"], out["lg_velo_std"])
    z_spin = _z(out["avg_spin"], out["lg_spin_mean"], out["lg_spin_std"])
    z_move = _z(out["avg_move"], out["lg_move_mean"], out["lg_move_std"])
    z_mean = pd.concat([z_velo, z_spin, z_move], axis=1).mean(axis=1, skipna=True)
    out["stuff_lite"] = (100 + 12 * z_mean).clip(50, 150)

    out = out.rename(columns={"pitcher": "pitcher_id"})
    out["pitcher_name"] = out["pitcher_id"].map(pitcher_names)
    return out[["pitcher_id", "pitcher_name", "pitch_type", "stand",
                "usage_pct", "avg_velo", "avg_spin",
                "avg_pfx_x", "avg_pfx_z", "stuff_lite"]]


def build_zone_hitter(df: pd.DataFrame, batter_names: pd.Series) -> pd.DataFrame:
    """Per (batter, zone 1-9): pitch count and xwOBA."""
    sub = df.dropna(subset=["batter", "zone"]).copy()
    sub = sub[sub["zone"].isin(range(1, 10))]
    g = sub.groupby(["batter", "zone"], observed=True)
    out = pd.DataFrame({
        "n": g.size(),
        "xwoba": g["estimated_woba_using_speedangle"].mean(),
    }).reset_index().rename(columns={"batter": "batter_id"})
    out["zone"] = out["zone"].astype(int)
    out["batter_name"] = out["batter_id"].map(batter_names)
    return out[["batter_id", "batter_name", "zone", "n", "xwoba"]]


def build_zone_pitcher(df: pd.DataFrame, pitcher_names: dict) -> pd.DataFrame:
    """Per (pitcher, zone 1-9): share of the pitcher's in-zone pitches in each zone."""
    sub = df.dropna(subset=["pitcher", "zone"]).copy()
    sub = sub[sub["zone"].isin(range(1, 10))]
    g = sub.groupby(["pitcher", "zone"], observed=True)
    n = g.size().rename("n")
    out = n.reset_index()
    totals = out.groupby("pitcher", observed=True)["n"].transform("sum")
    out["pitch_pct"] = 100 * out["n"] / totals
    out = out.rename(columns={"pitcher": "pitcher_id"})
    out["zone"] = out["zone"].astype(int)
    out["pitcher_name"] = out["pitcher_id"].map(pitcher_names)
    return out[["pitcher_id", "pitcher_name", "zone", "pitch_pct"]]


def build_hitter_game(pa_end: pd.DataFrame, batter_names: pd.Series) -> pd.DataFrame:
    """
    Per (batter, game): PA count and wOBA = sum(woba_value)/sum(woba_denom).
    woba_value/woba_denom are only populated on PA-end rows (verified in raw
    data), so no dedupe beyond pa_end is needed.
    """
    pe = pa_end.dropna(subset=["batter", "game_date"]).copy()
    pe["woba_value"] = pd.to_numeric(pe["woba_value"], errors="coerce")
    pe["woba_denom"] = pd.to_numeric(pe["woba_denom"], errors="coerce")
    g = pe.groupby(["batter", "game_date"], observed=True)
    wv = g["woba_value"].sum()
    wd = g["woba_denom"].sum()
    out = pd.DataFrame({
        "pa": g.size(),
        "woba": wv / wd.replace(0, np.nan),
    }).reset_index().rename(columns={"batter": "batter_id"})
    out["batter_name"] = out["batter_id"].map(batter_names)
    return out[["batter_id", "batter_name", "game_date", "pa", "woba"]]


def build_hitter_stand(df: pd.DataFrame) -> pd.DataFrame:
    """Most common batting stand per (batter, pitcher-handedness). For switch hitters."""
    sub = df.dropna(subset=["batter", "p_throws", "stand"]).copy()
    mode = (sub.groupby(["batter", "p_throws"], observed=True)["stand"]
               .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
               .reset_index()
               .rename(columns={"batter": "batter_id"}))
    return mode[["batter_id", "p_throws", "stand"]]


def build_league_pitch(df: pd.DataFrame) -> pd.DataFrame:
    """League-average xwOBA per pitch type (context for matchup ratings)."""
    sub = df.dropna(subset=["pitch_type"]).copy()
    g = sub.groupby("pitch_type", observed=True)["estimated_woba_using_speedangle"]
    out = pd.DataFrame({"lg_xwoba": g.mean(), "lg_n": g.size()}).reset_index()
    return out[["pitch_type", "lg_xwoba", "lg_n"]]


def build_players(df: pd.DataFrame, batter_names: pd.Series, pitcher_names: dict) -> pd.DataFrame:
    """
    Player registry. role: batter / pitcher / both.
    hand: most common p_throws for pitchers, most common stand for batters;
          for 'both', p_throws (throwing hand) is used as the single hand field.
    """
    batters = set(df["batter"].dropna().unique())
    pitchers = set(df["pitcher"].dropna().unique())

    def role(pid):
        is_b = pid in batters
        is_p = pid in pitchers
        return "both" if (is_b and is_p) else ("batter" if is_b else "pitcher")

    throw_hand = (df.dropna(subset=["pitcher", "p_throws"])
                    .groupby("pitcher", observed=True)["p_throws"]
                    .agg(lambda s: s.mode().iloc[0]))
    bat_hand = (df.dropna(subset=["batter", "stand"])
                  .groupby("batter", observed=True)["stand"]
                  .agg(lambda s: s.mode().iloc[0]))

    rows = []
    for pid in sorted(batters | pitchers):
        r = role(pid)
        name = batter_names.get(pid, pitcher_names.get(int(pid)))
        if r == "batter":
            hand = bat_hand.get(pid)
        elif r == "pitcher":
            hand = throw_hand.get(pid)
        else:  # both -> throwing hand
            hand = throw_hand.get(pid, bat_hand.get(pid))
        rows.append({"player_id": pid, "player_name": name, "role": r, "hand": hand})
    return pd.DataFrame(rows, columns=["player_id", "player_name", "role", "hand"])


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    df = load_raw()
    pa_end = pa_end_rows(df)
    print(f"PA-end rows: {len(pa_end):,}")

    # --- names ---
    # player_name in this pybaseball output is the PITCHER's name ("Last, First").
    # Sanity: expect ~1 distinct player_name per pitcher id.
    pname_counts = df.dropna(subset=["pitcher", "player_name"]).groupby("pitcher")["player_name"].nunique()
    multi = pname_counts[pname_counts > 1]
    print(f"Pitcher ids with >1 distinct player_name: {len(multi)}"
          + (f" (e.g. {multi.index[:5].tolist()})" if len(multi) else ""))

    def _flip(name: str) -> str:
        """'Last, First' -> 'First Last' (handles suffixes: 'Lynch IV, Daniel')."""
        parts = str(name).split(",")
        return f"{parts[1].strip()} {parts[0].strip()}" if len(parts) == 2 else str(name).strip()

    pitcher_pname = (df.dropna(subset=["pitcher", "player_name"])
                       .groupby("pitcher")["player_name"]
                       .agg(lambda s: _flip(s.mode().iloc[0])))

    # Canonical names from the MLB Stats API for every id.
    # Cached in data/processed/_name_map.json so re-runs don't re-hit the API.
    name_cache = PROC_DIR / "_name_map.json"
    if name_cache.exists():
        name_map = {int(k): v for k, v in json.loads(name_cache.read_text()).items()}
        print(f"Loaded {len(name_map):,} cached names from {name_cache.name}.")
    else:
        name_map = build_name_map(pd.concat([df["batter"], df["pitcher"]], ignore_index=True))
        name_cache.write_text(json.dumps({str(k): v for k, v in name_map.items()}))
        print(f"Cached {len(name_map):,} names to {name_cache.name}.")
    batter_names = pd.Series({int(pid): name_map[pid] for pid in df["batter"].dropna().unique()
                              if int(pid) in name_map})
    pitcher_names = {int(pid): name_map[pid] for pid in df["pitcher"].dropna().unique()
                     if int(pid) in name_map}
    # Fallback: if statsapi missed a pitcher, use the flipped player_name.
    for pid, pname in pitcher_pname.items():
        pitcher_names.setdefault(int(pid), pname)
    print(f"Name coverage: batters {len(batter_names)}/{df['batter'].nunique()}, "
          f"pitchers {len(pitcher_names)}/{df['pitcher'].nunique()}")

    # Cross-check: statcast player_name vs statsapi name for pitchers.
    def _norm(s): return str(s).lower().replace(".", "").replace(" ", "")
    mism = sum(1 for pid, pname in pitcher_pname.items()
               if int(pid) in name_map and _norm(pname) != _norm(name_map[int(pid)]))
    print(f"Pitcher name cross-check mismatches (player_name vs statsapi): {mism}")
    if mism:
        shown = 0
        for pid, pname in pitcher_pname.items():
            if int(pid) in name_map and _norm(pname) != _norm(name_map[int(pid)]):
                print(f"    id {pid}: statcast='{pname}' statsapi='{name_map[int(pid)]}'")
                shown += 1
                if shown >= 5:
                    break

    # --- build ---
    tables = {
        "hitter_pitch":  build_hitter_pitch(df, pa_end, batter_names),
        "pitcher_pitch": build_pitcher_pitch(df, pitcher_names),
        "zone_hitter":   build_zone_hitter(df, batter_names),
        "zone_pitcher":  build_zone_pitcher(df, pitcher_names),
        "hitter_game":   build_hitter_game(pa_end, batter_names),
        "hitter_stand":  build_hitter_stand(df),
        "league_pitch":  build_league_pitch(df),
        "players":       build_players(df, batter_names, pitcher_names),
    }
    for name, t in tables.items():
        # CSV (not parquet): the processed tables are committed to GitHub and
        # GitHub's file API only transports text reliably. ~1.5MB total.
        path = PROC_DIR / f"{name}.csv"
        t.to_csv(path, index=False)
        print(f"Wrote {path.name}: {t.shape[0]:,} rows x {t.shape[1]} cols")

    # --- verification summary ---
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"Total pitches (regular season): {len(df):,}")
    print(f"Date range: {df['game_date'].min()} -> {df['game_date'].max()}")
    print(f"Distinct batters: {df['batter'].nunique():,} | pitchers: {df['pitcher'].nunique():,}")
    print(f"Games: {df['game_pk'].nunique():,}")
    for name, t in tables.items():
        print(f"  {name:15s} {t.shape[0]:>8,} rows x {t.shape[1]:>2} cols  "
              f"cols={list(t.columns)}")
    # league wOBA sanity (~.310-.330)
    pe = pa_end.copy()
    pe["woba_value"] = pd.to_numeric(pe["woba_value"], errors="coerce")
    pe["woba_denom"] = pd.to_numeric(pe["woba_denom"], errors="coerce")
    lg_woba = pe["woba_value"].sum() / pe["woba_denom"].sum()
    print(f"League wOBA (PA-end rows): {lg_woba:.3f}  [sanity target ~.310-.330]")
    # processed dir size
    total_mb = sum(p.stat().st_size for p in PROC_DIR.glob("*.csv")) / 1e6
    print(f"processed/ size: {total_mb:.1f} MB")
    # spot-check: top hitters by pitch volume
    top = df["batter"].value_counts().head(3)
    for pid, n in top.items():
        nm = batter_names.get(pid, name_map.get(int(pid)))
        hp = tables["hitter_pitch"]
        sub = hp[hp["batter_id"] == pid].sort_values("n", ascending=False).head(3)
        print(f"\nSpot-check: {nm} (id {pid}), {n:,} pitches seen")
        for _, r in sub.iterrows():
            print(f"  vs {r['p_throws']} {r['pitch_type']}: n={r['n']}, "
                  f"whiff%={r['whiff_pct']:.1f}, xwOBA={r['xwoba']:.3f}, SLG={r['slg']:.3f}")
    print("\nDone.")


if __name__ == "__main__":
    sys.exit(main())
