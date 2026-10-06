"""Hitter-vs-pitcher matchup engine.

The core idea
--------------
A matchup rating answers: "If this hitter faced this pitcher many times,
what level of production (xwOBA) would we expect, relative to league average?"

We build it pitch-type by pitch-type:

1. Take the pitcher's arsenal against the hitter's batting side
   (each pitch type's usage%).
2. For each pitch type, take the hitter's historical xwOBA against that pitch
   type from the same pitcher-handedness split.
3. Shrink the hitter's pitch-type xwOBA toward the league-average xwOBA for
   that pitch type (see _shrink below for why).
4. Weight each shrunk pitch-type xwOBA by the pitcher's usage% and average.
5. Express the result on a 100 scale vs. overall league xwOBA:
   100 = league average, 110 = 10% above, 90 = 10% below.

All inputs are precomputed CSV aggregates (no live queries).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

# Default location of the precomputed aggregates, relative to the repo root
# (i.e. the parent of the engine/ package).
PROCESSED_DIR_DEFAULT = Path(__file__).resolve().parent.parent / "data" / "processed"

# Every CSV table the engine expects to find in the processed dir.
TABLES = (
    "hitter_pitch",    # batter_id, batter_name, pitch_type, p_throws, n, whiff_pct, xwoba, slg
    "pitcher_pitch",   # pitcher_id, pitcher_name, pitch_type, stand, usage_pct, avg_velo,
                       #   avg_spin, avg_pfx_x, avg_pfx_z, stuff_lite
    "zone_hitter",     # batter_id, batter_name, zone (1-9), n, xwoba
    "zone_pitcher",    # pitcher_id, pitcher_name, zone (1-9), pitch_pct
    "hitter_game",     # batter_id, batter_name, game_date, pa, woba
    "hitter_stand",    # batter_id, p_throws, stand (hitter's most common side vs that hand)
    "league_pitch",    # pitch_type, lg_xwoba, lg_n
    "players",         # player_id, player_name, role ('batter'/'pitcher'/'both'), hand
)

_BATTER_ROLES = ("batter", "both")
_PITCHER_ROLES = ("pitcher", "both")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def load_aggregates(processed_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load every aggregate table into a dict of DataFrames.

    Each table ships as one or more ~100KB CSV parts
    ({name}_part01.csv, {name}_part02.csv, ... — see _write_chunked() in
    data/build_aggregates.py); parts are concatenated back into a single
    DataFrame here.

    Cached (lru_cache) so repeated calls — e.g. one per matchup rating —
    don't re-read CSVs from disk. Raises FileNotFoundError with a clear
    message naming the missing table.
    """
    d = Path(processed_dir)
    if not d.is_dir():
        raise FileNotFoundError(
            f"Processed data directory not found: {d}\n"
            "Expected the precomputed CSV parts at data/processed/ "
            "(built by the data pipeline; see README.md)."
        )
    tables: dict[str, pd.DataFrame] = {}
    for name in TABLES:
        parts = sorted(d.glob(f"{name}_part*.csv"))
        if not parts:
            raise FileNotFoundError(
                f"Missing required table: {name} (no {name}_part*.csv in {d})\n"
                "Run the data pipeline first (see README.md) so that all of "
                f"{', '.join(t + '_part*.csv' for t in TABLES)} exist."
            )
        tables[name] = pd.concat(
            (pd.read_csv(p) for p in parts), ignore_index=True
        )
    return tables


def _resolve_dir(processed_dir: str | Path | None) -> Path:
    return Path(processed_dir) if processed_dir is not None else PROCESSED_DIR_DEFAULT


# ---------------------------------------------------------------------------
# Player helpers
# ---------------------------------------------------------------------------

def format_last_first(name: str) -> str:
    """Render a player name as 'Last, First' for dropdowns.

    Passes through names already in 'Last, First' form; otherwise splits on
    the final space ('Shohei Ohtani' -> 'Ohtani, Shohei').
    """
    name = str(name).strip()
    if ", " in name:
        return name
    parts = name.split()
    if len(parts) >= 2:
        return f"{parts[-1]}, {' '.join(parts[:-1])}"
    return name


def list_hitters(tables: dict[str, pd.DataFrame], min_pitches: int = 200) -> pd.DataFrame:
    """Hitters with at least `min_pitches` pitches of data, sorted 'Last, First'.

    Why a minimum: pitch-type splits get noisy fast at tiny samples. The
    rating already shrinks small samples toward league average, but the
    dropdown itself stays clean and every listed hitter has a trustworthy
    baseline.
    """
    hp = tables["hitter_pitch"]
    totals = hp.groupby(["batter_id", "batter_name"], as_index=False)["n"].sum()
    q = totals[totals["n"] >= min_pitches].copy()
    q["display"] = q["batter_name"].map(format_last_first)
    # Disambiguate rare duplicate display names with the id (keeps ids unique
    # as selectbox keys even if two players share a name).
    dupes = q.duplicated("display", keep=False)
    q.loc[dupes, "display"] = (
        q.loc[dupes, "display"] + "  ·  #" + q.loc[dupes, "batter_id"].astype(str)
    )
    return q.sort_values("display").reset_index(drop=True)


def list_pitchers(tables: dict[str, pd.DataFrame], min_pitches: int = 100) -> pd.DataFrame:
    """Pitchers with meaningful arsenal data, sorted 'Last, First'.

    Why a minimum: same reasoning as hitters — a rating built on a 12-pitch
    cameo tells you nothing.

    Schema note: pitcher_pitch carries usage_pct without raw pitch counts, so
    when no `n` column is present the >=100-pitch floor is approximated by
    repertoire coverage: >=4 (pitch_type x stand) rows, i.e. at least a
    two-pitch arsenal seen from both sides of the plate. Anything less is more
    likely a data artifact (e.g. a position player pitching) than a real
    pitcher workload. If an `n` column exists it is used exactly.
    """
    pp = tables["pitcher_pitch"]
    if "n" in pp.columns:
        totals = pp.groupby(["pitcher_id", "pitcher_name"], as_index=False)["n"].sum()
        q = totals[totals["n"] >= min_pitches][["pitcher_id", "pitcher_name"]].copy()
    else:
        coverage = (
            pp.groupby(["pitcher_id", "pitcher_name"], as_index=False)
            .size()
            .rename(columns={"size": "rows"})
        )
        q = coverage[coverage["rows"] >= 4][["pitcher_id", "pitcher_name"]].copy()
    q["display"] = q["pitcher_name"].map(format_last_first)
    dupes = q.duplicated("display", keep=False)
    q.loc[dupes, "display"] = (
        q.loc[dupes, "display"] + "  ·  #" + q.loc[dupes, "pitcher_id"].astype(str)
    )
    return q.sort_values("display").reset_index(drop=True)


def matchup_context(
    hitter_id: int, pitcher_id: int, processed_dir: str | Path | None = None
) -> dict:
    """Handedness context for a matchup: pitcher hand, hitter stand, names."""
    tables = load_aggregates(str(_resolve_dir(processed_dir)))
    players = tables["players"]

    prow = players[
        (players["player_id"] == pitcher_id) & players["role"].isin(_PITCHER_ROLES)
    ]
    if prow.empty:
        raise ValueError(f"pitcher_id {pitcher_id} not found among pitchers in players.csv.")
    p_hand = str(prow["hand"].mode().iloc[0])

    h_stand = _hitter_stand(tables, hitter_id, p_hand)

    h_name = players.loc[
        (players["player_id"] == hitter_id) & players["role"].isin(_BATTER_ROLES),
        "player_name",
    ]
    p_name = prow["player_name"].iloc[0]
    return {
        "p_hand": p_hand,
        "h_stand": h_stand,
        "hitter_name": str(h_name.iloc[0]) if not h_name.empty else f"#{hitter_id}",
        "pitcher_name": str(p_name),
    }


def _hitter_stand(tables: dict[str, pd.DataFrame], hitter_id: int, p_hand: str | None) -> str:
    """Hitter's batting side vs a pitcher hand, with sensible fallbacks."""
    hs = tables["hitter_stand"]
    rows = hs[hs["batter_id"] == hitter_id]
    if p_hand is not None:
        m = rows[rows["p_throws"] == p_hand]
        if not m.empty:
            return str(m["stand"].mode().iloc[0])
    if not rows.empty:
        return str(rows["stand"].mode().iloc[0])
    # No stand data at all: assume the platoon-advantage side (what a manager
    # would do). Documented, and only reached for degenerate inputs.
    return "L" if p_hand == "R" else "R"


# ---------------------------------------------------------------------------
# The rating
# ---------------------------------------------------------------------------

def _shrink(xw_hitter: float, n_hitter: int, xw_league: float, prior_pa: int) -> float:
    """Shrink a hitter's pitch-type xwOBA toward the league-average pitch-type xwOBA.

    Rationale (this is what real projection models do): a hitter's observed
    xwOBA on, say, 14 sliders is mostly noise — a .450 mark there doesn't mean
    he crushes sliders. Empirical-Bayes-style shrinkage blends the observed
    value with the league average, weighted by sample size:

        shrunk = (n * observed + prior * league_avg) / (n + prior)

    With prior_pa = 50, a 10-pitch sample is ~17% hitter / 83% league average,
    while a 500-pitch sample is ~91% hitter. Small samples regress hard toward
    league average; large samples speak for themselves.
    """
    if prior_pa < 0:
        raise ValueError("prior_pa must be >= 0.")
    return (n_hitter * xw_hitter + prior_pa * xw_league) / (n_hitter + prior_pa)


def matchup_rating(
    hitter_id: int,
    pitcher_id: int,
    prior_pa: int = 50,
    processed_dir: str | Path | None = None,
) -> tuple[float, pd.DataFrame]:
    """Rate a hitter-vs-pitcher matchup on a 100 scale (100 = league average).

    Returns (rating, breakdown) where breakdown is a per-pitch-type DataFrame
    with columns: pitch_type, hitter_pitches, hitter_xwoba (shrunk, 1-dec),
    hitter_whiff_pct (1-dec), pitcher_usage_pct (1-dec), pitcher_stuff_lite
    (1-dec), sorted by pitcher_usage_pct descending.

    Raises ValueError if the hitter or pitcher id is not found in the data.
    """
    tables = load_aggregates(str(_resolve_dir(processed_dir)))
    hp = tables["hitter_pitch"]
    pp = tables["pitcher_pitch"]
    lg = tables["league_pitch"]
    players = tables["players"]

    # --- Validate ids -------------------------------------------------------
    pitcher_rows = players[
        (players["player_id"] == pitcher_id) & players["role"].isin(_PITCHER_ROLES)
    ]
    pitcher_has_arsenal = (pp["pitcher_id"] == pitcher_id).any()
    if pitcher_rows.empty and not pitcher_has_arsenal:
        raise ValueError(
            f"pitcher_id {pitcher_id} not found: no matching pitcher in "
            "players.csv or pitcher_pitch.csv."
        )
    hitter_all_splits = hp[hp["batter_id"] == hitter_id]
    hitter_in_players = (
        (players["player_id"] == hitter_id) & players["role"].isin(_BATTER_ROLES)
    ).any()
    if hitter_all_splits.empty and not hitter_in_players:
        raise ValueError(
            f"hitter_id {hitter_id} not found: no matching hitter in "
            "players.csv or hitter_pitch.csv."
        )

    # --- Handedness context --------------------------------------------------
    # p_throws of the pitcher determines which hitter split we read; the
    # hitter's stand determines which pitcher split we read.
    p_hand: str | None = (
        str(pitcher_rows["hand"].mode().iloc[0]) if not pitcher_rows.empty else None
    )
    h_stand = _hitter_stand(tables, hitter_id, p_hand)

    # Hitter's pitch-type rows vs this pitcher hand; fall back to all hands
    # if the split is empty (avoids a bogus all-league-average rating).
    h = hitter_all_splits
    if p_hand is not None:
        split = hitter_all_splits[hitter_all_splits["p_throws"] == p_hand]
        if not split.empty:
            h = split

    # Pitcher's arsenal vs the hitter's stand; fall back to all stands.
    p = pp[(pp["pitcher_id"] == pitcher_id) & (pp["stand"] == h_stand)]
    if p.empty:
        p = pp[pp["pitcher_id"] == pitcher_id]
    if p.empty:
        raise ValueError(
            f"pitcher_id {pitcher_id} has no arsenal rows in pitcher_pitch.csv."
        )

    # --- League baselines ----------------------------------------------------
    if lg.empty:
        raise ValueError("league_pitch.csv is empty; cannot compute league baselines.")
    lg_map = dict(zip(lg["pitch_type"], lg["lg_xwoba"]))
    lg_overall = float((lg["lg_xwoba"] * lg["lg_n"]).sum() / lg["lg_n"].sum())
    if not np.isfinite(lg_overall) or lg_overall <= 0:
        raise ValueError("League-average xwOBA is not positive; check league_pitch.csv.")

    # --- Pitch-type loop ------------------------------------------------------
    rows: list[dict] = []
    w_sum = 0.0   # total usage weight
    xw_sum = 0.0  # usage-weighted shrunk xwOBA
    for _, pr in p.iterrows():
        pt = str(pr["pitch_type"])
        lg_pt = float(lg_map.get(pt, lg_overall))  # unknown pitch type -> overall avg

        hr = h[h["pitch_type"] == pt]
        n_h = int(hr["n"].sum()) if not hr.empty else 0
        # n-weighted mean xwOBA across any duplicate rows for this pitch type.
        xw_h = float((hr["n"] * hr["xwoba"]).sum() / n_h) if n_h else 0.0
        shrunk = _shrink(xw_h, n_h, lg_pt, prior_pa)

        w = float(pr["usage_pct"]) / 100.0
        w_sum += w
        xw_sum += w * shrunk

        whiff = float((hr["n"] * hr["whiff_pct"]).sum() / n_h) if n_h else np.nan
        rows.append(
            {
                "pitch_type": pt,
                "hitter_pitches": n_h,
                "hitter_xwoba": round(shrunk, 3),  # shrunk value; rounded to 1-dec below
                "hitter_whiff_pct": whiff,
                "pitcher_usage_pct": float(pr["usage_pct"]),
                "pitcher_stuff_lite": float(pr["stuff_lite"]),
            }
        )

    weighted_xwoba = xw_sum / w_sum if w_sum > 0 else lg_overall
    rating = round(100 * weighted_xwoba / lg_overall, 1)

    breakdown = (
        pd.DataFrame(rows)
        .sort_values("pitcher_usage_pct", ascending=False)
        .reset_index(drop=True)
    )
    # Display rounding: 1 decimal everywhere (spec). hitter_pitches stays int.
    breakdown["hitter_xwoba"] = breakdown["hitter_xwoba"].round(1)
    breakdown["hitter_whiff_pct"] = breakdown["hitter_whiff_pct"].round(1)
    breakdown["pitcher_usage_pct"] = breakdown["pitcher_usage_pct"].round(1)
    breakdown["pitcher_stuff_lite"] = breakdown["pitcher_stuff_lite"].round(1)
    return rating, breakdown
