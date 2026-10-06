# Matchup Lab — Game Planning Dashboard

A single-page Streamlit app that rates any hitter-vs-pitcher matchup on a
**100 scale** (100 = league-average expected production), with a per-pitch-type
arsenal breakdown, zone heatmaps for both players, and the hitter's rolling
wOBA trend.

Built as a portfolio project toward a pro-baseball front-office standard:
dark, Savant-grade presentation, honest methodology, and zero live queries —
everything runs off precomputed CSV aggregates committed in-repo.

## Features

- **Matchup rating card** — one big number: expected production vs. league
  average, color-coded red (elite) → blue (below average).
- **Arsenal vs. approach table** — every pitch in the pitcher's arsenal, the
  hitter's (shrinkage-adjusted) xwOBA and whiff% against it, usage%, and an
  approximate Stuff+ ("Stuff lite", 100 = average).
- **Zone fingerprints** — side-by-side 3×3 heatmaps: hitter xwOBA by zone and
  pitcher pitch% by zone.
- **Hitter form** — rolling 15-game wOBA trend.
- **Methodology expander** — data source, date range, sample minimums, and the
  shrinkage prior, all stated plainly.
- **Team-neutral** — no team filters anywhere.

## How the rating works (short version)

For each pitch type in the pitcher's arsenal (vs. the hitter's batting side):

1. Take the hitter's historical xwOBA against that pitch type from the same
   pitcher-handedness split.
2. **Shrink it** toward the league-average xwOBA for that pitch type with a
   50-PA prior — small samples regress to league average, the way real
   projection models handle noise.
3. Weight by the pitcher's usage% and average across the arsenal.
4. Divide by overall league xwOBA × 100. That's the rating.

See `engine/matchup.py` for the full commented implementation.

## Repo layout

```
.
├── app.py                  # Streamlit app ("Matchup Lab", single page)
├── engine/
│   ├── __init__.py
│   └── matchup.py          # load_aggregates(), matchup_rating(), dropdown helpers
├── data/
│   ├── processed/          # precomputed *.csv aggregates (committed, <100MB)
│   └── raw/                # raw Statcast pulls (gitignored, reproducible)
├── .streamlit/
│   └── config.toml         # dark pro theme
├── requirements.txt
├── .gitignore
└── README.md
```

## Data tables (`data/processed/`)

| File | Contents |
|---|---|
| `hitter_pitch.csv` | batter × pitch type × pitcher-hand: n, whiff%, xwOBA, SLG |
| `pitcher_pitch.csv` | pitcher × pitch type × batter-stand: usage%, velo/spin/movement, Stuff-lite |
| `zone_hitter.csv` | batter × zone (1–9): n, xwOBA |
| `zone_pitcher.csv` | pitcher × zone (1–9): pitch% |
| `hitter_game.csv` | batter × game: PA, wOBA |
| `hitter_stand.csv` | batter's most common side vs each pitcher hand |
| `league_pitch.csv` | league-average xwOBA per pitch type |
| `players.csv` | player id, name, role, hand |

## Data pipeline

The tables above were built by two scripts in `data/` (run in this order):

```bash
# 1. Pull September 2026 Statcast, one day at a time (~2-4 min; resume-safe)
/home/hatch/.tmp/pbvenv/bin/python data/pull_september.py
# 2. Build the 8 aggregate tables (<1 min on re-run)
python data/build_aggregates.py
```

Details: `pull_september.py` saves `data/raw/statcast_2026-09-DD.parquet` per day
(30 files, 108,683 pitches). `build_aggregates.py` keeps regular-season rows only
(`game_type == 'R'`; regular season ran 2026-09-01 → 2026-09-27, 106,356 pitches,
360 games, 476 batters, 552 pitchers) and writes the 8 tables.

Gotchas the app should know about:

- In this pybaseball output **`player_name` is the pitcher's name**
  ("Last, First") — verified against the MLB Stats API with 0 mismatches.
  Canonical "First Last" display names come from
  `https://statsapi.mlb.com/api/v1/people/<mlbam_id>` (no key needed); the
  1,022-name map is cached in `data/processed/_name_map.json` (internal file,
  not an app table) so re-runs don't re-hit the API.
- `pybaseball.playerid_reverse_lookup` is intentionally avoided: on first use it
  prompts to download the Chadwick register, which blocks non-interactive runs.
- `woba_value`/`woba_denom` only populate on each PA's final pitch (verified),
  so `hitter_game.woba = Σwoba_value/Σwoba_denom` needs no extra dedupe.
  League wOBA over the sample is 0.328 (sanity target ~.310–.330).
- `stuff_lite = clip(100 + 12·mean(z_velo, z_spin, z_move), 50, 150)` where each
  z is the (pitcher, pitch_type, stand) group's average velo/spin/movement
  scored against the league per-pitch-type distribution. Mean ≈ 99, SD ≈ 7.
- `pitcher_pitch` has no pitch-count column by design (spec); use
  `usage_pct` (sums to 100 per pitcher × stand) for volume.

## Run locally

```bash
# 1. Create and activate a virtual environment (example)
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Make sure data/processed/*.csv exists (built by the data pipeline)

# 4. Launch
streamlit run app.py
```

The app opens at http://localhost:8501.

## Deploy to Streamlit Community Cloud (free)

1. Push this repo to GitHub (make sure `data/processed/*.csv` are committed —
   total size is ~1.5MB, within GitHub/Streamlit limits).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **New app** → select the repo, branch, and `app.py` as the main file.
4. Deploy. That's it — `requirements.txt` is picked up automatically and the
   committed CSV files load straight from the repo. No secrets or external
   data connections needed.

## Notes & limitations

- Free data only; aggregates are precomputed, so the app never queries
  Statcast at runtime.
- "Stuff lite" is a z-scored velo/spin/movement approximation, **not** official Stuff+.
- Dropdowns are limited to hitters with ≥200 pitches and pitchers with
  ≥100 pitches of aggregate data (pitcher floor approximated by repertoire
  coverage where raw pitch counts aren't stored) to keep small-sample noise
  out of the menus.
- All displayed numbers are rounded to 1 decimal.
