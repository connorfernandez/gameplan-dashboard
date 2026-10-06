"""Matchup Lab — Game Planning Dashboard.

Single-page app: pick a hitter and a pitcher, get a 100-scale matchup rating
(100 = league-average expected production), a per-pitch-type breakdown, zone
heatmaps for both players, and the hitter's rolling wOBA trend.

Data: precomputed Statcast aggregates (CSV, committed in data/processed/).
No live queries happen inside the app. Team-neutral by design.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine.matchup import (
    format_last_first,
    list_hitters,
    list_pitchers,
    load_aggregates,
    matchup_context,
    matchup_rating,
)

st.set_page_config(
    page_title="Matchup Lab — Game Planning Dashboard",
    layout="wide",
)

PROCESSED_DIR = Path(__file__).parent / "data" / "processed"

# Sample minimums for the dropdowns. Pitch-type splits are noisy at tiny
# samples; the rating already shrinks toward league average, but the menus
# stay clean and every listed player has a trustworthy baseline.
MIN_HITTER_PITCHES = 200
MIN_PITCHER_PITCHES = 100


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading precomputed aggregates…")
def get_tables() -> dict[str, pd.DataFrame]:
    """Load all CSV tables (cached across reruns)."""
    return load_aggregates(str(PROCESSED_DIR))


try:
    tables = get_tables()
except FileNotFoundError as e:
    st.error(
        "Could not load the precomputed data tables.\n\n"
        f"{e}\n\n"
        "Expected `data/processed/*_part*.csv` next to `app.py` — run the data "
        "pipeline first (see README.md), then reload."
    )
    st.stop()

# Date range for the methodology note, computed from the actual data.
game_dates = pd.to_datetime(tables["hitter_game"]["game_date"], errors="coerce")
date_range_str = (
    f"{game_dates.min():%b %d, %Y} – {game_dates.max():%b %d, %Y}"
    if game_dates.notna().any()
    else "unknown"
)


# ---------------------------------------------------------------------------
# Header + selectors
# ---------------------------------------------------------------------------

st.title("Matchup Lab")
st.caption(
    "Game-planning dashboard · hitter-vs-pitcher expected production · "
    "Statcast aggregates, no live queries · team-neutral"
)

hitters = list_hitters(tables, MIN_HITTER_PITCHES)
pitchers = list_pitchers(tables, MIN_PITCHER_PITCHES)

if hitters.empty or pitchers.empty:
    st.error(
        "No qualified players found in the data. Hitters need "
        f">= {MIN_HITTER_PITCHES} pitches and pitchers need >= {MIN_PITCHER_PITCHES} "
        "pitches of aggregate data."
    )
    st.stop()

hitter_id_by_display = dict(zip(hitters["display"], hitters["batter_id"]))
pitcher_id_by_display = dict(zip(pitchers["display"], pitchers["pitcher_id"]))

col_h, col_p = st.columns(2)
with col_h:
    hitter_display = st.selectbox(
        f"Hitter  ·  ≥{MIN_HITTER_PITCHES} pitches in data",
        options=hitters["display"].tolist(),
        key="hitter",
    )
with col_p:
    pitcher_display = st.selectbox(
        f"Pitcher  ·  ≥{MIN_PITCHER_PITCHES} pitches in data",
        options=pitchers["display"].tolist(),
        key="pitcher",
    )

hitter_id = int(hitter_id_by_display[hitter_display])
pitcher_id = int(pitcher_id_by_display[pitcher_display])


# ---------------------------------------------------------------------------
# Rating
# ---------------------------------------------------------------------------

try:
    ctx = matchup_context(hitter_id, pitcher_id, processed_dir=PROCESSED_DIR)
    rating, breakdown = matchup_rating(
        hitter_id, pitcher_id, processed_dir=PROCESSED_DIR
    )
except ValueError as e:
    st.error(f"Could not compute this matchup: {e}")
    st.stop()


def _rating_color(r: float) -> str:
    """Red = elite, blue = below average (Savant-style)."""
    if r >= 110:
        return "#e63946"  # elite
    if r >= 105:
        return "#f4845f"  # above average
    if r >= 95:
        return "#e8e8e8"  # around average
    if r >= 90:
        return "#5aa9e6"  # below average
    return "#2f6fdd"      # well below average


def _rating_caption(r: float) -> str:
    diff = r - 100
    if diff > 0:
        return f"{r:.1f} = {diff:.0f}% above league-average expected production"
    if diff < 0:
        return f"{r:.1f} = {abs(diff):.0f}% below league-average expected production"
    return "100.0 = exactly league-average expected production"


color = _rating_color(rating)
st.markdown(
    f"""
    <div style="
        background-color:#161b26;
        border:1px solid #2a3242;
        border-left:6px solid {color};
        border-radius:12px;
        padding:28px 32px;
        margin:12px 0 20px 0;
        text-align:center;">
        <div style="font-size:0.85rem; letter-spacing:0.15em; color:#9aa3b2;
                    text-transform:uppercase; margin-bottom:6px;">
            Matchup rating
        </div>
        <div style="font-size:4.2rem; font-weight:800; line-height:1; color:{color};">
            {rating:.1f}
        </div>
        <div style="font-size:1.05rem; color:#c9d1de; margin-top:10px;">
            {_rating_caption(rating)}
        </div>
        <div style="font-size:0.85rem; color:#8b93a5; margin-top:6px;">
            {format_last_first(ctx['hitter_name'])} ({ctx['h_stand']}HB)
            &nbsp;vs&nbsp;
            {format_last_first(ctx['pitcher_name'])} ({ctx['p_hand']}HP)
            · 100 = league average
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Per-pitch-type breakdown
# ---------------------------------------------------------------------------

st.subheader("Arsenal vs. approach — by pitch type")
st.dataframe(
    breakdown,
    width="stretch",
    hide_index=True,
    column_config={
        "pitch_type": st.column_config.TextColumn("Pitch"),
        "hitter_pitches": st.column_config.NumberColumn("Hitter pitches", format="%d"),
        "hitter_xwoba": st.column_config.NumberColumn(
            "Hitter xwOBA*", format="%.1f",
            help="Shrunk toward league-average pitch-type xwOBA (50-PA prior).",
        ),
        "hitter_whiff_pct": st.column_config.NumberColumn("Hitter whiff %", format="%.1f"),
        "pitcher_usage_pct": st.column_config.NumberColumn("Pitcher usage %", format="%.1f"),
        "pitcher_stuff_lite": st.column_config.NumberColumn(
            "Stuff (lite)", format="%.1f",
            help="Approximate Stuff+ — 100 = average. See methodology.",
        ),
    },
)
st.caption("* Hitter xwOBA is shrunk toward the league-average xwOBA for that pitch type.")


# ---------------------------------------------------------------------------
# Zone heatmaps
# ---------------------------------------------------------------------------

def _zone_matrix(df: pd.DataFrame, id_col: str, pid: int, value_col: str) -> np.ndarray:
    """Pivot zone (1-9) values into a 3x3 matrix; rows top→bottom = 1-2-3 / 4-5-6 / 7-8-9.

    Missing zones stay NaN so they render blank instead of a fake zero.
    """
    m = np.full((3, 3), np.nan)
    sub = df[df[id_col] == pid]
    for _, r in sub.iterrows():
        z = int(r["zone"])
        if 1 <= z <= 9:
            m[(z - 1) // 3, (z - 1) % 3] = r[value_col]
    return m


def _zone_heatmap(matrix: np.ndarray, title: str, colorscale: str,
                  zmin: float, zmax: float, suffix: str) -> go.Figure:
    # 1-decimal annotations; blanks where the player has no data in that zone.
    text = np.where(
        np.isnan(matrix), "",
        np.char.add(np.round(matrix, 1).astype(str), suffix),
    )
    fig = go.Heatmap(
        z=matrix,
        colorscale=colorscale,
        zmin=zmin,
        zmax=zmax,
        text=text,
        texttemplate="%{text}",
        textfont={"size": 13, "color": "#0e1117"},
        hovertemplate=f"Zone %{{x}},%{{y}}: %{{z:.1f}}{suffix}<extra></extra>",
        xgap=3,
        ygap=3,
        colorbar={"title": suffix.strip(), "tickformat": ".1f"},
    )
    layout = go.Figure(fig)
    layout.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"showticklabels": False, "showgrid": False, "zeroline": False},
        yaxis={"showticklabels": False, "showgrid": False, "zeroline": False,
               "autorange": "reversed", "scaleanchor": "x"},
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        height=380,
    )
    return layout


st.subheader("Zone fingerprints")
col_z1, col_z2 = st.columns(2)
with col_z1:
    hitter_zone = _zone_matrix(tables["zone_hitter"], "batter_id", hitter_id, "xwoba")
    st.plotly_chart(
        _zone_heatmap(
            hitter_zone,
            f"Hitter xwOBA by zone — {hitter_display}",
            colorscale="RdBu_r",  # reversed: red = good
            zmin=0.2, zmax=0.4,
            suffix="",
        ),
        width="stretch",
    )
with col_z2:
    pitcher_zone = _zone_matrix(tables["zone_pitcher"], "pitcher_id", pitcher_id, "pitch_pct")
    st.plotly_chart(
        _zone_heatmap(
            pitcher_zone,
            f"Pitcher pitch % by zone — {pitcher_display}",
            colorscale="Blues",
            zmin=0.0, zmax=float(np.nanmax(pitcher_zone)) if np.isfinite(pitcher_zone).any() else 20.0,
            suffix="%",
        ),
        width="stretch",
    )
st.caption("Zones 1–9, top row = 1-2-3. Blank cells = no data in that zone.")


# ---------------------------------------------------------------------------
# Rolling wOBA trend
# ---------------------------------------------------------------------------

st.subheader("Hitter form — rolling 15-game wOBA")
hg = tables["hitter_game"]
hg_h = hg[hg["batter_id"] == hitter_id].copy()
if hg_h.empty:
    st.info("No game-level data for this hitter.")
else:
    hg_h["game_date"] = pd.to_datetime(hg_h["game_date"], errors="coerce")
    hg_h = hg_h.dropna(subset=["game_date"]).sort_values("game_date")
    hg_h["woba_roll15"] = hg_h["woba"].rolling(15, min_periods=1).mean().round(1)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=hg_h["game_date"],
            y=hg_h["woba_roll15"],
            mode="lines+markers",
            name="wOBA (15-game roll)",
            line={"color": "#e63946", "width": 2.5},
            marker={"size": 5},
            hovertemplate="%{x|%b %d, %Y}: %{y:.1f}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Game date",
        yaxis_title="wOBA",
        yaxis={"tickformat": ".1f"},
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        height=340,
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch")
    st.caption("15-game rolling mean of game wOBA (min 1 game at the start of the sample).")


# ---------------------------------------------------------------------------
# Methodology
# ---------------------------------------------------------------------------

with st.expander("Methodology & data notes"):
    st.markdown(
        f"""
        **What the rating means.** 100 = league-average expected production.
        Each pitch type in the pitcher's arsenal is scored by the hitter's
        historical xwOBA against that pitch type (same pitcher-handedness split),
        then weighted by how often the pitcher throws it.

        **Shrinkage.** Small samples lie. Every pitch-type xwOBA is shrunk toward
        the league-average xwOBA for that pitch type with a 50-PA prior — the
        same idea real projection models use. A 12-pitch sample barely moves the
        needle; a 500-pitch sample speaks for itself.

        **Stuff (lite).** An approximation of Stuff+: pitch velocity, spin, and
        movement z-scored against league average and scaled so ~100 is average.
        It is *not* official Stuff+.

        **Data.** Statcast via pybaseball, precomputed to CSV — the app makes
        no live queries. Date range in the current build: **{date_range_str}**.
        Team-neutral: no team filters anywhere in the app.

        **Sample minimums.** Dropdowns list hitters with ≥{MIN_HITTER_PITCHES}
        pitches and pitchers with ≥{MIN_PITCHER_PITCHES} pitches of aggregate
        data (pitcher floor approximated by repertoire coverage where raw pitch
        counts aren't stored). All numbers shown rounded to 1 decimal.
        """
    )

st.caption("Matchup Lab · precomputed Statcast aggregates · no live queries")
