# Baseball Ops Playbook — Connor's living glossary

*Plain-English notes on the concepts behind the dashboard. Grows with the project —
every new feature ships with a new entry. No math degree required.*

---

## The workspace & the repo
`~/workspace/gameplan-dashboard/` is the folder on Muse's computer where your entire
app project lives. Think of `~/workspace` as a desk, and `gameplan-dashboard/` as the
project folder sitting on it. When you push to GitHub, this folder *becomes* the
repository — same files, now public.

**What each piece does:**
- `app.py` — the app itself. What you see in the browser.
- `engine/matchup.py` — the brain. The matchup-rating math lives here.
- `data/pull_september.py` — the grocery run. Fetches raw Statcast pitch data.
- `data/build_aggregates.py` — the prep cook. Turns 106,000 raw pitches into the
  summary tables the app actually reads.
- `data/processed/*.parquet` — the filing cabinet. Ready-to-use tables (see below).
- `requirements.txt` — the shopping list of Python packages the app needs.
- `.streamlit/config.toml` — the interior design (dark theme, colors).

## Parquet
A file format for tables — like a supercharged spreadsheet file. A CSV writes every
number out longhand; parquet compresses and indexes the data, so it loads 10–50x
faster in a fraction of the space. Our 106,000 pitches fit in 460KB because of it.
Rule of thumb: CSVs are for humans to eyeball, parquet is for apps to read fast.

## Z-score
Answers: *"how many notches above or below average is this?"* Take league-average
fastball velocity (~94 mph) with a typical spread of ~2 mph. A 98 mph fastball is
(98 − 94) / 2 = **+2.0** — two notches above average. Negative works the same way:
a 90 mph heater is −2.0. Z-scores let you compare apples to oranges (mph vs. rpm
vs. inches of break) on one scale — which is exactly what our stuff-lite does.

## xwOBA (expected weighted on-base average)
The single best number for "how good was this hitter's contact?" It looks at exit
velocity and launch angle of every batted ball and asks: *how often does a ball hit
like that become a hit?* A 110-mph line drive is ~.900 xwOBA even if a fielder
caught it; a bloop single is ~.300 even though it counts as a hit. It strips out
luck and defense. League average hovers around .320. Our matchup rating is built
on it: we estimate the hitter's *expected* xwOBA against a given pitcher.

## Whiff%
Simple: swinging strikes divided by swings. The purest "can he miss bats?" number.
A 35% whiff rate on a slider is elite; under 20% is hittable.

## Stuff+
Created by Eno Sarris (FanGraphs/The Athletic). Grades a pitch on **physical traits
only** — velocity, spin, movement, release point — scaled so **100 = league average**.
It deliberately ignores results: a 100-mph fastball down the middle that gets
crushed still grades as great *stuff* (the pitcher just located it terribly).
Stabilizes around ~80 pitches. Siblings: **Location+** (command) and **Pitching+**
(stuff + location combined).

## Stuff-lite (ours)
Our prototype's version. Takes velocity, spin rate, and movement, converts each to
a z-score, and averages them, scaled so 100 = average. Same *idea* as Stuff+ with
simpler math: instead of a trained model learning how much each trait matters, we
just average them evenly. Good enough to rank arsenals; the roadmap below makes it
real.

## Shrinkage (regression to the mean)
Small samples lie. If a hitter is 3-for-8 lifetime against horizontal fastballs,
we don't trust .375 — we *shrink* it toward league average, more aggressively when
the sample is tiny. Our engine uses a 50-plate-appearance prior: think of it as
"start every hitter at league average, and let real pitches drag the number away
from it." This is what every pro model does, and it's why our ratings don't do
crazy things on 11 pitches.

## The data tables (what each one is, in baseball terms)
- `hitter_pitch` — each hitter's report card vs each pitch type (xwOBA, whiff%…).
- `pitcher_pitch` — each pitcher's arsenal sheet (what he throws, usage%, stuff-lite
  per pitch).
- `zone_hitter` / `zone_pitcher` — the heat-map data: where the hitter does damage,
  where the pitcher lives.
- `hitter_game` — game-by-game lines; powers the rolling trend charts.
- `league_pitch` — league averages by pitch type. This is what "100 = average" is
  measured against.
- `players` — the roster directory (ids, names, handedness, role).

## The matchup rating (how ours works)
For a hitter vs a pitcher: take the hitter's (shrunk) xwOBA against each pitch
type the pitcher throws, weight each by how often the pitcher throws it, and scale
so 100 = league average. A 119 means we'd expect that hitter to produce ~19% above
league average against that pitcher. It's a weighted, regressed expectation — not
a prediction of one at-bat, but of true talent in the matchup.

---

## Roadmap: stuff-lite → a real Stuff+ model (your curriculum)
1. **Done:** stuff-lite (z-score average). Concept learned: z-scores.
2. **Next:** train a simple regression that *learns* the weights — e.g., how much
   is 1 mph of velo worth vs 100 rpm of spin, based on what actually produced
   whiffs and weak contact across millions of pitches. Concept learned: regression.
3. **Then:** validate it like a pro — check that higher grades actually predict
   future whiffs and run prevention on data the model hasn't seen. Concept learned:
   out-of-sample validation.
4. **Later:** per-pitch-type models and arsenal-level aggregation, the way the
   public Stuff+ models do it.
