# UFO Spotter

Explore UFO sightings by region: what kinds of sightings an area has, and whether
there are **waves**: many similar reports close in space and time, some of which look like something *moving*.

> Waves are candidates, not proof of "the same object". Many have mundane explanations
> (fireworks, fireballs, rocket launches), and the app labels those.

Only classical machine learning is used: DBSCAN, PCA, Spearman correlation,
TF-IDF, truncated SVD (LSA), KMeans.

## Quick start
 
Uses [uv](https://docs.astral.sh/uv/) for the environment (Python version is pinned in `.python-version`).
Run every command from the **project root**: the notebooks use relative paths (`data/...`).
 
```bash
# 1. Environment (creates .venv from pyproject.toml / uv.lock)
uv sync
 
# 2. Data folder (notebook 01 does not create it)
mkdir -p data
 
# 3. Run the pipeline in order (see below), then start the app
uv run marimo run notebooks/07-app.py
```
 
Dependencies are managed in `pyproject.toml`. If a package is missing, add it with:
 
```bash
uv add marimo polars numpy scipy scikit-learn altair pyarrow geonamescache
```
 
Each notebook can be run in two ways:
 
```bash
uv run marimo edit notebooks/04-geocode.py     # interactive, in the browser
uv run python notebooks/04-geocode.py          # headless, as a plain script (good for a full re-run)
```

## Pipeline

Run the notebooks in this order. Each one reads the previous one's output from `data/`.

| # | Notebook | What it does | Writes to `data/` |
|---|---|---|---|
| 01 | `01-fetch-clean.py` | Downloads the NUFORC dataset and US state populations, cleans (year ≥ 1960, valid US states) | `data_raw.csv`, `population.csv`, `data_clean.csv` |
| 02 | `02-normalization.py` | Sightings per 100k inhabitants by state and year | `sightings_normalized.csv` |
| 03 | `03-clustering_example.py` | Warm-up: KMeans clustering of US *states* (standalone, not needed by the app) | – |
| 04 | `04-geocode.py` | City + state → latitude/longitude (GeoNames), reports the match rate | `ufo_geocoded.parquet`, `US.zip`, `US.txt` |
| 05 | `05-waves.py` | Finds waves with DBSCAN in (x, y, time) space, scores tracks, permutation test, parameter sensitivity | `sightings_waves.parquet`, `wave_summary.parquet` |
| 06 | `06-profiles.py` | Labels waves with likely explanations, region profiles (shape lift, hours, words) | `wave_summary_explained.parquet` |
| 06b | `06b-text-similarity.py` | TF-IDF/LSA text similarity inside waves vs. a same-state null distribution | `wave_scores.parquet`, `tfidf.npz`, `lsa.npy`, `terms.npy` |
| 07 | `07-app.py` | The app: map, sighting types per region, wave list, wave track viewer | – |

To re-download the raw data, run notebook 01 with the `new` argument:

```bash
marimo edit 01-fetch-clean.py -- new
```

## Using the app

1. Pick a **region**: a state, or a radius around a latitude/longitude.
2. Narrow by **years**, minimum wave size, or hide waves that have a mundane explanation.
3. Read the sections:
   - **Map**: waves in the region (bubble size = number of reports, colour = explanation).
   - **Sighting types**: KMeans (k adjustable) on the report texts, with typical shapes, distinctive words and an example.
   - **Distinctive shapes / hours**: compared with the rest of the US (lift > 1 = over-represented).
   - **Waves**: sorted by `mover_score`. Select a row to see the track in time order and the reports.

### Key columns

| Column | Meaning |
|---|---|
| `track_rho` | How well time orders the reports along the wave's main direction (0–1) |
| `shape_share` | Share of the most common shape in the wave |
| `text_z_lsa` | How much more alike the report texts are than random same-state reports (z-score) |
| `coherence` | ½ shape agreement + ½ clipped text z-score |
| `mover_score` | `coherence × track_rho`; a heuristic, **not** a probability |

## Caveats

- Locations are **city centroids**, not exact positions.
- Reporting follows population and media attention. Compare regions with normalized rates (notebook 02), not raw counts.
- Reporting volume jumped after the internet era, hence `YEAR_MIN = 2000` in notebook 05.
- Text similarity can partly reflect copied or same-author reports.

## Data sources

- NUFORC sighting reports: [LinkWentz/nuforc-dataset](https://github.com/LinkWentz/nuforc-dataset)
- Historical US state population: [JoshData/historical-state-population-csv](https://github.com/JoshData/historical-state-population-csv)
- Place coordinates: [GeoNames](https://www.geonames.org/) (CC BY 4.0)