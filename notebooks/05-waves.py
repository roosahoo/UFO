import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import numpy as np
    import pathlib
    from sklearn.cluster import DBSCAN
    from sklearn.decomposition import PCA
    from scipy.stats import spearmanr

    DATA_DIR = pathlib.Path.cwd() / "data"
    return DATA_DIR, DBSCAN, PCA, mo, np, pl, spearmanr


@app.cell
def _(mo):
    mo.md("""
    Wave= several reports close to each other in space and time (DBSCAN on
    x, y, t, where one day is converted to KM_PER_DAY kilometres so that EPS means
    "about X km or Y days"). We then score each wave for a track-like pattern:
    do the reports advance along one direction as time passes?

    Based on the sighting data, it is not possible to prove that the sightings involve the same UFO, but it is possible to identify waves of sightings (multiple reports occurring close to each other in both time and location) and determine whether the wave appears to follow a moving trajectory.

    !! So waves are candidates, not proof of the "same" object !! :D ^^
    """)
    return


@app.cell
def _():
    # Parameters (tweak these, then see the sensitivity grid at the bottom)

    YEAR_MIN = 2000        # reporting volume changed drastically with the internet
    EPS = 60.0             # neighbourhood radius in the combined km-space
    KM_PER_DAY = 20.0      # 1 day counts as this many km (lower = allows longer gaps)
    MIN_SAMPLES = 5        # minimum reports for a wave
    N_PERMUTATIONS = 3     # permutation runs (each is a full DBSCAN)
    return EPS, KM_PER_DAY, MIN_SAMPLES, N_PERMUTATIONS, YEAR_MIN


@app.cell
def _(DATA_DIR, YEAR_MIN, np, pl):
    # Load the geocoded sightings and build metric coordinates

    raw = pl.read_parquet(DATA_DIR / "ufo_geocoded.parquet")

    sightings = (
        raw
        .drop_nulls(["lat", "lon", "event_date", "hour"])
        .filter(pl.col("year") >= YEAR_MIN)
        .with_columns(pl.col("event_date").str.to_date("%Y-%m-%d"))
        .with_columns(
            (
                pl.col("event_date").cast(pl.Int32)
                + (pl.col("hour") + pl.col("minute") / 60.0) / 24.0
            ).alias("t_days")
        )
        .sort("t_days")
        .with_row_index("sid")
    )

    # Equirectangular projection around the mean latitude (good enough for the US)
    _lat0 = np.deg2rad(sightings["lat"].mean())
    KM_LAT = 111.0
    KM_LON = 111.0 * np.cos(_lat0)

    x_km = sightings["lon"].to_numpy() * KM_LON
    y_km = sightings["lat"].to_numpy() * KM_LAT
    t_days = sightings["t_days"].to_numpy()

    print(f"{sightings.height:,} geocoded sightings since {YEAR_MIN} "
          f"({raw.height:,} rows in file)")
    return KM_LAT, KM_LON, sightings, t_days, x_km, y_km


@app.cell
def _(DBSCAN, np):
    def find_waves(x, y, t, eps, km_per_day, min_samples):
        """Return DBSCAN labels (-1 = not in any wave) in x/y/time space."""
        X = np.column_stack([x, y, t * km_per_day])
        return DBSCAN(
            eps=eps, min_samples=min_samples, algorithm="kd_tree", n_jobs=-1
        ).fit_predict(X)

    return (find_waves,)


@app.cell
def _(
    EPS,
    KM_PER_DAY,
    MIN_SAMPLES,
    find_waves,
    pl,
    sightings,
    t_days,
    x_km,
    y_km,
):
    # Baseline run

    labels = find_waves(x_km, y_km, t_days, EPS, KM_PER_DAY, MIN_SAMPLES)
    waves_df = sightings.with_columns(pl.Series("wave", labels))

    _in_wave = (labels >= 0).mean()
    print(f"{_in_wave:.1%} of sightings belong to a wave; "
          f"{len(set(labels)) - (1 if -1 in labels else 0)} waves")
    return labels, waves_df


@app.cell
def _(KM_LAT, KM_LON, PCA, np, pl, spearmanr, waves_df):
    # Per-wave summary and track score

    def summarize_wave(g: pl.DataFrame) -> dict:
        t_h = g["t_days"].to_numpy() * 24.0
        xy = np.column_stack([g["lon"].to_numpy() * KM_LON, g["lat"].to_numpy() * KM_LAT])

        # Main spatial direction of the wave and how well time orders the reports along it
        # All reports in the same city centroid -> no spatial direction exists
        if np.ptp(xy, axis=0).max() > 0:
            main_axis = PCA(n_components=1).fit_transform(xy).ravel()
        else:
            main_axis = np.zeros(len(xy))
        if np.ptp(t_h) > 0 and np.ptp(main_axis) > 0:
            rho = abs(spearmanr(t_h, main_axis).statistic)
        else:
            rho = np.nan

        span_km = float(np.ptp(main_axis))
        hours = float(np.ptp(t_h))

        shapes = g["shape"].value_counts(sort=True)
        return {
            "n": g.height,
            "start": g["event_date"].min(),
            "end": g["event_date"].max(),
            "duration_days": hours / 24.0,
            "span_km": span_km,
            "speed_kmh": span_km / max(hours, 1.0),
            "track_rho": float(rho),
            "top_shape": shapes["shape"][0],
            "shape_share": shapes["count"][0] / g.height,
            "states": ",".join(sorted(g["state"].unique().to_list())[:4]),
            "lat": g["lat"].mean(),
            "lon": g["lon"].mean(),
        }

    wave_summary = pl.DataFrame(
        [
            {"wave": w[0], **summarize_wave(g)}
            for w, g in waves_df.filter(pl.col("wave") >= 0).group_by("wave")
        ]
    ).sort("n", descending=True)
    return (wave_summary,)


@app.cell
def _(mo, pl, wave_summary):
    # Flag track candidates (thresholds are a starting point, tune by reading summaries)

    candidates = wave_summary.with_columns(
        (
            (pl.col("track_rho") >= 0.7)
            & (pl.col("shape_share") >= 0.5)
            & (pl.col("n") >= 8)
            & (pl.col("span_km") >= 20)
        ).alias("track_candidate"),
        # Very long waves are usually busy cities chaining together, not one event
        (pl.col("duration_days") > 14).alias("chained"),
    )

    mo.vstack([
        mo.md(f"""
        ## Waves

        - Waves: **{candidates.height:,}**
        - Track candidates: **{candidates['track_candidate'].sum():,}**
        - Probably chained (> 14 days): **{candidates['chained'].sum():,}**
        """),
        mo.ui.table(candidates),
    ])
    return (candidates,)


@app.cell
def _(
    EPS,
    KM_PER_DAY,
    MIN_SAMPLES,
    N_PERMUTATIONS,
    find_waves,
    labels,
    mo,
    np,
    sightings,
    t_days,
    x_km,
    y_km,
):
    # Permutation test: shuffle times among sightings of the same year (keeps locations
    # and yearly volume). If real data forms clearly more/larger waves, they are not
    # just a density effect.

    _rng = np.random.default_rng(333)
    _years = sightings["year"].to_numpy()

    def _summ(lbl):
        _ids = lbl[lbl >= 0]
        _n = len(set(_ids))
        _share = float((lbl >= 0).mean())
        _largest = int(np.bincount(_ids).max()) if _n else 0
        return _n, _share, _largest

    _real = _summ(labels)
    _perm = []
    for _ in range(N_PERMUTATIONS):
        _t = t_days.copy()
        for _y in np.unique(_years):
            _idx = np.where(_years == _y)[0]
            _t[_idx] = _rng.permutation(_t[_idx])
        _perm.append(_summ(find_waves(x_km, y_km, _t, EPS, KM_PER_DAY, MIN_SAMPLES)))

    _p = np.array(_perm)
    mo.md(f"""
    ## Permutation test ({N_PERMUTATIONS} runs)

    | | waves | share in a wave | largest wave |
    |---|---|---|---|
    | **Real data** | {_real[0]:,} | {_real[1]:.1%} | {_real[2]:,} |
    | Shuffled (mean) | {_p[:, 0].mean():,.0f} | {_p[:, 1].mean():.1%} | {_p[:, 2].mean():,.0f} |

    Real >> shuffled means the waves are genuine space-time concentrations.
    """)
    return


@app.cell
def _(MIN_SAMPLES, find_waves, mo, np, pl, t_days, x_km, y_km):
    # Parameter sensitivity: stable structure should survive reasonable parameter changes

    _rows = []
    for _eps in (30.0, 60.0, 100.0):
        for _kpd in (10.0, 20.0, 40.0):
            _lbl = find_waves(x_km, y_km, t_days, _eps, _kpd, MIN_SAMPLES)
            _rows.append({
                "eps": _eps,
                "km_per_day": _kpd,
                "waves": len(set(_lbl[_lbl >= 0])),
                "share_in_wave": round(float((_lbl >= 0).mean()), 3),
                "largest_wave": int(np.bincount(_lbl[_lbl >= 0]).max()) if (_lbl >= 0).any() else 0,
            })

    mo.vstack([mo.md("## Parameter sensitivity"), mo.ui.table(pl.DataFrame(_rows))])
    return


@app.cell
def _(DATA_DIR, candidates, waves_df):
    # Save for the profile / app steps

    waves_df.write_parquet(DATA_DIR / "sightings_waves.parquet")
    candidates.write_parquet(DATA_DIR / "wave_summary.parquet")
    print("Saved sightings_waves.parquet and wave_summary.parquet")
    return


if __name__ == "__main__":
    app.run()
