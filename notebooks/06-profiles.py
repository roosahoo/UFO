import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import numpy as np
    import altair as alt
    import pathlib
    from sklearn.feature_extraction.text import CountVectorizer

    DATA_DIR = pathlib.Path.cwd() / "data"
    return CountVectorizer, DATA_DIR, alt, mo, np, pl


@app.cell
def _(mo):
    mo.md("""
    Explain waves= label each wave with a likely mundane explanation (fireworks, meteor/fireball, rocket launch) so that the unexplained ones stand out. ;-)

    Region profile= what kind of sightings does an area have compared to the rest of the country (shape *lift*, hour of day, month, distinctive words, waves).
    """)
    return


@app.cell
def _(DATA_DIR, pl):
    # Load outputs of the previous steps

    sightings = pl.read_parquet(DATA_DIR / "sightings_waves.parquet")
    waves = pl.read_parquet(DATA_DIR / "wave_summary.parquet")
    print(f"{sightings.height:,} sightings, {waves.height:,} waves")
    return sightings, waves


@app.cell
def _(DATA_DIR, pl):
    # Optional (maybe in the future?): rocket launches for matching. Lets put a CSV at data/launches.csv with columns if data can be found?? ;-|
    #   date (YYYY-MM-DD), name, lat, lon
    # (e.g. exported from the Launch Library 2 API). If it is missing, the launch rule is skipped.

    _path = DATA_DIR / "launches.csv"
    if _path.exists():
        launches = pl.read_csv(_path).with_columns(pl.col("date").str.to_date("%Y-%m-%d"))
        print(f"{launches.height:,} launches loaded")
    else:
        launches = None
        print("No data/launches.csv found, skipping launch matching")
    return (launches,)


@app.cell
def _(launches, np, pl, waves):
    # Rule-based explanation per wave. Order matters: the first matching rule wins.

    def _haversine_km(lat1, lon1, lat2, lon2):
        p = np.pi / 180
        a = (np.sin((lat2 - lat1) * p / 2) ** 2
             + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2)
        return 12742 * np.arcsin(np.sqrt(a))

    def explain(row: dict) -> str:
        start, end = row["start"], row["end"]

        # 1) Fireworks: waves overlapping July 4th or New Year's Eve
        _md_start, _md_end = (start.month, start.day), (end.month, end.day)
        # (only short waves; a month-long wave is not one fireworks night)
        if row["duration_days"] <= 4:
            if start.year == end.year and _md_start <= (7, 5) and _md_end >= (7, 3):
                return "fireworks (July 4th)"
            if _md_start >= (12, 30) or _md_end <= (1, 2):
                return "fireworks (New Year)"

        # 2) Rocket launch: launch date within +-1 day of the wave and within 1500 km
        if launches is not None:
            _hit = launches.filter(
                (pl.col("date") >= start - pl.duration(days=1))
                & (pl.col("date") <= end + pl.duration(days=1))
            )
            for _l in _hit.iter_rows(named=True):
                if _haversine_km(row["lat"], row["lon"], _l["lat"], _l["lon"]) <= 1500:
                    return f"rocket launch ({_l['name']})"

        # 3) Meteor / fireball: short, wide, mostly fireball-type reports
        if row["top_shape"] in ("Fireball", "Flash") and row["duration_days"] <= 1.5:
            return "possible meteor / fireball"

        return "unexplained"

    explained = waves.with_columns(
        pl.Series(
            "explanation", [explain(r) for r in waves.iter_rows(named=True)]
        )
    )
    return (explained,)


@app.cell
def _(explained, mo):
    _counts = explained.group_by("explanation").len().sort("len", descending=True)
    mo.vstack([
        mo.md("## Wave explanations"),
        mo.ui.table(_counts),
        mo.md("**Unexplained waves, largest first** (the interesting ones):"),
        mo.ui.table(
            explained.filter(explained["explanation"] == "unexplained")
            .sort("n", descending=True)
            .select("wave", "n", "start", "end", "duration_days", "span_km",
                    "track_rho", "top_shape", "shape_share", "states")
        ),
    ])
    return


@app.cell
def _(mo, sightings):
    # Region selector: either a whole state or a radius around a point

    _states = sorted(sightings["state"].unique().to_list())
    mode = mo.ui.radio(["State", "Radius"], value="State", label="Region type")
    state_pick = mo.ui.dropdown(_states, value="CA", label="State")
    center_lat = mo.ui.number(25, 49, value=34.05, step=0.01, label="Lat")
    center_lon = mo.ui.number(-125, -66, value=-118.24, step=0.01, label="Lon")
    radius_km = mo.ui.slider(10, 300, value=100, step=10, label="Radius (km)")

    mo.hstack([mode, state_pick, center_lat, center_lon, radius_km], justify="start")
    return center_lat, center_lon, mode, radius_km, state_pick


@app.cell
def _(center_lat, center_lon, mode, np, pl, radius_km, sightings, state_pick):
    # Split sightings into "in region" and "rest of the country"

    if mode.value == "State":
        _mask = (sightings["state"] == state_pick.value).to_numpy()
        region_name = state_pick.value
    else:
        _p = np.pi / 180
        _la, _lo = sightings["lat"].to_numpy(), sightings["lon"].to_numpy()
        _a = (np.sin((_la - center_lat.value) * _p / 2) ** 2
              + np.cos(center_lat.value * _p) * np.cos(_la * _p)
              * np.sin((_lo - center_lon.value) * _p / 2) ** 2)
        _mask = (12742 * np.arcsin(np.sqrt(_a))) <= radius_km.value
        region_name = f"{radius_km.value} km around ({center_lat.value:.2f}, {center_lon.value:.2f})"

    in_region = sightings.filter(pl.Series(_mask))
    rest = sightings.filter(pl.Series(~_mask))
    return in_region, region_name, rest


@app.cell
def _(in_region, mo, pl, region_name, rest):
    # Shape lift: regional share / national share. A raw percentage only says
    # "Light is the most common everywhere"; lift shows what is *distinctive*.

    _MIN_N = 10

    _reg = in_region.group_by("shape").agg(pl.len().alias("n_region"))
    _all = rest.group_by("shape").agg(pl.len().alias("n_rest"))

    shape_lift = (
        _reg.join(_all, on="shape", how="full", coalesce=True)
        .with_columns(pl.col("n_region", "n_rest").fill_null(0))
        .with_columns(
            (pl.col("n_region") / in_region.height).alias("share_region"),
            (pl.col("n_rest") / rest.height).alias("share_rest"),
        )
        .with_columns((pl.col("share_region") / pl.col("share_rest")).alias("lift"))
        .filter(pl.col("n_region") >= _MIN_N)
        .sort("lift", descending=True)
    )

    mo.vstack([
        mo.md(f"## Profile: {region_name} ({in_region.height:,} sightings)"),
        mo.md(f"**Shape lift** (shapes with at least {_MIN_N} sightings; > 1 = over-represented):"),
        mo.ui.table(shape_lift),
    ])
    return


@app.cell
def _(alt, in_region, mo, pl, rest):
    # Hour-of-day and month distributions (as shares, so region and rest are comparable)

    def _dist(df, col, label):
        return (
            df.group_by(col).len()
            .with_columns((pl.col("len") / df.height).alias("share"), pl.lit(label).alias("group"))
            .select(pl.col(col).alias("value"), "share", "group")
        )

    _hours = pl.concat([_dist(in_region, "hour", "region"), _dist(rest, "hour", "rest")])
    _months = pl.concat([_dist(in_region, "month", "region"), _dist(rest, "month", "rest")])

    def _chart(data, title, xtitle):
        return (
            alt.Chart(data)
            .mark_line(point=True)
            .encode(
                x=alt.X("value:O", title=xtitle),
                y=alt.Y("share:Q", title="Share of sightings", axis=alt.Axis(format="%")),
                color="group:N",
            )
            .properties(title=title, width=330, height=220)
        )

    mo.hstack([_chart(_hours, "Hour of day", "Hour"), _chart(_months, "Month", "Month")])
    return


@app.cell
def _(CountVectorizer, in_region, mo, np, pl, rest):
    # Distinctive words: document-frequency log-ratio, region vs rest of the country

    def _top_terms(n_terms=25, min_df_region=5):
        _docs_r = in_region["summary"].fill_null("").to_list()
        _docs_o = rest["summary"].fill_null("").to_list()
        _vec = CountVectorizer(stop_words="english", binary=True, min_df=20, max_df=0.5)
        _vec.fit(_docs_r + _docs_o)
        _dr = np.asarray(_vec.transform(_docs_r).sum(axis=0)).ravel()
        _do = np.asarray(_vec.transform(_docs_o).sum(axis=0)).ravel()
        _score = (np.log((_dr + 1) / (len(_docs_r) + 1))
                  - np.log((_do + 1) / (len(_docs_o) + 1)))
        _terms = _vec.get_feature_names_out()
        _keep = _dr >= min_df_region
        return (
            pl.DataFrame({"term": _terms, "n_region": _dr, "n_rest": _do, "log_ratio": _score})
            .filter(pl.Series(_keep))
            .sort("log_ratio", descending=True)
            .head(n_terms)
        )

    mo.vstack([mo.md("**Distinctive words in this region:**"), mo.ui.table(_top_terms())])
    return


@app.cell
def _(explained, in_region, mo, pl):
    # Waves that touch this region and example sightings with links

    _wave_ids = in_region.filter(pl.col("wave") >= 0)["wave"].unique()
    region_waves = explained.filter(pl.col("wave").is_in(_wave_ids.to_list())).sort("n", descending=True)

    _examples = (
        in_region.filter(pl.col("summary").is_not_null())
        .sample(n=min(10, in_region.height), seed=333)
        .select("event_date", "city", "state", "shape", "duration", "summary", "event_url")
    )

    mo.vstack([
        mo.md(f"**Waves in this region:** {region_waves.height:,}"),
        mo.ui.table(region_waves.select("wave", "n", "start", "end", "span_km",
                                         "top_shape", "explanation").head(30)),
        mo.md("**Random example sightings:**"),
        mo.ui.table(_examples),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In the test run, 61 of the 647 waves were explained by Fourth of July fireworks, 27 by New Year's celebrations, and 15 by meteors or fireballs. This left 544 unexplained waves. However, that does not necessarily mean they were unusual, since the launch rule was not yet included.
    """)
    return


@app.cell
def _(DATA_DIR, explained):
    # Save for the app step

    explained.write_parquet(DATA_DIR / "wave_summary_explained.parquet")
    print("Saved wave_summary_explained.parquet")
    return


if __name__ == "__main__":
    app.run()
