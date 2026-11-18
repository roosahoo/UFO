import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full", app_title="UFO Spotter")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import numpy as np
    import altair as alt
    import pathlib
    import scipy.sparse as sp
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    DATA_DIR = pathlib.Path.cwd() / "data"
    US_TOPO = "https://cdn.jsdelivr.net/npm/vega-datasets@v1.29.0/data/us-10m.json"
    return DATA_DIR, KMeans, US_TOPO, alt, mo, np, pl, silhouette_score, sp


@app.cell
def _(mo):
    mo.md("""
    # UFO Spotter

    Pick a region to see what kind of sightings it has, and whether there are
    waves (many similar reports close in space and time) that look like something
    moving. Waves are candidates, not proof of "the same object": many have a
    mundane explanation (fireworks, fireballs, launches) and are labelled as such.

    Methods: DBSCAN (waves), PCA + Spearman (track), TF-IDF + LSA (text similarity),
    KMeans (sighting types).
    """)
    return


@app.cell
def _(DATA_DIR, np, pl, sp):
    # Load everything produced by notebooks 04–06b

    sightings = pl.read_parquet(DATA_DIR / "sightings_waves.parquet")
    scores = pl.read_parquet(DATA_DIR / "wave_scores.parquet")
    X_tfidf = sp.load_npz(DATA_DIR / "tfidf.npz").tocsr()
    X_lsa = np.load(DATA_DIR / "lsa.npy")
    terms = np.load(DATA_DIR / "terms.npy")

    # Row i of the matrices must be row i of the sightings table
    assert sightings.height == X_tfidf.shape[0] == X_lsa.shape[0], "Re-run 06b: matrices out of sync"

    # Reference profile of the whole country
    national_mean_tfidf = np.asarray(X_tfidf.mean(axis=0)).ravel()
    return X_lsa, X_tfidf, national_mean_tfidf, scores, sightings, terms


@app.cell
def _(mo, sightings):
    # Controls

    _states = sorted(sightings["state"].unique().to_list())
    mode = mo.ui.radio(["State", "Radius around a point"], value="State", label="Region")
    state_pick = mo.ui.dropdown(_states, value="CA", label="State")
    center_lat = mo.ui.number(25, 49, value=34.05, step=0.01, label="Lat")
    center_lon = mo.ui.number(-125, -66, value=-118.24, step=0.01, label="Lon")
    radius_km = mo.ui.slider(10, 400, value=100, step=10, label="Radius (km)", show_value=True)

    _y0, _y1 = int(sightings["year"].min()), int(sightings["year"].max())
    years = mo.ui.range_slider(_y0, _y1, value=[_y0, _y1], step=1, label="Years", show_value=True)
    hide_explained = mo.ui.checkbox(label="Hide waves with a mundane explanation")
    min_n = mo.ui.slider(5, 30, value=5, step=1, label="Min reports per wave", show_value=True)
    k_types = mo.ui.slider(2, 8, value=5, step=1, label="Number of sighting types (k)", show_value=True)

    mo.vstack([
        mo.hstack([mode, state_pick, center_lat, center_lon, radius_km], justify="start"),
        mo.hstack([years, min_n, k_types, hide_explained], justify="start"),
    ])
    return (
        center_lat,
        center_lon,
        hide_explained,
        k_types,
        min_n,
        mode,
        radius_km,
        state_pick,
        years,
    )


@app.cell
def _(
    center_lat,
    center_lon,
    mode,
    np,
    pl,
    radius_km,
    sightings,
    state_pick,
    years,
):
    # Region mask (over ALL sightings so indices stay aligned with the matrices)

    _in_years = ((sightings["year"] >= years.value[0]) & (sightings["year"] <= years.value[1])).to_numpy()

    if mode.value == "State":
        _geo = (sightings["state"] == state_pick.value).to_numpy()
        region_name = state_pick.value
    else:
        _p = np.pi / 180
        _la, _lo = sightings["lat"].to_numpy(), sightings["lon"].to_numpy()
        _a = (np.sin((_la - center_lat.value) * _p / 2) ** 2
              + np.cos(center_lat.value * _p) * np.cos(_la * _p)
              * np.sin((_lo - center_lon.value) * _p / 2) ** 2)
        _geo = (12742 * np.arcsin(np.sqrt(_a))) <= radius_km.value
        region_name = f"{radius_km.value} km around ({center_lat.value:.2f}, {center_lon.value:.2f})"

    region_idx = np.where(_geo & _in_years)[0]
    region_df = sightings[region_idx]
    other_df = sightings.filter(pl.Series(~(_geo & _in_years)))
    return other_df, region_df, region_idx, region_name


@app.cell
def _(
    US_TOPO,
    alt,
    center_lat,
    center_lon,
    hide_explained,
    min_n,
    mo,
    mode,
    np,
    pl,
    radius_km,
    region_df,
    region_name,
    scores,
    years,
):
    # Waves touching the region + map

    _ids = region_df.filter(pl.col("wave") >= 0)["wave"].unique().to_list()
    region_waves = (
        scores.filter(pl.col("wave").is_in(_ids))
        .filter(pl.col("n") >= min_n.value)
        .filter((pl.col("start").dt.year() >= years.value[0]) & (pl.col("start").dt.year() <= years.value[1]))
        .sort("mover_score", descending=True)
    )
    if hide_explained.value:
        region_waves = region_waves.filter(pl.col("explanation") == "unexplained")

    # Zoomed projection: fit the map to the region
    if region_df.height:
        if mode.value == "State":
            _lat_c = float(region_df["lat"].median()); _lon_c = float(region_df["lon"].median())
            _span_km = max(float(np.ptp(region_df["lat"].to_numpy())) * 111, 100.0) * 1.15
        else:
            _lat_c, _lon_c = center_lat.value, center_lon.value
            _span_km = radius_km.value * 2.4
        _scale = 450 * 6371 * np.cos(np.deg2rad(_lat_c)) / _span_km
        _proj = dict(type="mercator", center=[_lon_c, _lat_c], scale=float(_scale),
                     translate=[300, 225])

        _base = (
            alt.Chart(alt.topo_feature(US_TOPO, "states"))
            .mark_geoshape(fill="#f2f2f2", stroke="#bbbbbb")
            .project(**_proj)
            .properties(width=600, height=450)
        )
        _sample = region_df.sample(n=min(2500, region_df.height), seed=333)
        _dots = (
            alt.Chart(_sample.select("lat", "lon", "shape", "city", "event_date"))
            .mark_circle(size=8, opacity=0.25, color="#888888")
            .encode(longitude="lon:Q", latitude="lat:Q", tooltip=["city", "shape", "event_date"])
            .project(**_proj)
        )
        _wv = (
            alt.Chart(region_waves.select("wave", "n", "lat", "lon", "explanation", "top_shape", "start", "mover_score"))
            .mark_circle(opacity=0.7, stroke="black", strokeWidth=0.5)
            .encode(
                longitude="lon:Q", latitude="lat:Q",
                size=alt.Size("n:Q", scale=alt.Scale(range=[40, 800]), title="Reports"),
                color=alt.Color("explanation:N", title="Explanation"),
                tooltip=["wave", "n", "start", "top_shape", "explanation", "mover_score"],
            )
            .project(**_proj)
        )
        region_map = (_base + _dots + _wv).properties(title=f"Waves in {region_name}")
    else:
        region_map = None

    mo.vstack([
        mo.md(f"## {region_name}: {region_df.height:,} sightings, {region_waves.height:,} waves"),
        region_map if region_map is not None else mo.md("_No sightings in this region._"),
    ])
    return (region_waves,)


@app.cell
def _(
    KMeans,
    X_lsa,
    X_tfidf,
    k_types,
    mo,
    national_mean_tfidf,
    np,
    pl,
    region_df,
    region_idx,
    silhouette_score,
    terms,
):
    # "What kinds of UFOs are here?" -> KMeans on LSA vectors of the region's reports

    if len(region_idx) < max(30, k_types.value * 10):
        types_view = mo.md("_Too few sightings in this region to find types._")
    else:
        _X = X_lsa[region_idx]
        _km = KMeans(n_clusters=k_types.value, random_state=333, n_init=10).fit(_X)
        _lbl = _km.labels_
        _sil = silhouette_score(_X, _lbl, sample_size=min(5000, len(_lbl)), random_state=333)
        _Xt = X_tfidf[region_idx]

        _rows = []
        for _c in range(k_types.value):
            _m = np.where(_lbl == _c)[0]
            # Distinctive terms: cluster mean TF-IDF minus national mean (generic words cancel out)
            _diff = np.asarray(_Xt[_m].mean(axis=0)).ravel() - national_mean_tfidf
            _top_terms = ", ".join(terms[np.argsort(_diff)[::-1][:8]])
            _shapes = region_df[_m]["shape"].value_counts(sort=True).head(3)
            _shape_txt = ", ".join(
                f"{s} {n / len(_m):.0%}" for s, n in zip(_shapes["shape"], _shapes["count"])
            )
            # Most typical report = closest to the cluster centre
            _near = _m[np.argmin(((_X[_m] - _km.cluster_centers_[_c]) ** 2).sum(axis=1))]
            _rep = region_df[int(_near)]
            _rows.append({
                "type": _c + 1,
                "reports": len(_m),
                "share": round(len(_m) / len(_lbl), 3),
                "typical shapes": _shape_txt,
                "typical words": _top_terms,
                "example": (_rep["summary"][0] or "")[:220],
                "link": _rep["event_url"][0],
            })

        types_view = mo.vstack([
            mo.md(f"""
            ## Sighting types

            KMeans (k={k_types.value}) on the report texts. Silhouette: **{_sil:.2f}**
            (near 0 = the types overlap a lot; short free-text reports usually score low,
            so read the words rather than trusting the boundaries).
            """),
            mo.ui.table(pl.DataFrame(_rows).sort("reports", descending=True), selection=None),
        ])

    types_view
    return


@app.cell
def _(alt, mo, other_df, pl, region_df, region_name):
    # Shape lift and hour-of-day vs the rest of the country

    if region_df.height < 30:
        profile_view = mo.md("")
    else:
        _MIN_N = 10
        _reg = region_df.group_by("shape").agg(pl.len().alias("n_region"))
        _oth = other_df.group_by("shape").agg(pl.len().alias("n_rest"))
        _lift = (
            _reg.join(_oth, on="shape", how="inner")
            .filter(pl.col("n_region") >= _MIN_N)
            .with_columns(
                ((pl.col("n_region") / region_df.height) / (pl.col("n_rest") / other_df.height)).alias("lift")
            )
            .sort("lift", descending=True)
        )
        _lift_chart = (
            alt.Chart(_lift)
            .mark_bar()
            .encode(
                y=alt.Y("shape:N", sort="-x", title=None),
                x=alt.X("lift:Q", title="Lift vs. rest of the US (1 = typical)"),
                color=alt.condition(alt.datum.lift > 1, alt.value("#e45756"), alt.value("#4c78a8")),
                tooltip=["shape", "n_region", "lift"],
            )
            .properties(title="Distinctive shapes", width=330, height=250)
        )

        def _dist(df, label):
            return (
                df.group_by("hour").len()
                .with_columns((pl.col("len") / df.height).alias("share"), pl.lit(label).alias("group"))
            )

        _hours = pl.concat([_dist(region_df, region_name), _dist(other_df, "rest of US")])
        _hour_chart = (
            alt.Chart(_hours)
            .mark_line(point=True)
            .encode(
                x=alt.X("hour:O", title="Hour of day"),
                y=alt.Y("share:Q", title="Share", axis=alt.Axis(format="%")),
                color=alt.Color("group:N", title=None),
            )
            .properties(title="When are they seen?", width=330, height=250)
        )
        profile_view = mo.hstack([_lift_chart, _hour_chart], justify="start")

    profile_view
    return


@app.cell
def _(mo, region_waves):
    # Wave list: select one row to inspect it

    wave_table = mo.ui.table(
        region_waves.select(
            "wave", "n", "start", "end", "span_km", "top_shape", "shape_share",
            "text_z_lsa", "track_rho", "mover_score", "explanation", "top_terms",
        ),
        selection="single",
        page_size=10,
    )
    mo.vstack([
        mo.md("""
        ## Waves

        Sorted by `mover_score` (alike reports that also advance along a line).
        **Select a row** to see its track and reports.
        """),
        wave_table,
    ])
    return (wave_table,)


@app.cell
def _(alt, mo, pl, scores, sightings, wave_table):
    # Selected wave: track on a map + the reports in time order

    if not wave_table.value.height:
        wave_view = mo.md("_Select a wave above._")
    else:
        _w = int(wave_table.value["wave"][0])
        _g = sightings.filter(pl.col("wave") == _w).sort("t_days").with_row_index("order", offset=1)
        _s = scores.filter(pl.col("wave") == _w).row(0, named=True)

        _track = (
            alt.Chart(_g.select("order", "lat", "lon", "city", "shape", "event_date", "hour"))
            .encode(
                x=alt.X("lon:Q", scale=alt.Scale(zero=False), title="Longitude"),
                y=alt.Y("lat:Q", scale=alt.Scale(zero=False), title="Latitude"),
                order="order:Q",
            )
        )
        _path = _track.mark_line(color="#999999", strokeDash=[4, 3])
        _pts = _track.mark_circle(size=140).encode(
            color=alt.Color("order:Q", scale=alt.Scale(scheme="viridis"), title="Time order"),
            tooltip=["order", "city", "shape", "event_date", "hour"],
        )
        _labels = _track.mark_text(dy=-12, fontSize=11).encode(text="order:Q")
        _chart = (_path + _pts + _labels).properties(width=520, height=380, title=f"Wave {_w}: reports in time order")

        wave_view = mo.vstack([
            mo.md(f"""
            ### Wave {_w}: {_s['explanation']}

            **{_s['n']}** reports, {_s['start']} → {_s['end']}, span **{_s['span_km']:.0f} km**.
            Shape agreement **{_s['shape_share']:.0%}** ({_s['top_shape']}), text z-score **{_s['text_z_lsa']:.1f}**,
            track correlation **{_s['track_rho']:.2f}**. Words: _{_s['top_terms']}_
            """),
            _chart,
            mo.ui.table(
                _g.select("order", "event_date", "hour", "city", "state", "shape", "duration", "summary", "event_url"),
                selection=None,
            ),
        ])

    wave_view
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    **Caveats.** Locations are city centroids. Reporting follows population and media
    attention, so busy places dominate. `mover_score` is a heuristic, not a probability.
    Data: NUFORC (unofficial?) reports.
    """)
    return


if __name__ == "__main__":
    app.run()
