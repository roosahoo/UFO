import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import pathlib
    import zipfile
    from urllib.request import urlretrieve

    DATA_DIR = pathlib.Path.cwd() / "data"
    DATA_DIR.mkdir(exist_ok=True)
    return DATA_DIR, mo, pl, urlretrieve, zipfile


@app.cell
def _(mo):
    mo.md("""
    The dataset has no coordinates, so we derive them from the `city` + `state` pair
    using the GeoNames US gazetteer (all populated places, not just large cities).
    The coordinate is the city centroid, which is accurate enough for wave detection.
    """)
    return


@app.cell
def _(DATA_DIR, urlretrieve, zipfile):
    # Download the GeoNames US dump (once). ~100 MB unzipped.

    GEONAMES_URL = "https://download.geonames.org/export/dump/US.zip"
    GEONAMES_ZIP = DATA_DIR / "US.zip"
    GEONAMES_TXT = DATA_DIR / "US.txt"

    geonames_ok = GEONAMES_TXT.exists()
    if not geonames_ok:
        try:
            urlretrieve(GEONAMES_URL, GEONAMES_ZIP)
            with zipfile.ZipFile(GEONAMES_ZIP) as _z:
                _z.extract("US.txt", DATA_DIR)
            geonames_ok = True
            print("GeoNames downloaded")
        except Exception as _e:
            print("GeoNames download failed, falling back to geonamescache:", _e)
    else:
        print("GeoNames already present")
    return GEONAMES_TXT, geonames_ok


@app.cell
def _(GEONAMES_TXT, geonames_ok, pl):
    # Gazetteer: (name, state, lat, lon, pop)

    GAZ_COLS = ["name", "state", "lat", "lon", "pop"]

    if geonames_ok:
        # GeoNames columns (0-indexed): 1 name, 2 asciiname, 4 lat, 5 lon,
        # 6 feature class, 10 admin1 (state), 14 population
        _raw = pl.read_csv(
            GEONAMES_TXT,
            separator="\t",
            has_header=False,
            quote_char=None,
            infer_schema_length=0,
        ).select(
            pl.col("column_1").alias("name"),
            pl.col("column_2").alias("ascii"),
            pl.col("column_4").alias("lat"),
            pl.col("column_5").alias("lon"),
            pl.col("column_6").alias("fclass"),
            pl.col("column_10").alias("state"),
            pl.col("column_14").alias("pop"),
        )
        _places = _raw.filter(pl.col("fclass") == "P").with_columns(
            pl.col("lat").cast(pl.Float64),
            pl.col("lon").cast(pl.Float64),
            pl.col("pop").cast(pl.Int64),
        )
        gazetteer = pl.concat(
            [
                _places.select(pl.col("name"), "state", "lat", "lon", "pop"),
                _places.select(pl.col("ascii").alias("name"), "state", "lat", "lon", "pop"),
            ]
        )
    else:
        import geonamescache

        _gc = geonamescache.GeonamesCache()
        gazetteer = pl.DataFrame(
            [
                {"name": c["name"], "state": c["admin1code"], "lat": c["latitude"],
                 "lon": c["longitude"], "pop": c["population"]}
                for c in _gc.get_cities().values()
                if c["countrycode"] == "US"
            ]
        )

    print("Gazetteer rows:", gazetteer.height)
    return (gazetteer,)


@app.cell
def _(pl):
    # Name normalization: same function for both sightings and gazetteer

    def normalize_name(col: str) -> pl.Expr:
        return (
            pl.col(col)
            .str.to_lowercase()
            .str.replace_all(r"\(.*?\)", " ")          # drop unrelevant
            .str.split("/").list.first()               # "A / B" -> "A"
            .str.replace_all(r"\bst\.?\s", "saint ")
            .str.replace_all(r"\bste\.?\s", "sainte ")
            .str.replace_all(r"\bft\.?\s", "fort ")
            .str.replace_all(r"\bmt\.?\s", "mount ")
            .str.replace(r"^n\.?\s", "north ")           # "N. Las Vegas"
            .str.replace_all(r"[^a-z0-9 ]", " ")
            .str.replace_all(r"\s+", " ")
            .str.strip_chars()
        )

    return (normalize_name,)


@app.cell
def _(gazetteer, normalize_name, pl):
    # One row per (name, state): the largest place wins (Clinton, Hudson...)

    gaz = (
        gazetteer
        .with_columns(normalize_name("name").alias("city_key"))
        .filter(pl.col("city_key") != "")
        .sort("pop", descending=True)
        .unique(subset=["city_key", "state"], keep="first", maintain_order=True)
        .select("city_key", "state", "lat", "lon")
    )
    print("Unique (name, state) pairs:", gaz.height)
    return (gaz,)


@app.cell
def _(gaz, normalize_name, pl):
    # Join coordinates onto the sightings

    ufo = pl.read_csv("data/data_clean.csv")

    # Rows whose city is tagged as foreign carry a wrong US state code
    _foreign = pl.col("city").str.contains(r"(?i)\((canada|mexico|uk|australia)")

    ufo_geo = (
        ufo
        .with_columns(
            normalize_name("city").alias("city_key"),
            _foreign.alias("is_foreign"),
        )
        .join(gaz, on=["city_key", "state"], how="left")
        .with_columns(
            pl.when(pl.col("is_foreign"))
            .then(None).otherwise(pl.col("lat")).alias("lat"),
            pl.when(pl.col("is_foreign"))
            .then(None).otherwise(pl.col("lon")).alias("lon"),
        )
    )

    assert ufo_geo.height == ufo.height, "Join duplicated rows!"
    return (ufo_geo,)


@app.cell
def _(mo, ufo_geo):
    # Match rate and quality check

    _total = ufo_geo.height
    _hit = ufo_geo["lat"].is_not_null().sum()
    _foreign = ufo_geo["is_foreign"].sum()
    _rate = _hit / _total

    mo.md(f"""
    ## Match rate

    - Total sightings: **{_total:,}**
    - Geocoded: **{_hit:,}** (**{_rate:.1%}**)
    - Tagged as foreign (dropped): {_foreign:,}
    - Verdict: {"sufficient (≥ 85 %)" if _rate >= 0.85 else "below 85 %, see the unmatched names below"}
    """)
    return


@app.cell
def _(mo, pl, ufo_geo):
    # Most common places left ungeocoded -> shows what to clean up

    unmatched = (
        ufo_geo
        .filter(pl.col("lat").is_null() & ~pl.col("is_foreign"))
        .group_by("city", "state")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .head(40)
    )
    mo.ui.table(unmatched)
    return


@app.cell
def _(mo, pl, ufo_geo):
    # Sanity check: coordinates fall in the right state (rough bbox check)

    _bounds = (
        ufo_geo.drop_nulls("lat")
        .group_by("state")
        .agg(
            pl.col("lat").min().alias("lat_min"), pl.col("lat").max().alias("lat_max"),
            pl.col("lon").min().alias("lon_min"), pl.col("lon").max().alias("lon_max"),
            pl.col("lat").std().alias("lat_std"),
        )
        .with_columns(((pl.col("lat_max") - pl.col("lat_min"))).alias("lat_span"))
        .sort("lat_span", descending=True)
    )
    mo.vstack([
        mo.md("**Sanity check:** states whose points are spread the widest "
              "(AK/HI/TX are naturally large, others suggest wrong matches)"),
        mo.ui.table(_bounds.head(12)),
    ])
    return


@app.cell
def _(DATA_DIR, ufo_geo):
    # Save for the next step

    out = ufo_geo.drop("is_foreign")
    out.write_parquet(DATA_DIR / "ufo_geocoded.parquet")
    print("Saved with coordinates:", out.drop_nulls("lat").height)
    return


if __name__ == "__main__":
    app.run()
