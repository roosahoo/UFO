import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import altair as alt
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    return KMeans, StandardScaler, alt, mo, pl


@app.cell
def _():
    # K-means clustering example to try out basic functionality
    return


@app.cell
def _(mo, pl):
    # Import data from CSVs

    df = pl.read_csv("data/data_clean.csv")
    normalized = pl.read_csv("data/sightings_normalized.csv")

    mo.ui.table(df)
    return df, normalized


@app.cell
def _(normalized, pl):
    # Normalized sighting rates  by state

    state_sightings = (
        normalized
        .group_by("state")
        .agg(
            pl.col("sightings_per_100k")
            .mean()
            .alias("avg_sightings_per_100k")
        )
    )
    return (state_sightings,)


@app.cell
def _(df, pl):
    # Proportions of sightings during nighttime

    night_sightings = (
        df
        .with_columns(
            (
                (pl.col("hour") > 21) |
                (pl.col("hour") < 6)
            ).cast(pl.Int8).alias("is_night")
        )
        .group_by("state")
        .agg(
            pl.col("is_night")
            .mean()
            .alias("night_share")
        )
    )
    return (night_sightings,)


@app.cell
def _(night_sightings, state_sightings):
    # Join the two dataframes

    features = (
        state_sightings
        .join(night_sightings, on="state", how="inner")
        .sort("state")
    )
    return (features,)


@app.cell
def _(StandardScaler, features):
    # Feature scaling

    scaler_columns = ["avg_sightings_per_100k", "night_share"]

    X = features.select(scaler_columns).to_numpy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return (X_scaled,)


@app.cell
def _(KMeans, X_scaled):
    # K-means model

    model = KMeans(
        n_clusters = 3,
        random_state = 333,
        n_init = 10
    )

    clusters = model.fit_predict(X_scaled)

    return (clusters,)


@app.cell
def _(clusters, features, pl):
    features2 = features.with_columns(
        pl.Series("cluster", clusters)
    )
    print(features2)
    return (features2,)


@app.cell
def _(alt, features2):
    chart = (
        alt.Chart(features2)
        .mark_circle(size=100)
        .encode(
            x=alt.X(
                "avg_sightings_per_100k:Q",
                title="Average annual sightings per 100k"
            ),
            y=alt.Y(
                "night_share:Q",
                title="Proportion of nighttime sightings"
            ),
            color=alt.Color(
                "cluster:N",
                title="Cluster"
            ),
            tooltip=[
                "state:N",
                "avg_sightings_per_100k:Q",
                "night_share:Q",
                "cluster:N"
            ]
        )
        .properties(
            width=650,
            height=400,
            title="Clustering US states by UFO reporting patterns"
        )
        .interactive()
    )

    chart
    return


if __name__ == "__main__":
    app.run()
