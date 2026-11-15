import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import altair as alt

    return mo, pl


@app.cell
def _(pl):
    # Load the CSVs to dataframes

    ufo_df = pl.read_csv("data/data_clean.csv")
    pop_df = pl.read_csv(
        "data/population.csv",
        new_columns = ["state", "year", "population"]
    )
    return pop_df, ufo_df


@app.cell
def _(mo, pop_df):
    mo.ui.table(pop_df)
    return


@app.cell
def _(mo, pl, ufo_df):
    # New aggregate dataframe for sightings per year and state

    sightings_per_state_year = (
        ufo_df
        .group_by(["state", "year"])
        .agg(
            pl.len().alias("sightings")
        )
        .sort(["state", "year"])
    )

    mo.ui.table(sightings_per_state_year)
    return (sightings_per_state_year,)


@app.cell
def _(mo, pl, pop_df, sightings_per_state_year):
    # Join the datasets

    joined = (
        pop_df
        .filter(pl.col("year").is_between(1960, 2021))
        .join(
            sightings_per_state_year,
            on=["state", "year"],
            how="left",
            validate="1:1"
        )
        .with_columns(
            pl.col("sightings").fill_null(0)
        )
        .with_columns(
            (
                pl.col("sightings")
                / pl.col("population")
                * 100_000
            ).alias("sightings_per_100k")
        )
    )

    mo.ui.table(joined)
    return (joined,)


@app.cell
def _(joined, mo, pl):
    # Normalized sightings, pretty useless rn but a proof of concept for future work

    normalized_df = joined.with_columns(
        (
            pl.col("sightings")
            / pl.col("population")
            * 100000
        ).alias("sightings_per_100k")
    )

    mo.ui.table(normalized_df)
    return


if __name__ == "__main__":
    app.run()
