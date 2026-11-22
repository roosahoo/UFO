import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import altair as alt
    alt.data_transformers.enable("vegafusion")
    return alt, mo, pl


@app.cell
def _(mo):
    # Check for cli_args

    if "new" in mo.cli_args():
        FORCE_NEW = True
    else:
        FORCE_NEW = False
    return (FORCE_NEW,)


@app.cell
def _(FORCE_NEW):
    # Fetch the NUFORC dataset

    import pathlib
    from urllib.request import urlretrieve

    DATASET_URL = "https://raw.githubusercontent.com/LinkWentz/nuforc-dataset/refs/heads/master/nuforc_events_complete.csv"

    ROOT_DIR = pathlib.Path.cwd()
    DATA_DIR = pathlib.Path.cwd() / "data"
    DATA_RAW_PATH = DATA_DIR / "data_raw.csv"
    DATA_CLEAN_PATH = DATA_DIR / "data_clean.csv"

    if not DATA_RAW_PATH.exists() or FORCE_NEW:
        urlretrieve(DATASET_URL, DATA_RAW_PATH)
        print("Success 👽")
    else:
        print("Target already located 🛸")
    return DATA_CLEAN_PATH, DATA_DIR, urlretrieve


@app.cell
def _(DATA_DIR, urlretrieve):
    # Fetch the dataset for historical US state population

    POPULATION_DATASET_URL = "https://raw.githubusercontent.com/JoshData/historical-state-population-csv/refs/heads/main/historical_state_population_by_year.csv"

    POPULATION_DATA_PATH = DATA_DIR / "population.csv"

    if not POPULATION_DATA_PATH.exists():
        urlretrieve(POPULATION_DATASET_URL, POPULATION_DATA_PATH)
        print("Success 🏢")
    else:
        print("Population data already exists 🏛️")
    return


@app.cell
def _(pl):
    # Some exploratory data analysis on the raw data

    df = pl.read_csv("data/data_raw.csv").with_columns(
        [
            pl.col("event_date").str.to_date("%Y-%m-%d"),
            pl.col("year", "month", "day", "hour", "minute").cast(pl.Int64)
        ]
    )

    print("Shape:", df.shape)
    print(df.schema)
    return (df,)


@app.cell
def _(df, mo):
    mo.ui.table(df)
    return


@app.cell
def _(df, pl):
    # Check for missing values

    missing = df.select(
        pl.all().null_count()
    ).transpose(
        include_header=True,
        header_name="column",
        column_names=["missing"]
    ).with_columns(
        (pl.col("missing") / df.height * 100).round(2).alias("percent")
    )
    return (missing,)


@app.cell
def _(missing):
    # Missing values per column

    missing
    return


@app.cell
def _(df, mo):
    # New dataframe without null values

    df_not_null = df.drop_nulls()

    removed_nulls = len(df) - len(df_not_null)
    mo.ui.table(df_not_null)
    return df_not_null, removed_nulls


@app.cell
def _(df_not_null, mo, pl):
    # New dataframe starting from the year 1960

    df_1960 = df_not_null.filter(
        pl.col("year") >= 1960
    )

    removed_pre_1960 = len(df_not_null) - len(df_1960)
    mo.ui.table(df_1960)
    return df_1960, removed_pre_1960


@app.cell
def _(df_1960, mo, pl):
    # Remove invalid states

    valid_states = [
        "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
        "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
        "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
        "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
        "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"
    ]

    df_semifinal = df_1960.filter(pl.col("state").is_in(valid_states))

    removed_invalid_states = len(df_1960) - len(df_semifinal)
    mo.ui.table(df_semifinal)
    return df_semifinal, removed_invalid_states


@app.cell
def _(df_semifinal, pl):
    # Remove events with invalid hours/minutes

    df_final = df_semifinal.filter(
        pl.col("hour").is_between(0,23) &
        pl.col("minute").is_between(0, 59)
    )

    removed_invalid_times = len(df_semifinal) - len(df_final)
    return df_final, removed_invalid_times


@app.cell
def _(DATA_CLEAN_PATH, df_final):
    # Write the new dataframe to a CSV

    df_final.write_csv(DATA_CLEAN_PATH, separator=",")
    return


@app.cell
def _(
    alt,
    df,
    pl,
    removed_invalid_states,
    removed_invalid_times,
    removed_nulls,
    removed_pre_1960,
):
    # Visualization of data removed

    removed_total = removed_invalid_states + removed_invalid_times + removed_nulls + removed_pre_1960
    visualized_total = len(df) - removed_total

    sizes = pl.DataFrame(
        {
            "dataset": ["total", "removed"],
            "size": [visualized_total, removed_total]
        }
    )

    alt.Chart(sizes).mark_arc().encode(theta="size", color="dataset").properties(
        title=alt.TitleParams(
            text="Data removed from the dataset during cleaning",
            subtitle=f"Removed: {removed_total}/{len(df)}"
        )
    )
    return


@app.cell
def _(
    alt,
    pl,
    removed_invalid_states,
    removed_invalid_times,
    removed_nulls,
    removed_pre_1960,
):
    reasons_for_removal = pl.DataFrame(
        {
            "reason": ["null values", "pre-1960", "invalid state", "invalid time"],
            "number": [removed_nulls, removed_pre_1960, removed_invalid_states, removed_invalid_times]
        }
    )

    alt.Chart(reasons_for_removal).mark_bar().encode(
        y=alt.Y("reason:N", title="Reason for removal"),
        x=alt.X("number:Q", title="Events removed")
    )
    return


@app.cell
def _(alt, df_final):
    alt.Chart(df_final).transform_calculate(
        decade="floor(datum.year / 10) * 10"
    ).transform_aggregate(
        count="count()",
        groupby=["decade", "shape"]
    ).transform_joinaggregate(
        total="sum(count)",
        groupby=["decade"]
    ).transform_calculate(
        percentage="datum.count / datum.total * 100"
    ).mark_rect().encode(
        x=alt.X("decade:O", title="Decade"),
        y=alt.Y("shape:N", title="Shape"),
        color=alt.Color("percentage:Q", title="Percentage of sightings"),
    ).properties(
        title="Sightings by shape and decade",
        width=480,
        height=640
    )
    return


@app.cell
def _(mo):
    mo.md("""
    # Why did lights replace the classic disk-shaped UFOs?? 👀🤯⁉️ note to self: investigate‼️
    """)
    return


@app.cell
def _(alt, df_final, pl):
    shape_yearly = (
        df_final
        .group_by(["year", "shape"])
        .agg(pl.len().alias("count"))
        .with_columns(
            total=pl.col("count").sum().over("year")
        )
        .with_columns(
            percentage=100 * pl.col("count") / pl.col("total")
        )
        .filter(pl.col("shape").is_in(["Disk", "Light"]))
        .sort("year")
    )
    alt.Chart(shape_yearly).mark_line(point=True).encode(
        x=alt.X("year:Q", title="Year"),
        y=alt.Y("percentage:Q", title="Percentage of all sightings"),
        color="shape:N",
        tooltip=["year", "shape", "count", "percentage"]
    ).properties(
        title="Disk vs light sightings over time"
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## wtf
    """)
    return


@app.cell
def _(alt, df_final):
    alt.Chart(df_final).transform_aggregate(
        count="count()",
        groupby=["year"]
    ).mark_line().encode(
        x=alt.X("year", title="Year"),
        y=alt.Y("count:Q")
    ).properties(
        title="Sightings per year"
    )
    return


if __name__ == "__main__":
    app.run()
