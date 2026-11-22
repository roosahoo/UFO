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

    mo.ui.table(df_not_null)
    return (df_not_null,)


@app.cell
def _(df_not_null, mo, pl):
    # New dataframe starting from the year 1960

    df_1960 = df_not_null.filter(
        pl.col("year") >= 1960
    )

    mo.ui.table(df_1960)
    return (df_1960,)


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

    mo.ui.table(df_semifinal)
    return (df_semifinal,)


@app.cell
def _(df_semifinal, pl):
    # Remove events with invalid hours/minutes

    df_final = df_semifinal.filter(
        pl.col("hour").is_between(0,23) &
        pl.col("minute").is_between(0, 59)
    )
    return (df_final,)


@app.cell
def _(DATA_CLEAN_PATH, df_final):
    # Write the new dataframe to a CSV

    df_final.write_csv(DATA_CLEAN_PATH, separator=",")
    return


if __name__ == "__main__":
    app.run()
