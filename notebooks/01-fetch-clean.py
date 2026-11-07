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
def _():
    # Fetch the NUFORC dataset

    import pathlib
    from urllib.request import urlretrieve

    DATASET_URL = "https://raw.githubusercontent.com/LinkWentz/nuforc-dataset/refs/heads/master/nuforc_events_complete.csv"

    ROOT_DIR = pathlib.Path.cwd()
    DATA_DIR = pathlib.Path.cwd() / "data"
    DATA_RAW_PATH = DATA_DIR / "data_raw.csv"
    DATA_CLEAN_PATH = DATA_DIR / "data_clean.csv"

    if not DATA_RAW_PATH.exists():
        urlretrieve(DATASET_URL, DATA_RAW_PATH)
        print("Success 👽")
    else:
        print("Target already located 🛸")
    return (DATA_CLEAN_PATH,)


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

    df_final = df_not_null.filter(
        pl.col("year") >= 1960
    )

    mo.ui.table(df_final)
    return (df_final,)


@app.cell
def _(DATA_CLEAN_PATH, df_final):
    # Write the new dataframe to a CSV

    df_final.write_csv(DATA_CLEAN_PATH, separator=",")
    return


if __name__ == "__main__":
    app.run()
