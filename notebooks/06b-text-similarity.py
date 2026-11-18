import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import numpy as np
    import pathlib
    import scipy.sparse as sp
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import Normalizer

    DATA_DIR = pathlib.Path.cwd() / "data"
    return DATA_DIR, Normalizer, TfidfVectorizer, TruncatedSVD, mo, np, pl, sp


@app.cell
def _(mo):
    mo.md("""
    (TF-IDF + LSA)

    A wave is only a convincing "same object" candidate if the *descriptions* resemble
    each other too. Only classical methods are used here: TF-IDF, truncated SVD (LSA)
    and cosine similarity.

    For each wave we compute the mean pairwise cosine similarity of its reports and compare
    it to a **null distribution**: random sets of the same size drawn from the same state.
    The z-score tells how much more alike the wave's reports are than chance.
    """)
    return


@app.cell
def _():
    # Parameters

    LSA_COMPONENTS = 100
    N_NULL = 50          # random same-size sets per wave for the null distribution
    SEED = 333
    return LSA_COMPONENTS, N_NULL, SEED


@app.cell
def _(DATA_DIR, pl):
    sightings = pl.read_parquet(DATA_DIR / "sightings_waves.parquet")
    waves = pl.read_parquet(DATA_DIR / "wave_summary_explained.parquet")

    # Clean the free text: drop NUFORC editor notes like "((NUFORC Note: ...))" and the
    # sighting's own city name. Without the latter, neighbouring reports look "similar"
    # merely because they all mention the same place.
    _clean = (
        sightings["summary"].fill_null("")
        .str.replace_all(r"\(\(.*?\)\)", " ")
        .str.to_lowercase()
        .to_list()
    )
    _cities = (
        sightings["city"].str.to_lowercase().str.replace_all(r"\(.*?\)", "").str.strip_chars().to_list()
    )
    texts = [t.replace(c, " ") if c else t for t, c in zip(_clean, _cities)]
    print(f"{len(texts):,} texts; {waves.height:,} waves")
    return sightings, texts, waves


@app.cell
def _(LSA_COMPONENTS, Normalizer, SEED, TfidfVectorizer, TruncatedSVD, texts):
    # TF-IDF on all texts (stable IDF), then LSA to capture synonyms in short texts

    vectorizer = TfidfVectorizer(
        stop_words="english",
        min_df=5,
        max_df=0.5,
        sublinear_tf=True,
        ngram_range=(1, 2),
    )
    X_tfidf = vectorizer.fit_transform(texts)          # rows are L2-normalized
    terms = vectorizer.get_feature_names_out()

    _svd = TruncatedSVD(n_components=LSA_COMPONENTS, random_state=SEED)
    X_lsa = Normalizer(copy=False).fit_transform(_svd.fit_transform(X_tfidf))

    print("TF-IDF:", X_tfidf.shape, "| LSA:", X_lsa.shape,
          f"| explained variance: {_svd.explained_variance_ratio_.sum():.1%}")
    return X_lsa, X_tfidf, terms


@app.cell
def _(np):
    def mean_pairwise_cosine(M) -> float:
        """Mean pairwise cosine of L2-normalized rows (sparse or dense).

        With unit-length rows: sum_{i != j} cos = ||sum of rows||^2 - n.
        """
        n = M.shape[0]
        if n < 2:
            return float("nan")
        s = np.asarray(M.sum(axis=0)).ravel()
        return float((s @ s - n) / (n * (n - 1)))

    return (mean_pairwise_cosine,)


@app.cell
def _(
    N_NULL,
    SEED,
    X_lsa,
    X_tfidf,
    mean_pairwise_cosine,
    np,
    pl,
    sightings,
    waves,
):
    # Per-wave similarity with a same-state null distribution

    _rng = np.random.default_rng(SEED)
    _state = sightings["state"].to_numpy()
    _wave_col = sightings["wave"].to_numpy()

    # Row indices per state of sightings that are NOT in any wave (the null pool)
    _pool = {
        s: np.where((_state == s) & (_wave_col < 0))[0] for s in np.unique(_state)
    }

    def _z(obs, null):
        sd = np.std(null)
        return float((obs - np.mean(null)) / sd) if sd > 0 else float("nan")

    _rows = []
    for _w in waves["wave"].to_list():
        _idx = np.where(_wave_col == _w)[0]
        _k = len(_idx)
        _st = pl.Series(_state[_idx]).mode()[0]
        _p = _pool.get(_st, np.array([], dtype=int))
        if len(_p) < _k:                       # tiny state pool: fall back to all non-wave rows
            _p = np.where(_wave_col < 0)[0]

        _obs_t = mean_pairwise_cosine(X_tfidf[_idx])
        _obs_l = mean_pairwise_cosine(X_lsa[_idx])
        _null_t, _null_l = [], []
        for _ in range(N_NULL):
            _s = _rng.choice(_p, size=_k, replace=False)
            _null_t.append(mean_pairwise_cosine(X_tfidf[_s]))
            _null_l.append(mean_pairwise_cosine(X_lsa[_s]))

        _rows.append({
            "wave": _w,
            "sim_tfidf": _obs_t,
            "sim_tfidf_null": float(np.mean(_null_t)),
            "text_z_tfidf": _z(_obs_t, _null_t),
            "sim_lsa": _obs_l,
            "sim_lsa_null": float(np.mean(_null_l)),
            "text_z_lsa": _z(_obs_l, _null_l),
        })

    text_scores = pl.DataFrame(_rows)
    return (text_scores,)


@app.cell
def _(X_tfidf, np, pl, sightings, terms, waves):
    # Top terms per wave (mean TF-IDF weight of the wave's reports)

    _wave_col = sightings["wave"].to_numpy()
    _top = []
    for _w in waves["wave"].to_list():
        _m = np.asarray(X_tfidf[np.where(_wave_col == _w)[0]].mean(axis=0)).ravel()
        _top.append({"wave": _w, "top_terms": ", ".join(terms[np.argsort(_m)[::-1][:6]])})

    top_terms = pl.DataFrame(_top)
    return (top_terms,)


@app.cell
def _(pl, text_scores, top_terms, waves):
    # Combine into one scored table. The combined scores are heuristics:
    #   coherence   = how alike the reports are (shape agreement + text z-score, both in 0..1)
    #   mover_score = coherence * track_rho  (alike reports that also advance along a line)

    scored = (
        waves
        .join(text_scores, on="wave", how="left")
        .join(top_terms, on="wave", how="left")
        .with_columns(
            (
                0.5 * pl.col("shape_share")
                + 0.5 * (pl.col("text_z_lsa").fill_nan(None).fill_null(0) / 5).clip(0, 1)
            ).alias("coherence")
        )
        .with_columns(
            (pl.col("coherence") * pl.col("track_rho").fill_nan(None).fill_null(0)).alias("mover_score")
        )
        .sort("mover_score", descending=True)
    )
    return (scored,)


@app.cell
def _(mo, pl, scored):
    _big = scored.filter(pl.col("n") >= 8)
    mo.vstack([
        mo.md(f"""
        ## Result

        - Waves where the reports are more alike than chance (LSA z > 3): **{scored.filter(pl.col('text_z_lsa') > 3).height:,}** of {scored.height:,}
        - Median z (TF-IDF): **{scored['text_z_tfidf'].median():.2f}**, median z (LSA): **{scored['text_z_lsa'].median():.2f}**

        If the median z is clearly above 0, report texts inside waves are more similar than
        random same-state reports. Top candidates (waves with ≥ 8 reports):
        """),
        mo.ui.table(
            _big.select("wave", "n", "start", "end", "span_km", "top_shape", "shape_share",
                        "text_z_tfidf", "text_z_lsa", "track_rho", "coherence",
                        "mover_score", "explanation", "top_terms").head(30)
        ),
    ])
    return


@app.cell
def _(DATA_DIR, X_lsa, X_tfidf, np, scored, sp, terms):
    # Save everything the app needs; row i of the matrices = row i of sightings_waves.parquet

    scored.write_parquet(DATA_DIR / "wave_scores.parquet")
    sp.save_npz(DATA_DIR / "tfidf.npz", X_tfidf.tocsr())
    np.save(DATA_DIR / "lsa.npy", X_lsa.astype(np.float32))
    np.save(DATA_DIR / "terms.npy", terms.astype(str))
    print("Saved wave_scores.parquet, tfidf.npz, lsa.npy, terms.npy")
    return


if __name__ == "__main__":
    app.run()
