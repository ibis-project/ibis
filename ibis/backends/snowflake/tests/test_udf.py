from __future__ import annotations

import pandas.testing as tm
import pytest
from pytest import param

import ibis.expr.datatypes as dt
from ibis import udf


@udf.scalar.builtin
def soundex(s: str) -> str:
    ...


@udf.scalar.builtin
def jarowinkler_similarity(a: str, b: str) -> float:
    ...


# TODO: allow multiple signatures
@udf.scalar.builtin(name="compress")
def compress_str(data: str, method: str) -> bytes:
    ...


@udf.scalar.builtin(name="compress")
def compress_bytes(data: bytes, method: str) -> bytes:
    ...


@pytest.mark.parametrize(
    ("func", "args"),
    [
        param(soundex, ("snow",), id="soundex"),
        param(jarowinkler_similarity, ("snow", "show"), id="jarowinkler_similarity"),
    ],
)
def test_builtin_scalar_udf(con, func, args):
    expr = func(*args)

    query = f"SELECT {func.__name__}({', '.join(map(repr, args))})"
    with con._safe_raw_sql(query) as cur:
        [(expected,)] = cur.fetchall()

    assert con.execute(expr) == expected


@pytest.mark.parametrize(
    ("func", "pyargs", "snowargs"),
    [
        param(compress_str, ("snow", "zstd"), ("'snow'", "'zstd'"), id="str"),
        param(compress_bytes, (b"snow", "zstd"), ("'snow'", "'zstd'"), id="bytes"),
    ],
)
def test_compress(con, func, pyargs, snowargs):
    expr = func(*pyargs)

    query = f"SELECT compress({', '.join(snowargs)})"
    with con._safe_raw_sql(query) as cur:
        [(expected,)] = cur.fetchall()

    assert con.execute(expr) == expected


@udf.agg.builtin
def minhash(x, y) -> dt.json:
    ...


@udf.agg.builtin
def approximate_jaccard_index(a) -> float:
    ...


def test_builtin_agg_udf(con):
    ft = con.tables.FUNCTIONAL_ALLTYPES.limit(2)
    ft = ft.select(mh=minhash(100, ft.string_col).over(group_by=ft.date_string_col))
    expr = ft.agg(aji=approximate_jaccard_index(ft.mh))

    result = expr.execute()
    query = """
    SELECT approximate_jaccard_index("mh") AS "aji"
    FROM (
        SELECT minhash(100, "string_col") OVER (PARTITION BY "date_string_col") AS "mh"
        FROM (
            SELECT * FROM "FUNCTIONAL_ALLTYPES" LIMIT 2
        )
    )
    """
    with con._safe_raw_sql(query) as cur:
        expected = cur.fetch_pandas_all()

    tm.assert_frame_equal(result, expected)


def test_xgboost_model(con):
    import ibis
    from ibis import _

    @udf.scalar.pandas(
        packages=("joblib", "xgboost"), imports=("@MODELS/model.joblib",)
    )
    def predict_price(
        carat_scaled: float, cut_encoded: int, color_encoded: int, clarity_encoded: int
    ) -> int:
        import sys

        import joblib
        import pandas as pd

        import_dir = sys._xoptions.get("snowflake_import_directory")
        model = joblib.load(f"{import_dir}model.joblib")
        df = pd.concat(
            [carat_scaled, cut_encoded, color_encoded, clarity_encoded], axis=1
        )
        df.columns = ["CARAT_SCALED", "CUT_ENCODED", "COLOR_ENCODED", "CLARITY_ENCODED"]
        return model.predict(df)

    def cases(value, mapping):
        """This should really be a top-level function or method."""
        expr = ibis.case()
        for k, v in mapping.items():
            expr = expr.when(value == k, v)
        return expr.end()

    diamonds = con.tables.DIAMONDS
    expr = diamonds.mutate(
        predicted_price=predict_price(
            (_.carat - _.carat.mean()) / _.carat.std(),
            cases(
                _.cut,
                {
                    c: i
                    for i, c in enumerate(
                        ("Fair", "Good", "Very Good", "Premium", "Ideal"), start=1
                    )
                },
            ),
            cases(_.color, {c: i for i, c in enumerate("DEFGHIJ", start=1)}),
            cases(
                _.clarity,
                {
                    c: i
                    for i, c in enumerate(
                        ("I1", "IF", "SI1", "SI2", "VS1", "VS2", "VVS1", "VVS2"),
                        start=1,
                    )
                },
            ),
        )
    )

    df = expr.execute()

    assert not df.empty
    assert "predicted_price" in df.columns
    assert len(df) == diamonds.count().execute()
