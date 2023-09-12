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
    with con.begin() as c:
        expected = c.exec_driver_sql(query).scalar()

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
    with con.begin() as c:
        expected = c.exec_driver_sql(query).scalar()

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
    with con.begin() as c:
        expected = c.exec_driver_sql(query).cursor.fetch_pandas_all()

    tm.assert_frame_equal(result, expected)
