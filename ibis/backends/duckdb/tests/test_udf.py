from __future__ import annotations

import pytest
from pytest import param

from ibis import udf


@udf.scalar.builtin
def hamming(a: str, b: str) -> int: ...


@udf.scalar.builtin
def jaccard(a: str, b: str) -> float: ...


@udf.scalar.builtin
def jaro_similarity(a: str, b: str) -> float: ...


@udf.scalar.builtin
def jaro_winkler_similarity(a: str, b: str) -> float: ...


@udf.scalar.builtin
def damerau_levenshtein(a: str, b: str) -> int: ...


@udf.scalar.builtin
def mismatches(a: str, b: str) -> int: ...


@pytest.mark.parametrize(
    "func",
    [
        hamming,
        jaccard,
        jaro_similarity,
        jaro_winkler_similarity,
        damerau_levenshtein,
        mismatches,
    ],
)
def test_builtin_scalar(con, func):
    a, b = "duck", "luck"
    expr = func(a, b)

    expected = con.raw_sql(f"SELECT {func.__name__}({a!r}, {b!r})").df().squeeze()
    assert con.execute(expr) == expected


def test_builtin_scalar_noargs(con):
    @udf.scalar.builtin
    def version() -> str: ...

    expr = version()
    assert con.execute(expr) == f"v{con.version}"


@udf.agg.builtin
def product(x, where: bool = True) -> float: ...


@udf.agg.builtin
def fsum(x, where: bool = True) -> float: ...


@udf.agg.builtin
def favg(x: float, where: bool = True) -> float: ...


@pytest.mark.parametrize("func", [product, fsum, favg])
def test_builtin_agg(con, func):
    import ibis

    raw_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    data = ibis.memtable({"a": raw_data})
    expr = func(data.a)

    expected = (
        con.raw_sql(f"SELECT {func.__name__}(a) FROM UNNEST({raw_data!r}) _ (a)")
        .df()
        .squeeze()
    )

    assert con.execute(expr) == expected


@udf.scalar.python
def dont_intercept_null(x: int) -> int:
    assert x is not None
    return x


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        param(dont_intercept_null(5), 5, id="notnull"),
        param(dont_intercept_null(None), None, id="null"),
        param(dont_intercept_null(5) + dont_intercept_null(None), None, id="mixed"),
    ],
)
def test_dont_intercept_null(con, expr, expected):
    assert con.execute(expr) == expected
