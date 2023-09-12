from __future__ import annotations

import pytest

from ibis import udf


@udf.scalar.builtin
def hamming(a: str, b: str) -> int:
    ...


@udf.scalar.builtin
def jaccard(a: str, b: str) -> float:
    ...


@udf.scalar.builtin
def jaro_similarity(a: str, b: str) -> float:
    ...


@udf.scalar.builtin
def jaro_winkler_similarity(a: str, b: str) -> float:
    ...


@udf.scalar.builtin
def damerau_levenshtein(a: str, b: str) -> int:
    ...


@udf.scalar.builtin
def mismatches(a: str, b: str) -> int:
    ...


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

    with con.begin() as c:
        expected = c.exec_driver_sql(f"SELECT {func.__name__}({a!r}, {b!r})").scalar()

    assert con.execute(expr) == expected


@udf.agg.builtin
def product(x, where: bool = True) -> float:
    ...


@udf.agg.builtin
def fsum(x, where: bool = True) -> float:
    ...


@udf.agg.builtin
def favg(x: float, where: bool = True) -> float:
    ...


@pytest.mark.parametrize("func", [product, fsum, favg])
def test_builtin_agg(con, func):
    import ibis

    raw_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    data = ibis.memtable({"a": raw_data})
    expr = func(data.a)

    with con.begin() as c:
        expected = c.exec_driver_sql(
            f"SELECT {func.__name__}(a) FROM UNNEST({raw_data!r}) _ (a)"
        ).scalar()

    assert con.execute(expr) == expected
