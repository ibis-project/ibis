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
def test_builtin(con, func):
    a, b = "duck", "luck"
    expr = func(a, b)

    with con.begin() as c:
        expected = c.exec_driver_sql(f"SELECT {func.__name__}({a!r}, {b!r})").scalar()

    assert con.execute(expr) == expected
