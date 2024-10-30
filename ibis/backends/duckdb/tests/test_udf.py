from __future__ import annotations

import duckdb
import pytest
from pytest import param

import ibis
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

    start, stop = 1, 11
    raw_data = list(map(float, range(start, stop)))
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


def test_kwargs_are_forwarded(con):
    def nullify_two(x: int) -> int:
        return None if x == 2 else x

    @udf.scalar.python
    def no_kwargs(x: int) -> int:
        return nullify_two(x)

    @udf.scalar.python(null_handling="special")
    def with_kwargs(x: int) -> int:
        return nullify_two(x)

    # If we return go Non-NULL -> Non-NULL, then passing null_handling="special"
    # will not change the result
    assert con.execute(no_kwargs(ibis.literal(1))) == 1
    assert con.execute(with_kwargs(ibis.literal(1))) == 1

    # But, if our UDF ever goes Non-NULL -> NULL, then we NEED to pass
    # null_handling="special", otherwise duckdb throws an error
    assert con.execute(with_kwargs(ibis.literal(2))) is None

    expr = no_kwargs(ibis.literal(2))
    with pytest.raises(duckdb.InvalidInputException):
        con.execute(expr)


def test_builtin_udf_uses_dialect():
    # in raw sqlglot, if you call regexp_extract, it will assume the
    # 3rd arg is "position" and not "groups". So when we make the UDF,
    # we need to make sure that we pass the dialect when creating the
    # sqlglot.expressions.func() object
    @udf.scalar.builtin(
        signature=(
            ("string", "string", "array<string>"),
            "struct<y: string, m: string, d: str>",
        ),
    )
    def regexp_extract(s, pattern, groups): ...

    e = regexp_extract("2023-04-15", r"(\d+)-(\d+)-(\d+)", ["y", "m", "d"])
    sql = str(ibis.to_sql(e, dialect="duckdb"))
    assert r"REGEXP_EXTRACT('2023-04-15', '(\d+)-(\d+)-(\d+)', ['y', 'm', 'd'])" in sql
