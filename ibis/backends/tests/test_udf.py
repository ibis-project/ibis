from __future__ import annotations

import pandas.testing as tm
import sqlalchemy as sa
from pytest import mark, param

from ibis import _, udf

no_python_udfs = mark.notimpl(
    [
        "bigquery",
        "clickhouse",
        "dask",
        "druid",
        "impala",
        "mssql",
        "mysql",
        "oracle",
        "pandas",
        "trino",
    ]
)


@no_python_udfs
@mark.notimpl(["pyspark"])
@mark.notyet(["datafusion"], raises=NotImplementedError)
def test_udf(batting):
    @udf.scalar.python
    def num_vowels(s: str, include_y: bool = False) -> int:
        return sum(map(s.lower().count, "aeiou" + ("y" * include_y)))

    batting = batting.limit(100)
    nvowels = num_vowels(batting.playerID)
    assert nvowels.op().__module__ == __name__
    assert type(nvowels.op()).__qualname__ == "num_vowels"

    expr = batting.group_by(id_len=nvowels).agg(n=_.count())
    result = expr.execute()
    assert not result.empty

    expr = batting.group_by(id_len=num_vowels(batting.playerID, include_y=True)).agg(
        n=_.count()
    )
    result = expr.execute()
    assert not result.empty


@no_python_udfs
@mark.notimpl(["pyspark"])
@mark.notyet(
    ["postgres"], raises=TypeError, reason="postgres only supports map<string, string>"
)
@mark.notimpl(["polars"])
@mark.notyet(["datafusion"], raises=NotImplementedError)
@mark.notyet(
    ["sqlite"],
    raises=sa.exc.OperationalError,
    reason="sqlite doesn't support map types",
)
def test_map_udf(batting):
    @udf.scalar.python
    def num_vowels_map(s: str, include_y: bool = False) -> dict[str, int]:
        y = "y" * include_y
        vowels = "aeiou" + y
        counter = dict.fromkeys(vowels, 0)
        for c in s:
            if c in vowels:
                counter[c] += 1

        return counter

    batting = batting.limit(100)

    expr = batting.select(vowel_dist=num_vowels_map(batting.playerID))
    df = expr.execute()
    assert not df.empty


@no_python_udfs
@mark.notimpl(["pyspark"])
@mark.notyet(
    ["postgres"], raises=TypeError, reason="postgres only supports map<string, string>"
)
@mark.notimpl(["polars"])
@mark.notyet(["datafusion"], raises=NotImplementedError)
@mark.notyet(["sqlite"], raises=TypeError, reason="sqlite doesn't support map types")
def test_map_merge_udf(batting):
    @udf.scalar.python
    def vowels_map(s: str) -> dict[str, int]:
        vowels = "aeiou"
        counter = dict.fromkeys(vowels, 0)
        for c in s:
            if c in vowels:
                counter[c] += 1

        return counter

    @udf.scalar.python
    def consonants_map(s: str) -> dict[str, int]:
        import string

        letters = frozenset(string.ascii_lowercase)
        consonants = letters - frozenset("aeiou")
        counter = dict.fromkeys(consonants, 0)

        for c in s:
            if c in consonants:
                counter[c] += 1

        return counter

    @udf.scalar.python
    def map_merge(x: dict[str, int], y: dict[str, int]) -> dict[str, int]:
        z = x.copy()
        z.update(y)
        return z

    batting = batting.limit(100)

    expr = batting.select(
        vowel_dist=map_merge(
            vowels_map(batting.playerID), consonants_map(batting.playerID)
        )
    )
    df = expr.execute()
    assert not df.empty


@udf.scalar.pandas
def add_one_pandas(s: int) -> int:  # s is series, int is the element type
    return s + 1


@udf.scalar.pyarrow
def add_one_pyarrow(s: int) -> int:  # s is series, int is the element type
    import pyarrow.compute as pac

    return pac.add(s, 1)


@no_python_udfs
@mark.notyet(
    ["postgres"],
    raises=NotImplementedError,
    reason="postgres only supports Python-native UDFs",
)
@mark.parametrize(
    "add_one",
    [
        param(
            add_one_pandas,
            marks=[
                mark.notyet(
                    ["duckdb", "datafusion", "polars", "sqlite"],
                    raises=NotImplementedError,
                    reason="backend doesn't support pandas UDFs",
                ),
            ],
        ),
        param(
            add_one_pyarrow,
            marks=[
                mark.notyet(
                    ["snowflake", "sqlite", "pyspark"],
                    raises=NotImplementedError,
                    reason="backend doesn't support pyarrow UDFs",
                )
            ],
        ),
    ],
)
def test_vectorized_udf(batting, add_one):
    batting = batting.limit(100)

    expr = (
        batting.select(year_id=lambda t: t.yearID)
        .mutate(next_year=lambda t: add_one(t.year_id))
        .order_by("year_id")
    )
    result = expr.execute()
    expected = (
        batting.select(year_id=lambda t: t.yearID)
        .execute()
        .assign(next_year=lambda df: df.year_id + 1)
        .sort_values(["year_id"])
        .reset_index(drop=True)
    )
    tm.assert_frame_equal(result, expected)
