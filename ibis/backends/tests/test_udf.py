from __future__ import annotations

import sys

from pytest import mark, param

import ibis.common.exceptions as com
from ibis import _, udf
from ibis.backends.tests.errors import Py4JJavaError, PySparkPythonException
from ibis.conftest import IS_SPARK_REMOTE

no_python_udfs = mark.notimpl(
    [
        "bigquery",
        "clickhouse",
        "druid",
        "exasol",
        "impala",
        "mssql",
        "mysql",
        "oracle",
        "trino",
        "risingwave",
        "databricks",
    ]
)
cloudpickle_version_mismatch = mark.notimpl(
    ["flink"],
    condition=sys.version_info >= (3, 11),
    raises=Py4JJavaError,
    reason="Docker image has Python 3.10, results in `cloudpickle` version mismatch",
)


@no_python_udfs
@cloudpickle_version_mismatch
@mark.notyet(["datafusion"], raises=NotImplementedError)
def test_udf(batting):
    @udf.scalar.python
    def num_vowels(s: str, include_y: bool = False) -> int:
        return sum(map(s.lower().count, "aeiou" + ("y" * include_y)))

    batting = batting.limit(100)
    nvowels = num_vowels(batting.playerID)
    assert nvowels.op().__module__ == __name__
    assert type(nvowels.op()).__qualname__.startswith("num_vowels")

    expr = batting.group_by(id_len=nvowels).agg(n=_.count())
    result = expr.execute()
    assert not result.empty

    expr = batting.group_by(id_len=num_vowels(batting.playerID, include_y=True)).agg(
        n=_.count()
    )
    result = expr.execute()
    assert not result.empty


@no_python_udfs
@cloudpickle_version_mismatch
@mark.notyet(
    ["postgres"], raises=TypeError, reason="postgres only supports map<string, string>"
)
@mark.notimpl(["polars"])
@mark.notyet(["datafusion"], raises=NotImplementedError)
@mark.notyet(
    ["sqlite"], raises=com.IbisTypeError, reason="sqlite doesn't support map types"
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
@cloudpickle_version_mismatch
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
@mark.notyet(
    ["pyspark"],
    condition=IS_SPARK_REMOTE,
    raises=PySparkPythonException,
    reason="remote udfs not yet tested due to environment complexities",
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
                cloudpickle_version_mismatch,
            ],
        ),
        param(
            add_one_pyarrow,
            marks=[
                mark.notyet(
                    ["snowflake", "sqlite", "flink"],
                    raises=NotImplementedError,
                    reason="backend doesn't support pyarrow UDFs",
                ),
                mark.xfail_version(pyspark=["pyspark<3.5"]),
            ],
        ),
    ],
)
def test_vectorized_udf(backend, batting, add_one):
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
    backend.assert_frame_equal(result, expected)
