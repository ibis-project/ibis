from __future__ import annotations

import sys

import pandas.testing as tm
import pytest

import ibis
import ibis.common.exceptions as com
from ibis import _
from ibis.backends.tests.errors import (
    PsycoPg2InternalError,
    Py4JJavaError,
    PyDruidProgrammingError,
)

no_randoms = [
    pytest.mark.notimpl(
        ["dask", "pandas", "polars"], raises=com.OperationNotDefinedError
    ),
    pytest.mark.notimpl("druid", raises=PyDruidProgrammingError),
    pytest.mark.notyet(
        "risingwave",
        raises=PsycoPg2InternalError,
        reason="function random() does not exist",
    ),
]

no_udfs = [
    pytest.mark.notyet("datafusion", raises=NotImplementedError),
    pytest.mark.notimpl(
        [
            "bigquery",
            "clickhouse",
            "dask",
            "druid",
            "exasol",
            "impala",
            "mssql",
            "mysql",
            "oracle",
            "pandas",
            "trino",
            "risingwave",
        ]
    ),
    pytest.mark.notimpl("pyspark", reason="only supports pandas UDFs"),
    pytest.mark.notyet(
        "flink",
        condition=sys.version_info >= (3, 11),
        raises=Py4JJavaError,
        reason="Docker image has Python 3.10, results in `cloudpickle` version mismatch",
    ),
]

no_uuids = [
    pytest.mark.notimpl(
        [
            "druid",
            "exasol",
            "oracle",
            "polars",
            "pyspark",
            "risingwave",
            "pandas",
            "dask",
        ],
        raises=com.OperationNotDefinedError,
    ),
    pytest.mark.notyet("mssql", reason="Unrelated bug: Incorrect syntax near '('"),
]


@ibis.udf.scalar.python(side_effects=True)
def my_random(x: float) -> float:
    # need to make the whole UDF self-contained for postgres to work
    import random

    return random.random()  # noqa: S311


mark_impures = pytest.mark.parametrize(
    "impure",
    [
        pytest.param(
            lambda _: ibis.random(),
            marks=no_randoms,
            id="random",
        ),
        pytest.param(
            lambda _: ibis.uuid().cast(str).contains("a").ifelse(1, 0),
            marks=[
                *no_uuids,
                pytest.mark.notyet("impala", reason="instances are uncorrelated"),
            ],
            id="uuid",
        ),
        pytest.param(
            lambda table: my_random(table.float_col),
            marks=[
                *no_udfs,
                pytest.mark.notyet(["flink"], reason="instances are uncorrelated"),
            ],
            id="udf",
        ),
    ],
)


@pytest.mark.notyet("sqlite", reason="instances are uncorrelated")
@mark_impures
def test_impure_correlated(alltypes, impure):
    # An "impure" expression is random(), uuid(), or some other non-deterministic UDF.
    # If we evaluate it for two different rows in the same relation,
    # we might get different results. This is expected.
    # But, as soon as we .select() it into a new relation, then that "locks in" the
    # value, and any further references to it will be the same.
    # eg if you look at the following SQL:
    # WITH
    #   t AS (SELECT random() AS common)
    # SELECT common as x, common as y FROM t
    # Then both x and y should have the same value.
    df = (
        alltypes.select(common=impure(alltypes))
        .select(x=_.common, y=_.common)
        .execute()
    )
    tm.assert_series_equal(df.x, df.y, check_names=False)


@pytest.mark.notyet("sqlite", reason="instances are uncorrelated")
@mark_impures
def test_chained_selections(alltypes, impure):
    # https://github.com/ibis-project/ibis/issues/8921#issue-2234327722
    # This is a slightly more complex version of test_impure_correlated.
    # consider this SQL:
    # WITH
    #   t AS (SELECT random() AS num)
    # SELECT num, num > 0.5 AS isbig FROM t
    # We would expect that the value of num and isbig are consistent,
    # since we "lock in" the value of num by selecting it into t.
    t = alltypes.select(num=impure(alltypes))
    t = t.mutate(isbig=(t.num > 0.5))
    df = t.execute()
    df["expected"] = df.num > 0.5
    tm.assert_series_equal(df.isbig, df.expected, check_names=False)


impure_params_uncorrelated = pytest.mark.parametrize(
    "impure",
    [
        pytest.param(
            lambda _: ibis.random(),
            marks=[
                *no_randoms,
                pytest.mark.notyet(
                    ["impala", "trino"], reason="instances are correlated"
                ),
            ],
            id="random",
        ),
        pytest.param(
            # make this a float so we can compare to .5
            lambda _: ibis.uuid().cast(str).contains("a").ifelse(1, 0),
            marks=[
                *no_uuids,
                pytest.mark.notyet(
                    ["mysql", "trino"], reason="instances are correlated"
                ),
            ],
            id="uuid",
        ),
        pytest.param(
            lambda table: my_random(table.float_col),
            marks=[
                *no_udfs,
                pytest.mark.notyet("duckdb", reason="instances are correlated"),
            ],
            id="udf",
        ),
    ],
)


@pytest.mark.notyet(["clickhouse"], reason="instances are correlated")
@impure_params_uncorrelated
def test_impure_uncorrelated_different_id(alltypes, impure):
    # This is the opposite of test_impure_correlated.
    # If we evaluate an impure expression for two different rows in the same relation,
    # the should be uncorrelated.
    # eg if you look at the following SQL:
    # select random() as x, random() as y
    # Then x and y should be uncorrelated.
    df = alltypes.select(x=impure(alltypes), y=impure(alltypes)).execute()
    assert (df.x != df.y).any()


@pytest.mark.notyet(["clickhouse"], reason="instances are correlated")
@impure_params_uncorrelated
def test_impure_uncorrelated_same_id(alltypes, impure):
    # Similar to test_impure_uncorrelated_different_id, but the two expressions
    # have the same ID. Still, they should be uncorrelated.
    common = impure(alltypes)
    df = alltypes.select(x=common, y=common).execute()
    assert (df.x != df.y).any()
