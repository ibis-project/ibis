from __future__ import annotations

import sys

import pytest

import ibis
import ibis.common.exceptions as com
from ibis import _
from ibis.backends.tests.errors import Py4JJavaError

tm = pytest.importorskip("pandas.testing")

# Concurrent execution of CREATE OR REPLACE FUNCTION in postgres fails
# This ensures that all tests in this module run in the same process as
# long as --dist=loadgroup is passed, which it is.
pytestmark = pytest.mark.xdist_group("impure")

no_randoms = [
    pytest.mark.notimpl(
        ["polars", "druid", "risingwave"], raises=com.OperationNotDefinedError
    ),
]

no_udfs = [
    pytest.mark.notyet(
        ["datafusion", "athena", "databricks"], raises=NotImplementedError
    ),
    pytest.mark.notimpl(
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
        ]
    ),
    pytest.mark.notyet(
        "flink",
        condition=sys.version_info >= (3, 11),
        raises=Py4JJavaError,
        reason="Docker image has Python 3.10, results in `cloudpickle` version mismatch",
    ),
]

no_uuids = [
    pytest.mark.notimpl(
        ["druid", "exasol", "oracle", "polars", "pyspark", "risingwave"],
        raises=com.OperationNotDefinedError,
    ),
    pytest.mark.notyet("mssql", reason="Unrelated bug: Incorrect syntax near '('"),
]


@ibis.udf.scalar.python(side_effects=True)
def my_random(x: float) -> float:  # noqa: ARG001
    # need to make the whole UDF self-contained for postgres to work
    import random

    return random.random()  # noqa: S311


mark_impures = pytest.mark.parametrize(
    "impure",
    [
        pytest.param(lambda _: ibis.random(), marks=no_randoms, id="random"),
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


# You can work around this by .cache()ing the table.
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
    expr = alltypes.select(common=impure(alltypes)).select(x=_.common, y=_.common)
    df = expr.execute()
    tm.assert_series_equal(df.x, df.y, check_names=False)


# You can work around this by .cache()ing the table.
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
                pytest.mark.notyet(["impala"], reason="instances are correlated"),
            ],
            id="random",
        ),
        pytest.param(
            # make this a float so we can compare to .5
            lambda _: ibis.uuid().cast(str).contains("a").ifelse(1, 0),
            marks=[
                *no_uuids,
                pytest.mark.notyet(
                    ["mysql"],
                    reason="instances are correlated; but sometimes this passes and it's not clear why",
                    strict=False,
                ),
            ],
            id="uuid",
        ),
        pytest.param(
            lambda table: my_random(table.float_col),
            marks=[
                *no_udfs,
                # no "impure" argument for pyspark and snowflake yet
                pytest.mark.notimpl(["pyspark", "snowflake"]),
            ],
            id="udf",
        ),
    ],
)


# You can work around this by doing .select().cache().select()
@pytest.mark.notyet(["clickhouse", "athena"], reason="instances are correlated")
@impure_params_uncorrelated
def test_impure_uncorrelated_different_id(alltypes, impure):
    # This is the opposite of test_impure_correlated.
    # If we evaluate an impure expression for two different rows in the same relation,
    # the should be uncorrelated.
    # eg if you look at the following SQL:
    # select random() as x, random() as y
    # Then x and y should be uncorrelated.
    expr = alltypes.select(x=impure(alltypes), y=impure(alltypes))
    df = expr.execute()
    assert (df.x != df.y).any()


# You can work around this by doing .select().cache().select()
@pytest.mark.notyet(["clickhouse", "athena"], reason="instances are correlated")
@impure_params_uncorrelated
def test_impure_uncorrelated_same_id(alltypes, impure):
    # Similar to test_impure_uncorrelated_different_id, but the two expressions
    # have the same ID. Still, they should be uncorrelated.
    common = impure(alltypes)
    expr = alltypes.select(x=common, y=common)
    df = expr.execute()
    assert (df.x != df.y).any()


@pytest.mark.notyet(
    [
        "duckdb",
        "clickhouse",
        "datafusion",
        "mysql",
        "impala",
        "mssql",
        "trino",
        "flink",
        "bigquery",
        "athena",
    ],
    raises=AssertionError,
    reason="instances are not correlated but ideally they would be",
)
@pytest.mark.notyet(
    ["sqlite"],
    raises=AssertionError,
    reason="instances are *sometimes* correlated but ideally they would always be",
    strict=False,
)
@pytest.mark.notimpl(
    ["polars", "risingwave", "druid", "exasol", "oracle", "pyspark"],
    raises=com.OperationNotDefinedError,
)
def test_self_join_with_generated_keys(con):
    # Even with CTEs in the generated SQL, the backends still
    # materialize a new value every time it is referenced.
    # This isn't ideal behavior, but there is nothing we can do about it
    # on the ibis side. The best you can do is to .cache() the table
    # right after you assign the uuid().
    # https://github.com/ibis-project/ibis/pull/9014#issuecomment-2399449665
    left = ibis.memtable({"idx": list(range(5))}).mutate(key=ibis.uuid())
    right = left.filter(left.idx < 3)
    expr = left.join(right, "key")
    result = con.execute(expr.count())
    assert result == 3
