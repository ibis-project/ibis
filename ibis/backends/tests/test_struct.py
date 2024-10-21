from __future__ import annotations

import contextlib
from collections.abc import Mapping

import pytest
from pytest import param

import ibis
import ibis.expr.datatypes as dt
from ibis import util
from ibis.backends.tests.errors import (
    DatabricksServerOperationError,
    PolarsColumnNotFoundError,
    PsycoPg2InternalError,
    PsycoPg2SyntaxError,
    Py4JJavaError,
    PySparkAnalysisException,
)
from ibis.common.exceptions import IbisError

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
tm = pytest.importorskip("pandas.testing")

pytestmark = [
    pytest.mark.never(["mysql", "sqlite", "mssql"], reason="No struct support"),
    pytest.mark.notyet(["impala"]),
    pytest.mark.notimpl(["druid", "oracle", "exasol"]),
]


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        param(
            "a",
            [1.0, 2.0, 2.0, 3.0, 3.0, np.nan, np.nan],
            id="a",
            marks=pytest.mark.notimpl(["snowflake"]),
        ),
        param(
            "b", ["apple", "banana", "banana", "orange", "orange", None, None], id="b"
        ),
        param(
            "c",
            [2, 2, 3, 3, 4, np.nan, np.nan],
            id="c",
            marks=pytest.mark.notimpl(["snowflake"]),
        ),
    ],
)
def test_single_field(struct, field, expected):
    expr = struct.select(field=lambda t: t.abc[field]).order_by("field")
    result = expr.execute()
    tm.assert_series_equal(result.field, pd.Series(expected, name="field"))


def test_all_fields(struct, struct_df):
    result = struct.abc.execute()
    expected = struct_df.abc

    assert {
        row if not isinstance(row, Mapping) else tuple(row.items()) for row in result
    } == {
        row if not isinstance(row, Mapping) else tuple(row.items()) for row in expected
    }


_SIMPLE_DICT = dict(a=1, b="2", c=3.0)
_STRUCT_LITERAL = ibis.struct(
    _SIMPLE_DICT,
    type="struct<a: int64, b: string, c: float64>",
)
_NULL_STRUCT_LITERAL = ibis.null().cast("struct<a: int64, b: string, c: float64>")


@pytest.mark.notimpl(["postgres", "risingwave"])
@pytest.mark.notyet(["datafusion"], raises=Exception, reason="unsupported syntax")
@pytest.mark.parametrize("field", ["a", "b", "c"])
def test_literal(backend, con, field):
    query = _STRUCT_LITERAL[field]
    dtype = query.type().to_pandas()
    result = pd.Series([con.execute(query)], dtype=dtype)
    result = result.replace({np.nan: None})
    expected = pd.Series([_SIMPLE_DICT[field]])
    backend.assert_series_equal(result, expected.astype(dtype))


@pytest.mark.notimpl(["postgres"])
@pytest.mark.notyet(["datafusion"], raises=Exception, reason="unsupported syntax")
@pytest.mark.parametrize("field", ["a", "b", "c"])
@pytest.mark.notyet(
    ["clickhouse"], reason="clickhouse doesn't support nullable nested types"
)
def test_null_literal(backend, con, field):
    query = _NULL_STRUCT_LITERAL[field]
    result = pd.Series([con.execute(query)])
    result = result.replace({np.nan: None})
    expected = pd.Series([None], dtype="object")
    backend.assert_series_equal(result, expected)


@pytest.mark.notimpl(["postgres", "risingwave"])
def test_struct_column(alltypes, df):
    t = alltypes
    expr = t.select(s=ibis.struct(dict(a=t.string_col, b=1, c=t.bigint_col)))
    assert expr.s.type() == dt.Struct(dict(a=dt.string, b=dt.int8, c=dt.int64))
    result = expr.execute()
    expected = pd.DataFrame(
        {"s": [dict(a=a, b=1, c=c) for a, c in zip(df.string_col, df.bigint_col)]}
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.notimpl(["postgres", "risingwave", "polars"])
def test_collect_into_struct(alltypes):
    from ibis import _

    t = alltypes
    expr = (
        t.filter(_.string_col.isin(("0", "1")))
        .group_by(group="string_col")
        .agg(
            val=lambda t: ibis.struct(
                dict(key=t.bigint_col.collect().cast("!array<int64>"))
            )
        )
    )
    result = expr.execute()
    assert result.shape == (2, 2)
    assert set(result.group) == {"0", "1"}
    val = result.val
    assert len(val.loc[result.group == "0"].iat[0]["key"]) == 730
    assert len(val.loc[result.group == "1"].iat[0]["key"]) == 730


@pytest.mark.notimpl(
    ["postgres"], reason="struct literals not implemented", raises=PsycoPg2SyntaxError
)
@pytest.mark.notimpl(
    ["risingwave"],
    reason="struct literals not implemented",
    raises=PsycoPg2InternalError,
)
@pytest.mark.notyet(["datafusion"], raises=Exception, reason="unsupported syntax")
@pytest.mark.notimpl(["flink"], raises=Py4JJavaError, reason="not implemented in ibis")
def test_field_access_after_case(con):
    s = ibis.struct({"a": 3})
    x = ibis.cases((True, s), else_=ibis.struct({"a": 4}))
    y = x.a
    assert con.to_pandas(y) == 3


@pytest.mark.notimpl(
    ["postgres"], reason="struct literals not implemented", raises=PsycoPg2SyntaxError
)
@pytest.mark.notimpl(["flink"], raises=IbisError, reason="not implemented in ibis")
@pytest.mark.parametrize(
    "nullable",
    [
        param(
            True,
            marks=pytest.mark.notyet(
                ["clickhouse"],
                raises=AssertionError,
                reason="clickhouse doesn't allow nullable nested types",
            ),
            id="nullable",
        ),
        param(
            False,
            marks=[
                pytest.mark.notyet(
                    ["polars"],
                    raises=AssertionError,
                    reason="polars doesn't support non-nullable types",
                ),
                pytest.mark.notyet(
                    ["risingwave"],
                    reason="non-nullable struct types not implemented",
                    raises=PsycoPg2InternalError,
                ),
                pytest.mark.notimpl(
                    ["pyspark"],
                    raises=AssertionError,
                    reason="non-nullable struct types not yet implemented in Ibis's PySpark backend",
                ),
            ],
            id="non-nullable",
        ),
    ],
)
@pytest.mark.notyet(
    ["trino"],
    raises=AssertionError,
    reason="trino returns unquoted and therefore unparsable struct field names, we fall back to dt.unknown",
)
@pytest.mark.notyet(
    ["snowflake"],
    raises=AssertionError,
    reason="snowflake doesn't have strongly typed structs",
)
@pytest.mark.notyet(["datafusion"], raises=Exception, reason="unsupported syntax")
@pytest.mark.notyet(
    ["databricks"],
    raises=DatabricksServerOperationError,
    reason="spaces are not allowed in column names",
)
def test_keyword_fields(con, nullable):
    schema = ibis.schema(
        {
            "struct": dt.Struct(
                {
                    "select": "int",
                    "from": "int",
                    "where": "int",
                    "order": "int",
                    "left join": "int",
                    "full outer join": "int",
                },
                nullable=nullable,
            )
        }
    )

    name = util.gen_name("struct_keywords")
    t = con.create_table(name, schema=schema)

    try:
        assert t.schema() == schema
        assert t.count().execute() == 0
    finally:
        with contextlib.suppress(NotImplementedError):
            con.drop_table(name, force=True)


@pytest.mark.notyet(
    ["postgres"],
    raises=PsycoPg2SyntaxError,
    reason="sqlglot doesn't implement structs for postgres correctly",
)
@pytest.mark.notyet(
    ["risingwave"],
    raises=PsycoPg2InternalError,
    reason="sqlglot doesn't implement structs for postgres correctly",
)
@pytest.mark.notyet(
    ["polars"],
    raises=PolarsColumnNotFoundError,
    reason="doesn't seem to support IN-style subqueries on structs",
)
@pytest.mark.xfail_version(
    pyspark=["pyspark<3.5"],
    reason="requires pyspark 3.5",
    raises=PySparkAnalysisException,
)
@pytest.mark.notimpl(
    ["flink"],
    raises=Py4JJavaError,
    reason="fails to parse due to an unsupported operation; flink docs say the syntax is supported",
)
@pytest.mark.notyet(["datafusion"], raises=Exception, reason="unsupported syntax")
def test_isin_struct(con):
    needle1 = ibis.struct({"x": 1, "y": 2})
    needle2 = ibis.struct({"x": 2, "y": 3})
    haystack_t = ibis.memtable({"xs": [1, 2, 3], "ys": [2, 3, 4]})
    haystack = ibis.struct({"x": haystack_t.xs, "y": haystack_t.ys})
    both = needle1.isin(haystack) | needle2.isin(haystack)
    result = con.execute(both)
    # TODO(cpcloud): ensure the type is consistent
    assert result is True or result is np.bool_(True)
