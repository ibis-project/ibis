from __future__ import annotations

import contextlib
from collections.abc import Mapping

import pytest
from pytest import param

import ibis
import ibis.expr.datatypes as dt
from ibis import util
from ibis.backends.tests.conftest import NO_STRUCT_SUPPORT_MARKS
from ibis.backends.tests.errors import (
    DatabricksServerOperationError,
    PsycoPg2InternalError,
    PsycoPg2ProgrammingError,
    PsycoPgSyntaxError,
    Py4JJavaError,
    PyAthenaDatabaseError,
    PyAthenaOperationalError,
    PySparkAnalysisException,
)
from ibis.common.exceptions import IbisError, UnsupportedOperationError

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
tm = pytest.importorskip("pandas.testing")

pytestmark = NO_STRUCT_SUPPORT_MARKS


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        param("a", [1.0, 2.0, 2.0, 3.0, 3.0, np.nan, np.nan], id="a"),
        param(
            "b", ["apple", "banana", "banana", "orange", "orange", None, None], id="b"
        ),
        param("c", [2, 2, 3, 3, 4, np.nan, np.nan], id="c"),
    ],
)
def test_single_field(struct, field, expected):
    expr = struct.select(field=lambda t: t.abc[field]).order_by("field")
    result = expr.execute()
    tm.assert_series_equal(
        result.field.replace(np.nan, None),
        pd.Series(expected, name="field").replace(np.nan, None),
        check_dtype=False,
    )


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
def test_literal(backend, con):
    dtype = _STRUCT_LITERAL.type().to_pandas()
    result = pd.Series([con.execute(_STRUCT_LITERAL)], dtype=dtype)
    result = result.replace({np.nan: None})
    expected = pd.Series([_SIMPLE_DICT])
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


@pytest.mark.notimpl(["postgres", "risingwave"])
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
    ["postgres"], reason="struct literals not implemented", raises=PsycoPgSyntaxError
)
@pytest.mark.notimpl(
    ["risingwave"],
    reason="struct literals not implemented",
    raises=PsycoPg2InternalError,
)
@pytest.mark.notimpl(["flink"], raises=Py4JJavaError, reason="not implemented in ibis")
def test_field_access_after_case(con):
    s = ibis.struct({"a": 3})
    x = ibis.cases((True, s), else_=ibis.struct({"a": 4}))
    y = x.a
    assert con.to_pandas(y) == 3


@pytest.mark.notimpl(
    ["postgres"], reason="struct literals not implemented", raises=PsycoPgSyntaxError
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
    ["athena"], raises=(AssertionError, PyAthenaOperationalError, PyAthenaDatabaseError)
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
    raises=PsycoPgSyntaxError,
    reason="sqlglot doesn't implement structs for postgres correctly",
)
@pytest.mark.notyet(
    ["risingwave"],
    raises=PsycoPg2InternalError,
    reason="sqlglot doesn't implement structs for postgres correctly",
)
@pytest.mark.notyet(
    ["polars"],
    raises=UnsupportedOperationError,
    reason="doesn't support IN-style subqueries on structs",
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


@pytest.mark.notyet(
    ["risingwave"],
    raises=PsycoPg2ProgrammingError,
    reason="can't adapt type for insert query",
)
@pytest.mark.notyet(
    ["postgres"],
    raises=PsycoPgSyntaxError,
    reason="sqlglot doesn't implement structs for postgres correctly",
)
def test_field_overwrite_always_prefers_unpacked(backend):
    t = backend.struct.filter(lambda t: t.abc.a.notnull())
    t1 = t.mutate(a=1)
    t2 = t1.select("a", "abc")
    t3 = t1.select("abc", "a")
    t4 = t2.select("abc", "a")

    expr1 = t1.unpack("abc")
    expr2 = t2.unpack("abc")
    expr3 = t3.unpack("abc")
    expr4 = t4.unpack("abc")

    expected = {1, 2, 3}

    assert expr1.columns == ("a", "b", "c")
    assert expr2.columns == ("a", "b", "c")
    assert expr3.columns == ("a", "b", "c")
    assert expr4.columns == ("a", "b", "c")

    assert set(expr1.execute().a) == expected
    assert set(expr2.execute().a) == expected
    assert set(expr3.execute().a) == expected
    assert set(expr4.execute().a) == expected
