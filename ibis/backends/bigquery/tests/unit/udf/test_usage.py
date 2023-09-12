from __future__ import annotations

import pytest
from pytest import param

import ibis
import ibis.expr.datatypes as dt
from ibis.backends.bigquery import udf
from ibis.backends.bigquery.udf import _udf_name_cache


def test_multiple_calls_redefinition(snapshot):
    _udf_name_cache.clear()

    @udf.python([dt.string], dt.double)
    def my_len(s):
        return s.length

    s = ibis.literal("abcd")
    expr = my_len(s) + my_len(s)

    @udf.python([dt.string], dt.double)
    def my_len(s):
        return s.length + 1

    expr = expr + my_len(s)

    sql = ibis.bigquery.compile(expr)
    snapshot.assert_match(sql, "out.sql")


@pytest.mark.parametrize(
    ("determinism",),
    [
        param(True),
        param(False),
        param(None),
    ],
)
def test_udf_determinism(snapshot, determinism):
    _udf_name_cache.clear()

    @udf.python([dt.string], dt.double, determinism=determinism)
    def my_len(s):
        return s.length

    s = ibis.literal("abcd")
    expr = my_len(s)

    sql = ibis.bigquery.compile(expr)
    snapshot.assert_match(sql, "out.sql")


def test_udf_sql(snapshot):
    _udf_name_cache.clear()

    format_t = udf.sql(
        "format_t",
        params={"input": dt.string},
        output_type=dt.double,
        sql_expression="FORMAT('%T', input)",
    )

    s = ibis.literal("abcd")
    expr = format_t(s)

    sql = ibis.bigquery.compile(expr)
    snapshot.assert_match(sql, "out.sql")


@pytest.mark.parametrize(
    ("argument_type", "return_type"),
    [
        param(
            dt.int64,
            dt.float64,
            marks=pytest.mark.xfail(raises=TypeError),
            id="int_float",
        ),
        param(
            dt.float64,
            dt.int64,
            marks=pytest.mark.xfail(raises=TypeError),
            id="float_int",
        ),
        # complex argument type, valid return type
        param(
            dt.Array(dt.int64),
            dt.float64,
            marks=pytest.mark.xfail(raises=TypeError),
            id="array_int_float",
        ),
        # valid argument type, complex invalid return type
        param(
            dt.float64,
            dt.Array(dt.int64),
            marks=pytest.mark.xfail(raises=TypeError),
            id="float_array_int",
        ),
        # both invalid
        param(
            dt.Array(dt.Array(dt.int64)),
            dt.int64,
            marks=pytest.mark.xfail(raises=TypeError),
            id="array_array_int_int",
        ),
        # struct type with nested integer, valid return type
        param(
            dt.Struct.from_tuples([("x", dt.Array(dt.int64))]),
            dt.float64,
            marks=pytest.mark.xfail(raises=TypeError),
            id="struct",
        ),
    ],
)
def test_udf_int64(argument_type, return_type):
    # invalid argument type, valid return type
    @udf.python([argument_type], return_type)
    def my_int64_add(x):
        return 1.0
