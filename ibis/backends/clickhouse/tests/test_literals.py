from __future__ import annotations

import pandas.testing as tm
import pytest
from pandas import Timestamp
from pytest import param

import ibis

pytest.importorskip("clickhouse_connect")


@pytest.mark.parametrize(
    "expr",
    [
        ibis.literal(Timestamp("2015-01-01 12:34:56")),
        ibis.literal(Timestamp("2015-01-01 12:34:56").to_pydatetime()),
        ibis.timestamp("2015-01-01 12:34:56"),
    ],
)
def test_timestamp_literals(con, expr, assert_sql):
    assert_sql(expr)
    assert con.execute(expr) == Timestamp("2015-01-01 12:34:56")


@pytest.mark.parametrize(
    "expr",
    [
        param(ibis.timestamp("2015-01-01 12:34:56.789"), id="millis"),
        param(ibis.timestamp("2015-01-01 12:34:56.789321"), id="micros"),
        param(ibis.timestamp("2015-01-01 12:34:56.789 UTC"), id="millis_tz"),
        param(ibis.timestamp("2015-01-01 12:34:56.789321 UTC"), id="micros_tz"),
    ],
)
def test_fine_grained_timestamp_literals(con, expr, assert_sql):
    assert_sql(expr)
    assert con.execute(expr) == expr.op().value


@pytest.mark.parametrize(
    "value",
    [
        param("simple", id="simple"),
        param("I can't", id="nested_quote"),
        param('An "escape"', id="nested_token"),
        param(5, id="int"),
        param(1.5, id="float"),
        param(True, id="true"),
        param(False, id="false"),
    ],
)
def test_string_numeric_boolean_literals(con, value, assert_sql):
    expr = ibis.literal(value)
    assert_sql(expr)
    assert con.execute(expr) == value


def test_array_params(con):
    t = con.tables.functional_alltypes
    param = ibis.param("int")
    expr = ibis.array([param + t.bigint_col])[0].name("result")
    result = con.execute(expr, params={param: 1})
    tm.assert_series_equal(result, (t.bigint_col + 1).name("result").execute())
