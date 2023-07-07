from __future__ import annotations

import pytest
from pandas import Timestamp
from pytest import param

import ibis

pytest.importorskip("clickhouse_connect")


@pytest.mark.parametrize(
    'expr',
    [
        ibis.literal(Timestamp('2015-01-01 12:34:56')),
        ibis.literal(Timestamp('2015-01-01 12:34:56').to_pydatetime()),
        ibis.timestamp('2015-01-01 12:34:56'),
    ],
)
def test_timestamp_literals(con, translate, expr):
    expected = "toDateTime('2015-01-01T12:34:56')"

    assert translate(expr.op()) == expected
    assert con.execute(expr) == Timestamp('2015-01-01 12:34:56')


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        param(
            ibis.timestamp('2015-01-01 12:34:56.789'),
            "toDateTime64('2015-01-01T12:34:56.789000', 3)",
            id="millis",
        ),
        param(
            ibis.timestamp('2015-01-01 12:34:56.789321'),
            "toDateTime64('2015-01-01T12:34:56.789321', 6)",
            id="micros",
        ),
        param(
            ibis.timestamp('2015-01-01 12:34:56.789 UTC'),
            "toDateTime64('2015-01-01T12:34:56.789000', 3, 'UTC')",
            id="millis_tz",
        ),
        param(
            ibis.timestamp('2015-01-01 12:34:56.789321 UTC'),
            "toDateTime64('2015-01-01T12:34:56.789321', 6, 'UTC')",
            id="micros_tz",
        ),
    ],
)
def test_subsecond_timestamp_literals(con, translate, expr, expected):
    assert translate(expr.op()) == expected
    assert con.execute(expr) == expr.op().value


@pytest.mark.parametrize(
    "value",
    [
        param("simple", id="simple"),
        param("I can't", id="nested_quote"),
        param('An "escape"', id="nested_token"),
    ],
)
def test_string_literals(con, translate, value, snapshot):
    expr = ibis.literal(value)
    result = translate(expr.op())
    snapshot.assert_match(result, "out.sql")
    assert con.execute(expr) == value


@pytest.mark.parametrize(('value', 'expected'), [(5, '5'), (1.5, '1.5')])
def test_number_literals(con, translate, value, expected):
    expr = ibis.literal(value)
    assert translate(expr.op()) == expected
    assert con.execute(expr) == value


@pytest.mark.parametrize(('value', 'expected'), [(True, '1'), (False, '0')])
def test_boolean_literals(con, translate, value, expected):
    expr = ibis.literal(value)
    assert translate(expr.op()) == expected
    assert con.execute(expr) == value
