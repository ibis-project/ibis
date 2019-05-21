import pytest
from pandas import Timestamp

import ibis
from ibis import literal as L


pytest.importorskip('clickhouse_driver')
pytestmark = pytest.mark.clickhouse


@pytest.mark.parametrize('expr', [
    L(Timestamp('2015-01-01 12:34:56')),
    L(Timestamp('2015-01-01 12:34:56').to_pydatetime()),
    ibis.timestamp('2015-01-01 12:34:56')
])
def test_timestamp_literals(con, translate, expr):
    expected = "toDateTime('2015-01-01 12:34:56')"

    assert translate(expr) == expected
    assert con.execute(expr) == Timestamp('2015-01-01 12:34:56')


@pytest.mark.parametrize(('value', 'expected'), [
    ('simple', "'simple'"),
    ('I can\'t', "'I can\\'t'"),
    ('An "escape"', "'An \"escape\"'")
])
def test_string_literals(con, translate, value, expected):
    expr = ibis.literal(value)
    assert translate(expr) == expected
    # TODO clickhouse-driver escaping problem
    # assert con.execute(expr) == expected


@pytest.mark.parametrize(('value', 'expected'), [
    (5, '5'),
    (1.5, '1.5'),
])
def test_number_literals(con, translate, value, expected):
    expr = ibis.literal(value)
    assert translate(expr) == expected
    assert con.execute(expr) == value


@pytest.mark.parametrize(('value', 'expected'), [
    (True, '1'),
    (False, '0'),
])
def test_boolean_literals(con, translate, value, expected):
    expr = ibis.literal(value)
    assert translate(expr) == expected
    assert con.execute(expr) == value
