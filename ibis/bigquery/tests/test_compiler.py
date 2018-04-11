import pytest

import ibis
import ibis.expr.datatypes as dt


def test_timestamp_accepts_date_literals(alltypes):
    date_string = '2009-03-01'
    param = ibis.param(dt.timestamp).name('param_0')
    expr = alltypes.mutate(param=param)
    params = {param: date_string}
    result = expr.compile(params=params)
    expected = """\
SELECT *, @param AS `param`
FROM testing.functional_alltypes"""
    assert result == expected


@pytest.mark.parametrize(
    ('distinct', 'expected_keyword'),
    [
        (True, 'DISTINCT'),
        (False, 'ALL'),
    ]
)
def test_union(alltypes, distinct, expected_keyword):
    expr = alltypes.union(alltypes, distinct=distinct)
    result = expr.compile()
    expected = """\
SELECT *
FROM testing.functional_alltypes
UNION {}
SELECT *
FROM testing.functional_alltypes""".format(expected_keyword)
    assert result == expected


def test_ieee_divide(alltypes):
    expr = alltypes.double_col / 0
    result = expr.compile()
    expected = """\
SELECT IEEE_DIVIDE(`double_col`, 0) AS `tmp`
FROM testing.functional_alltypes"""
    assert result == expected
