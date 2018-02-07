import ibis
import ibis.expr.datatypes as dt


def test_timestamp_accepts_date_literals(alltypes):
    date_string = '2009-03-01'
    param = ibis.param(dt.timestamp, name='param')
    expr = alltypes.mutate(param=param)
    params = {param: date_string}
    result = expr.compile(params=params)
    expected = """\
SELECT *, @param AS `param`
FROM testing.functional_alltypes"""
    assert result == expected
