import pytest

import ibis
import ibis.expr.datatypes as dt
import ibis.tests.util as tu


@pytest.mark.parametrize(('column', 'raw_value'), [
    ('double_col', 0.0),
    ('double_col', 10.1),
    ('float_col', 1.1),
    ('float_col', 2.2)
])
@tu.skipif_unsupported
def test_floating_scalar_parameter(backend, alltypes, df, column, raw_value):
    value = ibis.param(dt.double)
    expr = alltypes[column] + value
    expected = df[column] + raw_value

    result = expr.execute(params={value: raw_value})
    expected = backend.default_series_rename(expected).astype('float64')

    backend.assert_series_equal(result, expected)


@pytest.mark.parametrize(('start_string', 'end_string'), [
    ('2009-03-01', '2010-07-03'),
    ('2014-12-01', '2017-01-05')
])
@tu.skipif_unsupported
def test_date_scalar_parameter(backend, alltypes, df, start_string,
                               end_string):
    start, end = ibis.param(dt.date), ibis.param(dt.date)

    col = alltypes.timestamp_col.date()
    expr = col.between(start, end)
    expected_expr = col.between(start_string, end_string)

    result = expr.execute(params={start: start_string,
                                  end: end_string})
    expected = expected_expr.execute()

    backend.assert_series_equal(result, expected)


@tu.skipif_unsupported
def test_timestamp_accepts_date_literals(backend, alltypes):
    date_string = '2009-03-01'
    param = ibis.param(dt.timestamp)
    expr = alltypes.mutate(param=param)
    params = {param: date_string}
    assert expr.compile(params=params) is not None
