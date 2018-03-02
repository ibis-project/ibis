import pytest

import pandas as pd
import pandas.util.testing as tm

import ibis
import ibis.expr.datatypes as dt

udf = pytest.importorskip('ibis.bigquery.udf.udf')
pytest.importorskip('google.cloud.bigquery')


@pytest.fixture(scope='module')
def client():
    return ibis.bigquery.connect('ibis-gbq', 'testing')


@pytest.fixture(scope='module')
def alltypes(client):
    t = client.table('functional_alltypes')
    expr = t[t.bigint_col.isin([10, 20])].limit(10)
    return expr


@pytest.fixture(scope='module')
def df(alltypes):
    return alltypes.execute()


def test_udf(client, alltypes, df):
    @udf(input_type=[dt.double, dt.double], output_type=dt.double)
    def my_add(a, b):
        return a + b

    expr = my_add(alltypes.double_col, alltypes.double_col)
    result = expr.execute()
    assert not result.empty

    expected = (df.double_col + df.double_col).rename('tmp')
    tm.assert_series_equal(
        result.value_counts().sort_index(),
        expected.value_counts().sort_index()
    )


def test_udf_with_struct(client, alltypes, df):
    @udf(
        input_type=[dt.double, dt.double],
        output_type=dt.Struct.from_tuples([
            ('width', dt.double),
            ('height', dt.double)
        ])
    )
    def my_struct_thing(a, b):
        class Rectangle:
            def __init__(self, width, height):
                self.width = width
                self.height = height
        return Rectangle(a, b)

    assert my_struct_thing.js == '''\
CREATE TEMPORARY FUNCTION my_struct_thing(a FLOAT64, b FLOAT64)
RETURNS STRUCT<width FLOAT64, height FLOAT64>
LANGUAGE js AS """
'use strict';
function my_struct_thing(a, b) {
    class Rectangle {
        constructor(width, height) {
            this.width = width;
            this.height = height;
        }
    }
    return (new Rectangle(a, b));
}
return my_struct_thing(a, b);
""";'''

    expr = my_struct_thing(alltypes.double_col, alltypes.double_col)
    result = expr.execute()
    assert not result.empty

    expected = pd.Series(
        [{'width': c, 'height': c} for c in df.double_col],
        name='tmp'
    )
    tm.assert_series_equal(result, expected)


def test_udf_compose(client, alltypes, df):
    @udf([dt.double], dt.double)
    def add_one(x):
        return x + 1.0

    @udf([dt.double], dt.double)
    def times_two(x):
        return x * 2.0

    t = alltypes
    expr = times_two(add_one(t.double_col))
    result = expr.execute()
    expected = ((df.double_col + 1.0) * 2.0).rename('tmp')
    tm.assert_series_equal(result, expected)
