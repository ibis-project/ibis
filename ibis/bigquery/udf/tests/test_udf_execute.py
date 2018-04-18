import os

import pytest

import pandas as pd
import pandas.util.testing as tm

import ibis
import ibis.expr.datatypes as dt

pytest.importorskip('google.cloud.bigquery')

pytestmark = pytest.mark.bigquery

from ibis.bigquery.api import udf  # noqa: E402

PROJECT_ID = os.environ.get('GOOGLE_BIGQUERY_PROJECT_ID', 'ibis-gbq')
DATASET_ID = 'testing'


@pytest.fixture(scope='module')
def client():
    ga = pytest.importorskip('google.auth')

    try:
        return ibis.bigquery.connect(PROJECT_ID, DATASET_ID)
    except ga.exceptions.DefaultCredentialsError:
        pytest.skip("no credentials found, skipping")


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


def test_udf_scalar(client):
    @udf([dt.double, dt.double], dt.double)
    def my_add(x, y):
        return x + y

    expr = my_add(1, 2)
    sql = client.compile(expr)
    assert sql == '''\
CREATE TEMPORARY FUNCTION my_add(x FLOAT64, y FLOAT64)
RETURNS FLOAT64
LANGUAGE js AS """
'use strict';
function my_add(x, y) {
    return (x + y);
}
return my_add(x, y);
""";

SELECT my_add(1, 2) AS `tmp`'''
    result = client.execute(expr)
    assert result == 3
