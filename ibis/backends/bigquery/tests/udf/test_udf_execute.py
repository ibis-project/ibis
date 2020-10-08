import os

import pandas as pd
import pandas.util.testing as tm
import pytest
from pytest import param

import ibis
import ibis.expr.datatypes as dt
from ibis.compat import PY38

from . import udf  # noqa: E402

pytest.importorskip('google.cloud.bigquery')

if PY38:
    # ref: https://github.com/ibis-project/ibis/issues/2098
    # note: UDF is already skipt on CI
    pytestmark = [pytest.mark.bigquery, pytest.mark.udf]
else:
    pytestmark = pytest.mark.bigquery


PROJECT_ID = os.environ.get('GOOGLE_BIGQUERY_PROJECT_ID', 'ibis-gbq')
DATASET_ID = 'testing'


@pytest.fixture(scope='module')
def client():
    from .tests.conftest import connect

    return connect(PROJECT_ID, DATASET_ID)


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
        expected.value_counts().sort_index(),
    )


def test_udf_with_struct(client, alltypes, df):
    @udf(
        input_type=[dt.double, dt.double],
        output_type=dt.Struct.from_tuples(
            [('width', dt.double), ('height', dt.double)]
        ),
    )
    def my_struct_thing(a, b):
        class Rectangle:
            def __init__(self, width, height):
                self.width = width
                self.height = height

        return Rectangle(a, b)

    assert (
        my_struct_thing.js
        == '''\
CREATE TEMPORARY FUNCTION my_struct_thing_0(a FLOAT64, b FLOAT64)
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
    )

    expr = my_struct_thing(alltypes.double_col, alltypes.double_col)
    result = expr.execute()
    assert not result.empty

    expected = pd.Series(
        [{'width': c, 'height': c} for c in df.double_col], name='tmp'
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
    result = client.execute(expr)
    assert result == 3


def test_multiple_calls_has_one_definition(client):
    @udf([dt.string], dt.double)
    def my_str_len(s):
        return s.length

    s = ibis.literal('abcd')
    expr = my_str_len(s) + my_str_len(s)
    sql = client.compile(expr)
    expected = '''\
CREATE TEMPORARY FUNCTION my_str_len_0(s STRING)
RETURNS FLOAT64
LANGUAGE js AS """
'use strict';
function my_str_len(s) {
    return s.length;
}
return my_str_len(s);
""";

SELECT my_str_len_0('abcd') + my_str_len_0('abcd') AS `tmp`'''
    assert sql == expected
    result = client.execute(expr)
    assert result == 8.0


def test_udf_libraries(client):
    @udf(
        [dt.Array(dt.string)],
        dt.double,
        # whatever symbols are exported in the library are visible inside the
        # UDF, in this case lodash defines _ and we use that here
        libraries=['gs://ibis-testing-libraries/lodash.min.js'],
    )
    def string_length(strings):
        return _.sum(_.map(strings, lambda x: x.length))  # noqa: F821

    raw_data = ['aaa', 'bb', 'c']
    data = ibis.literal(raw_data)
    expr = string_length(data)
    result = client.execute(expr)
    expected = sum(map(len, raw_data))
    assert result == expected


def test_udf_with_len(client):
    @udf([dt.string], dt.double)
    def my_str_len(x):
        return len(x)

    @udf([dt.Array(dt.string)], dt.double)
    def my_array_len(x):
        return len(x)

    assert client.execute(my_str_len('aaa')) == 3
    assert client.execute(my_array_len(['aaa', 'bb'])) == 2


def test_multiple_calls_redefinition(client):
    @udf([dt.string], dt.double)
    def my_len(s):
        return s.length

    s = ibis.literal('abcd')
    expr = my_len(s) + my_len(s)

    @udf([dt.string], dt.double)
    def my_len(s):
        return s.length + 1

    expr = expr + my_len(s)

    sql = client.compile(expr)
    expected = '''\
CREATE TEMPORARY FUNCTION my_len_0(s STRING)
RETURNS FLOAT64
LANGUAGE js AS """
'use strict';
function my_len(s) {
    return s.length;
}
return my_len(s);
""";

CREATE TEMPORARY FUNCTION my_len_1(s STRING)
RETURNS FLOAT64
LANGUAGE js AS """
'use strict';
function my_len(s) {
    return (s.length + 1);
}
return my_len(s);
""";

SELECT (my_len_0('abcd') + my_len_0('abcd')) + my_len_1('abcd') AS `tmp`'''
    assert sql == expected


@pytest.mark.parametrize(
    ('argument_type', 'return_type'),
    [
        param(dt.int64, dt.float64, marks=pytest.mark.xfail(raises=TypeError)),
        param(dt.float64, dt.int64, marks=pytest.mark.xfail(raises=TypeError)),
        # complex argument type, valid return type
        param(
            dt.Array(dt.int64),
            dt.float64,
            marks=pytest.mark.xfail(raises=TypeError),
        ),
        # valid argument type, complex invalid return type
        param(
            dt.float64,
            dt.Array(dt.int64),
            marks=pytest.mark.xfail(raises=TypeError),
        ),
        # both invalid
        param(
            dt.Array(dt.Array(dt.int64)),
            dt.int64,
            marks=pytest.mark.xfail(raises=TypeError),
        ),
        # struct type with nested integer, valid return type
        param(
            dt.Struct.from_tuples([('x', dt.Array(dt.int64))]),
            dt.float64,
            marks=pytest.mark.xfail(raises=TypeError),
        ),
    ],
)
def test_udf_int64(client, argument_type, return_type):
    # invalid argument type, valid return type
    @udf([argument_type], return_type)
    def my_int64_add(x):
        return 1.0
