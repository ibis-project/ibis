import numpy as np
import pandas as pd
import pandas.util.testing as tm
import pyspark
import pytest
from pytest import param

import ibis

pytest.importorskip('pyspark')
pytestmark = pytest.mark.pyspark


if pyspark.__version__ < '3.0.0':
    null_representation = None
else:
    null_representation = np.nan


def test_array_length(client):
    table = client.table('array_table')

    result = table.mutate(length=table.array_int.length()).compile()

    expected = table.compile().toPandas()
    expected['length'] = (
        expected['array_int'].map(lambda a: len(a)).astype('int32')
    )
    tm.assert_frame_equal(result.toPandas(), expected)


def test_array_length_scalar(client):
    raw_value = [1, 2, 3]
    value = ibis.literal(raw_value)
    expr = value.length()
    result = client.execute(expr)
    expected = len(raw_value)
    assert result == expected


@pytest.mark.parametrize(
    ['start', 'stop'],
    [
        (1, 3),
        (1, 1),
        (2, 3),
        (2, 5),
        (None, 3),
        (None, None),
        (3, None),
        # negative slices are not supported
        param(
            -3,
            None,
            marks=pytest.mark.xfail(
                raises=ValueError, reason='Negative slicing not supported'
            ),
        ),
        param(
            None,
            -3,
            marks=pytest.mark.xfail(
                raises=ValueError, reason='Negative slicing not supported'
            ),
        ),
        param(
            -3,
            -1,
            marks=pytest.mark.xfail(
                raises=ValueError, reason='Negative slicing not supported'
            ),
        ),
    ],
)
def test_array_slice(client, start, stop):
    table = client.table('array_table')

    result = table.mutate(sliced=table.array_int[start:stop]).compile()

    expected = table.compile().toPandas()
    expected['sliced'] = expected['array_int'].map(lambda a: a[start:stop])
    tm.assert_frame_equal(result.toPandas(), expected)


@pytest.mark.parametrize(
    ['start', 'stop'],
    [
        (1, 3),
        (1, 1),
        (2, 3),
        (2, 5),
        (None, 3),
        (None, None),
        (3, None),
        # negative slices are not supported
        param(
            -3,
            None,
            marks=pytest.mark.xfail(
                raises=ValueError, reason='Negative slicing not supported'
            ),
        ),
        param(
            None,
            -3,
            marks=pytest.mark.xfail(
                raises=ValueError, reason='Negative slicing not supported'
            ),
        ),
        param(
            -3,
            -1,
            marks=pytest.mark.xfail(
                raises=ValueError, reason='Negative slicing not supported'
            ),
        ),
    ],
)
def test_array_slice_scalar(client, start, stop):
    raw_value = [-11, 42, 10]
    value = ibis.literal(raw_value)
    expr = value[start:stop]
    result = client.execute(expr)
    expected = raw_value[start:stop]
    assert result == expected


@pytest.mark.parametrize('index', [1, 3, 4, 11, -11])
def test_array_index(client, index):
    table = client.table('array_table')
    expr = table[table.array_int[index].name('indexed')]
    result = expr.execute()

    df = table.compile().toPandas()
    expected = pd.DataFrame(
        {
            'indexed': df.array_int.apply(
                lambda x: x[index]
                if -len(x) <= index < len(x)
                else null_representation
            )
        }
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize('index', [1, 3, 4, 11])
def test_array_index_scalar(client, index):
    raw_value = [-10, 1, 2, 42]
    value = ibis.literal(raw_value)
    expr = value[index]
    result = client.execute(expr)
    expected = (
        raw_value[index] if index < len(raw_value) else null_representation
    )
    assert result == expected or (np.isnan(result) and np.isnan(expected))


@pytest.mark.parametrize('op', [lambda x, y: x + y, lambda x, y: y + x])
def test_array_concat(client, op):
    table = client.table('array_table')
    x = table.array_int.cast('array<string>')
    y = table.array_str
    expr = op(x, y)
    result = expr.execute()

    df = table.compile().toPandas()
    expected = op(
        df.array_int.apply(lambda x: list(map(str, x))), df.array_str
    ).rename('tmp')
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize('op', [lambda x, y: x + y, lambda x, y: y + x])
def test_array_concat_scalar(client, op):
    raw_left = [1, 2, 3]
    raw_right = [3, 4]
    left = ibis.literal(raw_left)
    right = ibis.literal(raw_right)
    expr = op(left, right)
    result = client.execute(expr)
    assert result == op(raw_left, raw_right)


@pytest.mark.parametrize('n', [1, 3, 4, 7, -2])  # negative returns empty list
@pytest.mark.parametrize('mul', [lambda x, n: x * n, lambda x, n: n * x])
def test_array_repeat(client, n, mul):
    table = client.table('array_table')

    expr = table.projection([mul(table.array_int, n).name('repeated')])
    result = expr.execute()

    df = table.compile().toPandas()
    expected = pd.DataFrame({'repeated': df.array_int * n})
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize('n', [1, 3, 4, 7, -2])  # negative returns empty list
@pytest.mark.parametrize('mul', [lambda x, n: x * n, lambda x, n: n * x])
def test_array_repeat_scalar(client, n, mul):
    raw_array = [1, 2]
    array = ibis.literal(raw_array)
    expr = mul(array, n)
    result = client.execute(expr)
    expected = mul(raw_array, n)
    assert result == expected


def test_array_collect(client):
    table = client.table('array_table')
    expr = table.group_by(table.key).aggregate(
        collected=table.array_int.collect()
    )
    result = expr.execute().sort_values('key').reset_index(drop=True)

    df = table.compile().toPandas()
    expected = (
        df.groupby('key')
        .array_int.apply(list)
        .reset_index()
        .rename(columns={'array_int': 'collected'})
    )
    tm.assert_frame_equal(result, expected)
