import operator

import pandas as pd
import pandas.util.testing as tm

import pytest

import ibis
from ibis.common import IbisTypeError


def test_array_length(t, df):
    expr = t.projection([
        t.array_of_float64.length().name('array_of_float64_length'),
        t.array_of_int64.length().name('array_of_int64_length'),
        t.array_of_strings.length().name('array_of_strings_length'),
    ])
    result = expr.execute()
    expected = pd.DataFrame({
        'array_of_float64_length': [2, 1, 0],
        'array_of_int64_length': [2, 0, 1],
        'array_of_strings_length': [2, 0, 1],
    })

    tm.assert_frame_equal(result, expected)


def test_array_length_scalar(client):
    raw_value = [1, 2, 4]
    value = ibis.literal(raw_value)
    expr = value.length()
    result = client.execute(expr)
    expected = len(raw_value)
    assert result == expected


def test_array_collect(t, df):
    expr = t.group_by(
        t.dup_strings
    ).aggregate(collected=t.float64_with_zeros.collect())
    result = expr.execute().sort_values('dup_strings').reset_index(drop=True)
    expected = df.groupby(
        'dup_strings'
    ).float64_with_zeros.apply(list).reset_index().rename(
        columns={'float64_with_zeros': 'collected'}
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.xfail(
    raises=TypeError,
    reason=(
        'Pandas does not implement rolling for functions that do not return '
        'numbers'
    )
)
def test_array_collect_rolling_partitioned(t, df):
    window = ibis.trailing_window(2, order_by=t.plain_int64)
    colexpr = t.plain_float64.collect().over(window)
    expr = t['dup_strings', 'plain_int64', colexpr.name('collected')]
    result = expr.execute()
    expected = pd.DataFrame({
        'dup_strings': ['d', 'a', 'd'],
        'plain_int64': [1, 2, 3],
        'collected': [[4.0], [4.0, 5.0], [5.0, 6.0]],
    })[expr.columns]
    tm.assert_frame_equal(result, expected)


@pytest.mark.xfail(raises=IbisTypeError, reason='Not sure if this should work')
def test_array_collect_scalar(client):
    raw_value = 'abcd'
    value = ibis.literal(raw_value)
    expr = value.collect()
    result = client.execute(expr)
    expected = [raw_value]
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
        pytest.mark.xfail(
            (-3, None),
            raises=ValueError,
            reason='Negative slicing not supported'
        ),
        pytest.mark.xfail(
            (None, -3),
            raises=ValueError,
            reason='Negative slicing not supported'
        ),
        pytest.mark.xfail(
            (-3, -1),
            raises=ValueError,
            reason='Negative slicing not supported'
        ),
    ]
)
def test_array_slice(t, df, start, stop):
    expr = t.array_of_strings[start:stop]
    result = expr.execute()
    slicer = operator.itemgetter(slice(start, stop))
    expected = df.array_of_strings.apply(slicer)
    tm.assert_series_equal(result, expected)


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
        pytest.mark.xfail(
            (-3, None),
            raises=ValueError,
            reason='Negative slicing not supported'
        ),
        pytest.mark.xfail(
            (None, -3),
            raises=ValueError,
            reason='Negative slicing not supported'
        ),
        pytest.mark.xfail(
            (-3, -1),
            raises=ValueError,
            reason='Negative slicing not supported'
        ),
    ]
)
def test_array_slice_scalar(client, start, stop):
    raw_value = [-11, 42, 10]
    value = ibis.literal(raw_value)
    expr = value[start:stop]
    result = client.execute(expr)
    expected = raw_value[start:stop]
    assert result == expected


@pytest.mark.parametrize('index', [1, 3, 4, 11, -11])
def test_array_index(t, df, index):
    expr = t[t.array_of_float64[index].name('indexed')]
    result = expr.execute()
    expected = pd.DataFrame({
        'indexed': df.array_of_float64.apply(
            lambda x: x[index] if -len(x) <= index < len(x) else None
        )
    })
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize('index', [1, 3, 4, 11])
def test_array_index_scalar(client, index):
    raw_value = [-10, 1, 2, 42]
    value = ibis.literal(raw_value)
    expr = value[index]
    result = client.execute(expr)
    expected = raw_value[index] if index < len(raw_value) else None
    assert result == expected


@pytest.mark.parametrize('n', [1, 3, 4, 7, -2])  # negative returns empty list
@pytest.mark.parametrize('mul', [lambda x, n: x * n, lambda x, n: n * x])
def test_array_repeat(t, df, n, mul):
    expr = t.projection([mul(t.array_of_strings, n).name('repeated')])
    result = expr.execute()
    expected = pd.DataFrame({'repeated': df.array_of_strings * n})
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


@pytest.mark.parametrize('op', [lambda x, y: x + y, lambda x, y: y + x])
def test_array_concat(t, df, op):
    x = t.array_of_float64.cast('array<string>')
    y = t.array_of_strings
    expr = op(x, y)
    result = expr.execute()
    expected = op(
        df.array_of_float64.apply(lambda x: list(map(str, x))),
        df.array_of_strings
    )
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
