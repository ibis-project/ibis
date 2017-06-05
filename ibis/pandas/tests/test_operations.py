import operator
import datetime

import pytest

pytest.importorskip('multipledispatch')

import numpy as np  # noqa: E402

import pandas as pd  # noqa: E402
import pandas.util.testing as tm  # noqa: E402

import ibis  # noqa: E402
import ibis.expr.datatypes as dt  # noqa: E402
from ibis import literal as L  # noqa: E402
from ibis.pandas.api import connect, execute  # noqa: E402


@pytest.fixture
def df():
    return pd.DataFrame({
        'plain_int64': [1, 2, 3],
        'plain_strings': list('abc'),
        'plain_float64': [4.0, 5.0, 6.0],
        'plain_datetimes': pd.date_range('now', periods=3).values,
        'dup_strings': list('dad'),
        'float64_as_strings': ['1.0', '2', '3.234'],
        'int64_as_strings': list(map(str, range(1, 4))),
        'strings_with_space': list(' ad'),
        'int64_with_zeros': [0, 1, 0],
        'float64_with_zeros': [1.0, 0.0, 1.0],
    })


@pytest.fixture
def df1():
    return pd.DataFrame(
        {'key': list('abcd'), 'value': [3, 4, 5, 6], 'key2': list('eeff')}
    )


@pytest.fixture
def df2():
    return pd.DataFrame(
            {'key': list('ac'), 'other_value': [4.0, 6.0], 'key3': list('fe')}
    )


@pytest.fixture
def client(df, df1, df2):
    return connect(
        {'df': df, 'df1': df1, 'df2': df2, 'left': df1, 'right': df2}
    )


@pytest.fixture
def t(client):
    return client.table('df')


@pytest.fixture
def left(client):
    return client.table('left')


@pytest.fixture
def right(client):
    return client.table('right')


def test_table_column(t, df):
    expr = t.plain_int64
    result = expr.execute()
    tm.assert_series_equal(result, df.plain_int64)


def test_literal(client):
    assert client.execute(ibis.literal(1)) == 1


@pytest.mark.parametrize('from_', ['plain_float64', 'plain_int64'])
@pytest.mark.parametrize(
    ('to', 'expected'),
    [
        ('double', 'float64'),
        ('float', 'float32'),
        ('int8', 'int8'),
        ('int16', 'int16'),
        ('int32', 'int32'),
        ('int64', 'int64'),
        ('string', 'object'),
    ],
)
def test_cast_numeric(t, df, from_, to, expected):
    c = t[from_].cast(to)
    result = c.execute()
    assert str(result.dtype) == expected


@pytest.mark.parametrize('from_', ['float64_as_strings', 'int64_as_strings'])
@pytest.mark.parametrize(
    ('to', 'expected'),
    [
        ('double', 'float64'),
        ('string', 'object'),
    ]
)
def test_cast_string(t, df, from_, to, expected):
    c = t[from_].cast(to)
    result = c.execute()
    assert str(result.dtype) == expected


@pytest.mark.parametrize(
    ('to', 'expected'),
    [
        ('string', 'object'),
        ('int64', 'int64'),
        pytest.mark.xfail(('double', 'float64'), raises=TypeError),
    ]
)
def test_cast_timestamp(t, df, to, expected):
    c = t.plain_datetimes.cast(to)
    result = c.execute()
    assert str(result.dtype) == expected


def test_cast_date(t, df):
    assert t.plain_datetimes.type() == dt.timestamp

    expr = t.plain_datetimes.cast('date').cast('string')
    result = expr.execute()
    expected = df.plain_datetimes.dt.date.astype(str)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ('case_func', 'expected_func'),
    [
        (lambda v: v.strftime('%Y%m%d'), lambda vt: vt.strftime('%Y%m%d')),

        (lambda v: v.year(), lambda vt: vt.year),
        (lambda v: v.month(), lambda vt: vt.month),
        (lambda v: v.day(), lambda vt: vt.day),
        (lambda v: v.hour(), lambda vt: vt.hour),
        (lambda v: v.minute(), lambda vt: vt.minute),
        (lambda v: v.second(), lambda vt: vt.second),
        (lambda v: v.millisecond(), lambda vt: int(vt.microsecond / 1e3)),
    ] + [
        (
            operator.methodcaller('strftime', pattern),
            operator.methodcaller('strftime', pattern),
        ) for pattern in [
            '%Y%m%d %H',
            'DD BAR %w FOO "DD"',
            'DD BAR %w FOO "D',
            'DD BAR "%w" FOO "D',
            'DD BAR "%d" FOO "D',
            'DD BAR "%c" FOO "D',
            'DD BAR "%x" FOO "D',
            'DD BAR "%X" FOO "D',
        ]
    ]
)
def test_timestamp_functions(case_func, expected_func):
    v = L('2015-09-01 14:48:05.359').cast('timestamp')
    vt = datetime.datetime(
        year=2015, month=9, day=1,
        hour=14, minute=48, second=5, microsecond=359000
    )
    result = case_func(v)
    expected = expected_func(vt)
    assert execute(result) == expected


@pytest.mark.parametrize(
    'op',
    [
        # comparison
        operator.eq,
        operator.ne,
        operator.lt,
        operator.le,
        operator.gt,
        operator.ge,

        # arithmetic
        operator.add,
        operator.sub,
        operator.mul,
        operator.truediv,
        operator.floordiv,
        operator.mod,
        operator.pow,
    ]
)
def test_binary_operations(t, df, op):
    expr = op(t.plain_float64, t.plain_int64)
    result = expr.execute()
    tm.assert_series_equal(result, op(df.plain_float64, df.plain_int64))


@pytest.mark.parametrize('op', [operator.and_, operator.or_, operator.xor])
def test_binary_boolean_operations(t, df, op):
    expr = op(t.plain_int64 == 1, t.plain_int64 == 2)
    result = expr.execute()
    tm.assert_series_equal(
        result,
        op(df.plain_int64 == 1, df.plain_int64 == 2)
    )


@pytest.mark.parametrize(
    'how',
    [
        'inner',
        'left',
        'outer',

        pytest.mark.xfail('right', raises=KeyError),

        pytest.mark.xfail('semi', raises=NotImplementedError),
        pytest.mark.xfail('anti', raises=NotImplementedError),
    ]
)
def test_join(how, left, right, df1, df2):
    expr = left.join(right, left.key == right.key, how=how)
    result = expr.execute()
    expected = pd.merge(df1, df2, how=how, on='key')
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize('how', ['inner', 'left', 'outer'])
def test_join_with_multiple_predicates(how, left, right, df1, df2):
    expr = left.join(
        right, [left.key == right.key, left.key2 == right.key3], how=how
    )
    result = expr.execute()
    expected = pd.merge(
        df1, df2,
        how=how,
        left_on=['key', 'key2'],
        right_on=['key', 'key3'],
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize('how', ['inner', 'left', 'outer'])
def test_join_with_multiple_predicates_written_as_one(
    how, left, right, df1, df2
):
    predicate = (left.key == right.key) & (left.key2 == right.key3)
    expr = left.join(right, predicate, how=how)
    result = expr.execute()
    expected = pd.merge(
        df1, df2,
        how=how,
        left_on=['key', 'key2'],
        right_on=['key', 'key3'],
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize('how', ['inner', 'left', 'outer'])
def test_join_with_invalid_predicates(how, left, right):
    predicate = (left.key == right.key) & (left.key2 <= right.key3)
    expr = left.join(right, predicate, how=how)
    with pytest.raises(TypeError):
        expr.execute()

    predicate = left.key >= right.key
    expr = left.join(right, predicate, how=how)
    with pytest.raises(TypeError):
        expr.execute()


@pytest.mark.parametrize('how', ['inner', 'left', 'outer'])
def test_join_with_duplicate_non_key_columns(how, left, right, df1, df2):
    left = left.mutate(x=left.value * 2)
    right = right.mutate(x=right.other_value * 3)
    expr = left.join(right, left.key == right.key, how=how)
    with pytest.raises(ValueError):
        expr.execute()


@pytest.mark.parametrize('how', ['inner', 'left', 'outer'])
def test_join_with_duplicate_non_key_columns_not_selected(
    how, left, right, df1, df2
):
    left = left.mutate(x=left.value * 2)
    right = right.mutate(x=right.other_value * 3)
    right = right[['key', 'other_value']]
    expr = left.join(right, left.key == right.key, how=how)
    result = expr.execute()
    expected = pd.merge(
        df1.assign(x=df1.value * 2),
        df2[['key', 'other_value']],
        how=how,
        on='key',
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize('how', ['inner', 'left', 'outer'])
def test_join_with_post_expression_selection(how, left, right, df1, df2):
    join = left.join(right, left.key == right.key, how=how)
    expr = join[left.key, left.value, right.other_value]
    result = expr.execute()
    expected = pd.merge(df1, df2, on='key', how=how)[[
        'key', 'value', 'other_value'
    ]]
    tm.assert_frame_equal(result, expected)


def test_selection(t, df):
    expr = t[
        ((t.plain_strings == 'a') | (t.plain_int64 == 3)) &
        (t.dup_strings == 'd')
    ]
    result = expr.execute()
    expected = df[
        ((df.plain_strings == 'a') | (df.plain_int64 == 3)) &
        (df.dup_strings == 'd')
    ]
    tm.assert_frame_equal(result, expected)


def test_mutate(t, df):
    expr = t.mutate(x=t.plain_int64 + 1, y=t.plain_int64 * 2)
    result = expr.execute()
    expected = df.assign(x=df.plain_int64 + 1, y=df.plain_int64 * 2)
    tm.assert_frame_equal(result[expected.columns], expected[expected.columns])


@pytest.mark.parametrize(
    'where',
    [
        lambda t: t.dup_strings == 'd',
        lambda t: None,
    ]
)
def test_aggregation_group_by(t, df, where):
    ibis_where = where(t)
    expr = t.group_by(t.dup_strings).aggregate(
        avg_plain_int64=t.plain_int64.mean(where=ibis_where),
        sum_plain_float64=t.plain_float64.sum(where=ibis_where),
    )
    result = expr.execute()[['avg_plain_int64', 'sum_plain_float64']]

    pandas_where = where(df)
    mask = slice(None) if pandas_where is None else pandas_where
    expected = df.groupby('dup_strings').agg({
        'plain_int64': lambda x, mask=mask: x[mask].mean(),
        'plain_float64': lambda x, mask=mask: x[mask].sum(),
    }).reset_index().rename(
        columns={
            'plain_int64': 'avg_plain_int64',
            'plain_float64': 'sum_plain_float64',
        }
    )[['avg_plain_int64', 'sum_plain_float64']]
    tm.assert_frame_equal(result, expected)


def test_aggregation_without_group_by(t, df):
    expr = t.aggregate(
        avg_plain_int64=t.plain_int64.mean(),
        sum_plain_float64=t.plain_float64.sum()
    )
    result = expr.execute()[['avg_plain_int64', 'sum_plain_float64']]
    new_names = {
        'plain_float64': 'sum_plain_float64',
        'plain_int64': 'avg_plain_int64',
    }
    expected = pd.Series(
        [df['plain_int64'].mean(), df['plain_float64'].sum()],
        index=['plain_int64', 'plain_float64'],
    ).to_frame().T.rename(columns=new_names)
    tm.assert_frame_equal(result, expected)


@pytest.mark.xfail(raises=NotImplementedError)
def test_group_by_with_having(t, df):
    expr = t.group_by(t.dup_strings).having(
        t.plain_float64.sum() == 5
    ).aggregate(
        avg_a=t.plain_int64.mean(),
        sum_c=t.plain_float64.sum(),
    )
    result = expr.execute()[['avg_a', 'sum_c']]

    expected = df.groupby('dup_strings').agg({
        'a': 'mean',
        'c': 'sum',
    }).reset_index().rename(columns={'a': 'avg_a', 'c': 'sum_c'})
    expected = expected.loc[expected.sum_c == 5, ['avg_a', 'sum_c']]

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    'reduction',
    ['mean', 'sum', 'count', 'std', 'var']
)
@pytest.mark.parametrize(
    'where',
    [
        lambda t: (t.plain_strings == 'a') | (t.plain_strings == 'c'),
        lambda t: (t.dup_strings == 'd') & (
            (t.plain_int64 == 1) | (t.plain_int64 == 3)
        ),
        lambda t: None,
    ]
)
def test_reduction(t, df, reduction, where):
    func = getattr(t.plain_int64, reduction)
    mask = where(t)
    expr = func(where=mask)
    result = expr.execute()

    df_mask = where(df)
    expected_func = getattr(
        df.loc[df_mask if df_mask is not None else slice(None), 'plain_int64'],
        reduction,
    )
    expected = expected_func()
    assert result == expected


@pytest.mark.parametrize(
    'reduction',
    [
        lambda x: x.any(),
        lambda x: x.all(),
        lambda x: ~x.any(),
        lambda x: ~x.all(),
    ]
)
def test_boolean_aggregation(t, df, reduction):
    expr = reduction(t.plain_int64 == 1)
    result = expr.execute()
    expected = reduction(df.plain_int64 == 1)
    assert result == expected


@pytest.mark.parametrize('column', ['float64_with_zeros', 'int64_with_zeros'])
def test_null_if_zero(t, df, column):
    expr = t[column].nullifzero()
    result = expr.execute()
    expected = df[column].replace(0, np.nan)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ('case_func', 'expected_func'),
    [
        (lambda s: s.length(), lambda s: s.str.len()),
        (lambda s: s.substr(1, 2), lambda s: s.str[1:3]),
        (lambda s: s.strip(), lambda s: s.str.strip()),
        (lambda s: s.lstrip(), lambda s: s.str.lstrip()),
        (lambda s: s.rstrip(), lambda s: s.str.rstrip()),
        (
            lambda s: s.lpad(3, 'a'),
            lambda s: s.str.pad(3, side='left', fillchar='a')
        ),
        (
            lambda s: s.rpad(3, 'b'),
            lambda s: s.str.pad(3, side='right', fillchar='b')
        ),
        (lambda s: s.reverse(), lambda s: s.str[::-1]),
        (lambda s: s.lower(), lambda s: s.str.lower()),
        (lambda s: s.upper(), lambda s: s.str.upper()),
        (lambda s: s.capitalize(), lambda s: s.str.capitalize()),
        (lambda s: s.repeat(2), lambda s: s * 2),
        (lambda s: s.contains('a'), lambda s: s.str.contains('a')),
        (lambda s: ~s.contains('a'), lambda s: ~s.str.contains('a')),
        pytest.mark.xfail(
            (
                lambda s: s + s.rpad(3, 'a'),
                lambda s: s + s.str.pad(3, side='right', fillchar='a')
            ),
            raises=NotImplementedError,
            reason='Implement string concat with plus'
        ),
    ]
)
def test_string_ops(t, df, case_func, expected_func):
    expr = case_func(t.strings_with_space)
    result = expr.execute()
    series = expected_func(df.strings_with_space)
    tm.assert_series_equal(result, series)


def test_group_concat(t, df):
    expr = t.groupby(t.dup_strings).aggregate(
        foo=t.plain_int64.group_concat(',')
    )
    result = expr.execute()
    expected = df.groupby('dup_strings').plain_int64.apply(
        lambda x: ','.join(x.astype(str))
    ).to_frame().reset_index(drop=True).rename(columns={'plain_int64': 'foo'})
    tm.assert_frame_equal(result, expected)
