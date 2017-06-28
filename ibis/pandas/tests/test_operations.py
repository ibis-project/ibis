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


@pytest.fixture(params=[None, 'UTC', 'America/New_York'])
def tz(request):
    return request.param


@pytest.fixture
def df(tz):
    return pd.DataFrame({
        'plain_int64': list(range(1, 4)),
        'plain_strings': list('abc'),
        'plain_float64': [4.0, 5.0, 6.0],
        'plain_datetimes': pd.Series(
            pd.date_range('now', periods=3).values,
        ).dt.tz_localize(tz),
        'dup_strings': list('dad'),
        'dup_ints': [1, 2, 1],
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
    return ibis.pandas.connect(
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
        (
            dt.Timestamp('America/Los_Angeles'),
            'datetime64[ns, America/Los_Angeles]'
        )
    ]
)
def test_cast_timestamp_column(t, df, to, expected):
    c = t.plain_datetimes.cast(to)
    result = c.execute()
    assert str(result.dtype) == expected


@pytest.mark.parametrize(
    ('to', 'expected'),
    [
        ('string', str),
        ('int64', lambda x: x.value),
        pytest.mark.xfail(('double', float), raises=NotImplementedError),
        (
            dt.Timestamp('America/Los_Angeles'),
            lambda x: pd.Timestamp(x, tz='America/Los_Angeles')
        )
    ]
)
def test_cast_timestamp_scalar(to, expected, tz):
    literal_expr = ibis.literal(pd.Timestamp('now', tz=tz))
    value = literal_expr.cast(to)
    result = ibis.pandas.execute(value)
    raw = ibis.pandas.execute(literal_expr)
    assert result == expected(raw)


def test_timestamp_with_timezone_is_inferred_correctly(t, df, tz):
    assert t.plain_datetimes.type().equals(dt.Timestamp(tz))


def test_cast_date(t, df):
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
    assert ibis.pandas.execute(result) == expected


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
        lambda t: None,
        lambda t: t.dup_strings == 'd',
        lambda t: (t.dup_strings == 'd') | (t.plain_int64 < 100),
    ]
)
def test_aggregation_group_by(t, df, where):
    ibis_where = where(t)
    expr = t.group_by(t.dup_strings).aggregate(
        avg_plain_int64=t.plain_int64.mean(where=ibis_where),
        sum_plain_float64=t.plain_float64.sum(where=ibis_where),
        nunique_dup_ints=t.dup_ints.nunique(),
    )
    columns = ['avg_plain_int64', 'sum_plain_float64', 'nunique_dup_ints']
    result = expr.execute()

    pandas_where = where(df)
    mask = slice(None) if pandas_where is None else pandas_where
    expected = df.groupby('dup_strings').agg({
        'plain_int64': lambda x, mask=mask: x[mask].mean(),
        'plain_float64': lambda x, mask=mask: x[mask].sum(),
        'dup_ints': 'nunique',
    }).reset_index().rename(
        columns={
            'plain_int64': 'avg_plain_int64',
            'plain_float64': 'sum_plain_float64',
            'dup_ints': 'nunique_dup_ints',
        }
    )
    # TODO(phillipc): Why does pandas not return floating point values here?
    expected['avg_plain_int64'] = expected.avg_plain_int64.astype('float64')
    tm.assert_frame_equal(result[columns], expected[columns])


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


@pytest.mark.parametrize('reduction', ['mean', 'sum', 'count', 'std', 'var'])
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
    expected = df.groupby('dup_strings').apply(
        lambda df: ','.join(df.plain_int64.astype(str))
    ).reset_index().rename(columns={0: 'foo'})
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize('offset', [0, 2])
def test_frame_limit(t, df, offset):
    n = 5
    df_expr = t.limit(n, offset=offset)
    result = df_expr.execute()
    expected = df.iloc[offset:offset + n]
    tm.assert_frame_equal(result, expected)


@pytest.mark.xfail(
    raises=AttributeError, reason='TableColumn does not implement limit'
)
@pytest.mark.parametrize('offset', [0, 2])
def test_series_limit(t, df, offset):
    n = 5
    s_expr = t.plain_int64.limit(n, offset=offset)
    result = s_expr.execute()
    tm.assert_series_equal(result, df.plain_int64.iloc[offset:offset + n])


@pytest.mark.parametrize(
    ('key', 'pandas_by', 'pandas_ascending'),
    [
        (lambda t: [ibis.desc(t.plain_datetimes)], ['plain_datetimes'], False),
        (
            lambda t: [t.plain_datetimes, ibis.desc(t.plain_int64)],
            ['plain_datetimes', 'plain_int64'],
            [True, False]
        ),
        (
            lambda t: [ibis.desc(t.plain_int64 * 2)],
            ['plain_int64'],
            False,
        ),
    ]
)
def test_sort_by(t, df, key, pandas_by, pandas_ascending):
    expr = t.sort_by(key(t))
    result = expr.execute()
    expected = df.sort_values(pandas_by, ascending=pandas_ascending)
    tm.assert_frame_equal(result, expected)


def test_complex_sort_by(t, df):
    expr = t.sort_by([
        ibis.desc(t.plain_int64 * t.plain_float64), t.plain_float64
    ])
    result = expr.execute()
    expected = df.assign(
        foo=df.plain_int64 * df.plain_float64
    ).sort_values(['foo', 'plain_float64'], ascending=[False, True])

    tm.assert_frame_equal(result, expected.loc[:, expr.columns])


def test_distinct(t, df):
    expr = t.dup_strings.distinct()
    result = expr.execute()
    expected = pd.Series(df.dup_strings.unique(), name='dup_strings')
    tm.assert_series_equal(result, expected)


def test_count_distinct(t, df):
    expr = t.dup_strings.nunique()
    result = expr.execute()
    expected = df.dup_strings.nunique()
    assert result == expected


def test_value_counts(t, df):
    expr = t.dup_strings.value_counts()
    result = expr.execute()
    expected = df.dup_strings.value_counts().reset_index().rename(
        columns={'dup_strings': 'count'}
    ).rename(
        columns={'index': 'dup_strings'}
    ).sort_values(['dup_strings']).reset_index(drop=True)
    tm.assert_frame_equal(result, expected)


def test_table_count(t, df):
    expr = t.count()
    result = expr.execute()
    expected = len(df)
    assert result == expected


def test_weighted_average(t, df):
    expr = t.groupby(t.dup_strings).aggregate(
        avg=(t.plain_float64 * t.plain_int64).sum() / t.plain_int64.sum()
    )
    result = expr.execute()
    expected = df.groupby('dup_strings').apply(
        lambda df: (
            df.plain_int64 * df.plain_float64
        ).sum() / df.plain_int64.sum()
    ).reset_index().rename(columns={0: 'avg'})
    tm.assert_frame_equal(result, expected)


def test_group_by_multiple_keys(t, df):
    expr = t.groupby([t.dup_strings, t.dup_ints]).aggregate(
        avg_plain_float64=t.plain_float64.mean()
    )
    result = expr.execute()
    expected = df.groupby(['dup_strings', 'dup_ints']).agg(
        {'plain_float64': 'mean'}
    ).reset_index().rename(columns={'plain_float64': 'avg_plain_float64'})
    tm.assert_frame_equal(result, expected)


def test_mutate_after_group_by(t, df):
    gb = t.groupby(t.dup_strings).aggregate(
        avg_plain_float64=t.plain_float64.mean()
    )
    expr = gb.mutate(x=gb.avg_plain_float64)
    result = expr.execute()
    expected = df.groupby('dup_strings').agg(
        {'plain_float64': 'mean'}
    ).reset_index().rename(columns={'plain_float64': 'avg_plain_float64'})
    expected = expected.assign(x=expected.avg_plain_float64)
    tm.assert_frame_equal(result, expected)


def test_groupby_with_unnamed_arithmetic(t, df):
    expr = t.groupby(t.dup_strings).aggregate(
        naive_variance=(
            (t.plain_float64 ** 2).sum() - t.plain_float64.mean() ** 2
        ) / t.plain_float64.count()
    )
    result = expr.execute()
    expected = df.groupby('dup_strings').agg({
        'plain_float64': lambda x: ((x ** 2).sum() - x.mean() ** 2) / x.count()
    }).reset_index().rename(columns={'plain_float64': 'naive_variance'})
    tm.assert_frame_equal(result, expected)
