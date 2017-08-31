import operator
import pytest

import numpy as np

import pandas as pd
import pandas.util.testing as tm

import ibis
import ibis.expr.datatypes as dt
from ibis.common import IbisTypeError

pytest.importorskip('multipledispatch')

pytestmark = pytest.mark.pandas


def test_table_column(t, df):
    expr = t.plain_int64
    result = expr.execute()
    expected = df.plain_int64
    tm.assert_series_equal(result, expected)


def test_literal(client):
    assert client.execute(ibis.literal(1)) == 1


def test_read_with_undiscoverable_type(client):
    with pytest.raises(TypeError):
        client.table('df')


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
    tm.assert_frame_equal(result[expected.columns], expected)


@pytest.mark.parametrize('how', ['inner', 'left', 'outer'])
def test_join_project_left_table(how, left, right, df1, df2):
    expr = left.join(right, left.key == right.key, how=how)[left, right.key3]
    result = expr.execute()
    expected = pd.merge(df1, df2, how=how, on='key')[
        list(left.columns) + ['key3']
    ]
    tm.assert_frame_equal(result[expected.columns], expected)


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
    tm.assert_frame_equal(result[expected.columns], expected)


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
    tm.assert_frame_equal(result[expected.columns], expected)


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
    tm.assert_frame_equal(result[expected.columns], expected)


@pytest.mark.parametrize('how', ['inner', 'left', 'outer'])
def test_join_with_post_expression_selection(how, left, right, df1, df2):
    join = left.join(right, left.key == right.key, how=how)
    expr = join[left.key, left.value, right.other_value]
    result = expr.execute()
    expected = pd.merge(df1, df2, on='key', how=how)[[
        'key', 'value', 'other_value'
    ]]
    tm.assert_frame_equal(result[expected.columns], expected)


def test_selection(t, df):
    expr = t[
        ((t.plain_strings == 'a') | (t.plain_int64 == 3)) &
        (t.dup_strings == 'd')
    ]
    result = expr.execute()
    expected = df[
        ((df.plain_strings == 'a') | (df.plain_int64 == 3)) &
        (df.dup_strings == 'd')
    ].reset_index(drop=True)
    tm.assert_frame_equal(result[expected.columns], expected)


def test_mutate(t, df):
    expr = t.mutate(x=t.plain_int64 + 1, y=t.plain_int64 * 2)
    result = expr.execute()
    expected = df.assign(x=df.plain_int64 + 1, y=df.plain_int64 * 2)
    tm.assert_frame_equal(result[expected.columns], expected)


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
    result['avg_plain_int64'] = result.avg_plain_int64.astype('float64')
    tm.assert_frame_equal(result[expected.columns], expected)


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
    tm.assert_frame_equal(result[expected.columns], expected)


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

    tm.assert_frame_equal(result[expected.columns], expected)


def test_group_by_rename_key(t, df):
    expr = t.groupby(t.dup_strings.name('foo')).aggregate(
        dup_string_count=t.dup_strings.count()
    )

    assert 'foo' in expr.schema()
    result = expr.execute()
    assert 'foo' in result.columns

    expected = df.groupby('dup_strings').dup_strings.count().rename(
        'dup_string_count'
    ).reset_index().rename(columns={'dup_strings': 'foo'})
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


def test_group_concat(t, df):
    expr = t.groupby(t.dup_strings).aggregate(
        foo=t.plain_int64.group_concat(',')
    )
    result = expr.execute()
    expected = df.groupby('dup_strings').apply(
        lambda df: ','.join(df.plain_int64.astype(str))
    ).reset_index().rename(columns={0: 'foo'})
    tm.assert_frame_equal(result[expected.columns], expected)


@pytest.mark.parametrize('offset', [0, 2])
def test_frame_limit(t, df, offset):
    n = 5
    df_expr = t.limit(n, offset=offset)
    result = df_expr.execute()
    expected = df.iloc[offset:offset + n]
    tm.assert_frame_equal(result[expected.columns], expected)


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
        (lambda t, col: [ibis.desc(t[col])], lambda col: [col], False),
        (
            lambda t, col: [t[col], ibis.desc(t.plain_int64)],
            lambda col: [col, 'plain_int64'],
            [True, False]
        ),
        (
            lambda t, col: [ibis.desc(t.plain_int64 * 2)],
            lambda col: ['plain_int64'],
            False,
        ),
    ]
)
@pytest.mark.parametrize(
    'column',
    [
        'plain_datetimes_naive',
        'plain_datetimes_ny',
        'plain_datetimes_utc',
    ]
)
def test_sort_by(t, df, column, key, pandas_by, pandas_ascending):
    expr = t.sort_by(key(t, column))
    result = expr.execute()
    expected = df.sort_values(
        pandas_by(column), ascending=pandas_ascending
    ).reset_index(drop=True)
    tm.assert_frame_equal(result[expected.columns], expected)


def test_complex_sort_by(t, df):
    expr = t.sort_by([
        ibis.desc(t.plain_int64 * t.plain_float64), t.plain_float64
    ])
    result = expr.execute()
    expected = df.assign(
        foo=df.plain_int64 * df.plain_float64
    ).sort_values(['foo', 'plain_float64'], ascending=[False, True]).drop(
        ['foo'], axis=1
    ).reset_index(drop=True)

    tm.assert_frame_equal(result[expected.columns], expected)


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
    tm.assert_frame_equal(result[expected.columns], expected)


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
    tm.assert_frame_equal(result[expected.columns], expected)


def test_group_by_multiple_keys(t, df):
    expr = t.groupby([t.dup_strings, t.dup_ints]).aggregate(
        avg_plain_float64=t.plain_float64.mean()
    )
    result = expr.execute()
    expected = df.groupby(['dup_strings', 'dup_ints']).agg(
        {'plain_float64': 'mean'}
    ).reset_index().rename(columns={'plain_float64': 'avg_plain_float64'})
    tm.assert_frame_equal(result[expected.columns], expected)


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
    tm.assert_frame_equal(result[expected.columns], expected)


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
    tm.assert_frame_equal(result[expected.columns], expected)


def test_isnull(t, df):
    expr = t.strings_with_nulls.isnull()
    result = expr.execute()
    expected = df.strings_with_nulls.isnull()
    tm.assert_series_equal(result, expected)


def test_notnull(t, df):
    expr = t.strings_with_nulls.notnull()
    result = expr.execute()
    expected = df.strings_with_nulls.notnull()
    tm.assert_series_equal(result, expected)


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


@pytest.mark.parametrize('catop', [lambda x, y: x + y, lambda x, y: y + x])
def test_array_concat(t, df, catop):
    x = t.array_of_float64.cast('array<string>')
    y = t.array_of_strings
    expr = t[(x + y).name('catted')].catted
    result = expr.execute()
    expected = pd.DataFrame({
        'catted': (
            df.array_of_float64.apply(lambda x: list(map(str, x))) +
            df.array_of_strings
        )
    }).catted
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


@pytest.mark.parametrize('raw_value', [0.0, 1.0])
def test_scalar_parameter(t, df, raw_value):
    value = ibis.param(dt.double)
    expr = t.float64_with_zeros == value
    result = expr.execute(params={value: raw_value})
    expected = df.float64_with_zeros == raw_value
    tm.assert_series_equal(result, expected)
