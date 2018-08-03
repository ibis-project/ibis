import operator

from operator import methodcaller

import pytest

import numpy as np
import numpy.testing as npt

import pandas as pd
import pandas.util.testing as tm

import ibis
import ibis.expr.datatypes as dt

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


def test_project_scope_does_not_override(t, df):
    col = t.plain_int64
    expr = t[[
        col.name('new_col'),
        col.sum().over(ibis.window(group_by='dup_strings')).name('grouped')
    ]]
    result = expr.execute()
    expected = pd.concat(
        [
            df[['plain_int64', 'dup_strings']].rename(
                columns={'plain_int64': 'new_col'}),
            df.groupby('dup_strings').plain_int64.transform('sum').reset_index(
                drop=True).rename('grouped')
        ],
        axis=1
    )[['new_col', 'grouped']]
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    'where',
    [
        lambda t: None,
        lambda t: t.dup_strings == 'd',
        lambda t: (t.dup_strings == 'd') | (t.plain_int64 < 100),
    ]
)
@pytest.mark.parametrize(
    ('ibis_func', 'pandas_func'),
    [
        (methodcaller('abs'), np.abs),
        (methodcaller('ceil'), np.ceil),
        (methodcaller('exp'), np.exp),
        (methodcaller('floor'), np.floor),
        (methodcaller('ln'), np.log),
        (methodcaller('log10'), np.log10),
        (methodcaller('log', 2), lambda x: np.log(x) / np.log(2)),
        (methodcaller('log2'), np.log2),
        (methodcaller('round', 0), lambda x: x.round(0).astype('int64')),
        (methodcaller('round', -2), methodcaller('round', -2)),
        (methodcaller('round', 2), methodcaller('round', 2)),
        (methodcaller('round'), lambda x: x.round().astype('int64')),
        (methodcaller('sign'), np.sign),
        (methodcaller('sqrt'), np.sqrt),
    ]
)
def test_aggregation_group_by(t, df, where, ibis_func, pandas_func):
    ibis_where = where(t)
    expr = t.group_by(t.dup_strings).aggregate(
        avg_plain_int64=t.plain_int64.mean(where=ibis_where),
        sum_plain_float64=t.plain_float64.sum(where=ibis_where),
        mean_float64_positive=ibis_func(
            t.float64_positive
        ).mean(where=ibis_where),
        neg_mean_int64_with_zeros=(-t.int64_with_zeros).mean(where=ibis_where),
        nunique_dup_ints=t.dup_ints.nunique(),
    )
    result = expr.execute()

    pandas_where = where(df)
    mask = slice(None) if pandas_where is None else pandas_where
    expected = df.groupby('dup_strings').agg({
        'plain_int64': lambda x, mask=mask: x[mask].mean(),
        'plain_float64': lambda x, mask=mask: x[mask].sum(),
        'dup_ints': 'nunique',
        'float64_positive': (
            lambda x, mask=mask, func=pandas_func: func(x[mask]).mean()
        ),
        'int64_with_zeros': lambda x, mask=mask: (-x[mask]).mean(),
    }).reset_index().rename(
        columns={
            'plain_int64': 'avg_plain_int64',
            'plain_float64': 'sum_plain_float64',
            'dup_ints': 'nunique_dup_ints',
            'float64_positive': 'mean_float64_positive',
            'int64_with_zeros': 'neg_mean_int64_with_zeros',
        }
    )
    # TODO(phillipc): Why does pandas not return floating point values here?
    expected['avg_plain_int64'] = expected.avg_plain_int64.astype('float64')
    result['avg_plain_int64'] = result.avg_plain_int64.astype('float64')
    expected['neg_mean_int64_with_zeros'] = (
        expected.neg_mean_int64_with_zeros.astype('float64')
    )
    result['neg_mean_int64_with_zeros'] = (
        result.neg_mean_int64_with_zeros.astype('float64')
    )
    expected['mean_float64_positive'] = (
        expected.mean_float64_positive.astype('float64'))
    result['mean_float64_positive'] = result.mean_float64_positive.astype(
        'float64')
    lhs = result[expected.columns]
    rhs = expected
    tm.assert_frame_equal(lhs, rhs)


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


def test_group_by_with_having(t, df):
    expr = t.group_by(t.dup_strings).having(
        t.plain_float64.sum() == 5
    ).aggregate(
        avg_a=t.plain_int64.mean(),
        sum_c=t.plain_float64.sum(),
    )
    result = expr.execute()

    expected = df.groupby('dup_strings').agg({
        'plain_int64': 'mean',
        'plain_float64': 'sum',
    }).reset_index().rename(columns={
        'plain_int64': 'avg_a',
        'plain_float64': 'sum_c'
    })
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
    expected = df.iloc[offset:offset + n].reset_index(drop=True)
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


@pytest.mark.parametrize('raw_value', [0.0, 1.0])
def test_scalar_parameter(t, df, raw_value):
    value = ibis.param(dt.double)
    expr = t.float64_with_zeros == value
    result = expr.execute(params={value: raw_value})
    expected = df.float64_with_zeros == raw_value
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize('elements', [[1], (1,), {1}, frozenset({1})])
def test_isin(t, df, elements):
    expr = t.plain_float64.isin(elements)
    expected = df.plain_float64.isin(elements)
    result = expr.execute()
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize('elements', [[1], (1,), {1}, frozenset({1})])
def test_notin(t, df, elements):
    expr = t.plain_float64.notin(elements)
    expected = ~df.plain_float64.isin(elements)
    result = expr.execute()
    tm.assert_series_equal(result, expected)


def test_cast_on_group_by(t, df):
    expr = t.groupby(t.dup_strings).aggregate(
        casted=(t.float64_with_zeros == 0).cast('int64').sum()
    )
    result = expr.execute()
    expected = df.groupby('dup_strings').float64_with_zeros.apply(
        lambda s: (s == 0).astype('int64').sum()
    ).reset_index().rename(columns={'float64_with_zeros': 'casted'})
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    'op',
    [
        operator.add,
        operator.mul,
        operator.sub,
        operator.truediv,
        operator.floordiv,
        operator.mod,
        operator.pow,
    ],
    ids=operator.attrgetter('__name__'),
)
@pytest.mark.parametrize('args', [lambda c: (1.0, c), lambda c: (c, 1.0)])
def test_left_binary_op(t, df, op, args):
    expr = op(*args(t.float64_with_zeros))
    result = expr.execute()
    expected = op(*args(df.float64_with_zeros))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    'op',
    [
        operator.add,
        operator.mul,
        operator.sub,
        operator.truediv,
        operator.floordiv,
        operator.mod,
        operator.pow,
    ],
    ids=operator.attrgetter('__name__'),
)
@pytest.mark.parametrize('args', [lambda c: (1.0, c), lambda c: (c, 1.0)])
def test_left_binary_op_gb(t, df, op, args):
    expr = t.groupby('dup_strings').aggregate(
        foo=op(*args(t.float64_with_zeros)).sum()
    )
    result = expr.execute()
    expected = df.groupby('dup_strings').float64_with_zeros.apply(
        lambda s: op(*args(s)).sum()
    ).reset_index().rename(columns={'float64_with_zeros': 'foo'})
    tm.assert_frame_equal(result, expected)


def test_where_series(t, df):
    col_expr = t['plain_int64']
    result = ibis.where(col_expr > col_expr.mean(), col_expr, 0.0).execute()

    ser = df['plain_int64']
    expected = ser.where(ser > ser.mean(), other=0.0)

    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ('cond', 'expected_func'),
    [
        (True, lambda df: df['plain_int64']),
        (False, lambda df: pd.Series(np.repeat(3.0, len(df))))
    ]
)
def test_where_scalar(t, df, cond, expected_func):
    expr = ibis.where(cond, t['plain_int64'], 3.0)
    result = expr.execute()
    expected = expected_func(df)
    tm.assert_series_equal(result, expected)


def test_where_long(batting, batting_df):
    col_expr = batting['AB']
    result = ibis.where(col_expr > col_expr.mean(), col_expr, 0.0).execute()

    ser = batting_df['AB']
    expected = ser.where(ser > ser.mean(), other=0.0)

    tm.assert_series_equal(result, expected)


def test_round(t, df):
    precision = 2
    mult = 3.33333
    result = (t.count() * mult).round(precision).execute()
    expected = np.around(len(df) * mult, precision)
    npt.assert_almost_equal(result, expected, decimal=precision)


def test_quantile_groupby(batting, batting_df):
    def q_fun(x, quantile, interpolation):
        res = x.quantile(quantile, interpolation=interpolation).tolist()
        return [res for _ in range(len(x))]

    frac = 0.2
    intp = 'linear'
    result = (batting
              .groupby('teamID')
              .mutate(res=lambda x: x.RBI.quantile([frac, 1 - frac], intp))
              .res
              .execute())
    expected = (batting_df
                .groupby('teamID')
                .RBI
                .transform(q_fun, quantile=[frac, 1 - frac],
                           interpolation=intp)
                .rename('res'))
    tm.assert_series_equal(result, expected)


def test_summary_numeric(batting, batting_df):
    expr = batting.G.summary()
    result = expr.execute()
    assert len(result) == 1

    G = batting_df.G
    expected = dict(
        count=G.count(),
        nulls=G.isnull().sum(),
        min=G.min(),
        max=G.max(),
        sum=G.sum(),
        mean=G.mean(),
        approx_nunique=G.nunique(),
    )
    assert dict(result.iloc[0]) == expected


def test_summary_numeric_group_by(batting, batting_df):
    expr = batting.groupby('teamID').G.summary()
    result = expr.execute()
    expected = batting_df.groupby('teamID').G.apply(
        lambda s: pd.DataFrame(
            dict(
                count=s.count(),
                nulls=s.isnull().sum(),
                min=s.min(),
                max=s.max(),
                sum=s.sum(),
                mean=s.mean(),
                approx_nunique=s.nunique()
            ),
            index=[0]
        )
    ).reset_index(level=1, drop=True).reset_index()
    columns = expected.columns

    # TODO: fix isnull().sum() in the pandas backend: the type is incorrect
    tm.assert_frame_equal(result[columns], expected, check_dtype=False)


def test_summary_non_numeric(batting, batting_df):
    expr = batting.teamID.summary()
    result = expr.execute()
    assert len(result) == 1
    assert len(result.columns) == 3
    expected = dict(
        count=batting_df.teamID.count(),
        nulls=batting_df.teamID.isnull().sum(),
        uniques=batting_df.teamID.nunique(),
    )
    assert dict(result.iloc[0]) == expected


def test_summary_non_numeric_group_by(batting, batting_df):
    expr = batting.groupby('teamID').playerID.summary()
    result = expr.execute()
    expected = batting_df.groupby('teamID').playerID.apply(
        lambda s: pd.DataFrame(
            dict(
                count=s.count(),
                nulls=s.isnull().sum(),
                uniques=s.nunique()
            ),
            index=[0]
        )
    ).reset_index(level=1, drop=True).reset_index()
    columns = expected.columns
    tm.assert_frame_equal(result[columns], expected, check_dtype=False)


def test_searched_case_scalar(client):
    expr = ibis.case().when(True, 1).when(False, 2).end()
    result = client.execute(expr)
    expected = np.int8(1)
    assert result == expected


def test_searched_case_column(batting, batting_df):
    t = batting
    df = batting_df
    expr = (
        ibis.case()
            .when(t.RBI < 5, 'really bad team')
            .when(t.teamID == 'PH1', 'ph1 team')
            .else_(t.teamID)
            .end()
    )
    result = expr.execute()
    expected = pd.Series(
        np.select(
            [df.RBI < 5, df.teamID == 'PH1'],
            ['really bad team', 'ph1 team'],
            df.teamID
        )
    )
    tm.assert_series_equal(result, expected)


def test_simple_case_scalar(client):
    x = ibis.literal(2)
    expr = x.case().when(2, x - 1).when(3, x + 1).when(4, x + 2).end()
    result = client.execute(expr)
    expected = np.int8(1)
    assert result == expected


def test_simple_case_column(batting, batting_df):
    t = batting
    df = batting_df
    expr = (
        t.RBI.case()
             .when(5, 'five')
             .when(4, 'four')
             .when(3, 'three')
             .else_('could be good?')
             .end()
    )
    result = expr.execute()
    expected = pd.Series(
        np.select(
            [df.RBI == 5, df.RBI == 4, df.RBI == 3],
            ['five', 'four', 'three'],
            'could be good?'
        )
    )
    tm.assert_series_equal(result, expected)
