import numpy as np
import pandas as pd

import pytest
from pytest import param

import ibis
import ibis.common as com


def skip_if_invalid_operation(expr, valid_operations, con):
    try:
        con.compile(expr)
    except com.OperationNotDefinedError as e:
        pytest.skip('{} with client {}'.format(e, type(con).__name__))


@pytest.fixture(scope='function')
def result_func_default(request, con, alltypes, backend, valid_operations):
    func = request.param
    expr = func(alltypes)
    skip_if_invalid_operation(expr, valid_operations, con)
    return func


@pytest.fixture(scope='function')
def result_func_aggs(request, con, alltypes, valid_operations):
    func = request.param
    cond = request.getfixturevalue('ibis_cond')
    expr = func(alltypes, cond(alltypes))
    skip_if_invalid_operation(expr, valid_operations, con)
    return func


@pytest.fixture(scope='function')
def result_func_default_analytic(
    request, con, analytic_alltypes, backend, valid_operations
):
    func = request.param
    expr = analytic_alltypes.mutate(value=func)
    skip_if_invalid_operation(expr, valid_operations, con)
    return func


@pytest.mark.parametrize(
    'column', ['string_col', 'double_col', 'date_string_col']
)
def test_distinct_column(alltypes, df, column):
    expr = alltypes[column].distinct()
    result = expr.execute()
    expected = df[column].unique()
    assert set(result) == set(expected)


@pytest.mark.parametrize(
    ('result_func_default', 'expected_func'),
    [
        param(
            lambda t: t.string_col.contains('6'),
            lambda t: t.string_col.str.contains('6'),
            id='contains',
        ),
        param(
            lambda t: t.string_col.like('6%'),
            lambda t: t.string_col.str.contains('6.*'),
            id='like',
        ),
        param(
            lambda t: t.string_col.re_search(r'[[:digit:]]+'),
            lambda t: t.string_col.str.contains(r'\d+'),
            id='re_search',
        ),
        param(
            lambda t: t.string_col.re_extract(r'([[:digit:]]+)', 0),
            lambda t: t.string_col.str.extract(r'(\d+)', expand=False),
            id='re_extract',
        ),
        param(
            lambda t: t.string_col.re_replace(r'[[:digit:]]+', 'a'),
            lambda t: t.string_col.str.replace(r'\d+', 'a'),
            id='re_replace',
        ),
        param(
            lambda t: t.string_col.repeat(2),
            lambda t: t.string_col * 2,
            id='repeat'
        ),
        param(
            lambda t: t.string_col.translate('a', 'b'),
            lambda t: t.string_col.str.translate(dict(a='b')),
            id='translate',
        ),
        param(
            lambda t: t.string_col.find('a'),
            lambda t: t.string_col.str.find('a'),
            id='find'
        ),
        param(
            lambda t: t.string_col.lpad(10, 'a'),
            lambda t: t.string_col.str.pad(10, fillchar='a', side='left'),
            id='lpad'
        ),
        param(
            lambda t: t.string_col.rpad(10, 'a'),
            lambda t: t.string_col.str.pad(10, fillchar='a', side='right'),
            id='rpad',
        ),
        param(
            lambda t: t.string_col.find_in_set(['1']),
            lambda t: t.string_col.str.find('1'),
            id='find_in_set',
        ),
        param(
            lambda t: t.string_col.find_in_set(['a']),
            lambda t: t.string_col.str.find('a'),
            id='find_in_set_all_missing',
        ),
        param(
            lambda t: t.string_col.lower(),
            lambda t: t.string_col.str.lower(),
            id='lower'
        ),
        param(
            lambda t: t.string_col.upper(),
            lambda t: t.string_col.str.upper(),
            id='upper'
        ),
        param(
            lambda t: t.string_col.reverse(),
            lambda t: t.string_col.str[::-1],
            id='reverse'
        ),
        param(
            lambda t: t.string_col.ascii_str(),
            lambda t: t.string_col.map(ord),
            id='ascii_str'
        ),
        param(
            lambda t: t.string_col.length(),
            lambda t: t.string_col.str.len(),
            id='length'
        ),
        param(
            lambda t: t.string_col.strip(),
            lambda t: t.string_col.str.strip(),
            id='strip'
        ),
        param(
            lambda t: t.string_col.lstrip(),
            lambda t: t.string_col.str.lstrip(),
            id='lstrip'
        ),
        param(
            lambda t: t.string_col.rstrip(),
            lambda t: t.string_col.str.rstrip(),
            id='rstrip'
        ),
        param(
            lambda t: t.string_col.capitalize(),
            lambda t: t.string_col.str.capitalize(),
            id='capitalize',
        ),
        param(
            lambda t: t.date_string_col.substr(2, 3),
            lambda t: t.date_string_col.str[2:5],
            id='substr'
        ),
        param(
            lambda t: t.date_string_col.left(2),
            lambda t: t.date_string_col.str[:2],
            id='left',
        ),
        param(
            lambda t: t.date_string_col.right(2),
            lambda t: t.date_string_col.str[-2:],
            id='right',
        ),
        param(
            lambda t: t.date_string_col.split('/'),
            lambda t: t.date_string_col.str.split('/'),
            id='split',
        ),
        param(
            lambda t: ibis.literal('-').join(['a', t.string_col, 'c']),
            lambda t: 'a-' + t.string_col + '-c',
            id='join'
        )
    ],
    indirect=['result_func_default'],
)
def test_strings(alltypes, df, backend, result_func_default, expected_func):
    expr = result_func_default(alltypes)
    result = expr.execute()
    expected = backend.default_series_rename(expected_func(df))
    backend.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ('result_func_aggs', 'expected_func'),
    [
        param(
            lambda t, where: t.bool_col.count(where=where),
            lambda t, where: len(t.bool_col[where].dropna()),
            id='bool_col_count'
        ),
        param(
            lambda t, where: t.bool_col.any(),
            lambda t, where: t.bool_col.any(),
            id='bool_col_any'
        ),
        param(
            lambda t, where: t.bool_col.notany(),
            lambda t, where: ~t.bool_col.any(),
            id='bool_col_notany'
        ),
        param(
            lambda t, where: -t.bool_col.any(),
            lambda t, where: ~t.bool_col.any(),
            id='bool_col_any_negate'
        ),
        param(
            lambda t, where: t.bool_col.all(),
            lambda t, where: t.bool_col.all(),
            id='bool_col_all'
        ),
        param(
            lambda t, where: t.bool_col.notall(),
            lambda t, where: ~t.bool_col.all(),
            id='bool_col_notall'
        ),
        param(
            lambda t, where: -t.bool_col.all(),
            lambda t, where: ~t.bool_col.all(),
            id='bool_col_all_negate'
        ),
        param(
            lambda t, where: t.double_col.sum(),
            lambda t, where: t.double_col.sum(),
            id='double_col_sum',
        ),
        param(
            lambda t, where: t.double_col.mean(),
            lambda t, where: t.double_col.mean(),
            id='double_col_mean',
        ),
        param(
            lambda t, where: t.double_col.min(),
            lambda t, where: t.double_col.min(),
            id='double_col_min',
        ),
        param(
            lambda t, where: t.double_col.max(),
            lambda t, where: t.double_col.max(),
            id='double_col_max',
        ),
        param(
            lambda t, where: t.double_col.approx_median(),
            lambda t, where: t.double_col.median(),
            id='double_col_approx_median',
            marks=pytest.mark.xfail,
        ),
        param(
            lambda t, where: t.double_col.std(how='sample'),
            lambda t, where: t.double_col.std(ddof=1),
            id='double_col_std',
        ),
        param(
            lambda t, where: t.double_col.var(how='sample'),
            lambda t, where: t.double_col.var(ddof=1),
            id='double_col_var',
        ),
        param(
            lambda t, where: t.double_col.std(how='pop'),
            lambda t, where: t.double_col.std(ddof=0),
            id='double_col_std_pop',
        ),
        param(
            lambda t, where: t.double_col.var(how='pop'),
            lambda t, where: t.double_col.var(ddof=0),
            id='double_col_var_pop',
        ),
        param(
            lambda t, where: t.string_col.approx_nunique(),
            lambda t, where: t.string_col.nunique(),
            id='string_col_approx_nunique',
            marks=pytest.mark.xfail,
        ),
        param(
            lambda t, where: t.string_col.group_concat(','),
            lambda t, where: ','.join(t.string_col),
            id='string_col_group_concat',
            marks=pytest.mark.xfail,
        ),
    ],
    indirect=['result_func_aggs'],
)
@pytest.mark.parametrize(
    ('ibis_cond', 'pandas_cond'),
    [
        (lambda t: None, lambda t: slice(None)),
        (
            lambda t: t.string_col.isin(['1', '7']),
            lambda t: t.string_col.isin(['1', '7']),
        )
    ]
)
def test_aggregations(
    alltypes, df, backend, result_func_aggs, expected_func,
    ibis_cond, pandas_cond
):
    expr = result_func_aggs(alltypes, ibis_cond(alltypes))
    result = expr.execute()
    expected = expected_func(df, pandas_cond(df))
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    ('result_func_default_analytic', 'expected_func'),
    [
        param(
            lambda t: t.float_col.lag(),
            lambda t: t.float_col.shift(1),
            id='lag',
            marks=pytest.mark.xfail,
        ),
        param(
            lambda t: t.float_col.lead(),
            lambda t: t.float_col.shift(-1),
            id='lead',
            marks=pytest.mark.xfail,
        ),
        param(
            lambda t: t.float_col.rank(),
            lambda t: t,
            id='rank',
            marks=pytest.mark.xfail,
        ),
        param(
            lambda t: t.float_col.dense_rank(),
            lambda t: t,
            id='dense_rank',
            marks=pytest.mark.xfail,
        ),
        param(
            lambda t: t.float_col.percent_rank(),
            lambda t: t,
            id='percent_rank',
            marks=pytest.mark.xfail,
        ),
        param(
            lambda t: t.float_col.ntile(buckets=7),
            lambda t: t,
            id='ntile',
            marks=pytest.mark.xfail,
        ),
        param(
            lambda t: t.float_col.first(),
            lambda t: t.float_col.head(1),
            id='first',
            marks=pytest.mark.xfail,
        ),
        param(
            lambda t: t.float_col.last(),
            lambda t: t.float_col.tail(1),
            id='last',
            marks=pytest.mark.xfail,
        ),
        param(
            lambda t: t.float_col.first().over(ibis.window(preceding=10)),
            lambda t: t,
            id='first_preceding',
            marks=pytest.mark.xfail,
        ),
        param(
            lambda t: t.float_col.first().over(ibis.window(following=10)),
            lambda t: t,
            id='first_following',
            marks=pytest.mark.xfail,
        ),
        param(
            lambda t: ibis.row_number(),
            lambda t: pd.Series(np.arange(len(t))),
            id='row_number',
            marks=pytest.mark.xfail,
        ),
        param(
            lambda t: t.double_col.cumsum(),
            lambda t: t.double_col.cumsum(),
            id='cumsum',
            marks=pytest.mark.xfail,
        ),
        param(
            lambda t: t.double_col.cummean(),
            lambda t: t.double_col.expanding().mean().reset_index(
                drop=True, level=0
            ),
            id='cummean',
            marks=pytest.mark.xfail,
        ),
        param(
            lambda t: t.float_col.cummin(),
            lambda t: t.float_col.cummin(),
            id='cummin',
            marks=pytest.mark.xfail,
        ),
        param(
            lambda t: t.float_col.cummax(),
            lambda t: t.float_col.cummax(),
            id='cummax',
            marks=pytest.mark.xfail,
        ),
        param(
            lambda t: (t.double_col == 0).cumany(),
            lambda t: t.double_col.expanding().agg(
                lambda s: (s == 0).any()
            ).reset_index(drop=True, level=0).astype(bool),
            id='cumany',
            marks=pytest.mark.xfail,
        ),
        param(
            lambda t: (t.double_col == 0).cumall(),
            lambda t: t.double_col.expanding().agg(
                lambda s: (s == 0).all()
            ).reset_index(drop=True, level=0).astype(bool),
            id='cumall',
            marks=pytest.mark.xfail,
        ),
        param(
            lambda t: t.double_col.sum(),
            lambda gb: gb.double_col.transform('sum'),
            id='sum',
            marks=pytest.mark.xfail,
        ),
        param(
            lambda t: t.double_col.mean(),
            lambda gb: gb.double_col.transform('mean'),
            id='mean',
            marks=pytest.mark.xfail,
        ),
        param(
            lambda t: t.float_col.min(),
            lambda gb: gb.float_col.transform('min'),
            id='min',
            marks=pytest.mark.xfail,
        ),
        param(
            lambda t: t.float_col.max(),
            lambda gb: gb.float_col.transform('max'),
            id='max',
            marks=pytest.mark.xfail,
        ),
    ],
    indirect=['result_func_default_analytic'],
)
def test_analytic_functions(
    analytic_alltypes,
    df, con, backend, result_func_default_analytic, expected_func,
):
    if not backend.supports_window_operations:
        pytest.skip(
            'Backend {} does not support window operations'.format(backend)
        )
    expr = analytic_alltypes.mutate(value=result_func_default_analytic)

    try:
        raw_result = con.execute(expr)
    except com.OperationNotDefinedError as e:
        pytest.skip(str(e))

    result = raw_result.set_index('id').sort_index()

    gb = df.sort_values('id').groupby('string_col')
    expected = df.assign(value=expected_func(gb)).set_index('id').sort_index()
    left, right = result.value, expected.value
    backend.assert_series_equal(left, right)
