import numpy as np
import pandas as pd

import pytest
from pytest import param

import ibis
import ibis.common as com


@pytest.mark.parametrize(
    'column', ['string_col', 'double_col', 'date_string_col']
)
def test_distinct_column(alltypes, df, column):
    expr = alltypes[column].distinct()
    result = expr.execute()
    expected = df[column].unique()
    assert set(result) == set(expected)


def skip_if_invalid_operation(expr, valid_operations, con):
    op_type = type(expr.op())
    try:
        con.compile(expr)
    except com.OperationNotDefinedError:
        pytest.skip(
            'Operation {!r} is not defined for clients of type {!r}'.format(
                op_type.__name__, type(con).__name__
            )
        )


@pytest.fixture(scope='function')
def result_func_default(request, con, alltypes, backend, valid_operations):
    func = request.param
    expr = func(alltypes)
    skip_if_invalid_operation(expr, valid_operations, con)
    return func


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
            lambda t: t.string_col.find_in_set(['a']),
            lambda t: t.string_col.str.find('a'),
            id='find_in_set',
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


@pytest.fixture(scope='function')
def result_func_aggs(request, con, alltypes, valid_operations):
    func = request.param
    cond = request.getfixturevalue('ibis_cond')
    expr = func(alltypes, cond(alltypes))
    skip_if_invalid_operation(expr, valid_operations, con)
    return func


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
        # param(
            # lambda t, where: t.double_col.approx_median(),
            # lambda t, where: t.double_col.median(),
            # id='double_col_approx_median',
        # ),
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
        # param(
            # lambda t, where: t.string_col.approx_nunique(),
            # lambda t, where: t.string_col.nunique(),
            # id='string_col_approx_nunique'
        # ),
        # param(
            # lambda t, where: t.string_col.group_concat(','),
            # lambda t, where: ','.join(t.string_col),
            # id='string_col_group_concat'
        # ),
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
