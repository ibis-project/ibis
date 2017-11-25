import numpy as np

import pytest
from pytest import param


def test_sum(alltypes, df):
    expr = alltypes.double_col.sum()
    result = expr.execute()
    expected = df.double_col.sum()
    np.testing.assert_allclose(result, expected)


def test_distinct_column(alltypes, df):
    expr = alltypes.string_col.distinct()
    result = expr.execute()
    expected = df.string_col.unique()
    assert set(result) == set(expected)


@pytest.fixture(scope='function')
def result_func(request, con, alltypes, valid_operations):
    func = request.param
    result = func(alltypes)
    op_type = type(result.op())
    if op_type not in valid_operations:
        pytest.skip(
            'Operation {!r} is not defined for clients of type {!r}'.format(
                op_type.__name__, type(con).__name__
            )
        )
    return func


@pytest.mark.parametrize(
    ('result_func', 'expected_func'),
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
            lambda t: t.string_col.re_search(r'[\d]+'),
            lambda t: t.string_col.str.contains(r'[\d]+'),
            id='re_search',
        ),
        param(
            lambda t: t.string_col.re_extract(r'([\d]+)', 0),
            lambda t: t.string_col.str.extract(r'([\d]+)', expand=False),
            id='re_extract',
        ),
        param(
            lambda t: t.string_col.re_replace(r'[\d]+', 'a'),
            lambda t: t.string_col.str.replace(r'[\d]+', 'a'),
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
            lambda t: t.string_col.str.find('a').replace({-1: None}),
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
    ],
    indirect=['result_func'],
)
def test_strings(
    alltypes, df, result_func, expected_func, translator, backend
):
    expr = result_func(alltypes)
    result = expr.execute()
    expected = backend.default_series_rename(expected_func(df))
    backend.assert_series_equal(result, expected)
