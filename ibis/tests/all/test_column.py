import pytest
from pytest import param

import ibis


@pytest.mark.parametrize(
    'column', ['string_col', 'double_col', 'date_string_col']
)
def test_distinct_column(alltypes, df, column):
    expr = alltypes[column].distinct()
    result = expr.execute()
    expected = df[column].unique()
    assert set(result) == set(expected)


@pytest.fixture(scope='function')
def result_func(request, con, alltypes, backend, valid_operations):
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
        param(
            lambda t: t.string_col.capitalize(),
            lambda t: t.string_col.str.capitalize(),
            id='capitalize',
        ),
        param(
            lambda t: t.date_string_col.substr(2, 3),
            lambda t: t.date_string_col.str[2:2 + 3],
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
    indirect=['result_func'],
)
def test_strings(
    alltypes, df, result_func, expected_func, translator, backend
):
    expr = result_func(alltypes)
    result = expr.execute()
    expected = backend.default_series_rename(expected_func(df))
    backend.assert_series_equal(result, expected)


# def test_aggregations():

            # table.bool_col.count(),
            # d.sum(),
            # d.mean(),
            # d.min(),
            # d.max(),
            # s.approx_nunique(),
            # d.approx_median(),
            # s.group_concat(),

            # d.std(),
            # d.std(how='pop'),
            # d.var(),
            # d.var(how='pop'),

            # table.bool_col.any(),
            # table.bool_col.notany(),
            # -table.bool_col.any(),

            # table.bool_col.all(),
            # table.bool_col.notall(),
            # -table.bool_col.all(),

            # table.bool_col.count(where=cond),
            # d.sum(where=cond),
            # d.mean(where=cond),
            # d.min(where=cond),
            # d.max(where=cond),
            # d.std(where=cond),
            # d.var(where=cond),
    # def test_analytic_functions(self):

        # t = self.alltypes.limit(1000)

        # g = t.group_by('string_col').order_by('double_col')
        # f = t.float_col

        # exprs = [
            # f.lag(),
            # f.lead(),
            # f.rank(),
            # f.dense_rank(),
            # f.percent_rank(),
            # f.ntile(buckets=7),

            # f.first(),
            # f.last(),

            # f.first().over(ibis.window(preceding=10)),
            # f.first().over(ibis.window(following=10)),

            # ibis.row_number(),
            # f.cumsum(),
            # f.cummean(),
            # f.cummin(),
            # f.cummax(),

            # # boolean cumulative reductions
            # (f == 0).cumany(),
            # (f == 0).cumall(),

            # f.sum(),
            # f.mean(),
            # f.min(),
            # f.max()
        # ]
