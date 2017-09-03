import pytest
import numpy as np

from operator import methodcaller

from ibis import literal as L


@pytest.mark.parametrize(('reduction', 'func_translated'), [
    ('sum', 'sum'),
    ('count', 'count'),
    ('mean', 'avg'),
    ('max', 'max'),
    ('min', 'min'),
    ('std', 'stddevSamp'),
    ('var', 'varSamp')
])
def test_reduction_where(con, alltypes, translate, reduction, func_translated):
    template = '{0}If(`double_col`, `bigint_col` < 70)'
    expected = template.format(func_translated)

    method = getattr(alltypes.double_col, reduction)
    cond = alltypes.bigint_col < 70
    expr = method(where=cond)

    assert translate(expr) == expected
    assert isinstance(con.execute(expr), (np.float, np.uint))


def test_std_var_pop(con, alltypes, translate):
    cond = alltypes.bigint_col < 70
    expr1 = alltypes.double_col.std(where=cond, how='pop')
    expr2 = alltypes.double_col.var(where=cond, how='pop')

    assert translate(expr1) == 'stddevPopIf(`double_col`, `bigint_col` < 70)'
    assert translate(expr2) == 'varPopIf(`double_col`, `bigint_col` < 70)'
    assert isinstance(con.execute(expr1), np.float)
    assert isinstance(con.execute(expr2), np.float)


@pytest.mark.parametrize('reduction', [
    'sum',
    'count',
    'max',
    'min'
])
def test_reduction_invalid_where(con, alltypes, reduction):
    condbad_literal = L('T')

    with pytest.raises(TypeError):
        fn = methodcaller(reduction, where=condbad_literal)
        fn(alltypes.double_col)


# @pytest.mark.parametrize(
#     ('func', 'pandas_func'),
#     [
#         # tier and histogram
#         (
#             lambda d: d.bucket([0, 10, 25, 50, 100]),
#             lambda s: pd.cut(
#                 s, [0, 10, 25, 50, 100], right=False, labels=False,
#             )
#         ),
#         (
#             lambda d: d.bucket([0, 10, 25, 50], include_over=True),
#             lambda s: pd.cut(
#                 s, [0, 10, 25, 50, np.inf], right=False, labels=False
#             )
#         ),
#         (
#             lambda d: d.bucket([0, 10, 25, 50], close_extreme=False),
#             lambda s: pd.cut(s, [0, 10, 25, 50], right=False, labels=False),
#         ),
#         (
#             lambda d: d.bucket(
#                 [0, 10, 25, 50], closed='right', close_extreme=False
#             ),
#             lambda s: pd.cut(
#                 s, [0, 10, 25, 50],
#                 include_lowest=False,
#                 right=True,
#                 labels=False,
#             )
#         ),
#         (
#             lambda d: d.bucket([10, 25, 50, 100], include_under=True),
#             lambda s: pd.cut(
#                 s, [0, 10, 25, 50, 100], right=False, labels=False
#             ),
#         ),
#     ]
# )
# def test_bucket(alltypes, df, func, pandas_func):
#     expr = func(alltypes.double_col)
#     result = expr.execute()
#     expected = pandas_func(df.double_col)
#     tm.assert_series_equal(result, expected, check_names=False)


# def test_category_label(alltypes, df):
#     t = alltypes
#     d = t.double_col

#     bins = [0, 10, 25, 50, 100]
#     labels = ['a', 'b', 'c', 'd']
#     bucket = d.bucket(bins)
#     expr = bucket.label(labels)
#     result = expr.execute().astype('category', ordered=True)
#     result.name = 'double_col'

#     expected = pd.cut(df.double_col, bins, labels=labels, right=False)

#     tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(('func', 'pandas_func'), [
    (
        lambda t, cond: t.bool_col.count(),
        lambda df, cond: df.bool_col.count(),
    ),
    # (
    #     lambda t, cond: t.bool_col.nunique(),
    #     lambda df, cond: df.bool_col.nunique(),
    # ),
    (
        lambda t, cond: t.bool_col.approx_nunique(),
        lambda df, cond: df.bool_col.nunique(),
    ),
    # group_concat
    # (
    #     lambda t, cond: t.bool_col.any(),
    #     lambda df, cond: df.bool_col.any(),
    # ),
    # (
    #     lambda t, cond: t.bool_col.all(),
    #     lambda df, cond: df.bool_col.all(),
    # ),
    # (
    #     lambda t, cond: t.bool_col.notany(),
    #     lambda df, cond: ~df.bool_col.any(),
    # ),
    # (
    #     lambda t, cond: t.bool_col.notall(),
    #     lambda df, cond: ~df.bool_col.all(),
    # ),
    (
        lambda t, cond: t.double_col.sum(),
        lambda df, cond: df.double_col.sum(),
    ),
    (
        lambda t, cond: t.double_col.mean(),
        lambda df, cond: df.double_col.mean(),
    ),
    (
        lambda t, cond: t.int_col.approx_median(),
        lambda df, cond: df.int_col.median(),
    ),
    (
        lambda t, cond: t.double_col.min(),
        lambda df, cond: df.double_col.min(),
    ),
    (
        lambda t, cond: t.double_col.max(),
        lambda df, cond: df.double_col.max(),
    ),
    (
        lambda t, cond: t.double_col.var(),
        lambda df, cond: df.double_col.var(),
    ),
    (
        lambda t, cond: t.double_col.std(),
        lambda df, cond: df.double_col.std(),
    ),
    (
        lambda t, cond: t.double_col.var(how='sample'),
        lambda df, cond: df.double_col.var(ddof=1),
    ),
    (
        lambda t, cond: t.double_col.std(how='pop'),
        lambda df, cond: df.double_col.std(ddof=0),
    ),
    (
        lambda t, cond: t.bool_col.count(where=cond),
        lambda df, cond: df.bool_col[cond].count(),
    ),
    # (
    #     lambda t, cond: t.bool_col.nunique(where=cond),
    #     lambda df, cond: df.bool_col[cond].nunique(),
    # ),
    # (
    #     lambda t, cond: t.bool_col.approx_nunique(where=cond),
    #     lambda df, cond: df.bool_col[cond].nunique(),
    # ),
    (
        lambda t, cond: t.double_col.sum(where=cond),
        lambda df, cond: df.double_col[cond].sum(),
    ),
    (
        lambda t, cond: t.double_col.mean(where=cond),
        lambda df, cond: df.double_col[cond].mean(),
    ),
    (
        lambda t, cond: t.int_col.approx_median(where=cond),
        lambda df, cond: df.int_col[cond].median(),
    ),
    (
        lambda t, cond: t.double_col.min(where=cond),
        lambda df, cond: df.double_col[cond].min(),
    ),
    (
        lambda t, cond: t.double_col.max(where=cond),
        lambda df, cond: df.double_col[cond].max(),
    ),
    (
        lambda t, cond: t.double_col.var(where=cond),
        lambda df, cond: df.double_col[cond].var(),
    ),
    (
        lambda t, cond: t.double_col.std(where=cond),
        lambda df, cond: df.double_col[cond].std(),
    ),
    (
        lambda t, cond: t.double_col.var(where=cond, how='sample'),
        lambda df, cond: df.double_col[cond].var(),
    ),
    (
        lambda t, cond: t.double_col.std(where=cond, how='pop'),
        lambda df, cond: df.double_col[cond].std(ddof=0),
    )
])
def test_aggregations(alltypes, df, func, pandas_func, translate):
    table = alltypes.limit(100)
    count = table.count().execute()
    df = df.head(int(count))

    cond = table.string_col.isin(['1', '7'])
    mask = cond.execute().astype('bool')
    expr = func(table, cond)

    result = expr.execute()
    expected = pandas_func(df, mask)

    np.testing.assert_allclose(result, expected)


# def test_group_concat(alltypes, df):
#     expr = alltypes.string_col.group_concat()
#     result = expr.execute()
#     expected = ','.join(df.string_col.dropna())
#     assert result == expected


# TODO: requires CountDistinct to support condition
# def test_distinct_aggregates(alltypes, df, translate):
#     expr = alltypes.limit(100).double_col.nunique()
#     result = expr.execute()

#     assert translate(expr) == 'uniq(`double_col`)'
#     assert result == df.head(100).double_col.nunique()


@pytest.mark.parametrize('op', [
    methodcaller('sum'),
    methodcaller('mean'),
    methodcaller('min'),
    methodcaller('max'),
    methodcaller('std'),
    methodcaller('var')
])
def test_boolean_reduction(alltypes, op, df):
    result = op(alltypes.bool_col).execute()
    assert result == op(df.bool_col)


# def test_anonymus_aggregate(alltypes, df):
#     t = alltypes
#     expr = t[t.double_col > t.double_col.mean()]
#     result = expr.execute()
#     expected = df[df.double_col > df.double_col.mean()].reset_index(
#         drop=True
#     )
#     tm.assert_frame_equal(result, expected)


# def test_rank(con):
#     t = con.table('functional_alltypes')
#     expr = t.double_col.rank()
#     sqla_expr = expr.compile()
#     result = str(sqla_expr.compile(compile_kwargs=dict(literal_binds=True)))
#     expected = """\
#     assert result == expected


# def test_percent_rank(con):
#     t = con.table('functional_alltypes')
#     expr = t.double_col.percent_rank()
#     sqla_expr = expr.compile()
#     result = str(sqla_expr.compile(compile_kwargs=dict(literal_binds=True)))
#     expected = """\
#     assert result == expected


# def test_ntile(con):
#     t = con.table('functional_alltypes')
#     expr = t.double_col.ntile(7)
#     sqla_expr = expr.compile()
#     result = str(sqla_expr.compile(compile_kwargs=dict(literal_binds=True)))
#     expected = """\
#     assert result == expected
