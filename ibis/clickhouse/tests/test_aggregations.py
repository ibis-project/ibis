import pytest
import numpy as np
import pandas as pd
import pandas.util.testing as tm
from operator import methodcaller
from ibis import literal as L

pytest.importorskip('clickhouse_driver')
pytestmark = pytest.mark.clickhouse


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


@pytest.mark.parametrize(('func', 'pandas_func'), [
    (
        lambda t, cond: t.bool_col.count(),
        lambda df, cond: df.bool_col.count(),
    ),
    (
        lambda t, cond: t.bool_col.approx_nunique(),
        lambda df, cond: df.bool_col.nunique(),
    ),
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
        lambda df, cond: np.int32(df.int_col.median()),
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
    (
        lambda t, cond: t.double_col.sum(where=cond),
        lambda df, cond: df.double_col[cond].sum(),
    ),
    (
        lambda t, cond: t.double_col.mean(where=cond),
        lambda df, cond: df.double_col[cond].mean(),
    ),
    (
        lambda t, cond: t.float_col.approx_median(where=cond),
        lambda df, cond: df.float_col[cond].median(),
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


def test_anonymus_aggregate(alltypes, df, translate):
    t = alltypes
    expr = t[t.double_col > t.double_col.mean()]
    result = expr.execute().set_index('id')
    expected = df[df.double_col > df.double_col.mean()].set_index('id')
    tm.assert_frame_equal(result, expected, check_like=True)


def test_boolean_summary(alltypes):
    expr = alltypes.bool_col.summary()
    result = expr.execute()
    expected = pd.DataFrame(
        [[7300, 0, 0, 1, 3650, 0.5, 2]],
        columns=[
            'count',
            'nulls',
            'min',
            'max',
            'sum',
            'mean',
            'approx_nunique',
        ]
    )
    tm.assert_frame_equal(result, expected, check_column_type=False,
                          check_dtype=False)
