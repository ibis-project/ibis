import operator
import os
import string
import warnings
from datetime import date, datetime

import numpy as np
import pandas as pd
import pandas.util.testing as tm
import pytest
from pytest import param

import ibis
import ibis.config as config
import ibis.expr.datatypes as dt
import ibis.expr.types as ir
from ibis import literal as L
from ibis.expr.window import rows_with_max_lookback

sa = pytest.importorskip('sqlalchemy')
pytest.importorskip('psycopg2')

pytestmark = pytest.mark.postgres


@pytest.fixture
def guid(con):
    name = ibis.util.guid()
    try:
        yield name
    finally:
        con.drop_table(name, force=True)


@pytest.fixture
def guid2(con):
    name = ibis.util.guid()
    try:
        yield name
    finally:
        con.drop_table(name, force=True)


@pytest.mark.parametrize(
    ('left_func', 'right_func'),
    [
        param(
            lambda t: t.double_col.cast('int8'),
            lambda at: sa.cast(at.c.double_col, sa.SMALLINT),
            id='double_to_int8',
        ),
        param(
            lambda t: t.double_col.cast('int16'),
            lambda at: sa.cast(at.c.double_col, sa.SMALLINT),
            id='double_to_int16',
        ),
        param(
            lambda t: t.string_col.cast('double'),
            lambda at: sa.cast(
                at.c.string_col, sa.dialects.postgresql.DOUBLE_PRECISION
            ),
            id='string_to_double',
        ),
        param(
            lambda t: t.string_col.cast('float'),
            lambda at: sa.cast(at.c.string_col, sa.REAL),
            id='string_to_float',
        ),
        param(
            lambda t: t.string_col.cast('decimal'),
            lambda at: sa.cast(at.c.string_col, sa.NUMERIC(9, 0)),
            id='string_to_decimal_no_params',
        ),
        param(
            lambda t: t.string_col.cast('decimal(9, 3)'),
            lambda at: sa.cast(at.c.string_col, sa.NUMERIC(9, 3)),
            id='string_to_decimal_params',
        ),
    ],
)
def test_cast(alltypes, at, translate, left_func, right_func):
    left = left_func(alltypes)
    right = right_func(at)
    assert str(translate(left).compile()) == str(right.compile())


def test_date_cast(alltypes, at, translate):
    result = alltypes.date_string_col.cast('date')
    expected = sa.cast(at.c.date_string_col, sa.DATE)
    assert str(translate(result)) == str(expected)


@pytest.mark.parametrize(
    'column',
    [
        'index',
        'Unnamed: 0',
        'id',
        'bool_col',
        'tinyint_col',
        'smallint_col',
        'int_col',
        'bigint_col',
        'float_col',
        'double_col',
        'date_string_col',
        'string_col',
        'timestamp_col',
        'year',
        'month',
    ],
)
def test_noop_cast(alltypes, at, translate, column):
    col = alltypes[column]
    result = col.cast(col.type())
    expected = at.c[column]
    assert result.equals(col)
    assert str(translate(result)) == str(expected)


def test_timestamp_cast_noop(alltypes, at, translate):
    # See GH #592
    result1 = alltypes.timestamp_col.cast('timestamp')
    result2 = alltypes.int_col.cast('timestamp')

    assert isinstance(result1, ir.TimestampColumn)
    assert isinstance(result2, ir.TimestampColumn)

    expected1 = at.c.timestamp_col
    expected2 = sa.func.timezone('UTC', sa.func.to_timestamp(at.c.int_col))

    assert str(translate(result1)) == str(expected1)
    assert str(translate(result2)) == str(expected2)


@pytest.mark.parametrize(
    ('func', 'expected'),
    [
        param(operator.methodcaller('year'), 2015, id='year'),
        param(operator.methodcaller('month'), 9, id='month'),
        param(operator.methodcaller('day'), 1, id='day'),
        param(operator.methodcaller('hour'), 14, id='hour'),
        param(operator.methodcaller('minute'), 48, id='minute'),
        param(operator.methodcaller('second'), 5, id='second'),
        param(operator.methodcaller('millisecond'), 359, id='millisecond'),
        param(lambda x: x.day_of_week.index(), 1, id='day_of_week_index'),
        param(
            lambda x: x.day_of_week.full_name(),
            'Tuesday',
            id='day_of_week_full_name',
        ),
    ],
)
def test_simple_datetime_operations(con, func, expected, translate):
    value = ibis.timestamp('2015-09-01 14:48:05.359')
    assert con.execute(func(value)) == expected


@pytest.mark.parametrize(
    'pattern',
    [
        # there could be pathological failure at midnight somewhere, but
        # that's okay
        '%Y%m%d %H',
        # test quoting behavior
        'DD BAR %w FOO "DD"',
        'DD BAR %w FOO "D',
        'DD BAR "%w" FOO "D',
        'DD BAR "%d" FOO "D',
        param(
            'DD BAR "%c" FOO "D',
            marks=pytest.mark.xfail(
                condition=os.name == 'nt',
                reason='Locale-specific format specs not available on Windows',
            ),
        ),
        param(
            'DD BAR "%x" FOO "D',
            marks=pytest.mark.xfail(
                condition=os.name == 'nt',
                reason='Locale-specific format specs not available on Windows',
            ),
        ),
        param(
            'DD BAR "%X" FOO "D',
            marks=pytest.mark.xfail(
                condition=os.name == 'nt',
                reason='Locale-specific format specs not available on Windows',
            ),
        ),
    ],
)
def test_strftime(con, pattern):
    value = ibis.timestamp('2015-09-01 14:48:05.359')
    raw_value = datetime(
        year=2015,
        month=9,
        day=1,
        hour=14,
        minute=48,
        second=5,
        microsecond=359000,
    )
    assert con.execute(value.strftime(pattern)) == raw_value.strftime(pattern)


@pytest.mark.parametrize(
    ('func', 'left', 'right', 'expected'),
    [
        param(operator.add, L(3), L(4), 7, id='add'),
        param(operator.sub, L(3), L(4), -1, id='sub'),
        param(operator.mul, L(3), L(4), 12, id='mul'),
        param(operator.truediv, L(12), L(4), 3, id='truediv_no_remainder'),
        param(operator.pow, L(12), L(2), 144, id='pow'),
        param(operator.mod, L(12), L(5), 2, id='mod'),
        param(operator.truediv, L(7), L(2), 3.5, id='truediv_remainder'),
        param(operator.floordiv, L(7), L(2), 3, id='floordiv'),
        param(
            lambda x, y: x.floordiv(y), L(7), 2, 3, id='floordiv_no_literal'
        ),
        param(
            lambda x, y: x.rfloordiv(y), L(2), 7, 3, id='rfloordiv_no_literal'
        ),
    ],
)
def test_binary_arithmetic(con, func, left, right, expected):
    expr = func(left, right)
    result = con.execute(expr)
    assert result == expected


@pytest.mark.parametrize(
    ('value', 'expected'),
    [
        param(L('foo_bar'), 'text', id='text'),
        param(L(5), 'integer', id='integer'),
        param(ibis.NA, 'null', id='null'),
        # TODO(phillipc): should this really be double?
        param(L(1.2345), 'numeric', id='numeric'),
        param(
            L(
                datetime(
                    2015,
                    9,
                    1,
                    hour=14,
                    minute=48,
                    second=5,
                    microsecond=359000,
                )
            ),
            'timestamp without time zone',
            id='timestamp_without_time_zone',
        ),
        param(L(date(2015, 9, 1)), 'date', id='date'),
    ],
)
def test_typeof(con, value, expected):
    assert con.execute(value.typeof()) == expected


@pytest.mark.parametrize(('value', 'expected'), [(0, None), (5.5, 5.5)])
def test_nullifzero(con, value, expected):
    assert con.execute(L(value).nullifzero()) == expected


@pytest.mark.parametrize(('value', 'expected'), [('foo_bar', 7), ('', 0)])
def test_string_length(con, value, expected):
    assert con.execute(L(value).length()) == expected


@pytest.mark.parametrize(
    ('op', 'expected'),
    [
        param(operator.methodcaller('left', 3), 'foo', id='left'),
        param(operator.methodcaller('right', 3), 'bar', id='right'),
        param(operator.methodcaller('substr', 0, 3), 'foo', id='substr_0_3'),
        param(operator.methodcaller('substr', 4, 3), 'bar', id='substr_4, 3'),
        param(operator.methodcaller('substr', 1), 'oo_bar', id='substr_1'),
    ],
)
def test_string_substring(con, op, expected):
    value = L('foo_bar')
    assert con.execute(op(value)) == expected


@pytest.mark.parametrize(
    ('opname', 'expected'),
    [('lstrip', 'foo   '), ('rstrip', '   foo'), ('strip', 'foo')],
)
def test_string_strip(con, opname, expected):
    op = operator.methodcaller(opname)
    value = L('   foo   ')
    assert con.execute(op(value)) == expected


@pytest.mark.parametrize(
    ('opname', 'count', 'char', 'expected'),
    [('lpad', 6, ' ', '   foo'), ('rpad', 6, ' ', 'foo   ')],
)
def test_string_pad(con, opname, count, char, expected):
    op = operator.methodcaller(opname, count, char)
    value = L('foo')
    assert con.execute(op(value)) == expected


def test_string_reverse(con):
    assert con.execute(L('foo').reverse()) == 'oof'


def test_string_upper(con):
    assert con.execute(L('foo').upper()) == 'FOO'


def test_string_lower(con):
    assert con.execute(L('FOO').lower()) == 'foo'


@pytest.mark.parametrize(
    ('haystack', 'needle', 'expected'),
    [
        ('foobar', 'bar', True),
        ('foobar', 'foo', True),
        ('foobar', 'baz', False),
        ('100%', '%', True),
        ('a_b_c', '_', True),
    ],
)
def test_string_contains(con, haystack, needle, expected):
    value = L(haystack)
    expr = value.contains(needle)
    assert con.execute(expr) == expected


@pytest.mark.parametrize(
    ('value', 'expected'),
    [('foo bar foo', 'Foo Bar Foo'), ('foobar Foo', 'Foobar Foo')],
)
def test_capitalize(con, value, expected):
    assert con.execute(L(value).capitalize()) == expected


def test_repeat(con):
    expr = L('bar ').repeat(3)
    assert con.execute(expr) == 'bar bar bar '


def test_re_replace(con):
    expr = L('fudge|||chocolate||candy').re_replace('\\|{2,3}', ', ')
    assert con.execute(expr) == 'fudge, chocolate, candy'


def test_translate(con):
    expr = L('faab').translate('a', 'b')
    assert con.execute(expr) == 'fbbb'


@pytest.mark.parametrize(
    ('raw_value', 'expected'), [('a', 0), ('b', 1), ('d', -1), (None, 3)]
)
def test_find_in_set(con, raw_value, expected):
    value = L(raw_value, dt.string)
    haystack = ['a', 'b', 'c', None]
    expr = value.find_in_set(haystack)
    assert con.execute(expr) == expected


@pytest.mark.parametrize(
    ('raw_value', 'opname', 'expected'),
    [
        (None, 'isnull', True),
        (1, 'isnull', False),
        (None, 'notnull', False),
        (1, 'notnull', True),
    ],
)
def test_isnull_notnull(con, raw_value, opname, expected):
    lit = L(raw_value)
    op = operator.methodcaller(opname)
    expr = op(lit)
    assert con.execute(expr) == expected


@pytest.mark.parametrize(
    ('expr', 'expected'),
    [
        param(L('foobar').find('bar'), 3, id='find_pos'),
        param(L('foobar').find('baz'), -1, id='find_neg'),
        param(L('foobar').like('%bar'), True, id='like_left_pattern'),
        param(L('foobar').like('foo%'), True, id='like_right_pattern'),
        param(L('foobar').like('%baz%'), False, id='like_both_sides_pattern'),
        param(L('foobar').like(['%bar']), True, id='like_list_left_side'),
        param(L('foobar').like(['foo%']), True, id='like_list_right_side'),
        param(L('foobar').like(['%baz%']), False, id='like_list_both_sides'),
        param(
            L('foobar').like(['%bar', 'foo%']), True, id='like_list_multiple'
        ),
        param(L('foobarfoo').replace('foo', 'H'), 'HbarH', id='replace'),
        param(L('a').ascii_str(), ord('a'), id='ascii_str'),
    ],
)
def test_string_functions(con, expr, expected):
    assert con.execute(expr) == expected


@pytest.mark.parametrize(
    ('expr', 'expected'),
    [
        param(L('abcd').re_search('[a-z]'), True, id='re_search_match'),
        param(L('abcd').re_search(r'[\d]+'), False, id='re_search_no_match'),
        param(
            L('1222').re_search(r'[\d]+'), True, id='re_search_match_number'
        ),
    ],
)
def test_regexp(con, expr, expected):
    assert con.execute(expr) == expected


@pytest.mark.parametrize(
    ('expr', 'expected'),
    [
        param(
            L('abcd').re_extract('([a-z]+)', 0), 'abcd', id='re_extract_whole'
        ),
        param(
            L('abcd').re_extract('(ab)(cd)', 1), 'cd', id='re_extract_first'
        ),
        # valid group number but no match => empty string
        param(L('abcd').re_extract(r'(\d)', 0), '', id='re_extract_no_match'),
        # match but not a valid group number => NULL
        param(L('abcd').re_extract('abcd', 3), None, id='re_extract_match'),
    ],
)
def test_regexp_extract(con, expr, expected):
    assert con.execute(expr) == expected


@pytest.mark.parametrize(
    ('expr', 'expected'),
    [
        param(ibis.NA.fillna(5), 5, id='filled'),
        param(L(5).fillna(10), 5, id='not_filled'),
        param(L(5).nullif(5), None, id='nullif_null'),
        param(L(10).nullif(5), 10, id='nullif_not_null'),
    ],
)
def test_fillna_nullif(con, expr, expected):
    assert con.execute(expr) == expected


@pytest.mark.parametrize(
    ('expr', 'expected'),
    [
        param(ibis.coalesce(5, None, 4), 5, id='first'),
        param(ibis.coalesce(ibis.NA, 4, ibis.NA), 4, id='second'),
        param(ibis.coalesce(ibis.NA, ibis.NA, 3.14), 3.14, id='third'),
    ],
)
def test_coalesce(con, expr, expected):
    assert con.execute(expr) == expected


@pytest.mark.parametrize(
    ('expr', 'expected'),
    [
        param(ibis.coalesce(ibis.NA, ibis.NA), None, id='all_null'),
        param(
            ibis.coalesce(ibis.NA, ibis.NA, ibis.NA.cast('double')),
            None,
            id='all_nulls_with_one_cast',
        ),
        param(
            ibis.coalesce(
                ibis.NA.cast('int8'),
                ibis.NA.cast('int8'),
                ibis.NA.cast('int8'),
            ),
            None,
            id='all_nulls_with_all_cast',
        ),
    ],
)
def test_coalesce_all_na(con, expr, expected):
    assert con.execute(expr) == expected


def test_numeric_builtins_work(alltypes, df):
    expr = alltypes.double_col.fillna(0)
    result = expr.execute()
    expected = df.double_col.fillna(0)
    expected.name = 'tmp'
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ('op', 'pandas_op'),
    [
        param(
            lambda t: (t.double_col > 20).ifelse(10, -20),
            lambda df: pd.Series(
                np.where(df.double_col > 20, 10, -20), dtype='int8'
            ),
            id='simple',
        ),
        param(
            lambda t: (t.double_col > 20).ifelse(10, -20).abs(),
            lambda df: pd.Series(
                np.where(df.double_col > 20, 10, -20), dtype='int8'
            ).abs(),
            id='abs',
        ),
    ],
)
def test_ifelse(alltypes, df, op, pandas_op):
    expr = op(alltypes)
    result = expr.execute()
    result.name = None
    expected = pandas_op(df)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ('func', 'pandas_func'),
    [
        # tier and histogram
        param(
            lambda d: d.bucket([0, 10, 25, 50, 100]),
            lambda s: pd.cut(
                s, [0, 10, 25, 50, 100], right=False, labels=False
            ),
            id='include_over_false',
        ),
        param(
            lambda d: d.bucket([0, 10, 25, 50], include_over=True),
            lambda s: pd.cut(
                s, [0, 10, 25, 50, np.inf], right=False, labels=False
            ),
            id='include_over_true',
        ),
        param(
            lambda d: d.bucket([0, 10, 25, 50], close_extreme=False),
            lambda s: pd.cut(s, [0, 10, 25, 50], right=False, labels=False),
            id='close_extreme_false',
        ),
        param(
            lambda d: d.bucket(
                [0, 10, 25, 50], closed='right', close_extreme=False
            ),
            lambda s: pd.cut(
                s,
                [0, 10, 25, 50],
                include_lowest=False,
                right=True,
                labels=False,
            ),
            id='closed_right',
        ),
        param(
            lambda d: d.bucket([10, 25, 50, 100], include_under=True),
            lambda s: pd.cut(
                s, [0, 10, 25, 50, 100], right=False, labels=False
            ),
            id='include_under_true',
        ),
    ],
)
def test_bucket(alltypes, df, func, pandas_func):
    expr = func(alltypes.double_col)
    result = expr.execute()
    expected = pandas_func(df.double_col).astype('category')
    tm.assert_series_equal(result, expected, check_names=False)


def test_category_label(alltypes, df):
    t = alltypes
    d = t.double_col

    bins = [0, 10, 25, 50, 100]
    labels = ['a', 'b', 'c', 'd']
    bucket = d.bucket(bins)
    expr = bucket.label(labels)
    result = expr.execute()

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
    result = pd.Series(pd.Categorical(result, ordered=True))

    result.name = 'double_col'

    expected = pd.cut(df.double_col, bins, labels=labels, right=False)

    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ('distinct1', 'distinct2', 'expected1', 'expected2'),
    [
        (True, True, 'UNION', 'UNION'),
        (True, False, 'UNION', 'UNION ALL'),
        (False, True, 'UNION ALL', 'UNION'),
        (False, False, 'UNION ALL', 'UNION ALL'),
    ],
)
def test_union_cte(alltypes, distinct1, distinct2, expected1, expected2):
    t = alltypes
    expr1 = t.group_by(t.string_col).aggregate(metric=t.double_col.sum())
    expr2 = expr1.view()
    expr3 = expr1.view()
    expr = expr1.union(expr2, distinct=distinct1).union(
        expr3, distinct=distinct2
    )
    result = '\n'.join(
        map(
            lambda line: line.rstrip(),  # strip trailing whitespace
            str(
                expr.compile().compile(compile_kwargs=dict(literal_binds=True))
            ).splitlines(),
        )
    )
    expected = """\
WITH anon_1 AS
(SELECT t0.string_col AS string_col, sum(t0.double_col) AS metric
FROM functional_alltypes AS t0 GROUP BY t0.string_col),
anon_2 AS
(SELECT t0.string_col AS string_col, sum(t0.double_col) AS metric
FROM functional_alltypes AS t0 GROUP BY t0.string_col),
anon_3 AS
(SELECT t0.string_col AS string_col, sum(t0.double_col) AS metric
FROM functional_alltypes AS t0 GROUP BY t0.string_col)
 (SELECT anon_1.string_col, anon_1.metric
FROM anon_1 {} SELECT anon_2.string_col, anon_2.metric
FROM anon_2) {} SELECT anon_3.string_col, anon_3.metric
FROM anon_3""".format(
        expected1, expected2
    )
    assert str(result) == expected


@pytest.mark.parametrize(
    ('func', 'pandas_func'),
    [
        param(
            lambda t, cond: t.bool_col.count(),
            lambda df, cond: df.bool_col.count(),
            id='count',
        ),
        param(
            lambda t, cond: t.bool_col.any(),
            lambda df, cond: df.bool_col.any(),
            id='any',
        ),
        param(
            lambda t, cond: t.bool_col.all(),
            lambda df, cond: df.bool_col.all(),
            id='all',
        ),
        param(
            lambda t, cond: t.bool_col.notany(),
            lambda df, cond: ~df.bool_col.any(),
            id='notany',
        ),
        param(
            lambda t, cond: t.bool_col.notall(),
            lambda df, cond: ~df.bool_col.all(),
            id='notall',
        ),
        param(
            lambda t, cond: t.double_col.sum(),
            lambda df, cond: df.double_col.sum(),
            id='sum',
        ),
        param(
            lambda t, cond: t.double_col.mean(),
            lambda df, cond: df.double_col.mean(),
            id='mean',
        ),
        param(
            lambda t, cond: t.double_col.min(),
            lambda df, cond: df.double_col.min(),
            id='min',
        ),
        param(
            lambda t, cond: t.double_col.max(),
            lambda df, cond: df.double_col.max(),
            id='max',
        ),
        param(
            lambda t, cond: t.double_col.var(),
            lambda df, cond: df.double_col.var(),
            id='var',
        ),
        param(
            lambda t, cond: t.double_col.std(),
            lambda df, cond: df.double_col.std(),
            id='std',
        ),
        param(
            lambda t, cond: t.double_col.var(how='sample'),
            lambda df, cond: df.double_col.var(ddof=1),
            id='samp_var',
        ),
        param(
            lambda t, cond: t.double_col.std(how='pop'),
            lambda df, cond: df.double_col.std(ddof=0),
            id='pop_std',
        ),
        param(
            lambda t, cond: t.bool_col.count(where=cond),
            lambda df, cond: df.bool_col[cond].count(),
            id='count_where',
        ),
        param(
            lambda t, cond: t.double_col.sum(where=cond),
            lambda df, cond: df.double_col[cond].sum(),
            id='sum_where',
        ),
        param(
            lambda t, cond: t.double_col.mean(where=cond),
            lambda df, cond: df.double_col[cond].mean(),
            id='mean_where',
        ),
        param(
            lambda t, cond: t.double_col.min(where=cond),
            lambda df, cond: df.double_col[cond].min(),
            id='min_where',
        ),
        param(
            lambda t, cond: t.double_col.max(where=cond),
            lambda df, cond: df.double_col[cond].max(),
            id='max_where',
        ),
        param(
            lambda t, cond: t.double_col.var(where=cond),
            lambda df, cond: df.double_col[cond].var(),
            id='var_where',
        ),
        param(
            lambda t, cond: t.double_col.std(where=cond),
            lambda df, cond: df.double_col[cond].std(),
            id='std_where',
        ),
        param(
            lambda t, cond: t.double_col.var(where=cond, how='sample'),
            lambda df, cond: df.double_col[cond].var(),
            id='samp_var_where',
        ),
        param(
            lambda t, cond: t.double_col.std(where=cond, how='pop'),
            lambda df, cond: df.double_col[cond].std(ddof=0),
            id='pop_std_where',
        ),
    ],
)
def test_aggregations(alltypes, df, func, pandas_func):
    table = alltypes.limit(100)
    df = df.head(table.count().execute())

    cond = table.string_col.isin(['1', '7'])
    expr = func(table, cond)
    result = expr.execute()
    expected = pandas_func(df, cond.execute())

    np.testing.assert_allclose(result, expected)


def test_not_contains(alltypes, df):
    n = 100
    table = alltypes.limit(n)
    expr = table.string_col.notin(['1', '7'])
    result = expr.execute()
    expected = ~df.head(n).string_col.isin(['1', '7'])
    tm.assert_series_equal(result, expected, check_names=False)


def test_group_concat(alltypes, df):
    expr = alltypes.string_col.group_concat()
    result = expr.execute()
    expected = ','.join(df.string_col.dropna())
    assert result == expected


def test_distinct_aggregates(alltypes, df):
    expr = alltypes.limit(100).double_col.nunique()
    result = expr.execute()
    assert result == df.head(100).double_col.nunique()


def test_not_exists(alltypes, df):
    t = alltypes
    t2 = t.view()

    expr = t[~((t.string_col == t2.string_col).any())]
    result = expr.execute()

    left, right = df, t2.execute()
    expected = left[left.string_col != right.string_col]

    tm.assert_frame_equal(
        result, expected, check_index_type=False, check_dtype=False
    )


def test_interactive_repr_shows_error(alltypes):
    # #591. Doing this in PostgreSQL because so many built-in functions are
    # not available

    expr = alltypes.double_col.approx_median()

    with config.option_context('interactive', True):
        result = repr(expr)

    assert 'no translation rule' in result.lower()


def test_subquery(alltypes, df):
    t = alltypes

    expr = (
        t.mutate(d=t.double_col.fillna(0))
        .limit(1000)
        .group_by('string_col')
        .size()
    )
    result = expr.execute().sort_values('string_col').reset_index(drop=True)
    expected = (
        df.assign(d=df.double_col.fillna(0))
        .head(1000)
        .groupby('string_col')
        .string_col.count()
        .reset_index(name='count')
        .sort_values('string_col')
        .reset_index(drop=True)
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize('func', ['mean', 'sum', 'min', 'max'])
def test_simple_window(alltypes, func, df):
    t = alltypes
    f = getattr(t.double_col, func)
    df_f = getattr(df.double_col, func)
    result = (
        t.projection([(t.double_col - f()).name('double_col')])
        .execute()
        .double_col
    )
    expected = df.double_col - df_f()
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize('func', ['mean', 'sum', 'min', 'max'])
def test_rolling_window(alltypes, func, df):
    t = alltypes
    df = (
        df[['double_col', 'timestamp_col']]
        .sort_values('timestamp_col')
        .reset_index(drop=True)
    )
    window = ibis.window(order_by=t.timestamp_col, preceding=6, following=0)
    f = getattr(t.double_col, func)
    df_f = getattr(df.double_col.rolling(7, min_periods=0), func)
    result = (
        t.projection([f().over(window).name('double_col')])
        .execute()
        .double_col
    )
    expected = df_f()
    tm.assert_series_equal(result, expected)


def test_rolling_window_with_mlb(alltypes):
    t = alltypes
    window = ibis.trailing_window(
        preceding=rows_with_max_lookback(3, ibis.interval(days=5)),
        order_by=t.timestamp_col,
    )
    expr = t['double_col'].sum().over(window)
    with pytest.raises(NotImplementedError):
        expr.execute()


@pytest.mark.parametrize('func', ['mean', 'sum', 'min', 'max'])
def test_partitioned_window(alltypes, func, df):
    t = alltypes
    window = ibis.window(
        group_by=t.string_col,
        order_by=t.timestamp_col,
        preceding=6,
        following=0,
    )

    def roller(func):
        def rolled(df):
            torder = df.sort_values('timestamp_col')
            rolling = torder.double_col.rolling(7, min_periods=0)
            return getattr(rolling, func)()

        return rolled

    f = getattr(t.double_col, func)
    expr = f().over(window).name('double_col')
    result = t.projection([expr]).execute().double_col
    expected = (
        df.groupby('string_col').apply(roller(func)).reset_index(drop=True)
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize('func', ['sum', 'min', 'max'])
def test_cumulative_simple_window(alltypes, func, df):
    t = alltypes
    f = getattr(t.double_col, func)
    col = t.double_col - f().over(ibis.cumulative_window())
    expr = t.projection([col.name('double_col')])
    result = expr.execute().double_col
    expected = df.double_col - getattr(df.double_col, 'cum%s' % func)()
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize('func', ['sum', 'min', 'max'])
def test_cumulative_partitioned_window(alltypes, func, df):
    t = alltypes
    df = df.sort_values('string_col').reset_index(drop=True)
    window = ibis.cumulative_window(group_by=t.string_col)
    f = getattr(t.double_col, func)
    expr = t.projection([(t.double_col - f().over(window)).name('double_col')])
    result = expr.execute().double_col
    expected = df.groupby(df.string_col).double_col.transform(
        lambda c: c - getattr(c, 'cum%s' % func)()
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize('func', ['sum', 'min', 'max'])
def test_cumulative_ordered_window(alltypes, func, df):
    t = alltypes
    df = df.sort_values('timestamp_col').reset_index(drop=True)
    window = ibis.cumulative_window(order_by=t.timestamp_col)
    f = getattr(t.double_col, func)
    expr = t.projection([(t.double_col - f().over(window)).name('double_col')])
    result = expr.execute().double_col
    expected = df.double_col - getattr(df.double_col, 'cum%s' % func)()
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize('func', ['sum', 'min', 'max'])
def test_cumulative_partitioned_ordered_window(alltypes, func, df):
    t = alltypes
    df = df.sort_values(['string_col', 'timestamp_col']).reset_index(drop=True)
    window = ibis.cumulative_window(
        order_by=t.timestamp_col, group_by=t.string_col
    )
    f = getattr(t.double_col, func)
    expr = t.projection([(t.double_col - f().over(window)).name('double_col')])
    result = expr.execute().double_col
    method = operator.methodcaller('cum{}'.format(func))
    expected = df.groupby(df.string_col).double_col.transform(
        lambda c: c - method(c)
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(('func', 'shift_amount'), [('lead', -1), ('lag', 1)])
def test_analytic_shift_functions(alltypes, df, func, shift_amount):
    method = getattr(alltypes.double_col, func)
    expr = method(1)
    result = expr.execute().rename('double_col')
    expected = df.double_col.shift(shift_amount)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ('func', 'expected_index'), [('first', -1), ('last', 0)]
)
def test_first_last_value(alltypes, df, func, expected_index):
    col = alltypes.sort_by(ibis.desc(alltypes.string_col)).double_col
    method = getattr(col, func)
    expr = method()
    result = expr.execute().rename('double_col')
    expected = pd.Series(
        df.double_col.iloc[expected_index],
        index=pd.RangeIndex(len(df)),
        name='double_col',
    )
    tm.assert_series_equal(result, expected)


def test_null_column(alltypes):
    t = alltypes
    nrows = t.count().execute()
    expr = t.mutate(na_column=ibis.NA).na_column
    result = expr.execute()
    tm.assert_series_equal(result, pd.Series([None] * nrows, name='na_column'))


def test_null_column_union(alltypes, df):
    t = alltypes
    s = alltypes[['double_col']].mutate(string_col=ibis.NA.cast('string'))
    expr = t[['double_col', 'string_col']].union(s)
    result = expr.execute()
    nrows = t.count().execute()
    expected = pd.concat(
        [
            df[['double_col', 'string_col']],
            pd.concat(
                [
                    df[['double_col']],
                    pd.DataFrame({'string_col': [None] * nrows}),
                ],
                axis=1,
            ),
        ],
        axis=0,
        ignore_index=True,
    )
    tm.assert_frame_equal(result, expected)


def test_window_with_arithmetic(alltypes, df):
    t = alltypes
    w = ibis.window(order_by=t.timestamp_col)
    expr = t.mutate(new_col=ibis.row_number().over(w) / 2)

    df = (
        df[['timestamp_col']]
        .sort_values('timestamp_col')
        .reset_index(drop=True)
    )
    expected = df.assign(new_col=[x / 2.0 for x in range(len(df))])
    result = expr['timestamp_col', 'new_col'].execute()
    tm.assert_frame_equal(result, expected)


def test_anonymous_aggregate(alltypes, df):
    t = alltypes
    expr = t[t.double_col > t.double_col.mean()]
    result = expr.execute()
    expected = df[df.double_col > df.double_col.mean()].reset_index(drop=True)
    tm.assert_frame_equal(result, expected)


@pytest.fixture
def array_types(con):
    return con.table('array_types')


def test_array_length(array_types):
    expr = array_types.projection(
        [
            array_types.x.length().name('x_length'),
            array_types.y.length().name('y_length'),
            array_types.z.length().name('z_length'),
        ]
    )
    result = expr.execute()
    expected = pd.DataFrame(
        {
            'x_length': [3, 2, 2, 3, 3, 4],
            'y_length': [3, 2, 2, 3, 3, 4],
            'z_length': [3, 2, 2, 0, None, 4],
        }
    )

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ('column', 'value_type'),
    [('x', dt.int64), ('y', dt.string), ('z', dt.double)],
)
def test_array_schema(array_types, column, value_type):
    assert array_types[column].type() == dt.Array(value_type)


def test_array_collect(array_types):
    expr = array_types.group_by(array_types.grouper).aggregate(
        collected=lambda t: t.scalar_column.collect()
    )
    result = expr.execute().sort_values('grouper').reset_index(drop=True)
    expected = pd.DataFrame(
        {
            'grouper': list('abc'),
            'collected': [[1.0, 2.0, 3.0], [4.0, 5.0], [6.0]],
        }
    )[['grouper', 'collected']]
    tm.assert_frame_equal(result, expected, check_column_type=False)


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
        param(
            -3,
            None,
            marks=pytest.mark.xfail(
                raises=ValueError, reason='Negative slicing not supported'
            ),
        ),
        param(
            None,
            -3,
            marks=pytest.mark.xfail(
                raises=ValueError, reason='Negative slicing not supported'
            ),
        ),
        param(
            -3,
            -1,
            marks=pytest.mark.xfail(
                raises=ValueError, reason='Negative slicing not supported'
            ),
        ),
        param(
            -3,
            -1,
            marks=pytest.mark.xfail(
                raises=ValueError, reason='Negative slicing not supported'
            ),
        ),
    ],
)
def test_array_slice(array_types, start, stop):
    expr = array_types[array_types.y[start:stop].name('sliced')]
    result = expr.execute()
    expected = pd.DataFrame(
        {'sliced': array_types.y.execute().map(lambda x: x[start:stop])}
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize('index', [1, 3, 4, 11])
def test_array_index(array_types, index):
    expr = array_types[array_types.y[index].name('indexed')]
    result = expr.execute()
    expected = pd.DataFrame(
        {
            'indexed': array_types.y.execute().map(
                lambda x: x[index] if index < len(x) else None
            )
        }
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize('n', [1, 3, 4, 7, -2])
@pytest.mark.parametrize(
    'mul',
    [
        param(lambda x, n: x * n, id='mul'),
        param(lambda x, n: n * x, id='rmul'),
    ],
)
def test_array_repeat(array_types, n, mul):
    expr = array_types.projection([mul(array_types.x, n).name('repeated')])
    result = expr.execute()
    expected = pd.DataFrame(
        {'repeated': array_types.x.execute().map(lambda x, n=n: mul(x, n))}
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    'catop',
    [
        param(lambda x, y: x + y, id='concat'),
        param(lambda x, y: y + x, id='rconcat'),
    ],
)
def test_array_concat(array_types, catop):
    t = array_types
    x, y = t.x.cast('array<string>').name('x'), t.y
    expr = t.projection([catop(x, y).name('catted')])
    result = expr.execute()
    tuples = t.projection([x, y]).execute().itertuples(index=False)
    expected = pd.DataFrame({'catted': [catop(i, j) for i, j in tuples]})
    tm.assert_frame_equal(result, expected)


def test_array_concat_mixed_types(array_types):
    with pytest.raises(TypeError):
        array_types.x + array_types.x.cast('array<double>')


@pytest.fixture
def t(con, guid):
    con.raw_sql(
        """
        CREATE TABLE "{}" (
          id SERIAL PRIMARY KEY,
          name TEXT
        )
        """.format(
            guid
        )
    )
    return con.table(guid)


@pytest.fixture
def s(con, t, guid, guid2):
    assert t.op().name == guid
    assert t.op().name != guid2

    con.raw_sql(
        """
        CREATE TABLE "{}" (
          id SERIAL PRIMARY KEY,
          left_t_id INTEGER REFERENCES "{}",
          cost DOUBLE PRECISION
        )
        """.format(
            guid2, guid
        )
    )
    return con.table(guid2)


@pytest.fixture
def trunc(con, guid):
    con.raw_sql(
        """
        CREATE TABLE "{}" (
          id SERIAL PRIMARY KEY,
          name TEXT
        )
        """.format(
            guid
        )
    )
    con.raw_sql(
        """INSERT INTO "{}" (name) VALUES ('a'), ('b'), ('c')""".format(guid)
    )
    return con.table(guid)


def test_semi_join(t, s):
    t_a, s_a = t.op().sqla_table.alias('t0'), s.op().sqla_table.alias('t1')
    expr = t.semi_join(s, t.id == s.id)
    result = expr.compile().compile(compile_kwargs=dict(literal_binds=True))
    base = sa.select([t_a.c.id, t_a.c.name]).where(
        sa.exists(sa.select([1]).where(t_a.c.id == s_a.c.id))
    )
    expected = sa.select([base.c.id, base.c.name])
    assert str(result) == str(expected)


def test_anti_join(t, s):
    t_a, s_a = t.op().sqla_table.alias('t0'), s.op().sqla_table.alias('t1')
    expr = t.anti_join(s, t.id == s.id)
    result = expr.compile().compile(compile_kwargs=dict(literal_binds=True))
    expected = sa.select([sa.column('id'), sa.column('name')]).select_from(
        sa.select([t_a.c.id, t_a.c.name]).where(
            ~(sa.exists(sa.select([1]).where(t_a.c.id == s_a.c.id)))
        )
    )
    assert str(result) == str(expected)


def test_create_table_from_expr(con, trunc, guid2):
    con.create_table(guid2, expr=trunc)
    t = con.table(guid2)
    assert list(t.name.execute()) == list('abc')


def test_truncate_table(con, trunc):
    assert list(trunc.name.execute()) == list('abc')
    con.truncate_table(trunc.op().name)
    assert not len(trunc.execute())


def test_head(con):
    t = con.table('functional_alltypes')
    result = t.head().execute()
    expected = t.limit(5).execute()
    tm.assert_frame_equal(result, expected)


def test_identical_to(con, df):
    # TODO: abstract this testing logic out into parameterized fixtures
    t = con.table('functional_alltypes')
    dt = df[['tinyint_col', 'double_col']]
    expr = t.tinyint_col.identical_to(t.double_col)
    result = expr.execute()
    expected = (dt.tinyint_col.isnull() & dt.double_col.isnull()) | (
        dt.tinyint_col == dt.double_col
    )
    expected.name = result.name
    tm.assert_series_equal(result, expected)


def test_rank(con):
    t = con.table('functional_alltypes')
    expr = t.double_col.rank()
    sqla_expr = expr.compile()
    result = str(sqla_expr.compile(compile_kwargs=dict(literal_binds=True)))
    expected = (
        "SELECT rank() OVER (ORDER BY t0.double_col) - 1 AS tmp \n"
        "FROM functional_alltypes AS t0"
    )
    assert result == expected


def test_percent_rank(con):
    t = con.table('functional_alltypes')
    expr = t.double_col.percent_rank()
    sqla_expr = expr.compile()
    result = str(sqla_expr.compile(compile_kwargs=dict(literal_binds=True)))
    expected = (
        "SELECT percent_rank() OVER (ORDER BY t0.double_col) AS "
        "tmp \nFROM functional_alltypes AS t0"
    )
    assert result == expected


def test_ntile(con):
    t = con.table('functional_alltypes')
    expr = t.double_col.ntile(7)
    sqla_expr = expr.compile()
    result = str(sqla_expr.compile(compile_kwargs=dict(literal_binds=True)))
    expected = (
        "SELECT ntile(7) OVER (ORDER BY t0.double_col) - 1 AS tmp \n"
        "FROM functional_alltypes AS t0"
    )
    assert result == expected


@pytest.mark.parametrize('opname', ['invert', 'neg'])
def test_not_and_negate_bool(con, opname, df):
    op = getattr(operator, opname)
    t = con.table('functional_alltypes').limit(10)
    expr = t.projection([op(t.bool_col).name('bool_col')])
    result = expr.execute().bool_col
    expected = op(df.head(10).bool_col)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    'field',
    [
        'tinyint_col',
        'smallint_col',
        'int_col',
        'bigint_col',
        'float_col',
        'double_col',
        'year',
        'month',
    ],
)
def test_negate_non_boolean(con, field, df):
    t = con.table('functional_alltypes').limit(10)
    expr = t.projection([(-t[field]).name(field)])
    result = expr.execute()[field]
    expected = -df.head(10)[field]
    tm.assert_series_equal(result, expected)


def test_negate_boolean(con, df):
    t = con.table('functional_alltypes').limit(10)
    expr = t.projection([(-t.bool_col).name('bool_col')])
    result = expr.execute().bool_col
    expected = -df.head(10).bool_col
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ('opname', 'expected'),
    [
        ('year', {2009, 2010}),
        ('month', set(range(1, 13))),
        ('day', set(range(1, 32))),
    ],
)
def test_date_extract_field(db, opname, expected):
    op = operator.methodcaller(opname)
    t = db.functional_alltypes
    expr = op(t.timestamp_col.cast('date')).distinct()
    result = expr.execute().astype(int)
    assert set(result) == expected


@pytest.mark.parametrize('opname', ['sum', 'mean', 'min', 'max', 'std', 'var'])
def test_boolean_reduction(alltypes, opname, df):
    op = operator.methodcaller(opname)
    expr = op(alltypes.bool_col)
    result = expr.execute()
    assert result == op(df.bool_col)


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
        ],
    )

    type_conversions = {
        'count': 'int64',
        'nulls': 'int64',
        'min': 'bool',
        'max': 'bool',
        'sum': 'int64',
        'approx_nunique': 'int64',
    }
    for k, v in type_conversions.items():
        expected[k] = expected[k].astype(v)

    tm.assert_frame_equal(result, expected)


def test_timestamp_with_timezone(con):
    t = con.table('tzone')
    result = t.ts.execute()
    assert str(result.dtype.tz)


@pytest.fixture(
    params=[
        None,
        'UTC',
        'America/New_York',
        'America/Los_Angeles',
        'Europe/Paris',
        'Chile/Continental',
        'Asia/Tel_Aviv',
        'Asia/Tokyo',
        'Africa/Nairobi',
        'Australia/Sydney',
    ]
)
def tz(request):
    return request.param


@pytest.fixture
def tzone_compute(con, guid, tz):
    schema = ibis.schema(
        [('ts', dt.Timestamp(tz)), ('b', 'double'), ('c', 'string')]
    )
    con.create_table(guid, schema=schema)
    t = con.table(guid)

    n = 10
    df = pd.DataFrame(
        {
            'ts': pd.date_range('2017-04-01', periods=n, tz=tz).values,
            'b': np.arange(n).astype('float64'),
            'c': list(string.ascii_lowercase[:n]),
        }
    )

    df.to_sql(
        guid,
        con.con,
        index=False,
        if_exists='append',
        dtype={'ts': sa.TIMESTAMP(timezone=True), 'b': sa.FLOAT, 'c': sa.TEXT},
    )

    try:
        yield t
    finally:
        con.drop_table(guid)
        assert guid not in con.list_tables()


def test_ts_timezone_is_preserved(tzone_compute, tz):
    assert dt.Timestamp(tz).equals(tzone_compute.ts.type())


def test_timestamp_with_timezone_select(tzone_compute, tz):
    ts = tzone_compute.ts.execute()
    assert str(getattr(ts.dtype, 'tz', None)) == str(tz)


def test_timestamp_type_accepts_all_timezones(con):
    assert all(
        dt.Timestamp(row.name).timezone == row.name
        for row in con.con.execute('SELECT name FROM pg_timezone_names')
    )


@pytest.mark.parametrize(
    ('left', 'right', 'type'),
    [
        param(L('2017-04-01'), date(2017, 4, 2), dt.date, id='ibis_date'),
        param(date(2017, 4, 2), L('2017-04-01'), dt.date, id='python_date'),
        param(
            L('2017-04-01 01:02:33'),
            datetime(2017, 4, 1, 1, 3, 34),
            dt.timestamp,
            id='ibis_timestamp',
        ),
        param(
            datetime(2017, 4, 1, 1, 3, 34),
            L('2017-04-01 01:02:33'),
            dt.timestamp,
            id='python_datetime',
        ),
    ],
)
@pytest.mark.parametrize('opname', ['eq', 'ne', 'lt', 'le', 'gt', 'ge'])
def test_string_temporal_compare(con, opname, left, right, type):
    op = getattr(operator, opname)
    expr = op(left, right)
    result = con.execute(expr)
    left_raw = con.execute(L(left).cast(type))
    right_raw = con.execute(L(right).cast(type))
    expected = op(left_raw, right_raw)
    assert result == expected


@pytest.mark.parametrize(
    ('left', 'right'),
    [
        param(L('2017-03-31').cast(dt.date), date(2017, 4, 2), id='ibis_date'),
        param(
            date(2017, 3, 31), L('2017-04-02').cast(dt.date), id='python_date'
        ),
        param(
            L('2017-03-31 00:02:33').cast(dt.timestamp),
            datetime(2017, 4, 1, 1, 3, 34),
            id='ibis_timestamp',
        ),
        param(
            datetime(2017, 3, 31, 0, 2, 33),
            L('2017-04-01 01:03:34').cast(dt.timestamp),
            id='python_datetime',
        ),
    ],
)
@pytest.mark.parametrize(
    'op',
    [
        param(
            lambda left, right: ibis.timestamp('2017-04-01 00:02:34').between(
                left, right
            ),
            id='timestamp',
        ),
        param(
            lambda left, right: (
                ibis.timestamp('2017-04-01').cast(dt.date).between(left, right)
            ),
            id='date',
        ),
    ],
)
def test_string_temporal_compare_between(con, op, left, right):
    expr = op(left, right)
    result = con.execute(expr)
    assert isinstance(result, (bool, np.bool_))
    assert result


def test_scalar_parameter(con):
    start = ibis.param(dt.date)
    end = ibis.param(dt.date)
    t = con.table('functional_alltypes')
    col = t.date_string_col.cast('date')
    expr = col.between(start, end)
    start_string, end_string = '2009-03-01', '2010-07-03'
    result = expr.execute(params={start: start_string, end: end_string})
    expected = col.between(start_string, end_string).execute()
    tm.assert_series_equal(result, expected)


def test_string_to_binary_cast(con):
    t = con.table('functional_alltypes').limit(10)
    expr = t.string_col.cast('binary')
    result = expr.execute()
    sql_string = (
        "SELECT decode(string_col, 'escape') AS tmp "
        "FROM functional_alltypes LIMIT 10"
    )
    raw_data = [row[0][0] for row in con.raw_sql(sql_string).fetchall()]
    expected = pd.Series(raw_data, name='tmp')
    tm.assert_series_equal(result, expected)


def test_string_to_binary_round_trip(con):
    t = con.table('functional_alltypes').limit(10)
    expr = t.string_col.cast('binary').cast('string')
    result = expr.execute()
    sql_string = (
        "SELECT encode(decode(string_col, 'escape'), 'escape') AS tmp "
        "FROM functional_alltypes LIMIT 10"
    )
    expected = pd.Series(
        [row[0][0] for row in con.raw_sql(sql_string).fetchall()], name='tmp'
    )
    tm.assert_series_equal(result, expected)
