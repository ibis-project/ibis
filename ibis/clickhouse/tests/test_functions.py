import math
import operator
from datetime import date, datetime
from operator import methodcaller

import pandas as pd
import pandas.util.testing as tm
import pytest
from pytest import param

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.types as ir
from ibis import literal as L

clickhouse_driver = pytest.importorskip('clickhouse_driver')
pytestmark = pytest.mark.clickhouse


@pytest.mark.parametrize(
    ('to_type', 'expected'),
    [
        ('int8', 'CAST(`double_col` AS Int8)'),
        ('int16', 'CAST(`double_col` AS Int16)'),
        ('float', 'CAST(`double_col` AS Float32)'),
        # alltypes.double_col is non-nullable
        (dt.Double(nullable=False), '`double_col`'),
    ],
)
def test_cast_double_col(alltypes, translate, to_type, expected):
    expr = alltypes.double_col.cast(to_type)
    assert translate(expr) == expected


@pytest.mark.parametrize(
    ('to_type', 'expected'),
    [
        ('int8', 'CAST(`string_col` AS Int8)'),
        ('int16', 'CAST(`string_col` AS Int16)'),
        (dt.String(nullable=False), '`string_col`'),
        ('timestamp', 'CAST(`string_col` AS DateTime)'),
        ('date', 'CAST(`string_col` AS Date)'),
    ],
)
def test_cast_string_col(alltypes, translate, to_type, expected):
    expr = alltypes.string_col.cast(to_type)
    assert translate(expr) == expected


@pytest.mark.xfail(
    raises=AssertionError, reason='Clickhouse doesn\'t have decimal type'
)
def test_decimal_cast():
    assert False


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
def test_noop_cast(alltypes, translate, column):
    col = alltypes[column]
    result = col.cast(col.type())
    assert result.equals(col)
    assert translate(result) == '`{}`'.format(column)


def test_timestamp_cast_noop(alltypes, translate):
    target = dt.Timestamp(nullable=False)
    result1 = alltypes.timestamp_col.cast(target)
    result2 = alltypes.int_col.cast(target)

    assert isinstance(result1, ir.TimestampColumn)
    assert isinstance(result2, ir.TimestampColumn)

    assert translate(result1) == '`timestamp_col`'
    assert translate(result2) == 'CAST(`int_col` AS DateTime)'


def test_timestamp_now(con, translate):
    expr = ibis.now()
    # now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    assert translate(expr) == 'now()'
    # assert con.execute(expr) == now


@pytest.mark.parametrize(
    ('unit', 'expected'),
    [
        ('y', '2009-01-01'),
        param('m', '2009-05-01', marks=pytest.mark.xfail),
        ('d', '2009-05-17'),
        ('w', '2009-05-11'),
        ('h', '2009-05-17 12:00:00'),
        ('minute', '2009-05-17 12:34:00'),
    ],
)
def test_timestamp_truncate(con, translate, unit, expected):
    stamp = ibis.timestamp('2009-05-17 12:34:56')
    expr = stamp.truncate(unit)
    assert con.execute(expr) == pd.Timestamp(expected)


@pytest.mark.parametrize(
    ('func', 'expected'),
    [
        (methodcaller('year'), 2015),
        (methodcaller('month'), 9),
        (methodcaller('day'), 1),
        (methodcaller('hour'), 14),
        (methodcaller('minute'), 48),
        (methodcaller('second'), 5),
    ],
)
def test_simple_datetime_operations(con, func, expected):
    value = ibis.timestamp('2015-09-01 14:48:05.359')
    with pytest.raises(ValueError):
        con.execute(func(value))

    value = ibis.timestamp('2015-09-01 14:48:05')
    con.execute(func(value)) == expected


@pytest.mark.parametrize(('value', 'expected'), [(0, None), (5.5, 5.5)])
def test_nullifzero(con, value, expected):
    result = con.execute(L(value).nullifzero())
    if expected is None:
        assert pd.isnull(result)
    else:
        assert result == expected


@pytest.mark.parametrize(
    ('expr', 'expected'),
    [
        (L(None).isnull(), True),
        (L(1).isnull(), False),
        (L(None).notnull(), False),
        (L(1).notnull(), True),
    ],
)
def test_isnull_notnull(con, expr, expected):
    assert con.execute(expr) == expected


@pytest.mark.parametrize(
    ('expr', 'expected'),
    [
        (ibis.coalesce(5, None, 4), 5),
        (ibis.coalesce(ibis.NA, 4, ibis.NA), 4),
        (ibis.coalesce(ibis.NA, ibis.NA, 3.14), 3.14),
    ],
)
def test_coalesce(con, expr, expected):
    assert con.execute(expr) == expected


@pytest.mark.parametrize(
    ('expr', 'expected'),
    [
        (ibis.NA.fillna(5), 5),
        (L(5).fillna(10), 5),
        (L(5).nullif(5), None),
        (L(10).nullif(5), 10),
    ],
)
def test_fillna_nullif(con, expr, expected):
    result = con.execute(expr)
    if expected is None:
        assert pd.isnull(result)
    else:
        assert result == expected


@pytest.mark.parametrize(
    ('value', 'expected'),
    [
        (L('foo_bar'), 'String'),
        (L(5), 'UInt8'),
        (L(1.2345), 'Float64'),
        (L(datetime(2015, 9, 1, hour=14, minute=48, second=5)), 'DateTime'),
        (L(date(2015, 9, 1)), 'Date'),
        param(
            ibis.NA,
            'Null',
            marks=pytest.mark.xfail(
                raises=AssertionError,
                reason=(
                    'Client/server version mismatch not handled in the '
                    'clickhouse driver'
                ),
            ),
        ),
    ],
)
def test_typeof(con, value, expected):
    assert con.execute(value.typeof()) == expected


@pytest.mark.parametrize(('value', 'expected'), [('foo_bar', 7), ('', 0)])
def test_string_length(con, value, expected):
    assert con.execute(L(value).length()) == expected


@pytest.mark.parametrize(
    ('op', 'expected'),
    [
        (methodcaller('substr', 0, 3), 'foo'),
        (methodcaller('substr', 4, 3), 'bar'),
        (methodcaller('substr', 1), 'oo_bar'),
    ],
)
def test_string_substring(con, op, expected):
    value = L('foo_bar')
    assert con.execute(op(value)) == expected


def test_string_column_substring(con, alltypes, translate):
    expr = alltypes.string_col.substr(2)
    assert translate(expr) == 'substring(`string_col`, 2 + 1)'
    assert len(con.execute(expr))

    expr = alltypes.string_col.substr(0, 3)
    assert translate(expr) == 'substring(`string_col`, 0 + 1, 3)'
    assert len(con.execute(expr))


def test_string_reverse(con):
    assert con.execute(L('foo').reverse()) == 'oof'


def test_string_upper(con):
    assert con.execute(L('foo').upper()) == 'FOO'


def test_string_lower(con):
    assert con.execute(L('FOO').lower()) == 'foo'


def test_string_lenght(con):
    assert con.execute(L('FOO').length()) == 3


@pytest.mark.parametrize(
    ('value', 'op', 'expected'),
    [
        (L('foobar'), methodcaller('contains', 'bar'), True),
        (L('foobar'), methodcaller('contains', 'foo'), True),
        (L('foobar'), methodcaller('contains', 'baz'), False),
        (L('100%'), methodcaller('contains', '%'), True),
        (L('a_b_c'), methodcaller('contains', '_'), True),
    ],
)
def test_string_contains(con, op, value, expected):
    assert con.execute(op(value)) == expected


# TODO: clickhouse-driver escaping bug
def test_re_replace(con, translate):
    expr1 = L('Hello, World!').re_replace('.', '\\\\0\\\\0')
    expr2 = L('Hello, World!').re_replace('^', 'here: ')

    assert con.execute(expr1) == 'HHeelllloo,,  WWoorrlldd!!'
    assert con.execute(expr2) == 'here: Hello, World!'


@pytest.mark.parametrize(
    ('value', 'expected'),
    [(L('a'), 0), (L('b'), 1), (L('d'), -1)],  # TODO: what's the expected?
)
def test_find_in_set(con, value, expected, translate):
    vals = list('abc')
    expr = value.find_in_set(vals)
    assert con.execute(expr) == expected


def test_string_column_find_in_set(con, alltypes, translate):
    s = alltypes.string_col
    vals = list('abc')

    expr = s.find_in_set(vals)
    assert translate(expr) == "indexOf(['a','b','c'], `string_col`) - 1"
    assert len(con.execute(expr))


@pytest.mark.parametrize(
    ('url', 'extract', 'expected'),
    [
        (L('https://www.cloudera.com'), 'HOST', 'www.cloudera.com'),
        (L('https://www.cloudera.com'), 'PROTOCOL', 'https'),
        (
            L('https://www.youtube.com/watch?v=kEuEcWfewf8&t=10'),
            'PATH',
            '/watch',
        ),
        (
            L('https://www.youtube.com/watch?v=kEuEcWfewf8&t=10'),
            'QUERY',
            'v=kEuEcWfewf8&t=10',
        ),
    ],
)
def test_parse_url(con, translate, url, extract, expected):
    expr = url.parse_url(extract)
    assert con.execute(expr) == expected


def test_parse_url_query_parameter(con, translate):
    url = L('https://www.youtube.com/watch?v=kEuEcWfewf8&t=10')
    expr = url.parse_url('QUERY', 't')
    assert con.execute(expr) == '10'

    expr = url.parse_url('QUERY', 'v')
    assert con.execute(expr) == 'kEuEcWfewf8'


@pytest.mark.parametrize(
    ('expr', 'expected'),
    [
        (L('foobar').find('bar'), 3),
        (L('foobar').find('baz'), -1),
        (L('foobar').like('%bar'), True),
        (L('foobar').like('foo%'), True),
        (L('foobar').like('%baz%'), False),
        (L('foobar').like(['%bar']), True),
        (L('foobar').like(['foo%']), True),
        (L('foobar').like(['%baz%']), False),
        (L('foobar').like(['%bar', 'foo%']), True),
        (L('foobarfoo').replace('foo', 'H'), 'HbarH'),
    ],
)
def test_string_find_like(con, expr, expected):
    assert con.execute(expr) == expected


def test_string_column_like(con, alltypes, translate):
    expr = alltypes.string_col.like('foo%')
    assert translate(expr) == "`string_col` LIKE 'foo%'"
    assert len(con.execute(expr))

    expr = alltypes.string_col.like(['foo%', '%bar'])
    expected = "`string_col` LIKE 'foo%' OR `string_col` LIKE '%bar'"
    assert translate(expr) == expected
    assert len(con.execute(expr))


def test_string_column_find(con, alltypes, translate):
    s = alltypes.string_col

    expr = s.find('a')
    assert translate(expr) == "position(`string_col`, 'a') - 1"
    assert len(con.execute(expr))

    expr = s.find(s)
    assert translate(expr) == "position(`string_col`, `string_col`) - 1"
    assert len(con.execute(expr))


@pytest.mark.parametrize(
    ('call', 'expected'),
    [
        (methodcaller('log'), 'log(`double_col`)'),
        (methodcaller('log2'), 'log2(`double_col`)'),
        (methodcaller('log10'), 'log10(`double_col`)'),
        (methodcaller('round'), 'round(`double_col`)'),
        (methodcaller('round', 0), 'round(`double_col`, 0)'),
        (methodcaller('round', 2), 'round(`double_col`, 2)'),
        (methodcaller('exp'), 'exp(`double_col`)'),
        (methodcaller('abs'), 'abs(`double_col`)'),
        (methodcaller('ceil'), 'ceil(`double_col`)'),
        (methodcaller('floor'), 'floor(`double_col`)'),
        (methodcaller('sqrt'), 'sqrt(`double_col`)'),
        (
            methodcaller('sign'),
            'intDivOrZero(`double_col`, abs(`double_col`))',
        ),
    ],
)
def test_translate_math_functions(con, alltypes, translate, call, expected):
    expr = call(alltypes.double_col)
    assert translate(expr) == expected
    assert len(con.execute(expr))


@pytest.mark.parametrize(
    ('expr', 'expected'),
    [
        (L(-5).abs(), 5),
        (L(5).abs(), 5),
        (L(5.5).round(), 6.0),
        (L(5.556).round(2), 5.56),
        (L(5.556).ceil(), 6.0),
        (L(5.556).floor(), 5.0),
        (L(5.556).exp(), math.exp(5.556)),
        (L(5.556).sign(), 1),
        (L(-5.556).sign(), -1),
        (L(0).sign(), 0),
        (L(5.556).sqrt(), math.sqrt(5.556)),
        (L(5.556).log(2), math.log(5.556, 2)),
        (L(5.556).ln(), math.log(5.556)),
        (L(5.556).log2(), math.log(5.556, 2)),
        (L(5.556).log10(), math.log10(5.556)),
    ],
)
def test_math_functions(con, expr, expected, translate):
    assert con.execute(expr) == expected


def test_greatest(con, alltypes, translate):
    expr = ibis.greatest(alltypes.int_col, 10)

    assert translate(expr) == "greatest(`int_col`, 10)"
    assert len(con.execute(expr))

    expr = ibis.greatest(alltypes.int_col, alltypes.bigint_col)
    assert translate(expr) == "greatest(`int_col`, `bigint_col`)"
    assert len(con.execute(expr))


def test_least(con, alltypes, translate):
    expr = ibis.least(alltypes.int_col, 10)
    assert translate(expr) == "least(`int_col`, 10)"
    assert len(con.execute(expr))

    expr = ibis.least(alltypes.int_col, alltypes.bigint_col)
    assert translate(expr) == "least(`int_col`, `bigint_col`)"
    assert len(con.execute(expr))


# TODO: clickhouse-driver escaping bug
@pytest.mark.parametrize(
    ('expr', 'expected'),
    [
        (L('abcd').re_search('[a-z]'), True),
        (L('abcd').re_search(r'[\\d]+'), False),
        (L('1222').re_search(r'[\\d]+'), True),
    ],
)
def test_regexp(con, expr, expected):
    assert con.execute(expr) == expected


@pytest.mark.parametrize(
    ('expr', 'expected'),
    [
        (L('abcd').re_extract('([a-z]+)', 0), 'abcd'),
        # (L('abcd').re_extract('(ab)(cd)', 1), 'cd'),
        # valid group number but no match => empty string
        (L('abcd').re_extract(r'(\\d)', 0), ''),
        # match but not a valid group number => NULL
        # (L('abcd').re_extract('abcd', 3), None),
    ],
)
def test_regexp_extract(con, expr, expected, translate):
    assert con.execute(expr) == expected


def test_column_regexp_extract(con, alltypes, translate):
    expected = r"extractAll(`string_col`, '[\d]+')[3 + 1]"

    expr = alltypes.string_col.re_extract(r'[\d]+', 3)
    assert translate(expr) == expected
    assert len(con.execute(expr))


def test_column_regexp_replace(con, alltypes, translate):
    expected = r"replaceRegexpAll(`string_col`, '[\d]+', 'aaa')"

    expr = alltypes.string_col.re_replace(r'[\d]+', 'aaa')
    assert translate(expr) == expected
    assert len(con.execute(expr))


def test_numeric_builtins_work(con, alltypes, df, translate):
    expr = alltypes.double_col
    result = expr.execute()
    expected = df.double_col.fillna(0)
    tm.assert_series_equal(result, expected)


def test_null_column(alltypes, translate):
    t = alltypes
    nrows = t.count().execute()
    expr = t.mutate(na_column=ibis.NA).na_column
    result = expr.execute()
    expected = pd.Series([None] * nrows, name='na_column')
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ('attr', 'expected'),
    [
        (operator.methodcaller('year'), {2009, 2010}),
        (operator.methodcaller('month'), set(range(1, 13))),
        (operator.methodcaller('day'), set(range(1, 32))),
    ],
)
def test_date_extract_field(db, alltypes, attr, expected):
    t = alltypes
    expr = attr(t.timestamp_col.cast('date')).distinct()
    result = expr.execute().astype(int)
    assert set(result) == expected


def test_timestamp_from_integer(con, alltypes, translate):
    # timestamp_col has datetime type
    expr = alltypes.int_col.to_timestamp()
    assert translate(expr) == 'toDateTime(`int_col`)'
    assert len(con.execute(expr))


def test_count_distinct_with_filter(alltypes):
    expr = alltypes.string_col.nunique(
        where=alltypes.string_col.cast('int64') > 1
    )
    result = expr.execute()
    expected = alltypes.string_col.execute()
    expected = expected[expected.astype('int64') > 1].nunique()
    assert result == expected
