import pandas as pd
import pytest

import ibis
from ibis import literal as L
from ibis.backends.impala.compiler import ImpalaCompiler
from ibis.backends.impala.tests.conftest import translate


@pytest.fixture(scope="module")
def table(mockcon):
    return mockcon.table("alltypes")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param('simple', "'simple'", id="simple"),
        pytest.param('I can\'t', "'I can\\'t'", id="embedded_single_quote"),
        pytest.param(
            'An "escape"',
            "'An \"escape\"'",
            id="embedded_double_quote",
        ),
        pytest.param(5, '5', id="int"),
        pytest.param(1.5, '1.5', id="float"),
        pytest.param(True, 'TRUE', id="true"),
        pytest.param(False, 'FALSE', id="false"),
    ],
)
def test_literals(value, expected):
    expr = L(value)
    result = translate(expr)
    assert result == expected


@pytest.mark.parametrize(
    ("expr_fn", "expected"),
    [
        pytest.param(
            lambda c: c.precision(),
            'precision(`l_extendedprice`)',
            id="precision",
        ),
        pytest.param(
            lambda c: c.scale(),
            'scale(`l_extendedprice`)',
            id="scale",
        ),
    ],
)
def test_decimal_builtins(mockcon, expr_fn, expected):
    t = mockcon.table('tpch_lineitem')
    col = t.l_extendedprice
    expr = expr_fn(col)
    result = translate(expr)
    assert result == expected


def test_column_ref_table_aliases():
    context = ImpalaCompiler.make_context()

    table1 = ibis.table([('key1', 'string'), ('value1', 'double')])

    table2 = ibis.table([('key2', 'string'), ('value and2', 'double')])

    context.set_ref(table1, 't0')
    context.set_ref(table2, 't1')

    expr = table1['value1'] - table2['value and2']

    result = translate(expr, context=context)
    expected = 't0.`value1` - t1.`value and2`'
    assert result == expected


def test_column_ref_quoting():
    schema = [('has a space', 'double')]
    table = ibis.table(schema)
    translate(table['has a space'], named='`has a space`')


def test_identifier_quoting():
    schema = [('date', 'double'), ('table', 'string')]
    table = ibis.table(schema)
    translate(table['date'], named='`date`')
    translate(table['table'], named='`table`')


@pytest.mark.parametrize(
    ("expr_fn", "expected"),
    [
        pytest.param(
            lambda t: t.g.cast('double').name('g_dub'),
            'CAST(`g` AS double) AS `g_dub`',
            id="named_cast",
        ),
        pytest.param(
            lambda t: t.g.name('has a space'),
            '`g` AS `has a space`',
            id="named_spaces",
        ),
        pytest.param(
            lambda t: ((t.a - t.b) * t.a).name('expr'),
            '(`a` - `b`) * `a` AS `expr`',
            id="named_compound_expr",
        ),
    ],
)
def test_named_expressions(table, expr_fn, expected):
    expr = expr_fn(table)
    result = translate(expr, named=True)
    assert result == expected


@pytest.mark.parametrize(
    ("expr_fn", "expected"),
    [
        pytest.param(lambda t: t.a + t.b, '`a` + `b`', id="add"),
        pytest.param(lambda t: t.a - t.b, '`a` - `b`', id="sub"),
        pytest.param(lambda t: t.a * t.b, '`a` * `b`', id="mul"),
        pytest.param(lambda t: t.a / t.b, '`a` / `b`', id="div"),
        pytest.param(lambda t: t.a**t.b, 'pow(`a`, `b`)', id="pow"),
        pytest.param(lambda t: t.a < t.b, '`a` < `b`', id="lt"),
        pytest.param(lambda t: t.a <= t.b, '`a` <= `b`', id="le"),
        pytest.param(lambda t: t.a > t.b, '`a` > `b`', id="gt"),
        pytest.param(lambda t: t.a >= t.b, '`a` >= `b`', id="ge"),
        pytest.param(lambda t: t.a == t.b, '`a` = `b`', id="eq"),
        pytest.param(lambda t: t.a != t.b, '`a` != `b`', id="ne"),
        pytest.param(lambda t: t.h & (t.a > 0), '`h` AND (`a` > 0)', id="and"),
        pytest.param(lambda t: t.h | (t.a > 0), '`h` OR (`a` > 0)', id="or"),
        pytest.param(
            lambda t: t.h ^ (t.a > 0),
            '(`h` OR (`a` > 0)) AND NOT (`h` AND (`a` > 0))',
            id="xor",
        ),
    ],
)
def test_binary_infix_operators(table, expr_fn, expected):
    expr = expr_fn(table)
    result = translate(expr)
    assert result == expected


@pytest.mark.parametrize(
    ("expr_fn", "expected"),
    [
        pytest.param(
            lambda t: (t.a + t.b) + t.c,
            '(`a` + `b`) + `c`',
            id="parens_left",
        ),
        pytest.param(
            lambda t: t.a.log() + t.c,
            'ln(`a`) + `c`',
            id="function_call",
        ),
        pytest.param(
            lambda t: t.b + (-(t.a + t.c)),
            '`b` + (-(`a` + `c`))',
            id="negation",
        ),
    ],
)
def test_binary_infix_parenthesization(table, expr_fn, expected):
    expr = expr_fn(table)
    result = translate(expr)
    assert result == expected


def test_between(table):
    expr = table.f.between(0, 1)
    expected = '`f` BETWEEN 0 AND 1'
    result = translate(expr)
    assert result == expected


@pytest.mark.parametrize(
    ("expr_fn", "expected"),
    [
        pytest.param(lambda t: t['g'].isnull(), '`g` IS NULL', id="isnull"),
        pytest.param(
            lambda t: t['a'].notnull(), '`a` IS NOT NULL', id="notnull"
        ),
        pytest.param(
            lambda t: (t['a'] + t['b']).isnull(),
            '`a` + `b` IS NULL',
            id="compound_isnull",
        ),
    ],
)
def test_isnull_notnull(table, expr_fn, expected):
    expr = expr_fn(table)
    result = translate(expr)
    assert result == expected


@pytest.mark.parametrize(
    ("column", "to_type", "expected_type"),
    [
        ("a", "int16", "smallint"),
        ("a", "int32", "int"),
        ("a", "int64", "bigint"),
        ("a", "string", "string"),
        ("d", "int8", "tinyint"),
        ("g", "double", "double"),
        ("g", "timestamp", "timestamp"),
    ],
)
def test_casts(table, column, to_type, expected_type):
    expr = table[column].cast(to_type)
    result = translate(expr)
    assert result == f"CAST(`{column}` AS {expected_type})"


def test_misc_conditionals(table):
    expr = table.a.nullif(0)
    expected = 'nullif(`a`, 0)'
    result = translate(expr)
    assert result == expected


@pytest.mark.parametrize(
    ("expr_fn", "expected"),
    [
        pytest.param(
            lambda _: L('9.9999999').cast('decimal(38, 5)'),
            "CAST('9.9999999' AS decimal(38, 5))",
            id="literal",
        ),
        pytest.param(
            lambda t: t.f.cast('decimal(12, 2)'),
            "CAST(`f` AS decimal(12, 2))",
            id="column",
        ),
    ],
)
def test_decimal_casts(table, expr_fn, expected):
    expr = expr_fn(table)
    result = translate(expr)
    assert result == expected


@pytest.mark.parametrize(
    ("expr_fn", "expected"),
    [
        pytest.param(lambda t: -t['a'], '-`a`', id="negate_a"),
        pytest.param(lambda t: -t['f'], '-`f`', id="negate_f"),
        pytest.param(lambda t: -t['h'], 'NOT `h`', id="negate_bool"),
    ],
)
def test_negate(table, expr_fn, expected):
    expr = expr_fn(table)
    result = translate(expr)
    assert result == expected


@pytest.mark.parametrize(
    "field",
    [
        'year',
        'month',
        'day',
        'hour',
        'minute',
        'second',
        'millisecond',
    ],
)
def test_timestamp_extract_field(table, field):
    expr = getattr(table.i, field)()
    expected = f"extract(`i`, '{field}')"
    result = translate(expr)
    assert result == expected


def test_sql_extract(table):
    # integration with SQL translation
    expr = table[
        table.i.year().name('year'),
        table.i.month().name('month'),
        table.i.day().name('day'),
    ]

    result = ImpalaCompiler.to_sql(expr)
    expected = """\
SELECT extract(`i`, 'year') AS `year`, extract(`i`, 'month') AS `month`,
       extract(`i`, 'day') AS `day`
FROM alltypes"""
    assert result == expected


def test_timestamp_now():
    expr = ibis.now()
    result = translate(expr)
    assert result == "now()"


@pytest.mark.parametrize(
    ("unit", "compiled_unit"),
    [
        ('years', 'YEAR'),
        ('months', 'MONTH'),
        ('weeks', 'WEEK'),
        ('days', 'DAY'),
        ('hours', 'HOUR'),
        ('minutes', 'MINUTE'),
        ('seconds', 'SECOND'),
    ],
)
def test_timestamp_deltas(table, unit, compiled_unit):
    f = '`i`'

    K = 5

    offset = ibis.interval(**{unit: K})

    add_expr = table.i + offset
    result = translate(add_expr)
    assert result == f'date_add({f}, INTERVAL {K} {compiled_unit})'

    sub_expr = table.i - offset
    result = translate(sub_expr)
    assert result == f'date_sub({f}, INTERVAL {K} {compiled_unit})'


@pytest.mark.parametrize(
    ("expr_fn", "expected"),
    [
        pytest.param(
            lambda v: L(pd.Timestamp(v)),
            "'2015-01-01T12:34:56'",
            id="literal_pd_timestamp",
        ),
        pytest.param(
            lambda v: L(pd.Timestamp(v).to_pydatetime()),
            "'2015-01-01T12:34:56'",
            id="literal_pydatetime",
        ),
        pytest.param(
            lambda v: ibis.timestamp(v),
            "'2015-01-01T12:34:56'",
            id="ibis_timestamp_function",
        ),
    ],
)
def test_timestamp_literals(expr_fn, expected):
    expr = expr_fn('2015-01-01 12:34:56')
    result = translate(expr)
    assert result == expected


@pytest.mark.parametrize(
    ("expr_fn", "expected"),
    [
        pytest.param(
            lambda v: v.day_of_week.index(),
            "pmod(dayofweek('2015-09-01T01:00:23') - 2, 7)",
            id="index",
        ),
        pytest.param(
            lambda v: v.day_of_week.full_name(),
            "dayname('2015-09-01T01:00:23')",
            id="full_name",
        ),
    ],
)
def test_timestamp_day_of_week(expr_fn, expected):
    expr = expr_fn(ibis.timestamp('2015-09-01T01:00:23'))
    result = translate(expr)
    assert result == expected


@pytest.mark.parametrize(
    ("expr_fn", "expected"),
    [
        pytest.param(
            lambda col: col.to_timestamp(),
            'CAST(from_unixtime(`c`, "yyyy-MM-dd HH:mm:ss") ' 'AS timestamp)',
            id="default",
        ),
        pytest.param(
            lambda col: col.to_timestamp('ms'),
            (
                'CAST(from_unixtime(CAST(floor(`c` / 1000) AS int), '
                '"yyyy-MM-dd HH:mm:ss") '
                'AS timestamp)'
            ),
            id="ms",
        ),
        pytest.param(
            lambda col: col.to_timestamp('us'),
            (
                'CAST(from_unixtime(CAST(floor(`c` / 1000000) AS int), '
                '"yyyy-MM-dd HH:mm:ss") '
                'AS timestamp)'
            ),
            id="us",
        ),
    ],
)
def test_timestamp_from_integer(table, expr_fn, expected):
    expr = expr_fn(table.c)
    result = translate(expr)
    assert result == expected


def test_correlated_predicate_subquery(table):
    t0 = table
    t1 = t0.view()

    expr = t0.g == t1.g

    ctx = ImpalaCompiler.make_context()
    ctx.make_alias(t0)

    # Grab alias from parent context
    subctx = ctx.subcontext()
    subctx.make_alias(t1)
    subctx.make_alias(t0)

    result = translate(expr, context=subctx)
    expected = "t0.`g` = t1.`g`"
    assert result == expected


@pytest.mark.parametrize(
    ("expr_fn", "expected"),
    [
        pytest.param(lambda b: b.any(), 'max(`f` = 0)', id="any"),
        pytest.param(
            lambda b: -b.any(),
            'max(`f` = 0) = FALSE',
            id="not_any",
        ),
        pytest.param(lambda b: b.all(), 'min(`f` = 0)', id="all"),
        pytest.param(
            lambda b: -b.all(),
            'min(`f` = 0) = FALSE',
            id="not_all",
        ),
    ],
)
def test_any_all(table, expr_fn, expected):
    expr = expr_fn(table.f == 0)
    result = translate(expr)
    assert result == expected
