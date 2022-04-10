import pytest

from ibis import literal as L
from ibis.backends.impala.compiler import ImpalaCompiler
from ibis.backends.impala.tests.conftest import translate


@pytest.fixture(scope="module")
def table(mockcon):
    return mockcon.table("alltypes")


@pytest.mark.parametrize(
    ("expr_fn", "expected_fn"),
    [
        pytest.param(
            lambda t, values: t.g.isin(values),
            lambda values: f"`g` IN {tuple(values)}",
            id="in",
        ),
        pytest.param(
            lambda t, values: t.g.notin(values),
            lambda values: f"`g` NOT IN {tuple(values)}",
            id="not_in",
        ),
    ],
)
def test_field_in_literals(table, expr_fn, expected_fn):
    values = {'foo', 'bar', 'baz'}
    expr = expr_fn(table, values)
    expected = expected_fn(values)
    result = translate(expr)
    assert result == expected


@pytest.mark.parametrize(
    ("expr_fn", "expected"),
    [
        pytest.param(
            lambda t: L(2).isin([t.a, t.b, t.c]),
            '2 IN (`a`, `b`, `c`)',
            id="in",
        ),
        pytest.param(
            lambda t: L(2).notin([t.a, t.b, t.c]),
            '2 NOT IN (`a`, `b`, `c`)',
            id="not_in",
        ),
    ],
)
def test_literal_in_fields(table, expr_fn, expected):
    expr = expr_fn(table)
    result = translate(expr)
    assert result == expected


def test_isin_notin_in_select(table):
    values = {'foo', 'bar'}
    values_formatted = tuple(values)

    filtered = table[table.g.isin(values)]
    result = ImpalaCompiler.to_sql(filtered)
    expected = f"""SELECT *
FROM alltypes
WHERE `g` IN {values_formatted}"""
    assert result == expected

    filtered = table[table.g.notin(values)]
    result = ImpalaCompiler.to_sql(filtered)
    expected = f"""SELECT *
FROM alltypes
WHERE `g` NOT IN {values_formatted}"""
    assert result == expected
