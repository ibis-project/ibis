import pytest

import ibis
from ibis.backends.impala.tests.conftest import translate


@pytest.fixture(scope="module")
def table(mockcon):
    return mockcon.table("functional_alltypes")


@pytest.mark.parametrize(
    ("expr_fn", "expected"),
    [
        pytest.param(
            lambda t: ibis.row_number().over(
                ibis.window(order_by=t.float_col)
            ),
            '(row_number() OVER (ORDER BY `float_col`) - 1)',
        ),
        pytest.param(
            lambda t: t.string_col.lag(), 'lag(`string_col`)', id="lag_default"
        ),
        pytest.param(
            lambda t: t.string_col.lag(2), 'lag(`string_col`, 2)', id="lag_arg"
        ),
        pytest.param(
            lambda t: t.string_col.lag(default=0),
            'lag(`string_col`, 1, 0)',
            id="lag_explicit_default",
        ),
        pytest.param(
            lambda t: t.string_col.lead(),
            'lead(`string_col`)',
            id="lead_default",
        ),
        pytest.param(
            lambda t: t.string_col.lead(2),
            'lead(`string_col`, 2)',
            id="lead_arg",
        ),
        pytest.param(
            lambda t: t.string_col.lead(default=0),
            'lead(`string_col`, 1, 0)',
            id="lead_explicit_default",
        ),
        pytest.param(
            lambda t: t.double_col.first(),
            'first_value(`double_col`)',
            id="first",
        ),
        pytest.param(
            lambda t: t.double_col.last(),
            'last_value(`double_col`)',
            id="last",
        ),
        # (t.double_col.nth(4), 'first_value(lag(double_col, 4 - 1))')
        pytest.param(lambda t: t.double_col.ntile(3), 'ntile(3)', id="ntile"),
        pytest.param(
            lambda t: t.double_col.percent_rank(),
            'percent_rank()',
            id="percent_rank",
        ),
    ],
)
def test_analytic_exprs(table, expr_fn, expected):
    expr = expr_fn(table)
    result = translate(expr)
    assert result == expected
