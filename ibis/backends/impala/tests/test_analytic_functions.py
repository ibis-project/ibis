from __future__ import annotations

import pytest

from ibis.backends.impala.tests.conftest import translate


@pytest.fixture(scope="module")
def table(mockcon):
    return mockcon.table("functional_alltypes")


@pytest.mark.parametrize(
    "expr_fn",
    [
        pytest.param(lambda t: t.string_col.lag(), id="lag_default"),
        pytest.param(lambda t: t.string_col.lag(2), id="lag_arg"),
        pytest.param(lambda t: t.string_col.lag(default=0), id="lag_explicit_default"),
        pytest.param(lambda t: t.string_col.lead(), id="lead_default"),
        pytest.param(lambda t: t.string_col.lead(2), id="lead_arg"),
        pytest.param(
            lambda t: t.string_col.lead(default=0), id="lead_explicit_default"
        ),
        pytest.param(lambda t: t.double_col.first().over(order_by="id"), id="first"),
        pytest.param(lambda t: t.double_col.last().over(order_by="id"), id="last"),
        pytest.param(lambda t: t.double_col.ntile(3), id="ntile"),
        pytest.param(lambda t: t.double_col.percent_rank(), id="percent_rank"),
    ],
)
def test_analytic_exprs(table, expr_fn, snapshot):
    expr = expr_fn(table)
    result = translate(expr)
    snapshot.assert_match(result, "out.sql")
