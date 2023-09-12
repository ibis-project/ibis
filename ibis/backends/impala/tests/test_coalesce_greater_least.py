from __future__ import annotations

import pytest

import ibis
from ibis.backends.impala.tests.conftest import translate


@pytest.fixture(scope="module")
def table(mockcon):
    return mockcon.table("functional_alltypes")


@pytest.mark.parametrize(
    "expr_fn",
    [
        pytest.param(
            lambda t: ibis.coalesce(t.string_col, "foo"), id="coalesce_scalar"
        ),
        pytest.param(
            lambda t: ibis.coalesce(t.int_col, t.bigint_col), id="coalesce_columns"
        ),
        pytest.param(
            lambda t: ibis.greatest(t.string_col, "foo"), id="greatest_scalar"
        ),
        pytest.param(
            lambda t: ibis.greatest(t.int_col, t.bigint_col), id="greatest_columns"
        ),
        pytest.param(lambda t: ibis.least(t.string_col, "foo"), id="least_scalar"),
        pytest.param(lambda t: ibis.least(t.int_col, t.bigint_col), id="least_columns"),
    ],
)
def test_varargs_functions(table, expr_fn, snapshot):
    t = table
    expr = expr_fn(t)
    result = translate(expr)
    snapshot.assert_match(result, "out.sql")
