from __future__ import annotations

import pytest

from ibis import literal as L
from ibis.backends.impala.compiler import ImpalaCompiler
from ibis.backends.impala.tests.conftest import translate


@pytest.fixture(scope="module")
def table(mockcon):
    return mockcon.table("alltypes")


@pytest.mark.parametrize("method_name", ["isin", "notin"])
def test_field_in_literals(table, method_name, snapshot):
    values = ["foo", "bar", "baz"]
    method = getattr(table.g, method_name)
    expr = method(values)
    result = translate(expr)
    snapshot.assert_match(result, "out.sql")


@pytest.mark.parametrize("method_name", ["isin", "notin"])
def test_literal_in_fields(table, method_name, snapshot):
    values = [table.a, table.b, table.c]
    method = getattr(L(2), method_name)
    expr = method(values)
    result = translate(expr)
    snapshot.assert_match(result, "out.sql")


@pytest.mark.parametrize("method_name", ["isin", "notin"])
def test_isin_notin_in_select(table, method_name, snapshot):
    values = ["foo", "bar"]
    method = getattr(table.g, method_name)
    filtered = table[method(values)]
    result = ImpalaCompiler.to_sql(filtered)
    snapshot.assert_match(result, "out.sql")
