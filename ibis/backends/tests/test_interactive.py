# Copyright 2014 Cloudera Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import pytest

import ibis
from ibis import config


@pytest.fixture
def queries(monkeypatch):
    queries = []
    monkeypatch.setattr(ibis.options, "verbose", True)
    monkeypatch.setattr(ibis.options, "verbose_log", queries.append)
    monkeypatch.setattr(ibis.options, "interactive", True)
    return queries


@pytest.fixture(scope="module")
def table(backend):
    return backend.functional_alltypes


@pytest.mark.notimpl(["dask", "pandas", "polars"])
def test_interactive_execute_on_repr(table, queries):
    repr(table.bigint_col.sum())
    assert len(queries) >= 1


def test_repr_png_is_none_in_interactive(table):
    with config.option_context("interactive", True):
        assert table._repr_png_() is None


def test_repr_png_is_not_none_in_not_interactive(table):
    pytest.importorskip("ibis.expr.visualize")

    with (
        config.option_context("interactive", False),
        config.option_context("graphviz_repr", True),
    ):
        assert table._repr_png_() is not None


@pytest.mark.notimpl(["dask", "pandas", "polars"])
def test_default_limit(table, queries):
    repr(table.select("id", "bool_col"))

    assert len(queries) >= 1


@pytest.mark.notimpl(["dask", "pandas", "polars"])
def test_respect_set_limit(table, queries):
    repr(table.select("id", "bool_col").limit(10))

    assert len(queries) >= 1


@pytest.mark.notimpl(["dask", "pandas", "polars"])
def test_disable_query_limit(table, queries):
    assert ibis.options.sql.default_limit is None

    with config.option_context("sql.default_limit", 10):
        assert ibis.options.sql.default_limit == 10
        repr(table.select("id", "bool_col"))

    assert len(queries) >= 1


def test_interactive_non_compilable_repr_does_not_fail(table):
    """https://github.com/ibis-project/ibis/issues/170"""
    repr(table.string_col.topk(3))


def test_isin_rule_suppressed_exception_repr_not_fail(table):
    bool_clause = table["string_col"].notin(["1", "4", "7"])
    expr = table[bool_clause]["string_col"].value_counts()

    repr(expr)
