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

import shutil

import pytest

import ibis
import ibis.common.exceptions as exc


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


@pytest.mark.notimpl(["polars"])
def test_interactive_execute_on_repr(table, queries):
    repr(table.bigint_col.sum())
    assert len(queries) >= 1


def test_repr_png_is_none_in_interactive(table, monkeypatch):
    monkeypatch.setattr(ibis.options, "interactive", True)
    assert table._repr_png_() is None


def test_repr_png_is_not_none_in_not_interactive(table, monkeypatch):
    pytest.importorskip("ibis.expr.visualize")
    monkeypatch.setattr(ibis.options, "interactive", False)
    monkeypatch.setattr(ibis.options, "graphviz_repr", True)

    assert shutil.which("dot") is not None
    assert table._repr_png_() is not None


@pytest.mark.notimpl(["polars"])
def test_default_limit(table, queries):
    repr(table.select("id", "bool_col"))

    assert len(queries) >= 1


@pytest.mark.notimpl(["polars"])
def test_respect_set_limit(table, queries):
    repr(table.select("id", "bool_col").limit(10))

    assert len(queries) >= 1


@pytest.mark.notimpl(["polars"])
def test_disable_query_limit(table, queries, monkeypatch):
    assert ibis.options.sql.default_limit is None

    monkeypatch.setattr(ibis.options.sql, "default_limit", 10)

    assert ibis.options.sql.default_limit == 10

    repr(table.select("id", "bool_col"))

    assert len(queries) >= 1


def test_interactive_non_compilable_repr_does_not_fail(table):
    """https://github.com/ibis-project/ibis/issues/170"""
    repr(table.string_col.topk(3))


def test_isin_rule_suppressed_exception_repr_not_fail(table):
    bool_clause = table["string_col"].notin(["1", "4", "7"])
    expr = table.filter(bool_clause)["string_col"].value_counts()

    repr(expr)


def test_no_recursion_error(con, monkeypatch):
    monkeypatch.setattr(ibis.options, "interactive", True)
    monkeypatch.setattr(ibis.options, "default_backend", con)

    a = ibis.memtable({"a": [1]})
    b = ibis.memtable({"b": [1]})

    expr = a.count() + b.count()

    with pytest.raises(
        exc.RelationError, match="The scalar expression cannot be converted"
    ):
        repr(expr)


@pytest.mark.notimpl(
    ["impala", "flink", "pyspark"],
    reason="backend calls `execute` as part of pyarrow conversion",
)
def test_scalar_uses_pyarrow(con, table, monkeypatch, mocker):
    monkeypatch.setattr(ibis.options, "interactive", True)

    execute_spy = mocker.spy(con, "execute")
    to_pyarrow_spy = mocker.spy(con, "to_pyarrow")

    repr(table.limit(1).string_col)

    # pyarrow does get called
    to_pyarrow_spy.assert_called_once()

    # execute doesn't get called
    execute_spy.assert_not_called()
