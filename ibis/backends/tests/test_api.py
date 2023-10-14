from __future__ import annotations

import pytest
from pytest import param

import ibis.expr.types as ir
from ibis.backends.conftest import TEST_TABLES


def test_backend_name(backend):
    # backend is the TestConf for the backend
    assert backend.api.name == backend.name()


@pytest.mark.notimpl(
    ["druid"], raises=TypeError, reason="'NoneType' object is not iterable"
)
def test_version(backend):
    assert isinstance(backend.api.version, str)


# 1. `current_database` returns '.', but isn't listed in list_databases()
@pytest.mark.never(
    ["polars", "dask", "pandas", "druid", "oracle"],
    reason="backend does not support databases",
    raises=AttributeError,
)
@pytest.mark.notimpl(
    ["datafusion"],
    raises=NotImplementedError,
    reason="current_database isn't implemented",
)
@pytest.mark.never(
    ["bigquery"], raises=FutureWarning, reason="list_databases is deprecated"
)
def test_database_consistency(backend, con):
    databases = con.list_databases()
    assert isinstance(databases, list)
    assert len(databases) >= 1
    assert all(isinstance(database, str) for database in databases)

    # every backend has a different set of databases, not testing the
    # exact names for now
    current_database = con.current_database
    assert isinstance(current_database, str)
    if backend.name() == "snowflake":
        assert current_database.upper() in databases
    else:
        assert current_database in databases


def test_list_tables(con):
    tables = con.list_tables()
    assert isinstance(tables, list)
    # only table that is guaranteed to be in all backends
    key = "functional_alltypes"
    assert key in tables or key.upper() in tables
    assert all(isinstance(table, str) for table in tables)


def test_tables_accessor_mapping(backend, con):
    if con.name == "snowflake":
        pytest.skip("snowflake sometimes counts more tables than are around")

    name = backend.default_identifier_case_fn("functional_alltypes")

    assert isinstance(con.tables[name], ir.Table)

    with pytest.raises(KeyError, match="doesnt_exist"):
        con.tables["doesnt_exist"]

    # temporary might pop into existence in parallel test runs, in between the
    # first `list_tables` call and the second, so we check that there's a
    # non-empty intersection
    assert TEST_TABLES.keys() & set(map(str.lower, con.list_tables()))
    assert TEST_TABLES.keys() & set(map(str.lower, con.tables))


def test_tables_accessor_getattr(backend, con):
    name = backend.default_identifier_case_fn("functional_alltypes")
    assert isinstance(getattr(con.tables, name), ir.Table)

    with pytest.raises(AttributeError, match="doesnt_exist"):
        con.tables.doesnt_exist  # noqa: B018

    # Underscore/double-underscore attributes are never available, since many
    # python apis expect checking for the absence of these to be cheap.
    with pytest.raises(AttributeError, match="_private_attr"):
        con.tables._private_attr  # noqa: B018


def test_tables_accessor_tab_completion(backend, con):
    name = backend.default_identifier_case_fn("functional_alltypes")
    attrs = dir(con.tables)
    assert name in attrs
    assert "keys" in attrs  # type methods also present

    keys = con.tables._ipython_key_completions_()
    assert name in keys


def test_tables_accessor_repr(backend, con):
    name = backend.default_identifier_case_fn("functional_alltypes")
    result = repr(con.tables)
    assert f"- {name}" in result


@pytest.mark.parametrize(
    "expr_fn",
    [
        param(lambda t: t.limit(5).limit(10), id="small_big"),
        param(lambda t: t.limit(10).limit(5), id="big_small"),
    ],
)
def test_limit_chain(alltypes, expr_fn):
    expr = expr_fn(alltypes)
    result = expr.execute()
    assert len(result) == 5


@pytest.mark.parametrize(
    "expr_fn",
    [
        param(lambda t: t, id="alltypes table"),
        param(lambda t: t.join(t.view(), t.id == t.view().int_col), id="self join"),
    ],
)
def test_unbind(alltypes, expr_fn):
    expr = expr_fn(alltypes)
    assert expr.unbind() != expr
    assert expr.unbind().schema() == expr.schema()

    assert "Unbound" not in repr(expr)
    assert "Unbound" in repr(expr.unbind())
