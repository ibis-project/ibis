import pytest
from pytest import param

import ibis.expr.types as ir
from ibis.backends.conftest import TEST_TABLES


def test_backend_name(backend):
    # backend is the TestConf for the backend
    assert backend.api.name == backend.name()


def test_version(backend):
    assert isinstance(backend.api.version, str)


# 1. `current_database` returns '.', but isn't listed in list_databases()
@pytest.mark.never(["dask", "pandas"], reason="pass")
@pytest.mark.notimpl(["datafusion", "duckdb", "polars", "mssql", "trino"])
def test_database_consistency(backend, con):
    # every backend has a different set of databases, not testing the
    # exact names for now
    databases = con.list_databases()
    assert isinstance(databases, list)
    assert len(databases) >= 1
    assert all(isinstance(database, str) for database in databases)

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
    assert 'functional_alltypes' in tables
    assert all(isinstance(table, str) for table in tables)


def test_tables_accessor_mapping(con):
    if con.name == "snowflake":
        pytest.skip("snowflake sometimes counts more tables than are around")

    assert isinstance(con.tables["functional_alltypes"], ir.Table)

    with pytest.raises(KeyError, match="doesnt_exist"):
        con.tables["doesnt_exist"]

    # temporary might pop into existence in parallel test runs, in between the
    # first `list_tables` call and the second, so we check a subset relationship
    assert TEST_TABLES.keys() <= set(con.list_tables())
    assert TEST_TABLES.keys() <= set(con.tables)


def test_tables_accessor_getattr(con):
    assert isinstance(con.tables.functional_alltypes, ir.Table)

    with pytest.raises(AttributeError, match="doesnt_exist"):
        con.tables.doesnt_exist

    # Underscore/double-underscore attributes are never available, since many
    # python apis expect checking for the absence of these to be cheap.
    with pytest.raises(AttributeError, match="_private_attr"):
        con.tables._private_attr


def test_tables_accessor_tab_completion(con):
    attrs = dir(con.tables)
    assert 'functional_alltypes' in attrs
    assert 'keys' in attrs  # type methods also present

    keys = con.tables._ipython_key_completions_()
    assert 'functional_alltypes' in keys


def test_tables_accessor_repr(con):
    result = repr(con.tables)
    assert '- functional_alltypes' in result


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
