import pytest
from pytest import param

import ibis
import ibis.common.exceptions as exc


def test_backend_name(backend):
    # backend is the TestConf for the backend
    assert backend.api.name == backend.name()


def test_version(backend):
    assert isinstance(backend.api.version, str)


@pytest.mark.parametrize('table_name', ['functional_alltypes', 'unexisting'])
def test_exists_table(con, table_name):
    expected = table_name in con.list_tables()

    with pytest.warns(FutureWarning):
        actual = con.exists_table(table_name)

    assert actual == expected


# 1. `current_database` returns '.', but isn't listed in list_databases()
# 2. list_databases() returns directories which don't make sense as HDF5
#    databases
@pytest.mark.never(["dask", "pandas"], reason="pass")
@pytest.mark.notimpl(["datafusion", "duckdb"])
def test_database_consistency(con):
    # every backend has a different set of databases, not testing the
    # exact names for now
    databases = con.list_databases()
    assert isinstance(databases, list)
    assert len(databases) >= 1
    assert all(isinstance(database, str) for database in databases)

    current_database = con.current_database
    assert isinstance(current_database, str)
    assert current_database in databases


def test_list_tables(con):
    tables = con.list_tables()
    assert isinstance(tables, list)
    # only table that is guaranteed to be in all backends
    assert 'functional_alltypes' in tables
    assert all(isinstance(table, str) for table in tables)


def test_tables_accessor_mapping(con):
    assert isinstance(con.tables["functional_alltypes"], ibis.ir.Table)

    with pytest.raises(KeyError, match="doesnt_exist"):
        con.tables["doesnt_exist"]

    tables = con.list_tables()

    assert len(con.tables) == len(tables)
    assert sorted(con.tables) == sorted(tables)


def test_tables_accessor_getattr(con):
    assert isinstance(con.tables.functional_alltypes, ibis.ir.Table)

    with pytest.raises(AttributeError, match="doesnt_exist"):
        getattr(con.tables, "doesnt_exist")

    # Underscore/double-underscore attributes are never available, since many
    # python apis expect checking for the absence of these to be cheap.
    with pytest.raises(AttributeError, match="_private_attr"):
        getattr(con.tables, "_private_attr")


def test_tables_accessor_tab_completion(con):
    attrs = dir(con.tables)
    assert 'functional_alltypes' in attrs
    assert 'keys' in attrs  # type methods also present

    keys = con.tables._ipython_key_completions_()
    assert 'functional_alltypes' in keys


@pytest.mark.notimpl(["datafusion"], raises=exc.OperationNotDefinedError)
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
