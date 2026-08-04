"""Db2 backend-specific tests (connection, table management, queries).

These tests are run by the standard ibis CI matrix and require a running
Db2 instance (started via ``just up db2``).
"""

from __future__ import annotations

from ibis.util import gen_name


def test_connect(con):
    """Basic sanity check: connection object is a Db2 Backend."""
    from ibis.backends.db2 import Backend

    assert isinstance(con, Backend)


def test_list_databases(con):
    databases = con.list_databases()
    assert isinstance(databases, list)
    assert len(databases) > 0


def test_list_tables_like(con):
    tables = con.list_tables(like="functional%")
    assert "functional_alltypes" in tables


def test_table_expression(con):
    t = con.table("functional_alltypes")
    assert t is not None
    assert "id" in t.columns


def test_filter_and_execute(con):
    t = con.table("functional_alltypes")
    result = t.filter(t.id == 1).execute()
    assert len(result) >= 0  # just ensure it doesn't error


def test_aggregate(con):
    t = con.table("functional_alltypes")
    result = t.aggregate(n=t.count()).execute()
    assert result["n"].iat[0] > 0


def test_limit(con):
    t = con.table("functional_alltypes")
    result = t.limit(5).execute()
    assert len(result) == 5


def test_group_by(con):
    t = con.table("functional_alltypes")
    result = t.group_by("bool_col").aggregate(n=t.count()).execute()
    assert len(result) <= 2


def test_union(con):
    t = con.table("win")
    result = t.union(t).execute()
    assert len(result) == len(t.execute()) * 2


def test_create_and_drop_table_in_backend(con):
    name = gen_name("db2_backend_test")
    import pandas as pd

    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    tbl = con.create_table(name, df)
    try:
        assert tbl is not None
        result = tbl.execute()
        assert len(result) == 3
    finally:
        con.drop_table(name, force=True)
