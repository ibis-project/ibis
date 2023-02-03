import uuid
from pathlib import Path

import numpy as np
import pandas.testing as tm
import pytest

import ibis
import ibis.expr.types as ir
from ibis import config

pytest.importorskip("sqlalchemy")


def test_table(con):
    table = con.table('functional_alltypes')
    assert isinstance(table, ir.Table)


def test_column_execute(alltypes, df):
    expr = alltypes.double_col
    result = expr.execute()
    expected = df.double_col
    tm.assert_series_equal(result, expected)


def test_literal_execute(con):
    expr = ibis.literal('1234')
    result = con.execute(expr)
    assert result == '1234'


def test_simple_aggregate_execute(alltypes, df):
    expr = alltypes.double_col.sum()
    result = expr.execute()
    expected = df.double_col.sum()
    np.testing.assert_allclose(result, expected)


def test_list_tables(con):
    assert con.list_tables()
    assert len(con.list_tables(like='functional')) == 1


def test_attach_file(tmp_path):
    dbpath = str(tmp_path / "attached.db")
    path_client = ibis.sqlite.connect(dbpath)
    path_client.create_table("test", schema=ibis.schema(dict(a="int")))

    client = ibis.sqlite.connect()

    assert not client.list_tables()

    client.attach('baz', Path(dbpath))
    client.attach('bar', dbpath)

    foo_tables = client.list_tables(database='baz')
    bar_tables = client.list_tables(database='bar')

    assert foo_tables == ["test"]
    assert foo_tables == bar_tables


def test_compile_toplevel(snapshot):
    t = ibis.table([("a", "double")], name="t")

    expr = t.a.sum()
    result = ibis.sqlite.compile(expr)
    snapshot.assert_match(str(result), "out.sql")


def test_create_and_drop_table(con):
    t = con.table('functional_alltypes')
    name = str(uuid.uuid4())
    con.create_table(name, t.limit(5))
    new_table = con.table(name)
    tm.assert_frame_equal(new_table.execute(), t.limit(5).execute())
    con.drop_table(name)
    assert name not in con.list_tables()


def test_verbose_log_queries(con):
    queries = []

    with config.option_context('verbose', True):
        with config.option_context('verbose_log', queries.append):
            con.table('functional_alltypes')['year'].execute()

    assert len(queries) == 1
    (query,) = queries
    assert "SELECT t0.year" in query


def test_table_equality(dbpath):
    con1 = ibis.sqlite.connect(dbpath)
    batting1 = con1.table("batting")

    con2 = ibis.sqlite.connect(dbpath)
    batting2 = con2.table("batting")

    assert batting1.op() == batting2.op()
    assert batting1.equals(batting2)


def test_table_inequality(dbpath):
    con = ibis.sqlite.connect(dbpath)

    batting = con.table("batting")
    functional_alltypes = con.table("functional_alltypes")

    assert batting.op() != functional_alltypes.op()
    assert not batting.equals(functional_alltypes)
