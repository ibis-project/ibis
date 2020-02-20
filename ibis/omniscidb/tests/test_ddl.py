""" Tests for geo spatial data types"""
import pytest

import ibis

pymapd = pytest.importorskip('pymapd')
pytestmark = pytest.mark.omniscidb


@pytest.fixture
def new_schema():
    return ibis.schema([('a', 'float'), ('b', 'double'), ('c', 'int32')])


# table tests


def test_create_table_from_schema(con, new_schema, temp_table):
    con.create_table(temp_table, schema=new_schema)

    t = con.table(temp_table)

    for k, i_type in t.schema().items():
        assert new_schema[k] == i_type


def test_rename_table(con, temp_table, new_schema):
    temp_table_original = '{}_original'.format(temp_table)
    con.create_table(temp_table_original, schema=new_schema)

    t = con.table(temp_table_original)
    t.rename(temp_table)

    assert con.table(temp_table) is not None
    assert temp_table in con.list_tables()


# view tests


def test_create_drop_view(con, temp_view):
    # setup
    table_name = 'functional_alltypes'
    expr = con.table(table_name).limit(1)

    # create a new view
    con.create_view(temp_view, expr)
    # check if the view was created
    assert temp_view in con.list_tables()

    t_expr = con.table(table_name)
    v_expr = con.table(temp_view)
    # check if the view and the table has the same fields
    assert set(t_expr.schema().names) == set(v_expr.schema().names)
