import pandas as pd
import pytest
from pkg_resources import get_distribution, parse_version

import ibis
import ibis.common as com
import ibis.expr.types as ir
from ibis.tests.util import assert_equal

pytestmark = pytest.mark.mapd
pytest.importorskip('pymapd')


def test_table(alltypes):
    assert isinstance(alltypes, ir.TableExpr)


def test_array_execute(alltypes):
    d = alltypes.limit(10).double_col
    s = d.execute()
    assert isinstance(s, pd.Series)
    assert len(s) == 10


def test_literal_execute(alltypes):
    expr = alltypes[alltypes, ibis.literal('1234').name('lit')].limit(1)
    result = expr.execute()
    assert result.lit[0] == '1234'


def test_simple_aggregate_execute(alltypes):
    d = alltypes.double_col.sum().name('sum1')
    v = d.execute()
    assert isinstance(v, float)


def test_list_tables(con):
    assert len(con.list_tables()) > 0
    assert len(con.list_tables(like='functional_alltypes')) == 1


@pytest.mark.skipif(
    parse_version(get_distribution('pymapd').version) < parse_version('0.12'),
    reason='must have pymapd>=12 to connect to existing session'
)
def test_session_id_connection(session_con):
    new_connection = ibis.mapd.connect(
        protocol=session_con.protocol,
        host=session_con.host,
        port=session_con.port,
        session_id=session_con.con._session
    )
    assert new_connection.list_tables()


def test_compile_verify(alltypes):
    supported_expr = alltypes.double_col.sum()
    assert supported_expr.verify()


def test_database_layer(con, alltypes):
    db = con.database()
    t = db.functional_alltypes

    assert_equal(t, alltypes)

    assert db.list_tables() == con.list_tables()


def test_compile_toplevel():
    t = ibis.table([('foo', 'double')], name='t0')
    expr = t.foo.sum()
    result = ibis.mapd.compile(expr)
    expected = 'SELECT sum("foo") AS "sum"\nFROM t0'  # noqa
    assert str(result) == expected


def text_exists_table_with_database(
    con, alltypes, test_data_db, temp_table, temp_database
):
    tmp_db = test_data_db
    con.create_table(temp_table, alltypes, database=tmp_db)

    assert con.exists_table(temp_table, database=tmp_db)
    assert not con.exists_table(temp_table, database=temp_database)


def test_union_op(alltypes):
    t1 = alltypes
    t2 = alltypes
    expr = t1.union(t2)

    with pytest.raises(com.UnsupportedOperationError):
        expr.compile()

    t1 = alltypes.head()
    t2 = alltypes.head()
    expr = t1.union(t2)
    with pytest.raises(com.UnsupportedOperationError):
        expr.compile()


def test_create_table_schema(con):
    t_name = 'mytable'

    con.drop_table(t_name, force=True)

    schema = ibis.schema(
        [
            ('a', 'float'),
            ('b', 'double'),
            ('c', 'int32'),
            ('d', 'int64'),
            ('x', 'point'),
            ('y', 'linestring'),
            ('z', 'polygon'),
            ('w', 'multipolygon'),
        ]
    )

    con.create_table(t_name, schema=schema)

    try:
        t = con.table(t_name)

        assert isinstance(t.a, ir.FloatingColumn)
        assert isinstance(t.b, ir.FloatingColumn)
        assert isinstance(t.c, ir.IntegerColumn)
        assert isinstance(t.d, ir.IntegerColumn)
        assert isinstance(t.x, ir.PointColumn)
        assert isinstance(t.y, ir.LineStringColumn)
        assert isinstance(t.z, ir.PolygonColumn)
        assert isinstance(t.w, ir.MultiPolygonColumn)
    finally:
        con.drop_table(t_name)
