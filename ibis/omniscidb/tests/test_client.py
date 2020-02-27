import pathlib

import pandas as pd
import pytest
from pkg_resources import get_distribution, parse_version

import ibis
import ibis.common.exceptions as com
import ibis.expr.types as ir
from ibis.tests.util import assert_equal

pytestmark = pytest.mark.omniscidb
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
    reason='must have pymapd>=12 to connect to existing session',
)
def test_session_id_connection(session_con):
    new_connection = ibis.omniscidb.connect(
        protocol=session_con.protocol,
        host=session_con.host,
        port=session_con.port,
        session_id=session_con.con._session,
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
    result = ibis.omniscidb.compile(expr)
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
            ('c', 'int8'),
            ('d', 'int16'),
            ('e', 'int32'),
            ('f', 'int64'),
            ('x', 'point'),
            ('y', 'linestring'),
            ('z', 'polygon'),
            ('w', 'multipolygon'),
        ]
    )

    con.create_table(t_name, schema=schema)

    try:
        t = con.table(t_name)

        for k, i_type in t.schema().items():
            assert schema[k] == i_type
    finally:
        con.drop_table(t_name)


@pytest.mark.parametrize(
    'sql',
    [
        'select * from functional_alltypes limit 10--test',
        'select * from functional_alltypes \nlimit 10\n;',
        'select * from functional_alltypes \nlimit 10;',
        'select * from functional_alltypes \nlimit 10;--test',
    ],
)
def test_sql(con, sql):
    # execute the expression using SQL query
    con.sql(sql).execute()


def test_explain(con, alltypes):
    # execute the expression using SQL query
    con.explain(alltypes)


@pytest.mark.parametrize(
    'filename',
    [
        "/omnisci/test_read_csv.csv",
        pathlib.Path("/omnisci/test_read_csv.csv"),
    ],
)
def test_read_csv(con, filename):
    t_name = "test_read_csv"

    con.drop_table(t_name, force=True)
    schema = ibis.schema(
        [
            ('index', 'int64'),
            ('Unnamed__0', 'int64'),
            ('id', 'int32'),
            ('bool_col', 'bool'),
            ('tinyint_col', 'int16'),
            ('smallint_col', 'int16'),
            ('int_col', 'int32'),
            ('bigint_col', 'int64'),
            ('float_col', 'float32'),
            ('double_col', 'double'),
            ('date_string_col', 'string'),
            ('string_col', 'string'),
            ('timestamp_col', 'timestamp'),
            ('year_', 'int32'),
            ('month_', 'int32'),
        ]
    )
    con.create_table(t_name, schema=schema)

    # prepare csv file inside omnisci docker container
    # if the file exists, then it will be overwritten
    con._execute(
        "COPY (SELECT * FROM functional_alltypes) TO '{}'".format(filename)
    )

    try:
        db = con.database()
        table = db.table(t_name)
        table.read_csv(filename, header=False, quotechar='"', delimiter=",")

        df_read_csv = table.execute()
        df_expected = db.table("functional_alltypes").execute()

        pd.testing.assert_frame_equal(df_expected, df_read_csv)
    finally:
        con.drop_table(t_name)
