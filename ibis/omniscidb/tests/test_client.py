from typing import Optional

import mock
import pandas as pd
import pytest
from pkg_resources import get_distribution, parse_version

import ibis
import ibis.common.exceptions as com
import ibis.expr.types as ir
from ibis.tests.util import assert_equal

pymapd = pytest.importorskip('pymapd')

pytestmark = pytest.mark.omniscidb


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


@pytest.mark.parametrize(
    'dict_cols_with_types',
    [
        {},
        {'c': 'DOUBLE'},
        {'c': 'DOUBLE', 'd': 'TEXT', 'e': 'POINT', 'f': 'POLYGON'},
    ],
)
def test_add_column(con, dict_cols_with_types):
    table_name = 'my_table'

    con.drop_table(table_name, force=True)

    schema = ibis.schema([('a', 'float'), ('b', 'int8')])

    con.create_table(table_name, schema=schema)

    col_count = len(dict_cols_with_types)

    if col_count == 0:
        isException = False
        try:
            con.add_column(table_name, dict_cols_with_types)
        except com.IbisInputError:
            isException = True
        finally:
            con.drop_table(table_name)

        if not isException:
            assert False
    elif col_count == 1:
        schema_for_check = ibis.schema(
            [('a', 'float'), ('b', 'int8'), ('c', 'double')]
        )

        try:
            t = con.table(table_name)

            for k, i_type in t.schema().items():
                assert schema_for_check[k] == i_type
        finally:
            con.drop_table(table_name)
    else:
        schema_for_check = ibis.schema(
            [
                ('a', 'float'),
                ('b', 'int8'),
                ('c', 'double'),
                ('d', 'string'),
                ('e', 'point'),
                ('f', 'polygon'),
            ]
        )

        try:
            t = con.table(table_name)

            for k, i_type in t.schema().items():
                assert schema_for_check[k] == i_type
        finally:
            con.drop_table(table_name)


@pytest.mark.parametrize('column_names', [[], ['a'], ['a', 'b', 'c']])
def test_drop_column(con, column_names):
    table_name = 'my_table'

    con.drop_table(table_name, force=True)

    schema = ibis.schema(
        [('a', 'polygon'), ('b', 'point'), ('c', 'int8'), ('d', 'double')]
    )

    con.create_table(table_name, schema=schema)

    col_count = len(column_names)

    if col_count == 0:
        isException = False
        try:
            con.drop_column(table_name, column_names)
        except com.IbisInputError:
            isException = True
        finally:
            con.drop_table(table_name)

        if not isException:
            assert False
    elif col_count == 1:
        schema_for_check = ibis.schema(
            [('b', 'point'), ('c', 'int8'), ('d', 'double')]
        )

        try:
            t = con.table(table_name)

            for k, i_type in t.schema().items():
                assert schema_for_check[k] == i_type
        finally:
            con.drop_table(table_name)
    else:
        schema_for_check = ibis.schema([('d', 'double')])

        try:
            t = con.table(table_name)

            for k, i_type in t.schema().items():
                assert schema_for_check[k] == i_type
        finally:
            con.drop_table(table_name)


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


@pytest.mark.parametrize('ipc', [None, True, False])
@pytest.mark.parametrize('gpu_device', [None, 1])
def test_cpu_execution_type(
    mocker, con, ipc: Optional[bool], gpu_device: Optional[int]
):
    """Test the combination of ipc and gpu_device parameters for connection."""
    connection_info = {
        'host': con.host,
        'port': con.port,
        'user': con.user,
        'password': con.password,
        'database': con.db_name,
        'protocol': con.protocol,
        'ipc': ipc,
        'gpu_device': gpu_device,
    }

    if gpu_device and ipc is False:
        # test exception
        with pytest.raises(ibis.common.exceptions.IbisInputError):
            ibis.omniscidb.connect(**connection_info)
        return

    mocked_methods = []

    for mock_method_name in ('select_ipc', 'select_ipc_gpu'):
        mocked_method = mock.patch.object(
            pymapd.connection.Connection,
            mock_method_name,
            new=lambda *args, **kwargs: pd.DataFrame({'string_col': ['1']}),
        )

        mocked_method.start()
        mocked_methods.append(mocked_method)

    new_con = ibis.omniscidb.connect(**connection_info)
    assert new_con is not None
    assert new_con.ipc == ipc
    assert new_con.gpu_device == gpu_device

    expr = new_con.table('functional_alltypes')
    expr = expr[['string_col']].limit(1)

    assert expr.execute(ipc=True).shape[0] == 1
    assert expr.execute(ipc=False).shape[0] == 1
    assert expr.execute(ipc=None).shape[0] == 1

    for mocked_method in mocked_methods:
        mocked_method.stop()
