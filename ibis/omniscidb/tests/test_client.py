import pathlib
from typing import Optional

import mock
import pandas as pd
import pytest
from pkg_resources import get_distribution, parse_version
from pytest import param

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
    con, alltypes, test_data_db, table_name, database
):
    tmp_db = test_data_db
    con.create_table(table_name, alltypes, database=tmp_db)

    assert con.exists_table(table_name, database=tmp_db)
    assert not con.exists_table(table_name, database=database)


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


def test_add_zero_columns(table):
    cols_with_types = {}
    with pytest.raises(com.IbisInputError):
        table.add_columns(cols_with_types)


@pytest.mark.parametrize(
    'cols_with_types',
    [{'e': 'float64'}, {'e': 'string', 'f': 'point', 'g': 'polygon'}],
)
def test_add_columns(con, table, cols_with_types):
    schema_before = table.schema()

    table.add_columns(cols_with_types)

    res_tbl = con.table(table.name)

    schema_new_cols = ibis.schema(cols_with_types.items())
    old_schema_with_new_cols = schema_before.append(schema_new_cols)

    assert res_tbl.schema() == old_schema_with_new_cols


def test_drop_zero_columns(table):
    column_names = []
    with pytest.raises(com.IbisInputError):
        table.drop_columns(column_names)


@pytest.mark.parametrize('column_names', [['a'], ['a', 'b', 'c']])
def test_drop_columns(con, table, column_names):
    schema_before = table.schema()

    table.drop_columns(column_names)

    res_tbl = con.table(table.name)
    schema_with_dropped_cols = schema_before.delete(column_names)

    assert res_tbl.schema() == schema_with_dropped_cols


@pytest.mark.parametrize(
    'properties',
    [
        param(dict(), id='none',),
        param(dict(fragment_size=10000000), id='frag_size'),
        param(
            dict(fragment_size=0),
            id='frag_size0',
            marks=pytest.mark.xfail(
                raises=Exception,
                reason="FRAGMENT_SIZE must be a positive number",
            ),
        ),
        param(dict(max_rows=5), id='max_rows'),
        param(
            dict(max_rows=0),
            id='max_rows0',
            marks=pytest.mark.xfail(
                raises=Exception, reason="MAX_ROWS must be a positive number",
            ),
        ),
        param(
            dict(fragment_size=10000000, max_rows=5), id='frag_size-max_rows'
        ),
    ],
)
def test_create_table_schema(con, table_name, properties):
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

    con.create_table(table_name, schema=schema, **properties)

    t = con.table(table_name)

    for k, i_type in t.schema().items():
        assert schema[k] == i_type


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
    ["/tmp/test_read_csv.csv", pathlib.Path("/tmp/test_read_csv.csv")],
)
def test_read_csv(con, table_name, filename, alltypes, df_alltypes):
    schema = alltypes.schema()
    con.create_table(table_name, schema=schema)

    # prepare csv file inside omnisci docker container
    # if the file exists, then it will be overwritten
    con._execute(
        "COPY (SELECT * FROM functional_alltypes) TO '{}'".format(filename)
    )

    db = con.database()
    table = db.table(table_name)
    table.read_csv(filename, header=False, quotechar='"', delimiter=",")
    df_read_csv = table.execute()

    pd.testing.assert_frame_equal(df_alltypes, df_read_csv)


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


@pytest.mark.parametrize(
    'table_src',
    [
        ibis.schema(
            [('a', 'polygon'), ('b', 'point'), ('c', 'int8'), ('d', 'double')]
        ),
        ibis.table(
            schema=ibis.schema(
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
            ),
            name='functional_alltypes',
        ),
        param(
            pd.DataFrame(),
            marks=pytest.mark.xfail(reason="Not implemented yet."),
        ),
        param(
            pd.DataFrame([(1, 2.0, 'string'), (3, 4.0, 'string')]),
            marks=pytest.mark.xfail(reason="Not implemented yet."),
        ),
    ],
)
def test_create_table(con, table_name, table_src):

    if isinstance(table_src, ibis.Schema):
        con.create_table(table_name, schema=table_src)
    else:
        table_src.name = table_name
        con.create_table(table_name, obj=table_src)

    assert con.exists_table(table_name)
    if isinstance(table_src, ibis.Schema):
        assert con.get_schema(table_name) == table_src
    elif isinstance(table_src, pd.DataFrame):
        assert con.get_schema(table_name) == ibis.schema(
            types=table_src.dtypes()
        )
    elif isinstance(table_src, ibis.expr.types.TableExpr):
        assert con.get_schema(table_name) == table_src.schema()


@pytest.mark.parametrize('is_super', [False, True])
def test_create_user(con, user, is_super):
    name, password = user
    con.create_user(name, password, is_super)

    # if is_super=False, then it should throw an exception
    try:
        ibis.omniscidb.connect(
            protocol=con.protocol,
            host=con.host,
            port=con.port,
            user=name,
            password=password,
        )
    except Exception:
        assert not is_super


@pytest.mark.parametrize('is_super', [False, True])
def test_alter_user(con, is_super, user):
    name, password = user
    con.create_user(*user, is_super=False)
    con.alter_user(*user, is_super=is_super)

    # if is_super=False, then it should throw an exception
    try:
        ibis.omniscidb.connect(
            protocol=con.protocol,
            host=con.host,
            port=con.port,
            user=name,
            password=password,
        )
    except Exception:
        assert not is_super
