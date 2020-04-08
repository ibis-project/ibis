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


def test_add_zero_columns(test_table):
    cols_with_types = {}
    with pytest.raises(com.IbisInputError):
        test_table.add_columns(cols_with_types)


@pytest.mark.parametrize(
    'cols_with_types',
    [{'e': 'float64'}, {'e': 'string', 'f': 'point', 'g': 'polygon'}],
)
def test_add_columns(con, test_table, cols_with_types):
    schema_before = test_table.schema()

    test_table.add_columns(cols_with_types)

    res_tbl = con.table(test_table.name)

    schema_new_cols = ibis.schema(cols_with_types.items())
    old_schema_with_new_cols = schema_before.append(schema_new_cols)

    assert res_tbl.schema() == old_schema_with_new_cols


def test_drop_zero_columns(test_table):
    column_names = []
    with pytest.raises(com.IbisInputError):
        test_table.drop_columns(column_names)


@pytest.mark.xfail(
    reason="'ALTER TABLE <tbl_name> DROP <col_name>' is not supported yet by omnisci backend."
)
@pytest.mark.parametrize('column_names', [['a'], ['a', 'b', 'c']])
def test_drop_columns(con, test_table, column_names):
    schema_before = test_table.schema()

    test_table.drop_columns(column_names)

    res_tbl = con.table('test_table')
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
def test_create_table_schema(con, temp_table, properties):
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

    con.create_table(temp_table, schema=schema, **properties)

    t = con.table(temp_table)

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
def test_read_csv(con, temp_table, filename, alltypes, df_alltypes):
    schema = alltypes.schema()
    con.create_table(temp_table, schema=schema)

    # prepare csv file inside omnisci docker container
    # if the file exists, then it will be overwritten
    con._execute(
        "COPY (SELECT * FROM functional_alltypes) TO '{}'".format(filename)
    )

    db = con.database()
    table = db.table(temp_table)
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
def test_create_table(con, temp_table, table_src):

    if isinstance(table_src, ibis.Schema):
        con.create_table(temp_table, schema=table_src)
    else:
        table_src.name = temp_table
        con.create_table(temp_table, obj=table_src)

    assert con.exists_table(temp_table)
    if isinstance(table_src, ibis.Schema):
        assert con.get_schema(temp_table) == table_src
    elif isinstance(table_src, pd.DataFrame):
        assert con.get_schema(temp_table) == ibis.schema(
            types=table_src.dtypes()
        )
    elif isinstance(table_src, ibis.expr.types.TableExpr):
        assert con.get_schema(temp_table) == table_src.schema()


@pytest.mark.parametrize('force', [False, True])
def test_drop_table(con, temp_table, test_schema, force):
    _drop(
        con,
        temp_table,
        'table',
        dict({'force': force}),
        dict({'schema': test_schema}),
    )


def test_truncate_table(con, temp_table):
    con.create_table(temp_table, schema=con.get_schema('functional_alltypes'))
    # TODO: when #2119 will be merged, replase 'con._execute()'
    # with calling 'con.insert_into_select()'
    con._execute(
        "INSERT INTO {} SELECT * FROM functional_alltypes".format(temp_table)
    )

    db = con.database()
    table = db.table(temp_table)

    df_before, schema_before = table.execute(), table.schema()
    con.truncate_table(temp_table)
    df_after, schema_after = table.execute(), table.schema()

    assert con.exists_table(temp_table)
    assert schema_before == schema_after
    assert df_before.shape[0] != 0 and df_after.shape[0] == 0


@pytest.mark.parametrize(
    'expr',
    [
        [],
        ['invalid_collumn_name'],
        ['index', 'invalid_collumn_name'],
        [
            'index',
            'id',
            'bool_col',
            'tinyint_col',
            'float_col',
            'double_col',
            'string_col',
            'timestamp_col',
        ],
    ],
)
def test_create_view(con, temp_view, alltypes, expr):
    df_alltypes = alltypes.execute()

    # if list with selected cols contains invalid names
    # 'create_view' should raise exception
    try:
        con.create_view(temp_view, alltypes[expr])
    except com.IbisTypeError:
        assert not set(expr).issubset(df_alltypes.columns)
        return

    df_view = con._execute('SELECT * from {};'.format(temp_view)).to_df()

    # when list with selected columns is empty
    # in SQL notations it means - select all cols,
    # DataFrame[`empty_list`] means - select nothing,
    # so there is some logic here to properly process that situation
    pd.testing.assert_frame_equal(
        df_alltypes[expr if expr != [] else df_alltypes.columns],
        df_view,
        check_dtype=False,
    )


@pytest.mark.parametrize('force', [False, True])
def test_drop_view(con, test_view, force):
    assert con.exists_table(test_view)
    con.drop_view(test_view, force=force)
    assert not con.exists_table(test_view)

    # trying to drop non existing view and see,
    # if an exception occurred with force=False
    try:
        con.drop_view(test_view, force=force)
    except Exception:
        assert not force


@pytest.mark.parametrize('is_super', [False, True])
def test_create_user(con, test_user, is_super):
    name, password = test_user
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
def test_alter_user(con, is_super, test_user):
    name, password = test_user
    con.create_user(*test_user, is_super=False)
    con.alter_user(*test_user, is_super=is_super)

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


@pytest.mark.xfail(reason='\'exists_database\' not supported yet')
def test_create_database(con, temp_database):
    assert not con.exists_database(temp_database)
    con.create_database(temp_database)
    assert con.exists_database(temp_database)


@pytest.mark.xfail(reason='\'exists_database\' not supported yet')
@pytest.mark.parametrize('force', [False, True])
def test_drop_database(con, temp_database, force):
    _drop(con, temp_database, 'database', dict({'force': force}), dict())


def _drop(con, name, method_name, drop_kwargs, create_kwargs):
    # trying to drop non existing obj and see,
    # if an exception occurred with force=False
    try:
        getattr(con, 'drop_' + method_name)(name, **drop_kwargs)
    except Exception:
        assert not drop_kwargs['force']

    getattr(con, 'create_' + method_name)(name, **create_kwargs)
    assert getattr(con, 'exists_' + method_name)(name)
    getattr(con, 'drop_' + method_name)(name, **drop_kwargs)
    assert not getattr(con, 'exists_' + method_name)(name)
