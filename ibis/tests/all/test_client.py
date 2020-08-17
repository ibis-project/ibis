import pandas as pd
import pytest
from pkg_resources import parse_version

import ibis
import ibis.expr.datatypes as dt
import ibis.common.exceptions as com
from ibis.tests.backends import (
    BigQuery,
    Clickhouse,
    Impala,
    OmniSciDB,
    PySpark,
    Spark,
)


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


@pytest.fixture
def new_schema():
    return ibis.schema([('a', 'string'), ('b', 'bool'), ('c', 'int32')])


@pytest.mark.xfail_unsupported
def test_load_data_sqlalchemy(backend, con, temp_table):
    if not isinstance(con.dialect(), ibis.sql.alchemy.AlchemyDialect):
        pytest.skip('{} is not a SQL Alchemy Client.'.format(backend.name))

    sch = ibis.schema(
        [
            ('first_name', 'string'),
            ('last_name', 'string'),
            ('department_name', 'string'),
            ('salary', 'float64'),
        ]
    )

    df = pd.DataFrame(
        {
            'first_name': ['A', 'B', 'C'],
            'last_name': ['D', 'E', 'F'],
            'department_name': ['AA', 'BB', 'CC'],
            'salary': [100.0, 200.0, 300.0],
        }
    )
    con.create_table(temp_table, schema=sch)
    con.load_data(temp_table, df, if_exists='append')
    result = con.table(temp_table).execute()

    backend.assert_frame_equal(df, result)


@pytest.mark.xfail_unsupported
def test_version(backend, con):
    expected_type = (
        type(parse_version('1.0')),
        type(parse_version('1.0-legacy')),
    )
    assert isinstance(con.version, expected_type)


@pytest.mark.parametrize(
    ('expr_fn', 'expected'),
    [
        (lambda t: t.string_col, [('string_col', dt.String)]),
        (
            lambda t: t[t.string_col, t.bigint_col],
            [('string_col', dt.String), ('bigint_col', dt.Int64)],
        ),
    ],
)
def test_query_schema(backend, con, alltypes, expr_fn, expected):
    if not hasattr(con, '_build_ast'):
        pytest.skip(
            '{} backend has no _build_ast method'.format(
                type(backend).__name__
            )
        )

    expr = expr_fn(alltypes)

    # we might need a public API for it
    ast = con._build_ast(expr, backend.make_context())
    query = con.query_class(con, ast)
    schema = query.schema()

    # clickhouse columns has been defined as non-nullable
    # whereas other backends don't support non-nullable columns yet
    expected = ibis.schema(
        [
            (name, dtype(nullable=schema[name].nullable))
            for name, dtype in expected
        ]
    )
    assert query.schema().equals(expected)


@pytest.mark.parametrize(
    'sql',
    [
        'select * from functional_alltypes limit 10',
        'select * from functional_alltypes \nlimit 10\n',
    ],
)
@pytest.mark.xfail_backends((BigQuery,))
@pytest.mark.xfail_unsupported
def test_sql(backend, con, sql):
    if not hasattr(con, 'sql') or not hasattr(con, '_get_schema_using_query'):
        pytest.skip('Backend {} does not support sql method'.format(backend))

    # execute the expression using SQL query
    con.sql(sql).execute()


# test table


@pytest.mark.xfail_unsupported
def test_create_table_from_schema(con, backend, new_schema, table_name):
    if not hasattr(con, 'create_table') or not hasattr(con, 'drop_table'):
        pytest.xfail(
            '{} backend doesn\'t have create_table or drop_table methods.'
        )

    con.create_table(table_name, schema=new_schema)

    t = con.table(table_name)

    for k, i_type in t.schema().items():
        assert new_schema[k] == i_type


@pytest.mark.xfail_unsupported
def test_rename_table(con, backend, table_name, new_schema):
    if not hasattr(con, 'rename_table'):
        pytest.xfail('{} backend doesn\'t have rename_table method.')

    temp_table_original = '{}_original'.format(table_name)
    con.create_table(temp_table_original, schema=new_schema)

    t = con.table(temp_table_original)
    t.rename(table_name)

    assert con.table(table_name) is not None
    assert table_name in con.list_tables()


@pytest.mark.xfail_unsupported
@pytest.mark.xfail_backends([Impala, PySpark, Spark])
def test_nullable_input_output(con, backend, table_name):
    # - Impala, PySpark and Spark non-nullable issues #2138 and #2137
    if not hasattr(con, 'create_table') or not hasattr(con, 'drop_table'):
        pytest.xfail(
            '{} backend doesn\'t have create_table or drop_table methods.'
        )

    sch = ibis.schema(
        [
            ('foo', 'int64'),
            ('bar', ibis.expr.datatypes.int64(nullable=False)),
            ('baz', 'boolean*'),
        ]
    )

    con.create_table(table_name, schema=sch)

    t = con.table(table_name)

    assert t.schema().types[0].nullable
    assert not t.schema().types[1].nullable
    assert t.schema().types[2].nullable


@pytest.mark.parametrize('force', [False, True])
def test_drop_table(con, table_name, schema, force):
    _drop(
        con,
        name=table_name,
        method_name='table',
        drop_kwargs={'force': force},
        create_kwargs={'schema': schema},
    )


def test_truncate_table(con, table_name):
    con.create_table(table_name, schema=con.get_schema('functional_alltypes'))
    con._execute(
        "INSERT INTO {} SELECT * FROM functional_alltypes".format(table_name)
    )

    db = con.database()
    table = db.table(table_name)

    df_before, schema_before = table.execute(), table.schema()
    con.truncate_table(table_name)
    df_after, schema_after = table.execute(), table.schema()

    assert con.exists_table(table_name)
    assert schema_before == schema_after
    assert df_before.shape[0] != 0 and df_after.shape[0] == 0


# view tests


@pytest.mark.xfail_unsupported
@pytest.mark.xfail_backends([PySpark, Spark])
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
def test_create_view(con, view, alltypes, expr):
    df_alltypes = alltypes.execute()

    # if list with selected cols contains invalid names
    # 'create_view' should raise exception
    try:
        con.create_view(view, alltypes[expr])
    except com.IbisTypeError:
        assert not set(expr).issubset(df_alltypes.columns)
        return

    df_view = con._execute('SELECT * from {};'.format(view)).to_df()

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
def test_drop_view(con, new_view, force):
    assert con.exists_table(new_view)
    con.drop_view(new_view, force=force)
    assert not con.exists_table(new_view)

    # trying to drop non existing view and see,
    # if an exception occurred with force=False
    try:
        con.drop_view(new_view, force=force)
    except Exception:
        assert not force


@pytest.mark.only_on_backends(
    [BigQuery, Clickhouse, Impala, OmniSciDB, Spark, BigQuery],
    reason="run only if backend is sql-based",
)
def test_separate_database(con, alternate_current_database, current_data_db):
    # using alternate_current_database switches "con" current
    #  database to a temporary one until a test is over
    tmp_db = con.database(alternate_current_database)
    # verifying we can open another db which isn't equal to current
    db = con.database(current_data_db)
    assert db.name == current_data_db
    assert tmp_db.name == alternate_current_database


# database test


@pytest.mark.xfail_backends([OmniSciDB])
def test_create_database(con, database):
    assert not con.exists_database(database)
    con.create_database(database)
    assert con.exists_database(database)


@pytest.mark.xfail_backends([OmniSciDB])
@pytest.mark.parametrize('force', [False, True])
def test_drop_database(con, database, force):
    con.create_database(database)
    assert con.exists_database(database)
    _drop(
        con,
        name=database,
        method_name='database',
        drop_kwargs={'force': force},
        create_kwargs={},
    )
    assert not con.exists_database(database)
