import pandas as pd
import pandas.testing as tm
import pytest

import ibis
import ibis.expr.datatypes as dt
from ibis.util import guid

SQLALCHEMY_BACKENDS = ['sqlite', 'postgres', 'mysql']


@pytest.fixture
def new_schema():
    return ibis.schema([('a', 'string'), ('b', 'bool'), ('c', 'int32')])


@pytest.mark.only_on_backends(
    SQLALCHEMY_BACKENDS,
    reason="run only if backend is SQLAlchemy based",
)
@pytest.mark.xfail_unsupported
def test_load_data_sqlalchemy(backend, con, temp_table):
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
    if not hasattr(con, 'compiler'):
        pytest.skip(
            '{} backend has no `compiler` attribute'.format(
                type(backend).__name__
            )
        )

    expr = expr_fn(alltypes)

    # we might need a public API for it
    ast = con.compiler.to_ast(expr, backend.make_context())
    schema = con.ast_schema(ast)

    # clickhouse columns has been defined as non-nullable
    # whereas other backends don't support non-nullable columns yet
    expected = ibis.schema(
        [
            (name, dtype(nullable=schema[name].nullable))
            for name, dtype in expected
        ]
    )
    assert schema.equals(expected)


@pytest.mark.parametrize(
    'sql',
    [
        'select * from functional_alltypes limit 10',
        'select * from functional_alltypes \nlimit 10\n',
    ],
)
@pytest.mark.xfail_backends(('bigquery',))
@pytest.mark.xfail_unsupported
def test_sql(backend, con, sql):
    if not hasattr(con, 'sql') or not hasattr(con, '_get_schema_using_query'):
        pytest.skip(f'Backend {backend} does not support sql method')

    # execute the expression using SQL query
    con.sql(sql).execute()


# test table


@pytest.mark.xfail_unsupported
@pytest.mark.xfail_backends(['pandas', 'dask'])
def test_create_table_from_schema(con, backend, new_schema, temp_table):
    # xfailing pandas and dask: #3020
    con.create_table(temp_table, schema=new_schema)

    t = con.table(temp_table)

    for k, i_type in t.schema().items():
        assert new_schema[k] == i_type


@pytest.mark.xfail_unsupported
def test_rename_table(con, backend, temp_table, new_schema):
    if not hasattr(con, 'rename_table'):
        pytest.xfail('{} backend doesn\'t have rename_table method.')

    temp_table_original = f'{temp_table}_original'
    con.create_table(temp_table_original, schema=new_schema)

    t = con.table(temp_table_original)
    t.rename(temp_table)

    assert con.table(temp_table) is not None
    assert temp_table in con.list_tables()


@pytest.mark.xfail_unsupported
@pytest.mark.xfail_backends(['impala', 'pyspark', 'spark', 'pandas', 'dask'])
def test_nullable_input_output(con, backend, temp_table):
    # - Impala, PySpark and Spark non-nullable issues #2138 and #2137
    # xfailing pandas and dask: #3020
    sch = ibis.schema(
        [
            ('foo', 'int64'),
            ('bar', ibis.expr.datatypes.int64(nullable=False)),
            ('baz', 'boolean'),
        ]
    )

    con.create_table(temp_table, schema=sch)

    t = con.table(temp_table)

    assert t.schema().types[0].nullable
    assert not t.schema().types[1].nullable
    assert t.schema().types[2].nullable


# view tests


@pytest.mark.xfail_unsupported
@pytest.mark.xfail_backends(['pyspark', 'spark'])
def test_create_drop_view(con, backend, temp_view):
    # pyspark and spark skipt because table actually is a temporary view

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


@pytest.mark.only_on_backends(
    ['bigquery', 'clickhouse', 'impala', 'omniscidb', 'spark'],
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


def _create_temp_table_with_schema(con, temp_table_name, schema, data=None):

    con.drop_table(temp_table_name, force=True)
    con.create_table(temp_table_name, schema=schema)
    temporary = con.table(temp_table_name)
    assert len(temporary.execute()) == 0

    if data is not None and isinstance(data, pd.DataFrame):
        con.load_data(temp_table_name, data, if_exists='append')
        assert len(temporary.execute()) == len(data.index)
        tm.assert_frame_equal(temporary.execute(), data)

    return temporary


@pytest.mark.only_on_backends(
    SQLALCHEMY_BACKENDS,
    reason="run only if backend is SQLAlchemy based",
)
def test_insert_no_overwrite_from_dataframe(
    con, test_employee_schema, test_employee_data_2
):

    temp_table = f'temp_to_table_{guid()}'
    temporary = _create_temp_table_with_schema(
        con,
        temp_table,
        test_employee_schema,
    )

    con.insert(temp_table, obj=test_employee_data_2, overwrite=False)
    assert len(temporary.execute()) == 3
    tm.assert_frame_equal(temporary.execute(), test_employee_data_2)


@pytest.mark.only_on_backends(
    SQLALCHEMY_BACKENDS,
    reason="run only if backend is SQLAlchemy based",
)
def test_insert_overwrite_from_dataframe(
    con, test_employee_schema, test_employee_data_1, test_employee_data_2
):

    temp_table = f'temp_to_table_{guid()}'
    temporary = _create_temp_table_with_schema(
        con,
        temp_table,
        test_employee_schema,
        data=test_employee_data_1,
    )

    con.insert(temp_table, obj=test_employee_data_2, overwrite=True)
    assert len(temporary.execute()) == 3
    tm.assert_frame_equal(temporary.execute(), test_employee_data_2)


@pytest.mark.only_on_backends(
    SQLALCHEMY_BACKENDS,
    reason="run only if backend is SQLAlchemy based",
)
def test_insert_no_overwite_from_expr(
    con, test_employee_schema, test_employee_data_2
):

    temp_table = f'temp_to_table_{guid()}'
    temporary = _create_temp_table_with_schema(
        con,
        temp_table,
        test_employee_schema,
    )

    from_table_name = f'temp_from_table_{guid()}'
    from_table = _create_temp_table_with_schema(
        con,
        from_table_name,
        test_employee_schema,
        data=test_employee_data_2,
    )

    con.insert(temp_table, obj=from_table, overwrite=False)
    assert len(temporary.execute()) == 3
    tm.assert_frame_equal(temporary.execute(), from_table.execute())


@pytest.mark.only_on_backends(
    SQLALCHEMY_BACKENDS,
    reason="run only if backend is SQLAlchemy based",
)
def test_insert_overwrite_from_expr(
    con, test_employee_schema, test_employee_data_1, test_employee_data_2
):

    temp_table = f'temp_to_table_{guid()}'
    temporary = _create_temp_table_with_schema(
        con,
        temp_table,
        test_employee_schema,
        data=test_employee_data_1,
    )

    from_table_name = f'temp_from_table_{guid()}'
    from_table = _create_temp_table_with_schema(
        con,
        from_table_name,
        test_employee_schema,
        data=test_employee_data_2,
    )

    con.insert(temp_table, obj=from_table, overwrite=True)
    assert len(temporary.execute()) == 3
    tm.assert_frame_equal(temporary.execute(), from_table.execute())


@pytest.mark.only_on_backends(
    SQLALCHEMY_BACKENDS,
    reason="run only if backend is SQLAlchemy based",
)
def test_list_databases(con):
    # Every backend has its own databases
    TEST_DATABASES = {
        'sqlite': ['main', 'base'],
        'postgres': ['postgres', 'ibis_testing'],
        'mysql': ['ibis_testing', 'information_schema'],
    }
    assert con.list_databases() == TEST_DATABASES[con.name]


@pytest.mark.only_on_backends(
    set(SQLALCHEMY_BACKENDS) - {'postgres'},
    reason="run only if backend is SQLAlchemy based, except postgres which "
    "which has schemas different than databases",
)
def test_list_schemas(con):
    with pytest.warns(FutureWarning):
        schemas = con.list_schemas()
    assert schemas == con.list_databases()


def test_verify(con, backend):
    if not hasattr(con, 'compiler'):
        pytest.skip(
            '{} backend has no `compiler` attribute'.format(
                type(backend).__name__
            )
        )
    expr = con.table('functional_alltypes').double_col.sum()

    with pytest.warns(FutureWarning):
        assert expr.verify()

    with pytest.warns(FutureWarning):
        assert backend.api.verify(expr)

    # There is no expression that can't be compiled to any backend
    # Testing `not verify()` only for an expression not supported in postgres
    if backend.api.name == 'postgres':
        expr = con.table('functional_alltypes').double_col.approx_median()
        with pytest.warns(FutureWarning):
            assert not expr.verify()

        with pytest.warns(FutureWarning):
            assert not backend.api.verify(expr)
