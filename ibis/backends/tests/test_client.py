import pandas as pd
import pandas.testing as tm
import pytest
from pytest import mark

import ibis
import ibis.expr.datatypes as dt
from ibis.util import guid


@pytest.fixture
def new_schema():
    return ibis.schema([('a', 'string'), ('b', 'bool'), ('c', 'int32')])


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


def test_load_data_sqlalchemy(
    alchemy_backend, alchemy_con, alchemy_temp_table
):
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
    alchemy_con.create_table(alchemy_temp_table, schema=sch)
    alchemy_con.load_data(alchemy_temp_table, df, if_exists='append')
    result = alchemy_con.table(alchemy_temp_table).execute()

    alchemy_backend.assert_frame_equal(df, result)


@mark.parametrize(
    ('expr_fn', 'expected'),
    [
        (lambda t: t.string_col, [('string_col', dt.String)]),
        (
            lambda t: t[t.string_col, t.bigint_col],
            [('string_col', dt.String), ('bigint_col', dt.Int64)],
        ),
    ],
)
@mark.notimpl(["datafusion"])
def test_query_schema(ddl_backend, ddl_con, expr_fn, expected):
    expr = expr_fn(ddl_backend.functional_alltypes)

    # we might need a public API for it
    ast = ddl_con.compiler.to_ast(expr, ddl_backend.make_context())
    schema = ddl_con.ast_schema(ast)

    # clickhouse columns has been defined as non-nullable
    # whereas other backends don't support non-nullable columns yet
    expected = ibis.schema(
        [
            (name, dtype(nullable=schema[name].nullable))
            for name, dtype in expected
        ]
    )
    assert schema.equals(expected)


@mark.parametrize(
    'sql',
    [
        'select * from functional_alltypes limit 10',
        'select * from functional_alltypes \nlimit 10\n',
    ],
)
@mark.notimpl(["postgres", "mysql", "datafusion", "sqlite"])
def test_sql(ddl_backend, ddl_con, sql):
    # execute the expression using SQL query
    ddl_con.sql(sql).execute()


@mark.notimpl(["datafusion", "clickhouse"])
def test_create_table_from_schema(rw_con, rw_backend, new_schema, temp_table):
    rw_con.create_table(temp_table, schema=new_schema)

    t = rw_con.table(temp_table)

    for k, i_type in t.schema().items():
        assert new_schema[k] == i_type


@mark.notimpl(
    [
        "postgres",
        "sqlite",
        "mysql",
        "pandas",
        "dask",
        "datafusion",
        "clickhouse",
    ]
)
def test_rename_table(rw_con, temp_table, new_schema):
    temp_table_original = f'{temp_table}_original'
    rw_con.create_table(temp_table_original, schema=new_schema)

    t = rw_con.table(temp_table_original)
    t.rename(temp_table)

    assert rw_con.table(temp_table) is not None
    assert temp_table in rw_con.list_tables()


@mark.notimpl(["datafusion", "clickhouse"])
@mark.never(["impala", "pyspark"], reason="No non-nullable datatypes")
def test_nullable_input_output(rw_con, temp_table):
    sch = ibis.schema(
        [
            ('foo', 'int64'),
            ('bar', ibis.expr.datatypes.int64(nullable=False)),
            ('baz', 'boolean'),
        ]
    )

    rw_con.create_table(temp_table, schema=sch)

    t = rw_con.table(temp_table)

    assert t.schema().types[0].nullable
    assert not t.schema().types[1].nullable
    assert t.schema().types[2].nullable


# view tests


@mark.only_on_backends(["impala"])
def test_create_drop_view(ddl_con, ddl_backend, temp_view):
    # setup
    table_name = 'functional_alltypes'
    expr = ddl_con.table(table_name).limit(1)

    # create a new view
    ddl_con.create_view(temp_view, expr)
    # check if the view was created
    assert temp_view in ddl_con.list_tables()

    t_expr = ddl_con.table(table_name)
    v_expr = ddl_con.table(temp_view)
    # check if the view and the table has the same fields
    assert set(t_expr.schema().names) == set(v_expr.schema().names)


@mark.notimpl(["postgres", "mysql", "clickhouse", "datafusion"])
def test_separate_database(
    ddl_con, alternate_current_database, current_data_db
):
    # using alternate_current_database switches "con" current
    #  database to a temporary one until a test is over
    tmp_db = ddl_con.database(alternate_current_database)
    # verifying we can open another db which isn't equal to current
    db = ddl_con.database(current_data_db)
    assert db.name == current_data_db
    assert tmp_db.name == alternate_current_database


def test_insert_no_overwrite_from_dataframe(
    alchemy_backend, alchemy_con, test_employee_schema, test_employee_data_2
):
    temp_table = f'temp_to_table_{guid()}'
    temporary = _create_temp_table_with_schema(
        alchemy_con,
        temp_table,
        test_employee_schema,
    )

    alchemy_con.insert(temp_table, obj=test_employee_data_2, overwrite=False)
    assert len(temporary.execute()) == 3
    tm.assert_frame_equal(temporary.execute(), test_employee_data_2)


def test_insert_overwrite_from_dataframe(
    alchemy_backend,
    alchemy_con,
    test_employee_schema,
    test_employee_data_1,
    test_employee_data_2,
):

    temp_table = f'temp_to_table_{guid()}'
    temporary = _create_temp_table_with_schema(
        alchemy_con,
        temp_table,
        test_employee_schema,
        data=test_employee_data_1,
    )

    alchemy_con.insert(temp_table, obj=test_employee_data_2, overwrite=True)
    assert len(temporary.execute()) == 3
    tm.assert_frame_equal(temporary.execute(), test_employee_data_2)


def test_insert_no_overwite_from_expr(
    alchemy_backend, alchemy_con, test_employee_schema, test_employee_data_2
):
    temp_table = f'temp_to_table_{guid()}'
    temporary = _create_temp_table_with_schema(
        alchemy_con,
        temp_table,
        test_employee_schema,
    )

    from_table_name = f'temp_from_table_{guid()}'
    from_table = _create_temp_table_with_schema(
        alchemy_con,
        from_table_name,
        test_employee_schema,
        data=test_employee_data_2,
    )

    alchemy_con.insert(temp_table, obj=from_table, overwrite=False)
    assert len(temporary.execute()) == 3
    tm.assert_frame_equal(temporary.execute(), from_table.execute())


def test_insert_overwrite_from_expr(
    alchemy_backend,
    alchemy_con,
    test_employee_schema,
    test_employee_data_1,
    test_employee_data_2,
):
    temp_table = f'temp_to_table_{guid()}'
    temporary = _create_temp_table_with_schema(
        alchemy_con,
        temp_table,
        test_employee_schema,
        data=test_employee_data_1,
    )

    from_table_name = f'temp_from_table_{guid()}'
    from_table = _create_temp_table_with_schema(
        alchemy_con,
        from_table_name,
        test_employee_schema,
        data=test_employee_data_2,
    )

    alchemy_con.insert(temp_table, obj=from_table, overwrite=True)
    assert len(temporary.execute()) == 3
    tm.assert_frame_equal(temporary.execute(), from_table.execute())


def test_list_databases(alchemy_backend, alchemy_con):
    # Every backend has its own databases
    TEST_DATABASES = {
        'sqlite': ['main', 'base'],
        'postgres': ['postgres', 'ibis_testing'],
        'mysql': ['ibis_testing', 'information_schema'],
    }
    assert alchemy_con.list_databases() == TEST_DATABASES[alchemy_con.name]


@mark.never(
    ["postgres"], reason="schemas and databases are different in postgres"
)
def test_list_schemas(alchemy_backend, alchemy_con):
    with pytest.warns(FutureWarning):
        schemas = alchemy_con.list_schemas()
    assert schemas == alchemy_con.list_databases()


def test_verify(ddl_backend, ddl_con):
    expr = ddl_con.table('functional_alltypes').double_col.sum()

    with pytest.warns(FutureWarning):
        assert expr.verify()

    with pytest.warns(FutureWarning):
        assert ddl_backend.api.verify(expr)


@mark.only_on_backends(["postgres"])
def test_not_verify(alchemy_con, alchemy_backend):
    # There is no expression that can't be compiled to any backend
    # Testing `not verify()` only for an expression not supported in postgres
    expr = alchemy_con.table('functional_alltypes').double_col.approx_median()
    with pytest.warns(FutureWarning):
        assert not expr.verify()

    with pytest.warns(FutureWarning):
        assert not alchemy_backend.api.verify(expr)
