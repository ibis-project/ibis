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
    assert temporary.execute().empty

    if data is not None and isinstance(data, pd.DataFrame):
        con.load_data(temp_table_name, data, if_exists="append")
        result = temporary.execute()
        assert len(result) == len(data.index)
        tm.assert_frame_equal(result, data)

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
@mark.notimpl(["datafusion", "duckdb", "mysql", "postgres", "sqlite"])
def test_sql(ddl_con, sql):
    # execute the expression using SQL query
    ddl_con.sql(sql).execute()


@mark.notimpl(["clickhouse", "datafusion"])
def test_create_table_from_schema(con, new_schema, temp_table):
    con.create_table(temp_table, schema=new_schema)

    t = con.table(temp_table)

    for k, i_type in t.schema().items():
        assert new_schema[k] == i_type


@mark.notimpl(
    [
        "clickhouse",
        "dask",
        "datafusion",
        "duckdb",
        "mysql",
        "pandas",
        "postgres",
        "sqlite",
    ]
)
def test_rename_table(con, temp_table, new_schema):
    temp_table_original = f'{temp_table}_original'
    con.create_table(temp_table_original, schema=new_schema)
    try:
        t = con.table(temp_table_original)
        t.rename(temp_table)

        assert con.table(temp_table) is not None
        assert temp_table in con.list_tables()
    finally:
        con.drop_table(temp_table_original, force=True)
        con.drop_table(temp_table, force=True)


@mark.notimpl(["clickhouse", "datafusion"])
@mark.never(["impala", "pyspark"], reason="No non-nullable datatypes")
def test_nullable_input_output(con, temp_table):
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


@mark.notimpl(
    ["clickhouse", "datafusion", "duckdb", "mysql", "postgres", "sqlite"]
)
@mark.notyet(["pyspark"])
def test_create_drop_view(ddl_con, temp_view):
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


@pytest.fixture
def employee_empty_temp_table(alchemy_con, test_employee_schema):
    temp_table_name = f"temp_to_table_{guid()}"
    _create_temp_table_with_schema(
        alchemy_con,
        temp_table_name,
        test_employee_schema,
    )
    try:
        yield temp_table_name
    finally:
        alchemy_con.drop_table(temp_table_name)


@pytest.fixture
def employee_data_1_temp_table(
    alchemy_con,
    test_employee_schema,
    test_employee_data_1,
):
    temp_table_name = f"temp_to_table_{guid()}"
    _create_temp_table_with_schema(
        alchemy_con,
        temp_table_name,
        test_employee_schema,
        data=test_employee_data_1,
    )
    try:
        yield temp_table_name
    finally:
        alchemy_con.drop_table(temp_table_name)


@pytest.fixture
def employee_data_2_temp_table(
    alchemy_con,
    test_employee_schema,
    test_employee_data_2,
):
    temp_table_name = f"temp_to_table_{guid()}"
    _create_temp_table_with_schema(
        alchemy_con,
        temp_table_name,
        test_employee_schema,
        data=test_employee_data_2,
    )
    try:
        yield temp_table_name
    finally:
        alchemy_con.drop_table(temp_table_name)


def test_insert_no_overwrite_from_dataframe(
    alchemy_con,
    test_employee_data_2,
    employee_empty_temp_table,
):
    temporary = alchemy_con.table(employee_empty_temp_table)
    alchemy_con.insert(
        employee_empty_temp_table,
        obj=test_employee_data_2,
        overwrite=False,
    )
    result = temporary.execute()
    assert len(result) == 3
    tm.assert_frame_equal(result, test_employee_data_2)


def test_insert_overwrite_from_dataframe(
    alchemy_con,
    employee_data_1_temp_table,
    test_employee_data_2,
):
    temporary = alchemy_con.table(employee_data_1_temp_table)

    alchemy_con.insert(
        employee_data_1_temp_table,
        obj=test_employee_data_2,
        overwrite=True,
    )
    result = temporary.execute()
    assert len(result) == 3
    tm.assert_frame_equal(result, test_employee_data_2)


def test_insert_no_overwite_from_expr(
    alchemy_con,
    employee_empty_temp_table,
    employee_data_2_temp_table,
):
    temporary = alchemy_con.table(employee_empty_temp_table)
    from_table = alchemy_con.table(employee_data_2_temp_table)

    alchemy_con.insert(
        employee_empty_temp_table,
        obj=from_table,
        overwrite=False,
    )
    result = temporary.execute()
    assert len(result) == 3
    tm.assert_frame_equal(result, from_table.execute())


def test_insert_overwrite_from_expr(
    alchemy_con,
    employee_data_1_temp_table,
    employee_data_2_temp_table,
):
    temporary = alchemy_con.table(employee_data_1_temp_table)
    from_table = alchemy_con.table(employee_data_2_temp_table)

    alchemy_con.insert(
        employee_data_1_temp_table,
        obj=from_table,
        overwrite=True,
    )
    result = temporary.execute()
    assert len(result) == 3
    tm.assert_frame_equal(result, from_table.execute())


def test_list_databases(alchemy_con):
    # Every backend has its own databases
    TEST_DATABASES = {
        'sqlite': ['main'],
        'postgres': ['postgres', 'ibis_testing'],
        'mysql': ['ibis_testing', 'information_schema'],
        'duckdb': ['information_schema', 'main', 'temp'],
    }
    assert alchemy_con.list_databases() == TEST_DATABASES[alchemy_con.name]


def test_list_schemas(alchemy_con):
    with pytest.warns(FutureWarning):
        alchemy_con.list_schemas()


def test_verify(ddl_backend, ddl_con):
    expr = ddl_con.table('functional_alltypes').double_col.sum()

    with pytest.warns(FutureWarning):
        assert expr.verify()

    with pytest.warns(FutureWarning):
        assert ddl_backend.api.verify(expr)


@pytest.mark.never(["duckdb"], reason="duckdb supports approximate median")
def test_not_verify(alchemy_con, alchemy_backend):
    # There is no expression that can't be compiled to any backend
    # Testing `not verify()` only for an expression not supported in postgres
    expr = alchemy_con.table('functional_alltypes').double_col.approx_median()
    with pytest.warns(FutureWarning):
        assert not expr.verify()

    with pytest.warns(FutureWarning):
        assert not alchemy_backend.api.verify(expr)
